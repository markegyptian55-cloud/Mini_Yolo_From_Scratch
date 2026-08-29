"""Training loop for MiniYOLO-v2."""
import csv
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.v2.engine.validator import Validator
from src.v2.losses.loss import DetectionLoss
from src.v2.utils.ema import ModelEMA, de_parallel
from src.v2.utils.general import AverageMeter, colorstr, cosine_lr
from src.v2.utils.optim import build_optimizer, prog_alpha


class Trainer:
    def __init__(self, model, train_loader, val_loader, hyp, args, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hyp = hyp
        self.args = args
        self.device = device
        self.save_dir = Path(save_dir)
        self.wdir = self.save_dir / "weights"
        self.wdir.mkdir(parents=True, exist_ok=True)

        self.epochs = args.epochs
        self.nb = len(train_loader)
        self.nbs = int(hyp.get("nominal_batch", 64))
        self.accumulate = max(1, round(self.nbs / args.batch))
        # weight decay is specified for the nominal batch; scale it to the real one
        wd = float(hyp["weight_decay"]) * args.batch * self.accumulate / self.nbs

        self.optimizer = build_optimizer(model, name=args.optimizer,
                                         lr=float(hyp["lr0"]),
                                         momentum=float(hyp["momentum"]),
                                         weight_decay=wd,
                                         muon_ratio=float(hyp.get("muon_ratio", 0.5)))
        self.lf = cosine_lr(float(hyp["lrf"]), self.epochs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

        self.amp = bool(args.amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.ema = ModelEMA(model, decay=float(hyp.get("ema_decay", 0.9999)),
                            tau=float(hyp.get("ema_tau", 2000)))
        self.criterion = DetectionLoss(model, hyp, device)

        self.validator = Validator(val_loader, model.nc, model.names, device,
                                   conf=float(args.val_conf), iou=float(args.val_iou),
                                   max_det=args.max_det, e2e=not args.no_e2e)

        self.best_fitness = 0.0
        self.start_epoch = 0
        self.patience = args.patience
        self.no_improve = 0
        # Rule 1 layout: when the experiment folder already contains its
        # "REPORTS EXPI-<N>" directory, the epoch log belongs inside it as
        # training_log.csv, so the run lands in the final structure with nothing to
        # move afterwards. Otherwise fall back to the flat results.csv.
        rep = next(iter(sorted(self.save_dir.glob("REPORTS EXPI-*"))), None)
        self.report_dir = rep
        self.csv_path = (rep / "training_log.csv") if rep else (self.save_dir / "results.csv")

        # Per-epoch wall clock, appended after every epoch. Also survives --resume:
        # resume() reloads it from the CSV so the summary covers the whole run, not just
        # the segment after the interruption (experiment 1 was interrupted mid-run).
        self.epoch_times = []
        self.t_wall_start = None

        self.imgsz = args.imgsz
        self.ms_lo = int(round(self.imgsz * float(hyp.get("multi_scale_lo", 1.0)) / 32) * 32)
        self.ms_hi = int(round(self.imgsz * float(hyp.get("multi_scale_hi", 1.0)) / 32) * 32)

    # ------------------------------------------------------------ checkpoint
    def save_ckpt(self, epoch, fitness, best=False, name=None):
        ckpt = {
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "model": de_parallel(self.model).state_dict(),
            "ema": self.ema.ema.state_dict(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "hyp": self.hyp,
            "args": vars(self.args),
            "nc": self.model.nc,
            "names": self.model.names,
            "scale": self.model.scale,
            # reg_max decides the head's output width. Written so every loader can
            # rebuild the right architecture; absent in pre-experiment-3 checkpoints,
            # where the correct fallback is 1 (the old scalar head).
            "reg_max": getattr(de_parallel(self.model), "reg_max", 1),
            "fitness": fitness,
        }
        torch.save(ckpt, self.wdir / "last.pt")
        if best:
            torch.save(ckpt, self.wdir / "best.pt")
        if name:
            torch.save(ckpt, self.wdir / name)

    def resume(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        de_parallel(self.model).load_state_dict(ckpt["model"])
        self.ema.ema.load_state_dict(ckpt["ema"])
        self.ema.updates = ckpt.get("updates", 0)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_fitness = ckpt.get("best_fitness", 0.0)
        self.start_epoch = ckpt["epoch"] + 1
        for _ in range(self.start_epoch):
            self.scheduler.step()
        # Recover the epoch timings already on disk so training_summary.txt reports the
        # true total across the interruption rather than only the resumed segment.
        if self.csv_path.exists():
            try:
                with open(self.csv_path, newline="", encoding="utf-8") as f:
                    self.epoch_times = [float(r["epoch_seconds"]) for r in csv.DictReader(f)
                                        if r.get("epoch_seconds")]
                if self.epoch_times:
                    print(f"{colorstr('resume')} recovered {len(self.epoch_times)} epoch "
                          f"timings ({sum(self.epoch_times) / 3600:.2f} h) from "
                          f"{self.csv_path.name}")
            except Exception:
                self.epoch_times = []
        print(f"{colorstr('resume')} from {path} at epoch {self.start_epoch}")

    def _log_csv(self, row):
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(list(row.keys()))
            w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row.values()])

    def write_training_summary(self, dt_hours, epochs_done, res, interrupted=False):
        """Wall-clock and configuration summary -> REPORTS EXPI-<N>/training_summary.txt.

        `dt_hours` is this process's wall clock. When a run was resumed, the sum of the
        recovered per-epoch timings is the honest total and is reported alongside it --
        the two differ and pretending otherwise would understate a resumed run.
        """
        out = (self.report_dir or self.save_dir) / "training_summary.txt"
        et = self.epoch_times
        aug_keys = ("mosaic", "close_mosaic", "mixup", "scale", "degrees", "shear",
                    "translate", "perspective", "fliplr", "flipud", "hsv_h", "hsv_s",
                    "hsv_v", "gray", "blur", "erasing")
        L = [
            "=" * 74,
            "TRAINING SUMMARY",
            "=" * 74,
            f"run                     : {self.save_dir}",
            f"weights                 : {self.wdir / 'best.pt'}",
            f"finished                : {time.strftime('%Y-%m-%d %H:%M:%S')}"
            + ("   [INTERRUPTED / EARLY STOP]" if interrupted else ""),
            "",
            "-- wall clock -------------------------------------------------------",
            f"this process            : {dt_hours:.3f} h  ({dt_hours * 60:.1f} min)",
        ]
        if et:
            tot = sum(et) / 3600.0
            L += [
                f"sum of epoch timings    : {tot:.3f} h   <- true total across any --resume",
                f"epochs completed        : {len(et)} of {self.epochs}",
                f"mean epoch              : {sum(et) / len(et):.1f} s",
                f"fastest / slowest epoch : {min(et):.1f} s / {max(et):.1f} s",
                f"first / last epoch      : {et[0]:.1f} s / {et[-1]:.1f} s",
            ]
        else:
            L.append("epochs completed        : 0 (no epoch finished)")
        L += [
            "",
            "-- result -----------------------------------------------------------",
            f"best fitness            : {self.best_fitness:.6f}",
            f"epochs run              : {epochs_done} of {self.epochs} requested",
        ]
        for k, label in (("mAP50", "mAP50"), ("mAP50-95", "mAP50-95"),
                         ("precision", "precision"), ("recall", "recall")):
            if res and k in res:
                L.append(f"final val {label:<14s}: {res[k]:.4f}")
        L += [
            "",
            "-- configuration ----------------------------------------------------",
            "Full detail is in args.yaml and hyp.yaml beside this file; the fields that",
            "most often differ between experiments are repeated here so a reader does not",
            "have to open three files to know what was run.",
            "",
            f"scale                   : {self.args.scale}",
            f"imgsz                   : {self.args.imgsz}"
            f"   (multi-scale {self.ms_lo}-{self.ms_hi})",
            f"reg_max                 : {getattr(de_parallel(self.model), 'reg_max', 1)}",
            f"batch / nominal / accum : {self.args.batch} / {self.nbs} / {self.accumulate}",
            f"optimizer               : {self.args.optimizer}",
            f"seed                    : {self.args.seed}",
            f"amp                     : {self.amp}",
            f"train images            : {len(self.train_loader.dataset)}",
            f"val images              : {len(self.val_loader.dataset)}",
            "",
            "loss gains              : " + "  ".join(
                f"{k}={self.hyp.get(k)}" for k in ("box", "cls", "l1", "dfl")),
            "",
            "augmentation            :",
        ]
        for k in aug_keys:
            if k in self.hyp:
                L.append(f"  {k:<20s}  {self.hyp[k]}")
        L.append("")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(L), encoding="utf-8")
        print(f"{colorstr('summary')} {out}")
        return out

    # ------------------------------------------------------------ main loop
    def train(self):
        nw = max(round(float(self.hyp["warmup_epochs"]) * self.nb), 100)
        nw = min(nw, max(self.nb * self.epochs // 2, 1))
        close_mosaic_epoch = self.epochs - int(self.hyp.get("close_mosaic", 0))
        last_opt_step = -1
        t_start = time.time()

        print(f"{colorstr('train')} {self.epochs} epochs, {self.nb} iters/epoch, "
              f"batch={self.args.batch} accumulate={self.accumulate} "
              f"(nominal {self.nbs}), imgsz={self.imgsz} "
              f"multi-scale=[{self.ms_lo},{self.ms_hi}], amp={self.amp}, "
              f"optimizer={self.args.optimizer}")

        for epoch in range(self.start_epoch, self.epochs):
            t_epoch = time.time()
            self.model.train()
            alpha = prog_alpha(epoch, self.epochs,
                               float(self.hyp.get("prog_alpha_init", 0.8)),
                               float(self.hyp.get("prog_alpha_final", 0.1)))

            if epoch == close_mosaic_epoch and self.train_loader.dataset.mosaic:
                print(f"{colorstr('yellow', 'bold', 'closing mosaic')} at epoch {epoch}")
                self.train_loader.dataset.mosaic = False
                self.train_loader.dataset.hyp["mixup"] = 0.0
                self.train_loader.dataset.hyp["erasing"] = 0.0

            meters = {k: AverageMeter() for k in ("box", "cls", "l1", "dfl", "o2m", "o2o")}
            pbar = tqdm(enumerate(self.train_loader), total=self.nb,
                        desc=f"{epoch + 1}/{self.epochs}")
            self.optimizer.zero_grad(set_to_none=True)

            for i, (imgs, targets, _paths, _shapes) in pbar:
                ni = i + self.nb * epoch

                # ---- warmup
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.nbs / self.args.batch]).round()))
                    for j, g in enumerate(self.optimizer.param_groups):
                        warm_lr = float(self.hyp["warmup_bias_lr"]) if j == 1 else 0.0
                        g["lr"] = np.interp(ni, xi, [warm_lr, float(self.hyp["lr0"]) * self.lf(epoch)])
                        if "momentum" in g:
                            g["momentum"] = np.interp(
                                ni, xi, [float(self.hyp["warmup_momentum"]), float(self.hyp["momentum"])])
                        elif "betas" in g:
                            b = list(g["betas"])
                            b[0] = float(np.interp(ni, xi, [float(self.hyp["warmup_momentum"]),
                                                            float(self.hyp["momentum"])]))
                            g["betas"] = tuple(b)

                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0

                # ---- multi-scale
                if self.ms_hi > self.ms_lo:
                    sz = int(np.random.randint(self.ms_lo // 32, self.ms_hi // 32 + 1) * 32)
                    if sz != imgs.shape[-1]:
                        imgs = F.interpolate(imgs, size=(sz, sz), mode="bilinear",
                                             align_corners=False)

                with torch.amp.autocast("cuda", enabled=self.amp):
                    preds = self.model(imgs)
                    loss, items = self.criterion(preds, targets, alpha)

                self.scaler.scale(loss).backward()

                if ni - last_opt_step >= self.accumulate:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.ema.update(self.model)
                    last_opt_step = ni

                for k in meters:
                    meters[k].update(items[k])
                mem = f"{torch.cuda.memory_reserved() / 1e9:.2f}G" if torch.cuda.is_available() else "-"
                pbar.set_postfix_str(
                    f"box={meters['box'].avg:.3f} cls={meters['cls'].avg:.3f} "
                    f"l1={meters['l1'].avg:.3f} dfl={meters['dfl'].avg:.3f} a={alpha:.2f} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e} sz={imgs.shape[-1]} {mem}")

            self.scheduler.step()

            # Split the epoch clock: train time is what a hyperparameter change moves,
            # val time is fixed overhead. Reporting only the sum hides which one grew.
            t_train = time.time() - t_epoch

            # ---- validate
            self.ema.update_attr(self.model, include=("nc", "names", "scale", "strides"))
            res = self.validator(self.ema.ema, desc=f"val {epoch + 1}")
            fitness = res.get("fitness", 0.0)

            epoch_s = time.time() - t_epoch
            val_s = epoch_s - t_train
            self.epoch_times.append(epoch_s)
            elapsed_h = (time.time() - t_start) / 3600
            # remaining epochs x the mean of the last 5, so a mosaic-close slowdown shows up
            recent = self.epoch_times[-5:]
            eta_h = (self.epochs - epoch - 1) * (sum(recent) / len(recent)) / 3600

            print(f"  epoch {epoch + 1:>3d}  mAP50={res.get('mAP50', 0):.4f}  "
                  f"mAP50-95={res.get('mAP50-95', 0):.4f}  P={res.get('precision', 0):.3f}  "
                  f"R={res.get('recall', 0):.3f}  {res.get('ms_per_image', 0):.2f} ms/img  "
                  f"{epoch_s:.1f}s (train {t_train:.1f}s + val {val_s:.1f}s)  "
                  f"elapsed {elapsed_h:.2f}h  ETA {eta_h:.2f}h")

            self._log_csv({
                "epoch": epoch + 1,
                "box_loss": meters["box"].avg, "cls_loss": meters["cls"].avg,
                "l1_loss": meters["l1"].avg, "dfl_loss": meters["dfl"].avg,
                "loss_o2m": meters["o2m"].avg,
                "loss_o2o": meters["o2o"].avg, "prog_alpha": alpha,
                "lr": self.optimizer.param_groups[0]["lr"],
                "precision": res.get("precision", 0.0), "recall": res.get("recall", 0.0),
                "mAP50": res.get("mAP50", 0.0), "mAP50_95": res.get("mAP50-95", 0.0),
                "fitness": fitness, "ms_per_image": res.get("ms_per_image", 0.0),
                "epoch_seconds": epoch_s, "train_seconds": t_train,
                "val_seconds": val_s, "elapsed_hours": elapsed_h,
            })

            try:
                from src.v2.utils.plots import plot_results
                if self.report_dir:
                    (self.report_dir / "plots").mkdir(parents=True, exist_ok=True)
                    plot_results(self.csv_path,
                                 self.report_dir / "plots" / "01_training_curves.png")
                else:
                    plot_results(self.csv_path)
            except Exception:
                pass

            best = fitness > self.best_fitness
            if best:
                self.best_fitness = fitness
                self.no_improve = 0
            else:
                self.no_improve += 1
            self.save_ckpt(epoch, fitness, best=best)

            if self.patience and self.no_improve >= self.patience:
                print(f"{colorstr('early stop')} no improvement for {self.patience} epochs")
                break

        dt = (time.time() - t_start) / 3600
        print(f"\n{colorstr('done')} {dt:.2f} h. best fitness={self.best_fitness:.4f}")
        print(f"weights: {self.wdir / 'best.pt'}")

        # final report on the best EMA weights
        ckpt = torch.load(self.wdir / "best.pt", map_location=self.device, weights_only=False)
        self.ema.ema.load_state_dict(ckpt["ema"])
        res = self.validator(self.ema.ema, desc="final val")
        print("\n" + self.validator.metrics.table())

        self.write_training_summary(dt, len(self.epoch_times), res,
                                    interrupted=len(self.epoch_times) < self.epochs)
        return res
