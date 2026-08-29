"""Validation loop: mAP50 / mAP50-95 plus latency."""
import time

import torch
from tqdm import tqdm

from src.v2.utils.boxes import non_max_suppression, xywh2xyxy
from src.v2.utils.metrics import DetMetrics


class Validator:
    def __init__(self, dataloader, nc, names, device, conf=0.001, iou=0.7,
                 max_det=300, e2e=True, verbose=True):
        self.dl = dataloader
        self.nc = nc
        self.names = names
        self.device = device
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.e2e = e2e
        self.verbose = verbose

    @torch.no_grad()
    def __call__(self, model, half=False, desc="val"):
        was_training = model.training
        model.eval()
        model.head.e2e = self.e2e
        model.head.max_det = self.max_det
        metrics = DetMetrics(self.nc, self.names, device=self.device)

        t_infer, n_img = 0.0, 0
        pbar = tqdm(self.dl, desc=desc, leave=False) if self.verbose else self.dl
        for imgs, targets, _paths, _shapes in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            imgs = imgs.half() if half else imgs.float()
            imgs /= 255.0
            bs, _, h, w = imgs.shape

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            preds = model(imgs)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_infer += time.time() - t0
            n_img += bs

            if self.e2e:
                dets = []
                for b in range(bs):
                    d = preds[b].float()
                    d = d[d[:, 4] > self.conf]
                    if d.shape[0]:
                        d = torch.cat((xywh2xyxy(d[:, :4]), d[:, 4:]), 1)
                    dets.append(d)
            else:
                dets = non_max_suppression(preds.float(), self.conf, self.iou,
                                           max_det=self.max_det)

            targets = targets.to(self.device)
            scale = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
            for b in range(bs):
                t = targets[targets[:, 0] == b]
                if t.shape[0]:
                    lb = torch.cat((t[:, 1:2], xywh2xyxy(t[:, 2:6] * scale)), 1)
                else:
                    lb = torch.zeros((0, 5), device=self.device)
                metrics.process_batch(dets[b], lb)

        res = metrics.compute()
        res["ms_per_image"] = 1000.0 * t_infer / max(n_img, 1)
        res["fps"] = max(n_img, 1) / max(t_infer, 1e-9)
        self.metrics = metrics
        if was_training:
            model.train()
        return res
