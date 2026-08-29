"""YOLO-format dataset with mosaic / mixup / perspective pipeline.

Images are returned as uint8 CHW tensors; the /255 conversion happens on the GPU
in the trainer, which is measurably faster than doing it per-worker on CPU.
"""
import glob
import hashlib
import os
import random
from pathlib import Path

import cv2

# One cv2 thread per worker. The default (28 here) oversubscribes the CPU the
# moment DataLoader workers multiply, costing ~25% of per-sample time.
cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.v2.data.augment import (augment_hsv, cutout, letterbox, mixup, random_blur,
                                 random_gray, random_perspective)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _hash(paths):
    h = hashlib.sha256(str(sum(os.path.getsize(p) for p in paths)).encode())
    h.update("".join(paths).encode())
    return h.hexdigest()


def _parse_label_file(path, nc):
    """Return (n, 5) float32 [cls, xc, yc, w, h] normalised; supports polygon rows."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                c = int(float(parts[0]))
            except ValueError:
                continue
            if not 0 <= c < nc:
                continue
            vals = [float(v) for v in parts[1:]]
            if len(vals) == 4:
                xc, yc, w, h = vals
            else:  # polygon -> enclosing box
                if len(vals) % 2:
                    vals = vals[:-1]
                xs, ys = vals[0::2], vals[1::2]
                xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                w, h = max(xs) - min(xs), max(ys) - min(ys)
            if w <= 0 or h <= 0:
                continue
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                continue
            rows.append([c, xc, yc, min(w, 1.0), min(h, 1.0)])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), np.float32)


class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, nc=3, imgsz=384, hyp=None, augment=False,
                 cache_ram=False, prefix=""):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = hyp or {}
        self.nc = nc
        self.prefix = prefix

        files = sorted(p for p in glob.glob(str(self.img_dir / "*"))
                       if Path(p).suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"no images found in {self.img_dir}")
        self.im_files = files
        self.label_files = [str(self.label_dir / (Path(p).stem + ".txt")) for p in files]

        self.labels = self._load_labels()
        self.n = len(self.im_files)
        self.indices = list(range(self.n))

        self.mosaic = bool(augment and float(self.hyp.get("mosaic", 0.0)) > 0)
        self.ims = [None] * self.n
        self.cache_ram = cache_ram
        if cache_ram:
            self._cache_images()

        nb = sum(len(x) for x in self.labels)
        empty = sum(1 for x in self.labels if len(x) == 0)
        print(f"{prefix}{self.n} images, {nb} boxes, {empty} background images")

    # ------------------------------------------------------------- labels
    def _load_labels(self):
        cache_path = self.label_dir.parent / f"{self.label_dir.name}.v2cache.npy"
        key = _hash(self.im_files)
        if cache_path.exists():
            try:
                d = np.load(cache_path, allow_pickle=True).item()
                if d.get("hash") == key and d.get("nc") == self.nc:
                    return d["labels"]
            except Exception:
                pass
        labels = []
        for lf in tqdm(self.label_files, desc=f"{self.prefix}scanning labels", leave=False):
            labels.append(_parse_label_file(lf, self.nc) if os.path.exists(lf)
                          else np.zeros((0, 5), np.float32))
        try:
            np.save(cache_path, {"hash": key, "nc": self.nc, "labels": labels})
        except Exception:
            pass
        return labels

    def _cache_images(self):
        for i in tqdm(range(self.n), desc=f"{self.prefix}caching images to RAM", leave=False):
            self.ims[i] = self._read(i)

    def _read(self, i):
        im = cv2.imread(self.im_files[i])
        if im is None:
            raise FileNotFoundError(f"unreadable image {self.im_files[i]}")
        return im

    # ------------------------------------------------------------- loading
    def load_image(self, i):
        """Read and resize so the long side == imgsz. Returns im, (h0, w0), (h, w)."""
        im = self.ims[i] if self.ims[i] is not None else self._read(i)
        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (min(int(w0 * r + 0.5), self.imgsz),
                                 min(int(h0 * r + 0.5), self.imgsz)), interpolation=interp)
        return im, (h0, w0), im.shape[:2]

    def load_mosaic(self, index):
        """4-image mosaic on a 2*imgsz canvas. Returns img4, labels4 (n, 5) xyxy px."""
        s = self.imgsz
        labels4 = []
        yc, xc = (int(random.uniform(s // 2, 2 * s - s // 2)) for _ in range(2))
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)

        for i, idx in enumerate(indices):
            img, _, (h, w) = self.load_image(idx)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b

            lb = self.labels[idx].copy()
            if lb.size:
                lb = self._xywhn2xyxy(lb, w, h, padw, padh)
            labels4.append(lb)

        labels4 = np.concatenate(labels4, 0) if labels4 else np.zeros((0, 5), np.float32)
        if labels4.size:
            np.clip(labels4[:, 1:5], 0, 2 * s, out=labels4[:, 1:5])
        return img4, labels4

    @staticmethod
    def _xywhn2xyxy(lb, w, h, padw=0, padh=0):
        out = lb.copy()
        out[:, 1] = w * (lb[:, 1] - lb[:, 3] / 2) + padw
        out[:, 2] = h * (lb[:, 2] - lb[:, 4] / 2) + padh
        out[:, 3] = w * (lb[:, 1] + lb[:, 3] / 2) + padw
        out[:, 4] = h * (lb[:, 2] + lb[:, 4] / 2) + padh
        return out

    @staticmethod
    def _xyxy2xywhn(lb, w, h, eps=1e-3):
        out = lb.copy()
        out[:, 1] = ((lb[:, 1] + lb[:, 3]) / 2) / w
        out[:, 2] = ((lb[:, 2] + lb[:, 4]) / 2) / h
        out[:, 3] = (lb[:, 3] - lb[:, 1]) / w
        out[:, 4] = (lb[:, 4] - lb[:, 2]) / h
        keep = (out[:, 3] > eps) & (out[:, 4] > eps)
        return out[keep]

    # ------------------------------------------------------------- item
    def __len__(self):
        return self.n

    def __getitem__(self, index):
        hyp = self.hyp
        if self.mosaic and random.random() < float(hyp.get("mosaic", 0.0)):
            img, labels = self.load_mosaic(index)
            img, labels = random_perspective(
                img, labels,
                degrees=float(hyp.get("degrees", 0.0)),
                translate=float(hyp.get("translate", 0.1)),
                scale=float(hyp.get("scale", 0.5)),
                shear=float(hyp.get("shear", 0.0)),
                perspective=float(hyp.get("perspective", 0.0)),
                border=(-self.imgsz // 2, -self.imgsz // 2))
            if random.random() < float(hyp.get("mixup", 0.0)):
                img2, labels2 = self.load_mosaic(random.randint(0, self.n - 1))
                img2, labels2 = random_perspective(
                    img2, labels2,
                    degrees=float(hyp.get("degrees", 0.0)),
                    translate=float(hyp.get("translate", 0.1)),
                    scale=float(hyp.get("scale", 0.5)),
                    shear=float(hyp.get("shear", 0.0)),
                    perspective=float(hyp.get("perspective", 0.0)),
                    border=(-self.imgsz // 2, -self.imgsz // 2))
                img, labels = mixup(img, labels, img2, labels2)
        else:
            img, (h0, w0), (h, w) = self.load_image(index)
            img, ratio, pad = letterbox(img, self.imgsz, scaleup=self.augment)
            labels = self.labels[index].copy()
            if labels.size:
                labels = self._xywhn2xyxy(labels, ratio * w, ratio * h, pad[0], pad[1])
            if self.augment:
                img, labels = random_perspective(
                    img, labels,
                    degrees=float(hyp.get("degrees", 0.0)),
                    translate=float(hyp.get("translate", 0.1)),
                    scale=float(hyp.get("scale", 0.5)),
                    shear=float(hyp.get("shear", 0.0)),
                    perspective=float(hyp.get("perspective", 0.0)))

        if self.augment:
            img = augment_hsv(img, float(hyp.get("hsv_h", 0.015)),
                              float(hyp.get("hsv_s", 0.7)), float(hyp.get("hsv_v", 0.4)))
            img = random_gray(img, float(hyp.get("gray", 0.0)))
            img = random_blur(img, float(hyp.get("blur", 0.0)))
            if random.random() < float(hyp.get("fliplr", 0.0)):
                img = np.fliplr(img)
                if labels.size:
                    w = img.shape[1]
                    x1 = labels[:, 1].copy()
                    labels[:, 1] = w - labels[:, 3]
                    labels[:, 3] = w - x1
            if random.random() < float(hyp.get("flipud", 0.0)):
                img = np.flipud(img)
                if labels.size:
                    h = img.shape[0]
                    y1 = labels[:, 2].copy()
                    labels[:, 2] = h - labels[:, 4]
                    labels[:, 4] = h - y1
            img = np.ascontiguousarray(img)
            img, labels = cutout(img, labels, float(hyp.get("erasing", 0.0)))

        h, w = img.shape[:2]
        labels = self._xyxy2xywhn(labels, w, h) if labels.size else np.zeros((0, 5), np.float32)

        nl = len(labels)
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])  # BGR HWC -> RGB CHW
        return torch.from_numpy(img), labels_out, self.im_files[index], (h, w)

    @staticmethod
    def collate_fn(batch):
        ims, labels, paths, shapes = zip(*batch)
        for i, lb in enumerate(labels):
            lb[:, 0] = i
        return torch.stack(ims, 0), torch.cat(labels, 0), paths, shapes
