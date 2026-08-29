"""Augmentation primitives (cv2/numpy, uint8 BGR in -> uint8 BGR out).

Rationale for the non-standard entries, given a driver-monitoring stream:
  * random_gray   -- IR / night-vision cameras produce single-channel frames.
  * random_blur   -- motion blur and defocus are the dominant streaming artefacts.
  * cutout        -- hands over the mouth, sunglasses, steering wheel occlusion.
"""
import math
import random

import cv2

# One cv2 thread per worker. The default (28 here) oversubscribes the CPU the
# moment DataLoader workers multiply, costing ~25% of per-sample time.
cv2.setNumThreads(0)
import numpy as np


# ---------------------------------------------------------------- geometry
def letterbox(im, new_shape=(384, 384), color=(114, 114, 114), scaleup=True, stride=32,
              auto=False):
    """Resize + pad, preserving aspect ratio. Returns im, ratio, (dw, dh)."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """Filter boxes destroyed by a warp. box1/box2: (4, n) xyxy before/after."""
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def random_perspective(im, targets=(), degrees=5.0, translate=0.1, scale=0.5, shear=2.0,
                       perspective=0.0, border=(0, 0)):
    """Random rotate/scale/shear/translate. targets: (n, 5) [cls, x1, y1, x2, y2] px."""
    height = im.shape[0] + border[0] * 2
    width = im.shape[1] + border[1] * 2

    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2
    C[1, 2] = -im.shape[0] / 2

    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)

        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]
    return im, targets


# ---------------------------------------------------------------- photometric
def augment_hsv(im, hgain=0.015, sgain=0.7, vgain=0.4):
    if not (hgain or sgain or vgain):
        return im
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    dtype = im.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)


def random_gray(im, p=0.05):
    """Simulate IR / monochrome cameras."""
    if random.random() < p:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return im


def random_blur(im, p=0.03):
    """Motion blur or defocus, both common on a live video feed."""
    if random.random() >= p:
        return im
    if random.random() < 0.5:
        k = random.choice([3, 5, 7])
        return cv2.GaussianBlur(im, (k, k), 0)
    k = random.choice([5, 7, 9])
    kernel = np.zeros((k, k), np.float32)
    if random.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= k
    return cv2.filter2D(im, -1, kernel)


def cutout(im, labels, p=0.3):
    """Random erasing with label bookkeeping (drops boxes >60% occluded).

    labels: (n, 5) [cls, x1, y1, x2, y2] in pixels.
    """
    if random.random() >= p:
        return im, labels
    h, w = im.shape[:2]
    scales = [0.125] * 2 + [0.0625] * 4 + [0.03125] * 8
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)
        im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = _bbox_ioa(box, labels[:, 1:5])
            labels = labels[ioa < 0.60]
    return im, labels


def _bbox_ioa(box1, box2, eps=1e-7):
    """Intersection over box2 area. box1 (4,), box2 (n, 4) xyxy."""
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    inter = (np.minimum(box1[2], b2_x2) - np.maximum(box1[0], b2_x1)).clip(0) * \
            (np.minimum(box1[3], b2_y2) - np.maximum(box1[1], b2_y1)).clip(0)
    return inter / ((b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps)


def mixup(im1, labels1, im2, labels2):
    """Beta(32, 32) blend -- the YOLO recipe, a mild ~50/50 mix."""
    r = np.random.beta(32.0, 32.0)
    im = (im1 * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels1, labels2), 0)
    return im, labels
