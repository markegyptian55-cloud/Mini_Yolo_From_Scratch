"""Box geometry utilities for MiniYOLO-v2 (anchor-free, DFL-free)."""
import math
import torch
import torchvision


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor points (in feature-map units) and a per-anchor stride tensor.

    Returns:
        anchor_points: (A, 2) cell centres, units = feature-map cells
        stride_tensor: (A, 1)
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2], feats[i].shape[3]
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    """ltrb distances -> boxes. distance: (..., 4) = [l, t, r, b]."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points, bbox, reg_max=None):
    """xyxy box -> ltrb distances from anchor point."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    d = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        d = d.clamp_(0, reg_max - 0.01)
    return d


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    dw, dh = x[..., 2] / 2, x[..., 3] / 2
    y[..., 0] = x[..., 0] - dw
    y[..., 1] = x[..., 1] - dh
    y[..., 2] = x[..., 0] + dw
    y[..., 3] = x[..., 1] + dh
    return y


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def bbox_iou(box1, box2, xywh=False, CIoU=False, DIoU=False, GIoU=False, eps=1e-7):
    """IoU between aligned box sets. box1/box2 broadcastable, last dim = 4."""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    a = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * a)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


def box_iou(box1, box2, eps=1e-7):
    """Pairwise IoU. box1 (N,4), box2 (M,4) xyxy -> (N, M)."""
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (a2.minimum(b2) - a1.maximum(b1)).clamp_(0).prod(2)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1[:, None] + area2 - inter + eps)


def non_max_suppression(pred, conf_thres=0.25, iou_thres=0.7, max_det=300,
                        agnostic=False, classes=None):
    """Classic NMS for the one-to-many fallback path.

    pred: (B, 4 + nc, A) with boxes already in xywh image units.
    Returns list of (n, 6) tensors [x1, y1, x2, y2, conf, cls].
    """
    bs = pred.shape[0]
    nc = pred.shape[1] - 4
    xc = pred[:, 4:].amax(1) > conf_thres
    pred = pred.transpose(-1, -2)  # (B, A, 4+nc)
    out = [torch.zeros((0, 6), device=pred.device)] * bs
    for i, x in enumerate(pred):
        x = x[xc[i]]
        if not x.shape[0]:
            continue
        box, cls = x.split((4, nc), 1)
        box = xywh2xyxy(box)
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        if not x.shape[0]:
            continue
        c = x[:, 5:6] * (0 if agnostic else 7680)
        keep = torchvision.ops.nms(x[:, :4] + c, x[:, 4], iou_thres)[:max_det]
        out[i] = x[keep]
    return out


def scale_boxes(boxes, from_shape, to_shape, ratio_pad=None):
    """Rescale xyxy boxes from letterboxed `from_shape` back to original `to_shape`."""
    if ratio_pad is None:
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
        pad = ((from_shape[1] - to_shape[1] * gain) / 2,
               (from_shape[0] - to_shape[0] * gain) / 2)
    else:
        gain, pad = ratio_pad
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, to_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, to_shape[0])
    return boxes
