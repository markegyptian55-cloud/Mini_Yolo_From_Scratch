"""ADAS-style HUD renderer for demo videos: compact nav bar (model, driver
state, PERCLOS / blink / yawn telemetry, FPS) + semantic-colored detection boxes.

Ported from a reference demo_video.py (sibling project, same 3-class
convention: 0=closed_eye, 1=open_eye, 2=yawning) and adapted to this
project's detection format -- an (N,6) ndarray of x1,y1,x2,y2,conf,cls.

This file only *draws*. All temporal reasoning -- PERCLOS, blink vs microsleep,
yawn frequency -- lives in `src/v2/temporal.py` so it can be tested without a
frame buffer. Pass a `DriverStateMonitor` into `render_frame` to get the full
panel; without one the HUD falls back to the older heuristic `FatigueTracker`.
"""
from collections import Counter, deque

import cv2
import numpy as np

COLOR_MAP = {
    0: (45, 55, 235),    # RED   -> closed_eye
    1: (65, 225, 95),    # GREEN -> open_eye
    2: (0, 180, 255),    # AMBER -> yawning
}
CLASS_SHORT = {0: "C", 1: "O", 2: "Y"}


def class_short_for(names):
    """Short per-class label codes derived from the dataset's own names.

    Falls back to the first letter of the name, so an unexpected class still renders
    something meaningful instead of the wrong fixed letter.
    """
    if not names:
        return dict(CLASS_SHORT)
    out = {}
    for i, n in enumerate(names):
        low = str(n).strip().lower()
        if low.startswith("close") or "closed" in low:
            out[i] = "C"
        elif low.startswith("open"):
            out[i] = "O"
        elif low.startswith("yawn"):
            out[i] = "Y"
        else:
            out[i] = str(n)[:1].upper() or "?"
    return out

COLOR_BG = (12, 15, 20)
COLOR_MUTED = (120, 130, 140)
COLOR_BORDER = (55, 70, 65)
COLOR_CYAN = (220, 200, 60)
COLOR_BLACK = (0, 0, 0)

ALERT_COLORS = {"SAFE": (65, 225, 95), "WARNING": (0, 180, 255), "CRITICAL": (45, 55, 235)}

FONT = cv2.FONT_HERSHEY_DUPLEX
STATE_HISTORY_SIZE = 10
FATIGUE_WINDOW = 45  # ~1.5s @ 30fps -- simple heuristic smoothing window


def clamp(val, lo, hi):
    return max(lo, min(val, hi))


def scale(base, fw, ref=640):
    return max(1, int(base * fw / ref))


def font_scale(base, fw, ref=640):
    return max(0.25, base * fw / ref)


def draw_alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.82):
    h, w = frame.shape[:2]
    x1, x2 = int(clamp(x1, 0, w)), int(clamp(x2, 0, w))
    y1, y2 = int(clamp(y1, 0, h)), int(clamp(y2, 0, h))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def put_text_on_panel(frame, text, pos, fscale, color, thickness=1):
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(frame, text, (x, y), FONT, fscale, color, thickness, cv2.LINE_AA)


def put_text_on_video(frame, text, pos, fscale, color, thickness=1):
    """Drop-shadow text for legibility directly on raw video."""
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(frame, text, (x + 2, y + 2), FONT, fscale, COLOR_BLACK, thickness + 1, cv2.LINE_8)
    cv2.putText(frame, text, (x, y), FONT, fscale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------- driver state

def determine_driver_state(dets):
    if len(dets) == 0:
        return {"label": "NO DETECTION", "color": COLOR_MUTED, "short": "UNKNOWN"}
    class_ids = dets[:, 5].astype(int)
    if 2 in class_ids:
        return {"label": "YAWNING DETECTED", "color": COLOR_MAP[2], "short": "WARNING"}
    if 0 in class_ids:
        return {"label": "EYES CLOSED", "color": COLOR_MAP[0], "short": "DROWSY"}
    return {"label": "ATTENTIVE", "color": COLOR_MAP[1], "short": "NORMAL"}


def stabilize_driver_state(current_state, state_history):
    state_history.append(current_state["short"])
    if len(state_history) < 3:
        return current_state
    stable, _ = Counter(state_history).most_common(1)[0]
    if stable == "WARNING":
        return {"label": "YAWNING DETECTED", "color": COLOR_MAP[2], "short": "WARNING"}
    if stable == "DROWSY":
        return {"label": "EYES CLOSED", "color": COLOR_MAP[0], "short": "DROWSY"}
    if stable == "NORMAL":
        return {"label": "ATTENTIVE", "color": COLOR_MAP[1], "short": "NORMAL"}
    return current_state


class FatigueTracker:
    """Legacy rolling-window heuristic fatigue score.

    Superseded by `src.v2.temporal.DriverStateMonitor`, which measures PERCLOS and
    real blink/microsleep durations instead of blending two frame fractions. Kept
    so older callers keep working -- NOT clinically validated, just a smooth signal
    for the demo HUD: closed-eye frames weigh more than yawns, both decay out of
    the window after FATIGUE_WINDOW frames."""

    def __init__(self, window=FATIGUE_WINDOW):
        self.closed = deque(maxlen=window)
        self.yawn = deque(maxlen=window)

    def update(self, class_ids):
        self.closed.append(1 if 0 in class_ids else 0)
        self.yawn.append(1 if 2 in class_ids else 0)

    def score(self):
        if not self.closed:
            return 0.0
        closed_frac = sum(self.closed) / len(self.closed)
        yawn_frac = sum(self.yawn) / len(self.yawn)
        return float(clamp(0.6 * closed_frac + 0.4 * yawn_frac, 0.0, 1.0))

    def alert_level(self):
        s = self.score()
        if s < 0.2:
            return "SAFE"
        if s < 0.5:
            return "WARNING"
        return "CRITICAL"


# ---------------------------------------------------------------- drawing

def nav_bar_rect(frame, tstate=None):
    """(x1, y1, x2, y2) the nav panel will occupy.

    Exported because the panel is drawn *over* the boxes, so label placement has to
    know where it lands -- the telemetry panel is taller than the old fatigue panel
    and would otherwise bury the labels of any detection in the top-left corner.
    """
    fh, fw = frame.shape[:2]
    pad = scale(8, fw)
    bar_w = min(scale(255 if tstate else 230, fw), fw // 2 - pad)
    bar_h = min(scale(164 if tstate else 112, fw), fh // 2 - pad)
    return pad, pad, pad + bar_w, pad + bar_h


def draw_navigation_bar(frame, driver_state, fps, frame_idx, model_name, fatigue_score,
                        alert_level, tstate=None):
    """Nav bar. `tstate` is a DriverStateMonitor.update() dict; when present the panel
    grows two telemetry rows and the fatigue row is replaced by real PERCLOS."""
    fh, fw = frame.shape[:2]
    pad = scale(8, fw)
    bar_x, bar_y, bar_x2, bar_y2 = nav_bar_rect(frame, tstate)
    bar_w, bar_h = bar_x2 - bar_x, bar_y2 - bar_y

    draw_alpha_rect(frame, bar_x, bar_y, bar_x + bar_w, bar_y + bar_h, COLOR_BG, alpha=0.84)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_BORDER, 1, cv2.LINE_AA)

    lx = bar_x + scale(10, fw)
    cy = bar_y + scale(20, fw)
    line_h = scale(26, fw)
    fs_title, fs_label, fs_small = font_scale(0.42, fw), font_scale(0.38, fw), font_scale(0.34, fw)

    model_short = model_name if len(model_name) <= 18 else model_name[:16] + ".."
    put_text_on_panel(frame, f"ADAS - {model_short}", (lx, cy), fs_title, COLOR_CYAN, 1)
    cy += line_h

    state_color = driver_state["color"]
    dot_r = max(3, scale(4, fw))
    cv2.circle(frame, (lx + dot_r, cy - scale(3, fw)), dot_r, state_color, -1, cv2.LINE_AA)
    put_text_on_panel(frame, driver_state["label"], (lx + dot_r * 2 + scale(6, fw), cy), fs_label, state_color, 1)
    cy += line_h

    alert_color = ALERT_COLORS.get(alert_level, COLOR_MUTED)
    if tstate is None:
        put_text_on_panel(frame, f"FATIGUE {int(fatigue_score * 100):3d}%  {alert_level}",
                          (lx, cy), fs_small, alert_color, 1)
        cy += line_h
    else:
        p = tstate["perclos"]
        p_txt = "--" if p is None else f"{p * 100:.0f}%"
        put_text_on_panel(frame, f"PERCLOS {p_txt:>4s}  {alert_level}",
                          (lx, cy), fs_small, alert_color, 1)
        cy += line_h
        put_text_on_panel(
            frame,
            f"BLINK {tstate['blinks_per_min']:4.1f}/m  YAWN {tstate['yawns_per_min']:4.1f}/m",
            (lx, cy), fs_small, COLOR_MUTED, 1)
        cy += line_h
        # closure timer only while the eyes are actually shut -- a static "0.0s" is noise
        closure = tstate["closure_s"]
        if tstate["microsleep_active"]:
            put_text_on_panel(frame, f"MICROSLEEP {closure:4.1f}s  x{tstate['microsleep_count']}",
                              (lx, cy), fs_small, ALERT_COLORS["CRITICAL"], 1)
        elif closure > 0:
            put_text_on_panel(frame, f"EYES SHUT {closure:4.1f}s",
                              (lx, cy), fs_small, COLOR_MAP[0], 1)
        elif tstate["yawn_s"] > 0:
            put_text_on_panel(frame, f"YAWNING {tstate['yawn_s']:4.1f}s",
                              (lx, cy), fs_small, COLOR_MAP[2], 1)
        else:
            put_text_on_panel(frame, f"EYE TRACK {tstate['coverage'] * 100:3.0f}%",
                              (lx, cy), fs_small, COLOR_MUTED, 1)
        cy += line_h

    fps_str, frm_str = f"FPS {fps:04.1f}", f"F{frame_idx:05d}"
    put_text_on_panel(frame, fps_str, (lx, cy), fs_small, COLOR_MUTED, 1)
    (tw, _), _ = cv2.getTextSize(frm_str, FONT, fs_small, 1)
    put_text_on_panel(frame, frm_str, (bar_x + bar_w - scale(10, fw) - tw, cy), fs_small, COLOR_MUTED, 1)


def draw_detection_boxes(frame, dets, names, avoid=None):
    """avoid: optional (x1, y1, x2, y2) the labels must not be written into."""
    if len(dets) == 0:
        return
    fh, fw = frame.shape[:2]
    for x1, y1, x2, y2, conf, cls in dets:
        cls_id = int(cls)
        color = COLOR_MAP.get(cls_id, (65, 225, 95))
        rx1, ry1 = int(clamp(x1, 0, fw - 1)), int(clamp(y1, 0, fh - 1))
        rx2, ry2 = int(clamp(x2, 0, fw - 1)), int(clamp(y2, 0, fh - 1))
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 1, cv2.LINE_AA)

        short = CLASS_SHORT.get(cls_id, names[cls_id][0].upper() if cls_id < len(names) else "?")
        label = f"{short} {int(conf * 100)}%"
        fs = font_scale(0.34, fw)
        (tw, th), _ = cv2.getTextSize(label, FONT, fs, 1)
        tx, ty = rx1, ry1 - 4
        if ty - th < 0:
            ty = ry2 + th + 4
        if avoid is not None:
            ax1, ay1, ax2, ay2 = avoid
            # label box (tx, ty-th) .. (tx+tw, ty) vs the panel
            overlaps = (tx < ax2 and tx + tw > ax1 and ty - th < ay2 and ty > ay1)
            if overlaps:
                ty = ry2 + th + 4                      # try below the box
                if ty < ay2 and tx < ax2 and tx + tw > ax1:
                    tx = ax2 + scale(6, fw)            # still buried: push right of it
        put_text_on_video(frame, label, (tx, ty), fs, color, 1)


def draw_microsleep_banner(frame, tstate):
    """Full-width alarm strip. Deliberately impossible to miss and impossible to
    confuse with a box label -- a microsleep is the one event that must interrupt."""
    fh, fw = frame.shape[:2]
    h = scale(34, fw)
    y1 = fh - h - scale(8, fw)
    draw_alpha_rect(frame, 0, y1, fw, y1 + h, ALERT_COLORS["CRITICAL"], alpha=0.88)
    txt = f"! MICROSLEEP {tstate['closure_s']:.1f}s -- WAKE DRIVER"
    fs = font_scale(0.52, fw)
    (tw, th), _ = cv2.getTextSize(txt, FONT, fs, 2)
    put_text_on_panel(frame, txt, ((fw - tw) // 2, y1 + (h + th) // 2), fs, (255, 255, 255), 2)


def render_frame(frame, dets, names, model_name, frame_idx, fps, state_history,
                 fatigue_tracker=None, monitor=None):
    """Draw boxes + nav bar on `frame` in place.

    monitor: optional `src.v2.temporal.DriverStateMonitor`. When given it drives the
    alert level (real PERCLOS / microsleep timing) and `fatigue_tracker` is only used
    for the legacy 0-100 score, if one was passed at all.

    Returns (driver_state, tstate) where tstate is the monitor dict or None.
    """
    class_ids = dets[:, 5].astype(int) if len(dets) else np.array([], dtype=int)
    if fatigue_tracker is not None:
        fatigue_tracker.update(class_ids)

    tstate = monitor.update(dets) if monitor is not None else None

    raw_state = determine_driver_state(dets)
    driver_state = stabilize_driver_state(raw_state, state_history)
    if tstate is not None and tstate["microsleep_active"]:
        # the state machine outranks the per-frame majority vote
        driver_state = {"label": "MICROSLEEP", "color": ALERT_COLORS["CRITICAL"],
                        "short": "MICROSLEEP"}

    score = fatigue_tracker.score() if fatigue_tracker is not None else 0.0
    level = (tstate["alert_level"] if tstate is not None
             else (fatigue_tracker.alert_level() if fatigue_tracker is not None else "SAFE"))

    draw_detection_boxes(frame, dets, names, avoid=nav_bar_rect(frame, tstate))
    draw_navigation_bar(frame, driver_state, fps, frame_idx, model_name,
                        score, level, tstate=tstate)
    if tstate is not None and tstate["microsleep_active"]:
        draw_microsleep_banner(frame, tstate)
    return driver_state, tstate
