"""Temporal driver-fatigue logic for video streams.

The detector is per-frame and stateless. Fatigue is not: a single closed-eye frame
means nothing, a closed-eye frame that is the 45th in a row means the driver is
asleep. This module is the state machine in between, and it is deliberately kept
out of both the model and the renderer so it can be unit-tested, replayed offline,
and reused by any front end.

Three signals, all standard in the driver-monitoring literature:

* **PERCLOS** -- the fraction of time the eyes are closed over a rolling window
  (60 s by default). The single most validated drowsiness proxy. Computed only over
  frames where an eye was actually detected; frames with no visible eye go to a
  separate `coverage` figure rather than silently inflating or deflating the score.

* **Blinks vs microsleeps** -- both are "eyes closed", separated by duration alone.
  A natural blink is 100-400 ms. A closure past 1.5 s is a microsleep and fires an
  alarm *the moment the threshold is crossed*, not when the eyes reopen: waiting for
  the end of the event to report it would defeat the point.

* **Yawn frequency** -- continuous yawn duration plus a count per minute. A yawn that
  lasts a couple of frames is detector noise, so a run must exceed `yawn_min_ms` to
  be counted as an occurrence.

Timing is derived from the frame index and the stream FPS, so an offline pass over a
recorded clip produces exactly the same numbers every run. Pass an explicit `t` to
use a wall clock instead on a live feed.

Thresholds here are the conventional ones from published DMS work. They are NOT
clinically validated for this project, and no threshold in this file has been tuned
against labelled drowsiness ground truth -- there is none in this dataset. Treat the
output as an instrumented signal, not a medical determination.
"""
from collections import deque

# Default index layout. Every dataset this project has used puts the classes in this
# order, but the order is a property of the *dataset*, not of this module -- so nothing
# here assumes it. `resolve_class_ids` derives the real mapping from the class names in
# the checkpoint, and the monitor is constructed with that. Getting this wrong silently
# would compute PERCLOS from the wrong class, which is the worst kind of bug: plausible
# numbers, completely meaningless.
CLOSED_EYE, OPEN_EYE, YAWNING = 0, 1, 2

# Accepted spellings per role. Roboflow exports in particular vary a lot: Chapter 1 used
# ('closed_eye','open_eye','yawning'); the Chapter 2 export shipped ('close','open','yawn').
_ALIASES = {
    "closed": ("closed_eye", "closed", "close", "closed-eye", "eye_closed", "closedeye"),
    "open": ("open_eye", "open", "opened", "open-eye", "eye_open", "openeye"),
    "yawn": ("yawning", "yawn", "yawns", "mouth_open", "yawning_mouth"),
}


def resolve_class_ids(names):
    """names -> (closed_idx, open_idx, yawn_idx).

    Raises ValueError rather than guessing when a role cannot be matched. A wrong index
    here produces confident, wrong fatigue telemetry, so failing loudly is correct.
    """
    if not names:
        raise ValueError("no class names supplied; cannot resolve fatigue class indices")
    lowered = [str(n).strip().lower().replace(" ", "_") for n in names]
    out = {}
    for role, aliases in _ALIASES.items():
        hits = [i for i, n in enumerate(lowered) if n in aliases]
        if len(hits) != 1:
            raise ValueError(
                "cannot resolve the '%s' class from names=%r (matched %d candidates). "
                "Add the spelling to _ALIASES in src/v2/temporal.py rather than "
                "renaming label files." % (role, list(names), len(hits)))
        out[role] = hits[0]
    return out["closed"], out["open"], out["yawn"]

# Alert ladder, worst-first. Exposed so the HUD and any caller agree on ordering.
LEVELS = ("SAFE", "WARNING", "CRITICAL")


class DriverStateMonitor:
    """Rolling temporal state for one continuous video stream.

    Feed it one frame of detections at a time with :meth:`update`, which returns a
    dict describing the driver right now. Construct a new instance per stream --
    there is no reset-on-seek, because a seek breaks the temporal assumption the
    whole class rests on.
    """

    def __init__(self, fps=30.0,
                 names=None,
                 perclos_window_s=60.0,
                 blink_min_ms=100.0,
                 blink_max_ms=400.0,
                 microsleep_ms=1500.0,
                 yawn_min_ms=400.0,
                 yawn_window_s=60.0,
                 conf_thres=0.30,
                 perclos_warn=0.15,
                 perclos_alarm=0.30,
                 yawn_rate_warn=2.0):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.dt = 1.0 / self.fps
        self.perclos_window_s = float(perclos_window_s)
        self.blink_min_s = blink_min_ms / 1000.0
        self.blink_max_s = blink_max_ms / 1000.0
        self.microsleep_s = microsleep_ms / 1000.0
        self.yawn_min_s = yawn_min_ms / 1000.0
        self.yawn_window_s = float(yawn_window_s)
        self.conf_thres = float(conf_thres)
        self.perclos_warn = float(perclos_warn)
        self.perclos_alarm = float(perclos_alarm)
        self.yawn_rate_warn = float(yawn_rate_warn)

        # Class indices come from the dataset when names are supplied, otherwise fall
        # back to the conventional layout. Stored per-instance so one process can
        # monitor two streams with different class orders.
        if names:
            self.closed_id, self.open_id, self.yawn_id = resolve_class_ids(names)
        else:
            self.closed_id, self.open_id, self.yawn_id = CLOSED_EYE, OPEN_EYE, YAWNING

        # (timestamp, eye_closed) for frames where an eye was visible
        self._eye = deque()
        # (timestamp, eye_seen) for every frame -- drives the coverage figure
        self._seen = deque()
        self._blink_times = deque()      # end timestamps of completed blinks
        self._yawn_times = deque()       # end timestamps of completed yawns
        self._microsleep_times = deque()

        self.frame_idx = 0
        self.t = 0.0
        self.closure_start = None        # timestamp the current closure began
        self.yawn_start = None
        self.microsleep_latched = False  # so one long closure fires one event
        self.blink_count = 0
        self.yawn_count = 0
        self.microsleep_count = 0
        self.longest_closure_s = 0.0

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _best_conf(dets, cls_id, conf_thres):
        """Highest confidence among detections of one class, or 0.0."""
        best = 0.0
        for d in dets:
            if int(d[5]) == cls_id and float(d[4]) >= conf_thres and float(d[4]) > best:
                best = float(d[4])
        return best

    def _trim(self):
        lo = self.t - self.perclos_window_s
        while self._eye and self._eye[0][0] < lo:
            self._eye.popleft()
        while self._seen and self._seen[0][0] < lo:
            self._seen.popleft()
        lo_y = self.t - self.yawn_window_s
        for q in (self._blink_times, self._yawn_times, self._microsleep_times):
            while q and q[0] < lo_y:
                q.popleft()

    # ---------------------------------------------------------------- update
    def update(self, dets, t=None):
        """Advance one frame.

        dets: iterable of (x1, y1, x2, y2, conf, cls) -- an (N, 6) ndarray works.
        t:    optional wall-clock seconds. Omitted -> derived from frame index / fps.
        """
        self.t = float(t) if t is not None else self.frame_idx * self.dt
        self.frame_idx += 1

        dets = [] if dets is None or len(dets) == 0 else dets
        c_conf = self._best_conf(dets, self.closed_id, self.conf_thres)
        o_conf = self._best_conf(dets, self.open_id, self.conf_thres)
        y_conf = self._best_conf(dets, self.yawn_id, self.conf_thres)

        eye_seen = (c_conf > 0.0) or (o_conf > 0.0)
        # Both eyes can be detected in different states on a turned head; the more
        # confident detection wins. Ties go to "closed" -- under-reporting a closure
        # is the more dangerous error in this application.
        eye_closed = eye_seen and c_conf >= o_conf
        yawning = y_conf > 0.0

        self._seen.append((self.t, eye_seen))
        if eye_seen:
            self._eye.append((self.t, eye_closed))

        # ---- closure state machine
        microsleep_now = False
        if eye_closed:
            if self.closure_start is None:
                self.closure_start = self.t
                self.microsleep_latched = False
            dur = self.t - self.closure_start + self.dt
            if dur >= self.microsleep_s and not self.microsleep_latched:
                # fire on threshold crossing, not on eye reopening
                self.microsleep_latched = True
                self.microsleep_count += 1
                self._microsleep_times.append(self.t)
            microsleep_now = self.microsleep_latched
        else:
            # A frame with no eye visible does not end a closure -- the face simply
            # left the detector. Only a confirmed open eye does.
            if self.closure_start is not None and o_conf > 0.0:
                dur = self.t - self.closure_start
                self.longest_closure_s = max(self.longest_closure_s, dur)
                if self.blink_min_s <= dur <= self.blink_max_s:
                    self.blink_count += 1
                    self._blink_times.append(self.t)
                self.closure_start = None
                self.microsleep_latched = False
            elif self.closure_start is not None and self.microsleep_latched:
                microsleep_now = True   # face lost mid-microsleep: stay latched

        # ---- yawn state machine
        if yawning:
            if self.yawn_start is None:
                self.yawn_start = self.t
        elif self.yawn_start is not None:
            dur = self.t - self.yawn_start
            if dur >= self.yawn_min_s:
                self.yawn_count += 1
                self._yawn_times.append(self.t)
            self.yawn_start = None

        self._trim()
        return self.state(microsleep_now=microsleep_now, eye_seen=eye_seen,
                          eye_closed=eye_closed, yawning=yawning)

    # ---------------------------------------------------------------- readout
    def perclos(self):
        """Closed fraction over the window. None until an eye has been seen."""
        if not self._eye:
            return None
        return sum(1 for _, c in self._eye if c) / len(self._eye)

    def coverage(self):
        """Fraction of windowed frames in which an eye was detected at all."""
        if not self._seen:
            return 0.0
        return sum(1 for _, s in self._seen if s) / len(self._seen)

    def window_span_s(self):
        """Seconds actually accumulated -- less than the nominal window early on."""
        if not self._seen:
            return 0.0
        return self._seen[-1][0] - self._seen[0][0] + self.dt

    def _per_minute(self, q):
        span = self.window_span_s()
        if span <= 0.0:
            return 0.0
        return len(q) * 60.0 / span

    def blinks_per_min(self):
        return self._per_minute(self._blink_times)

    def yawns_per_min(self):
        return self._per_minute(self._yawn_times)

    def current_closure_s(self):
        return 0.0 if self.closure_start is None else self.t - self.closure_start + self.dt

    def current_yawn_s(self):
        return 0.0 if self.yawn_start is None else self.t - self.yawn_start + self.dt

    def state(self, microsleep_now=False, eye_seen=None, eye_closed=None, yawning=None):
        p = self.perclos()
        level, reason = "SAFE", "nominal"

        if p is not None:
            if p >= self.perclos_alarm:
                level, reason = "CRITICAL", f"PERCLOS {p * 100:.0f}%"
            elif p >= self.perclos_warn:
                level, reason = "WARNING", f"PERCLOS {p * 100:.0f}%"
        else:
            reason = "no eye detected yet"

        ypm = self.yawns_per_min()
        if ypm >= self.yawn_rate_warn and level == "SAFE":
            level, reason = "WARNING", f"{ypm:.1f} yawns/min"

        # A microsleep in progress outranks every windowed statistic: the windows are
        # averages, and this is happening now.
        if microsleep_now:
            level = "CRITICAL"
            reason = f"MICROSLEEP {self.current_closure_s():.1f}s"

        return {
            "t": self.t,
            "frame": self.frame_idx,
            "eye_seen": eye_seen,
            "eye_closed": eye_closed,
            "yawning": yawning,
            "perclos": p,
            "coverage": self.coverage(),
            "window_s": self.window_span_s(),
            "closure_s": self.current_closure_s(),
            "yawn_s": self.current_yawn_s(),
            "blinks_per_min": self.blinks_per_min(),
            "yawns_per_min": ypm,
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "microsleep_count": self.microsleep_count,
            "microsleep_active": bool(microsleep_now),
            "alert_level": level,
            "alert_reason": reason,
        }

    def summary(self):
        """End-of-stream totals. Rates here are over the whole stream, not the window."""
        total_s = self.frame_idx * self.dt
        per_min = (lambda n: n * 60.0 / total_s) if total_s > 0 else (lambda n: 0.0)
        return {
            "frames": self.frame_idx,
            "duration_s": total_s,
            "fps": self.fps,
            "blinks": self.blink_count,
            "blinks_per_min": per_min(self.blink_count),
            "yawns": self.yawn_count,
            "yawns_per_min": per_min(self.yawn_count),
            "microsleeps": self.microsleep_count,
            "perclos_final": self.perclos(),
            "eye_coverage": self.coverage(),
            "longest_closure_s": max(self.longest_closure_s, self.current_closure_s()),
        }
