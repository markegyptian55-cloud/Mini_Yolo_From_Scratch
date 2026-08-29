"""Behavioural tests for src/v2/temporal.py -- no pytest dependency, just run it:

    python -m src.v2.tests.test_temporal

Every case is a synthetic 30 fps timeline, so the expected frame indices in the
assertions are exact, not approximate.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.temporal import DriverStateMonitor    # noqa: E402

FPS = 30.0


def det(cls, conf=0.8):
    return [[10, 10, 50, 50, conf, cls]]


def run(seq):
    """seq: list of (n_frames, cls_or_None). Returns (monitor, last_state)."""
    m = DriverStateMonitor(fps=FPS)
    st = None
    for n, c in seq:
        for _ in range(n):
            st = m.update([] if c is None else det(c))
    return m, st


fails = []


def check(name, got, want):
    ok = got == want if not isinstance(want, float) else abs(got - want) < 1e-6
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        fails.append(name)


# --- 1. a 200 ms blink (6 frames) is a blink, not a microsleep
m, st = run([(30, 1), (6, 0), (30, 1)])
check("blink counted", m.blink_count, 1)
check("no microsleep from blink", m.microsleep_count, 0)

# --- 2. a 67 ms flicker (2 frames) is below blink_min -> not counted
m, st = run([(30, 1), (2, 0), (30, 1)])
check("sub-100ms flicker ignored", m.blink_count, 0)

# --- 3. a 500 ms closure is longer than a blink but short of microsleep -> neither
m, st = run([(30, 1), (15, 0), (30, 1)])
check("500ms closure is not a blink", m.blink_count, 0)
check("500ms closure is not a microsleep", m.microsleep_count, 0)

# --- 4. a 2 s closure fires exactly one microsleep, at the 1.5 s crossing
m = DriverStateMonitor(fps=FPS)
for _ in range(30):
    m.update(det(1))
fire_frame = None
for i in range(60):                       # 60 frames = 2.0 s closed
    s = m.update(det(0))
    if s["microsleep_active"] and fire_frame is None:
        fire_frame = i + 1
for _ in range(30):
    m.update(det(1))
check("one microsleep event", m.microsleep_count, 1)
check("fires at 1.5s (frame 45 of closure)", fire_frame, 45)

# --- 5. alarm is CRITICAL while latched
m2 = DriverStateMonitor(fps=FPS)
for _ in range(30):
    m2.update(det(1))
for _ in range(50):
    s = m2.update(det(0))
check("critical during microsleep", s["alert_level"], "CRITICAL")
check("reason names microsleep", s["alert_reason"].startswith("MICROSLEEP"), True)

# --- 6. PERCLOS: 30 closed of 120 eye-frames = 0.25
m, st = run([(90, 1), (30, 0)])
check("perclos 30/120", round(st["perclos"], 6), 0.25)
check("perclos >= warn -> not SAFE", st["alert_level"] in ("WARNING", "CRITICAL"), True)

# --- 7. frames with no detection do not enter the PERCLOS denominator
m, st = run([(60, 1), (60, None)])
check("perclos ignores blind frames", round(st["perclos"], 6), 0.0)
check("coverage halves", round(st["coverage"], 4), 0.5)

# --- 8. a face lost mid-microsleep keeps the alarm latched
m3 = DriverStateMonitor(fps=FPS)
for _ in range(60):
    m3.update(det(0))          # 2 s closed -> latched
s = m3.update([])              # detector loses the face
check("latch survives lost face", s["microsleep_active"], True)
s = m3.update(det(1))          # confirmed open eye clears it
check("open eye clears latch", s["microsleep_active"], False)

# --- 9. yawns: two 1 s yawns -> count 2; a 2-frame blip is not a yawn
m, st = run([(30, 1), (30, 2), (30, 1), (30, 2), (30, 1), (2, 2), (30, 1)])
check("two yawns counted", m.yawn_count, 2)

# --- 10. yawn rate per minute over the window
#     2 yawns inside a ~7.5 s window -> 16 /min
check("yawns_per_min > 2 triggers rate warn", st["yawns_per_min"] > 2.0, True)

# --- 11. continuous yawn duration is live
m4 = DriverStateMonitor(fps=FPS)
for _ in range(45):
    s = m4.update(det(2))
check("live yawn duration 1.5s", round(s["yawn_s"], 3), 1.5)

# --- 12. summary totals
m, st = run([(30, 1), (6, 0), (30, 1), (30, 2), (30, 1)])
sm = m.summary()
check("summary frames", sm["frames"], 126)
check("summary duration", round(sm["duration_s"], 2), 4.2)
check("summary blinks", sm["blinks"], 1)
check("summary yawns", sm["yawns"], 1)
check("summary microsleeps", sm["microsleeps"], 0)
check("longest closure 0.2s", round(sm["longest_closure_s"], 3), 0.2)

# --- 13. a clean stream is SAFE
m, st = run([(300, 1)])
check("all-open stream is SAFE", st["alert_level"], "SAFE")
check("perclos 0", st["perclos"], 0.0)

# --- 14. detections under conf_thres are ignored
m, st = run([(30, 1), (30, None)])
m5 = DriverStateMonitor(fps=FPS, conf_thres=0.9)
s = m5.update(det(0, conf=0.5))
check("low-conf det ignored", s["eye_seen"], False)

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
