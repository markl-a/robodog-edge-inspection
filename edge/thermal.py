"""Thermal check of a three-phase breaker from a small radiometric camera (160x120, Lepton-class).

Rule of thumb from NETA MTS (Table 100.18), comparing *similar components under similar load*:
    dT < 1 C   -> normal
    1-3 C      -> possible deficiency, investigate
    4-15 C     -> probable deficiency, repair as time permits
    > 15 C     -> major discrepancy, repair immediately

Hard part on a legged robot is not the threshold: the robot never stops at exactly the same spot, so ROIs
calibrated at commissioning drift off the terminals. We register each frame to the waypoint's reference
frame (phase correlation) before reading ROIs.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

H, W = 120, 160
PHASES = {"A": (44, 36, 12, 12), "B": (44, 74, 12, 12), "C": (44, 112, 12, 12)}   # terminal lug ROIs (y, x, h, w) at commissioning
SEVERITY = [(1.0, "normal"), (4.0, "possible"), (15.0, "probable"), (float("inf"), "major")]


def classify(delta_t: float) -> str:
    for upper, name in SEVERITY:
        if delta_t < upper:
            return name
    return "major"


def render(rng: np.random.Generator, phase_rise: dict[str, float], hotspot: dict[str, float] | None = None,
           shift=(0.0, 0.0), ambient: float = 26.0) -> np.ndarray:
    """Temperature frame in deg C.

    phase_rise: rise of each terminal over ambient (load-driven, similar across healthy phases).
    hotspot:    extra rise of a small (4x4 px) spot at a loose / corroded lug - what a real fault looks like.
    shift:      (dy, dx) pose error of the robot at the waypoint.
    """
    yy, _ = np.mgrid[0:H, 0:W].astype(np.float32)
    f = (ambient + 0.8 * (yy / H)).astype(np.float32)                                            # panel gradient
    cv2.rectangle(f, (20, 24), (140, 104), ambient + 1.5, -1)                                    # breaker body
    for ph, (y, x, h, w) in PHASES.items():
        t = ambient + phase_rise[ph]
        cv2.rectangle(f, (x, y), (x + w - 1, y + h - 1), t, -1)                                  # terminal
        cv2.rectangle(f, (x + 3, y + h), (x + w - 4, y + h + 34), ambient + 0.6 * phase_rise[ph], -1)   # cable
        extra = (hotspot or {}).get(ph, 0.0)
        if extra:
            cv2.rectangle(f, (x + 4, y + 2), (x + 7, y + 5), t + extra, -1)                      # lug hot spot
    f = cv2.GaussianBlur(f, (0, 0), 0.8)                                                         # optics
    m = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
    f = cv2.warpAffine(f, m, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return f + rng.normal(0, 0.08, f.shape).astype(np.float32)                                  # NETD-class noise


def register(frame: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Return (dy, dx) such that frame ~= reference shifted by (dy, dx)."""
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    norm = lambda im: cv2.Laplacian(cv2.GaussianBlur(im, (0, 0), 1.0), cv2.CV_32F)      # structure, not absolute temp
    (dx, dy), _ = cv2.phaseCorrelate(norm(reference), norm(frame), win)
    return float(dy), float(dx)


@dataclass
class ThermalResult:
    hotspots: dict[str, float]      # 95th-percentile temperature per phase ROI
    delta_t: float                  # hottest phase minus median phase
    severity: str
    hottest_phase: str
    shift: tuple[float, float]


def inspect(frame: np.ndarray, reference: np.ndarray | None = None) -> ThermalResult:
    dy, dx = register(frame, reference) if reference is not None else (0.0, 0.0)
    hs = {}
    for ph, (y, x, h, w) in PHASES.items():
        y0, x0 = int(round(y + dy)), int(round(x + dx))
        roi = frame[max(0, y0):y0 + h, max(0, x0):x0 + w]
        hs[ph] = round(float(np.percentile(roi, 95)), 2)
    hottest = max(hs, key=hs.get)
    dt = round(hs[hottest] - float(np.median(list(hs.values()))), 2)
    return ThermalResult(hs, dt, classify(dt), hottest, (round(dy, 2), round(dx, 2)))


def scenario(rng: np.random.Generator):
    """Random load and an optional lug fault on one phase."""
    base = rng.uniform(8, 20)
    rises = {ph: base + rng.normal(0, 0.25) for ph in PHASES}
    fault = rng.choice(["none", "possible", "probable", "major"], p=[0.4, 0.2, 0.25, 0.15])
    extra = {"none": 0.0, "possible": rng.uniform(2.0, 3.5), "probable": rng.uniform(6, 14), "major": rng.uniform(20, 40)}[fault]
    return rises, {str(rng.choice(list(PHASES))): extra} if extra else {}


def evaluate(n: int = 400, seed: int = 5, max_shift: float = 10.0) -> dict:
    """Truth = what the calibrated ROIs read when the robot stops exactly on the waypoint (same noise draw)."""
    rng = np.random.default_rng(seed)
    reference = render(np.random.default_rng(0), {ph: 12.0 for ph in PHASES})
    stats = {"no_registration": [], "registered": []}
    actionable = 0
    for i in range(n):
        rises, hot = scenario(rng)
        shift = tuple(rng.uniform(-max_shift, max_shift, 2))
        truth = inspect(render(np.random.default_rng(1000 + i), rises, hot))
        frame = render(np.random.default_rng(1000 + i), rises, hot, shift=shift)
        for key, ref in (("no_registration", None), ("registered", reference)):
            r = inspect(frame, ref)
            missed = truth.severity in ("probable", "major") and r.severity in ("normal", "possible")
            stats[key].append((r.severity == truth.severity, abs(r.delta_t - truth.delta_t), missed))
        actionable += truth.severity in ("probable", "major")
    out = {"n": n, "max_pose_shift_px": max_shift, "actionable_cases": actionable}
    for key, rows in stats.items():
        ok, err, missed = zip(*rows)
        out[key] = {"severity_acc": round(float(np.mean(ok)), 3), "dT_mae_C": round(float(np.mean(err)), 2),
                    "missed_actionable": int(sum(missed))}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate(), indent=1))
