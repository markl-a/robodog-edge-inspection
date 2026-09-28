"""Analog pressure-gauge reading from a patrol-robot camera crop.

Two readers share one calibration (GaugeSpec):
  * classic  - locate the dial (Hough), then find the darkest radial ray = needle. No training, explainable.
  * cnn      - tiny regression CNN (see edge/train_gauge.py), exported to ONNX and INT8-quantized for the edge box.

Synthetic images are rendered here so the whole pipeline is reproducible without a plant or a robot.
Angles use the math convention (0 deg = +x, counter-clockwise positive); image y points down.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GaugeSpec:
    min_value: float = 0.0
    max_value: float = 10.0          # bar
    start_deg: float = 225.0         # needle angle at min_value (7:30 o'clock)
    sweep_deg: float = 270.0         # clockwise sweep to max_value (4:30 o'clock)
    unit: str = "bar"
    accuracy_class: float = 1.6      # EN 837-1 class: permissible error = 1.6 % of full scale

    @property
    def span(self) -> float:
        return self.max_value - self.min_value

    def value_to_angle(self, v: float) -> float:
        return self.start_deg - (v - self.min_value) / self.span * self.sweep_deg

    def angle_to_value(self, deg: float) -> float:
        """Map a needle angle back to a value; angles in the dead zone snap to the nearest end stop."""
        cw = (self.start_deg - deg) % 360.0          # clockwise distance from the min stop
        if cw > self.sweep_deg:                       # dead zone at the bottom of the dial
            cw = self.sweep_deg if cw - self.sweep_deg < (360.0 - cw) else 0.0
        return self.min_value + cw / self.sweep_deg * self.span

    @property
    def tolerance(self) -> float:
        return self.accuracy_class / 100.0 * self.span


SPEC = GaugeSpec()
RAY_SPAN = (0.18, 0.65)   # fraction of dial radius sampled per candidate ray. Starting close to the hub matters:
                          # the needle is dark all the way in, a cable crossing the dial usually is not.


def render(value: float, rng: np.random.Generator, size: int = 96, spec: GaugeSpec = SPEC, hard: bool = False) -> np.ndarray:
    """Grayscale uint8 image of a gauge showing `value`, with the nuisances a robot camera sees."""
    img = np.full((size, size), int(rng.integers(150, 200)), np.uint8)
    c = np.array([size / 2, size / 2]) + rng.uniform(-4, 4, 2)
    r = size * rng.uniform(0.38, 0.44)
    cx, cy = int(round(c[0])), int(round(c[1]))
    cv2.circle(img, (cx, cy), int(r), int(rng.integers(225, 255)), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(r), 40, 2, cv2.LINE_AA)
    for k in range(11):                                              # major ticks every 1 bar
        a = np.deg2rad(spec.value_to_angle(spec.min_value + k * spec.span / 10))
        p1 = c + 0.80 * r * np.array([np.cos(a), -np.sin(a)])
        p2 = c + 0.95 * r * np.array([np.cos(a), -np.sin(a)])
        cv2.line(img, tuple(p1.round().astype(int)), tuple(p2.round().astype(int)), 60, 1, cv2.LINE_AA)
    a = np.deg2rad(spec.value_to_angle(value))
    tip = c + 0.72 * r * np.array([np.cos(a), -np.sin(a)])
    tail = c - 0.15 * r * np.array([np.cos(a), -np.sin(a)])
    cv2.line(img, tuple(tail.round().astype(int)), tuple(tip.round().astype(int)), int(rng.integers(10, 50)), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, int(0.07 * r)), 30, -1, cv2.LINE_AA)

    if hard and rng.random() < 0.5:                                  # cable / pipe in front of the dial
        a = rng.uniform(0, np.pi)
        off = rng.uniform(0.15, 0.5) * r * rng.choice([-1, 1])
        n = np.array([np.cos(a), np.sin(a)])
        d = np.array([-n[1], n[0]])
        p0 = c + off * n - 1.2 * r * d
        p1 = c + off * n + 1.2 * r * d
        cv2.line(img, tuple(p0.round().astype(int)), tuple(p1.round().astype(int)), int(rng.integers(15, 60)), int(rng.integers(2, 4)), cv2.LINE_AA)

    f = img.astype(np.float32)
    if hard or rng.random() < 0.3:                                   # specular glare on the glass
        g = np.zeros_like(f)
        gc = tuple((c + rng.uniform(-0.5, 0.5, 2) * r).astype(int))
        cv2.ellipse(g, gc, (int(r * rng.uniform(0.2, 0.45)), int(r * rng.uniform(0.08, 0.2))), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        f += cv2.GaussianBlur(g, (0, 0), 3) * rng.uniform(40, 110 if hard else 70)
    f = f * rng.uniform(0.7, 1.15) + rng.uniform(-25, 15)            # exposure
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 1.4 if hard else 1.0))   # focus / motion blur while the robot settles
    f += rng.normal(0, rng.uniform(2, 9 if hard else 6), f.shape)    # sensor noise
    return np.clip(f, 0, 255).astype(np.uint8)


@dataclass
class Reading:
    value: float
    confidence: float     # 0..1, how clearly one ray stands out as the needle
    method: str


def read_classic(img: np.ndarray, spec: GaugeSpec = SPEC) -> Reading:
    size = img.shape[0]
    blur = cv2.medianBlur(img, 3)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=size, param1=80, param2=20,
                               minRadius=int(size * 0.3), maxRadius=int(size * 0.48))
    if circles is not None:
        cx, cy, r = circles[0, 0]
    else:                                                            # fall back to the calibrated ROI centre
        cx, cy, r = size / 2, size / 2, size * 0.41
    angles = np.deg2rad(np.arange(0, 360, 1.0))
    radii = np.linspace(*RAY_SPAN, 12) * r                            # inside the tick ring; see RAY_SPAN
    xs = cx + np.outer(np.cos(angles), radii)
    ys = cy - np.outer(np.sin(angles), radii)
    samples = cv2.remap(blur.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)
    profile = samples.mean(axis=1)
    profile = np.convolve(np.r_[profile[-2:], profile, profile[:2]], np.ones(5) / 5, mode="valid")
    i = int(np.argmin(profile))
    # sub-degree refinement: parabola through the minimum and its neighbours
    l, m, rr = profile[i - 1], profile[i], profile[(i + 1) % 360]
    denom = l - 2 * m + rr
    deg = i + (0.5 * (l - rr) / denom if denom > 1e-6 else 0.0)
    # confidence = how much the needle stands out from the next-darkest direction (>20 deg away).
    # A cable or shadow across the dial produces a second dark ray and drives this towards 0.
    base = np.median(profile)
    far = np.abs((np.arange(360) - i + 180) % 360 - 180) > 20
    depth1, depth2 = base - m, base - profile[far].min()
    conf = float(np.clip(1 - depth2 / (depth1 + 1e-6), 0, 1))
    return Reading(round(float(spec.angle_to_value(deg % 360)), 3), round(conf, 3), "classic")


def dataset(n: int, seed: int, size: int = 64, hard: bool = False, spec: GaugeSpec = SPEC):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(spec.min_value, spec.max_value, n)
    imgs = np.stack([render(v, rng, size=size, hard=hard) for v in vals])
    return imgs, vals
