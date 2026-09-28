"""Patrol mission runner for a quadruped inspection robot.

The robot is behind a small interface (`Robot`) so the same mission logic runs against the simulator here,
or a vendor SDK / ROS 2 bridge on the real dog (not included - see docs/deploy.md).

Design choices:
  * The edge sends *measurements + confidence*, not alarms. Alarm limits live on the platform so changing a
    limit never needs a robot software update.
  * Low-confidence readings are retried from a slightly different pose; if still unclear the result goes out
    as NEEDS_REVIEW with the image reference - it is never silently dropped or guessed.
  * Before every checkpoint the robot checks it can still get home with margin; if not, it docks and marks
    the rest SKIPPED (reason=battery), so operators see exactly what was not inspected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Protocol

import numpy as np

from . import gauge, thermal
from .uplink import Outbox, result_key


@dataclass
class Checkpoint:
    id: str
    kind: str                         # "gauge" | "thermal"
    xy: tuple[float, float]           # metres, site map frame


class Robot(Protocol):
    robot_id: str
    xy: tuple[float, float]

    def battery_pct(self) -> float: ...
    def go_to(self, xy: tuple[float, float]) -> bool: ...
    def capture(self, cp: Checkpoint, attempt: int) -> np.ndarray: ...
    def dock(self) -> None: ...


@dataclass
class SimRobot:
    """Simulated dog: battery drain per metre, occasional blocked path, pose error at each stop."""
    plant: dict                       # checkpoint id -> ground truth (gauge value or thermal scenario)
    robot_id: str = "dog-01"
    xy: tuple[float, float] = (0.0, 0.0)
    battery: float = 100.0
    drain_per_m: float = 0.08
    drain_per_capture: float = 0.05
    blocked: set = field(default_factory=set)
    glare: set = field(default_factory=set)       # checkpoints where the first look hits heavy glare
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.dock_xy = self.xy

    def battery_pct(self) -> float:
        return self.battery

    def go_to(self, xy):
        if xy in self.blocked:
            return False
        self.battery -= math.dist(self.xy, xy) * self.drain_per_m
        self.xy = xy
        return True

    def capture(self, cp, attempt):
        self.battery -= self.drain_per_capture
        truth = self.plant[cp.id]
        if cp.kind == "gauge":
            return gauge.render(truth, self.rng, size=96, hard=cp.id in self.glare and attempt == 0)
        rises, hot = truth
        return thermal.render(self.rng, rises, hot, shift=tuple(self.rng.uniform(-8, 8, 2)))

    def dock(self):
        self.go_to(self.dock_xy)


GaugeReader = Callable[[np.ndarray], gauge.Reading]


@dataclass
class MissionRunner:
    robot: Robot
    outbox: Outbox
    thermal_refs: dict[str, np.ndarray]
    read_gauge: GaugeReader = gauge.read_classic
    min_confidence: float = 0.3
    max_attempts: int = 3
    reserve_pct: float = 15.0
    safety_factor: float = 1.3
    drain_per_m: float = 0.08          # planner's model of the robot, not the robot's true value

    def run(self, mission_id: str, checkpoints: list[Checkpoint], now: Callable[[], datetime] | None = None) -> dict:
        now = now or (lambda: datetime.now(timezone.utc))
        dock = self.robot.xy
        summary = {"OK": 0, "NEEDS_REVIEW": 0, "SKIPPED": 0}
        for i, cp in enumerate(checkpoints):
            need = (math.dist(self.robot.xy, cp.xy) + math.dist(cp.xy, dock)) * self.drain_per_m * self.safety_factor
            if self.robot.battery_pct() - need < self.reserve_pct:
                for rest in checkpoints[i:]:
                    self._emit(mission_id, rest, now, status="SKIPPED", detail={"reason": "battery"})
                    summary["SKIPPED"] += 1
                break
            if not self.robot.go_to(cp.xy):
                self._emit(mission_id, cp, now, status="SKIPPED", detail={"reason": "unreachable"})
                summary["SKIPPED"] += 1
                continue
            status = self._inspect(mission_id, cp, now)
            summary[status] += 1
        self.robot.dock()
        summary["battery_left_pct"] = round(self.robot.battery_pct(), 1)
        return summary

    def _inspect(self, mission_id, cp, now) -> str:
        if cp.kind == "gauge":
            best = None
            for attempt in range(self.max_attempts):
                r = self.read_gauge(self.robot.capture(cp, attempt))
                if best is None or r.confidence > best.confidence:
                    best = r
                if r.confidence >= self.min_confidence:
                    break
            status = "OK" if best.confidence >= self.min_confidence else "NEEDS_REVIEW"
            self._emit(mission_id, cp, now, status=status, value=best.value, unit=gauge.SPEC.unit,
                       confidence=best.confidence, detail={"attempts": attempt + 1, "method": best.method})
            return status
        res = thermal.inspect(self.robot.capture(cp, 0), self.thermal_refs[cp.id])
        # registration that needs a huge shift means the robot is not where it thinks it is
        status = "OK" if max(map(abs, res.shift)) < 20 else "NEEDS_REVIEW"
        self._emit(mission_id, cp, now, status=status, value=res.delta_t, unit="degC_dT", confidence=1.0,
                   detail={"severity": res.severity, "hottest_phase": res.hottest_phase, "hotspots": res.hotspots, "shift_px": res.shift})
        return status

    def _emit(self, mission_id, cp, now, status, value=None, unit=None, confidence=None, detail=None):
        self.outbox.put({"idempotencyKey": result_key(mission_id, cp.id), "robotId": self.robot.robot_id,
                         "missionId": mission_id, "checkpointId": cp.id, "kind": cp.kind, "status": status,
                         "value": value, "unit": unit, "confidence": confidence, "detail": detail or {},
                         "capturedAt": now().isoformat()})
