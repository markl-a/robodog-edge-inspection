"""End-to-end demo: patrol a small substation route, lose the network midway, recover, and write reports.

python -m edge.demo        -> results/patrol_demo.json, results/thermal_report.json
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .mission import Checkpoint, MissionRunner, SimRobot
from .thermal import PHASES, evaluate, render
from .uplink import Outbox

OUT = Path(__file__).resolve().parent.parent / "results"

ROUTE = [
    Checkpoint("B1-PG-01", "gauge", (12.0, 3.0)),      # chilled-water supply pressure
    Checkpoint("B1-PG-02", "gauge", (18.0, 3.0)),      # return pressure
    Checkpoint("B1-SW-01", "thermal", (25.0, 9.0)),    # LV breaker, feeder 1
    Checkpoint("B1-SW-02", "thermal", (28.0, 9.0)),    # LV breaker, feeder 2
    Checkpoint("B1-PG-03", "gauge", (40.0, 15.0)),     # fire-water header (behind a cable tray)
    Checkpoint("B1-PG-04", "gauge", (55.0, 20.0)),     # far end of the plant room
]
PLANT = {
    "B1-PG-01": 4.2, "B1-PG-02": 3.6, "B1-PG-03": 7.9, "B1-PG-04": 2.1,
    "B1-SW-01": ({p: 14.0 for p in PHASES}, {}),
    "B1-SW-02": ({p: 14.0 for p in PHASES}, {"C": 11.0}),   # loose lug on phase C
}


class FakePlatform:
    def __init__(self):
        self.stored, self.up = {}, False

    def __call__(self, rows):
        if not self.up:
            raise ConnectionError("plant-room Wi-Fi down")
        for r in rows:
            self.stored.setdefault(r["idempotencyKey"], r)
        return {r["idempotencyKey"] for r in rows}


def main():
    OUT.mkdir(exist_ok=True)
    refs = {cp.id: render(np.random.default_rng(0), {p: 12.0 for p in PHASES}) for cp in ROUTE if cp.kind == "thermal"}
    t = [datetime(2026, 9, 28, 2, 0, tzinfo=timezone.utc)]

    def clock():
        t[0] += timedelta(seconds=40)
        return t[0]

    robot = SimRobot(PLANT, seed=3, glare={"B1-PG-03"}, blocked={(55.0, 20.0)})
    outbox, platform = Outbox(), FakePlatform()
    summary = MissionRunner(robot, outbox, refs).run("2026-09-28-night", ROUTE, now=clock)
    during = outbox.flush(platform)            # link down in the plant room
    platform.up = True
    after = outbox.flush(platform)             # back at the dock
    results = sorted(platform.stored.values(), key=lambda r: r["capturedAt"])
    for r in results:
        if r["kind"] == "gauge" and r["value"] is not None:
            r["truth"] = PLANT[r["checkpointId"]]
    demo = {"summary": summary, "flush_in_plant_room": during, "flush_at_dock": after, "results": results}
    (OUT / "patrol_demo.json").write_text(json.dumps(demo, indent=2))
    (OUT / "thermal_report.json").write_text(json.dumps(evaluate(), indent=2))
    print(json.dumps(summary), json.dumps(during), json.dumps(after))
    for r in results:
        print(f'{r["checkpointId"]:9} {r["status"]:12} {r["value"]!s:7} {r.get("truth", "")!s:5} {r["detail"]}')


if __name__ == "__main__":
    main()
