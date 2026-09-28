import numpy as np

from edge.mission import Checkpoint, MissionRunner, SimRobot
from edge.thermal import PHASES, render
from edge.uplink import Outbox, result_key

CPS = [Checkpoint("G1", "gauge", (10.0, 0.0)), Checkpoint("T1", "thermal", (20.0, 5.0)), Checkpoint("G2", "gauge", (30.0, 0.0))]
REFS = {"T1": render(np.random.default_rng(0), {p: 12.0 for p in PHASES})}
PLANT = {"G1": 6.2, "G2": 3.1, "T1": ({p: 12.0 for p in PHASES}, {"A": 10.0})}


def run(robot, **kw):
    ob = Outbox()
    summary = MissionRunner(robot, ob, REFS, **kw).run("m-1", CPS)
    return summary, {r["checkpointId"]: r for r in ob.pending()}


def test_full_patrol_reports_every_checkpoint():
    summary, res = run(SimRobot(PLANT, seed=1))
    assert set(res) == {"G1", "T1", "G2"}
    assert abs(res["G1"]["value"] - 6.2) < 0.3 and abs(res["G2"]["value"] - 3.1) < 0.3
    assert res["T1"]["detail"]["hottest_phase"] == "A" and res["T1"]["detail"]["severity"] in ("probable", "major")
    assert res["G1"]["unit"] == "bar" and "status" in res["G1"]          # measurement, not an alarm decision


def test_low_battery_docks_and_marks_rest_skipped():
    summary, res = run(SimRobot(PLANT, battery=21.0, seed=1))
    assert res["G2"]["status"] == "SKIPPED" and res["G2"]["detail"]["reason"] == "battery"
    assert summary["battery_left_pct"] > 15.0 - 1e-9 - 5      # got home with margin


def test_unreachable_checkpoint_is_reported_not_dropped():
    summary, res = run(SimRobot(PLANT, blocked={(20.0, 5.0)}, seed=1))
    assert res["T1"]["status"] == "SKIPPED" and res["T1"]["detail"]["reason"] == "unreachable"
    assert res["G2"]["status"] == "OK"


def test_occluded_first_look_is_retried():
    retried = 0
    for seed in range(40):
        _, res = run(SimRobot(PLANT, glare={"G1"}, seed=seed))
        if res["G1"]["detail"]["attempts"] > 1:
            retried += 1
            assert abs(res["G1"]["value"] - 6.2) < 0.5
    assert retried > 0


def test_idempotency_key_stable_per_mission_and_checkpoint():
    assert result_key("m-1", "G1") == result_key("m-1", "G1") != result_key("m-2", "G1")
