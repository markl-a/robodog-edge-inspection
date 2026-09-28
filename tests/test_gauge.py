import numpy as np
import pytest

from edge.gauge import SPEC, dataset, read_classic


@pytest.mark.parametrize("v", [0.0, 2.5, 5.0, 7.5, 10.0])
def test_angle_value_round_trip(v):
    assert SPEC.angle_to_value(SPEC.value_to_angle(v) % 360) == pytest.approx(v, abs=1e-9)


def test_dead_zone_snaps_to_nearest_stop():
    assert SPEC.angle_to_value(260.0) == 0.0      # just past the min stop (clockwise-negative side)
    assert SPEC.angle_to_value(280.0) == 10.0     # just past the max stop


def test_classic_reader_mostly_within_gauge_accuracy_class():
    imgs, vals = dataset(150, seed=99, size=96)
    err = np.abs(np.array([read_classic(i).value for i in imgs]) - vals)
    assert (err <= SPEC.tolerance).mean() > 0.9
    assert err.max() < 0.6


def test_confidence_gate_catches_most_gross_errors_on_occluded_dials():
    # seed 98 contains a known failure: a cable almost parallel to the needle (results/failure_case.png)
    imgs, vals = dataset(300, seed=98, size=96, hard=True)
    rs = [read_classic(i) for i in imgs]
    err = np.abs(np.array([r.value for r in rs]) - vals)
    conf = np.array([r.confidence for r in rs])
    gross = err > 1.0
    assert gross.sum() >= 1
    assert (gross & (conf < 0.3)).sum() / gross.sum() >= 0.5
    assert (gross & (conf >= 0.3)).mean() < 0.005          # confident-and-wrong stays rare
