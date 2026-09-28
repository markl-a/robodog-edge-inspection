import numpy as np
import pytest

from edge.thermal import PHASES, classify, inspect, register, render


@pytest.mark.parametrize("dt,sev", [(0.5, "normal"), (2.0, "possible"), (8.0, "probable"), (16.0, "major")])
def test_neta_bands(dt, sev):
    assert classify(dt) == sev


def test_registration_recovers_robot_pose_error():
    ref = render(np.random.default_rng(0), {p: 12.0 for p in PHASES})
    frame = render(np.random.default_rng(1), {p: 15.0 for p in PHASES}, shift=(6.0, -7.5))
    dy, dx = register(frame, ref)
    assert dy == pytest.approx(6.0, abs=0.6) and dx == pytest.approx(-7.5, abs=0.6)


def test_hot_lug_found_only_with_registration():
    ref = render(np.random.default_rng(0), {p: 12.0 for p in PHASES})
    frame = render(np.random.default_rng(2), {p: 12.0 for p in PHASES}, {"B": 12.0}, shift=(7.0, 8.0))
    good = inspect(frame, ref)
    assert good.hottest_phase == "B" and good.severity in ("probable", "major")
    assert inspect(frame).severity in ("normal", "possible")   # fixed ROIs miss the spot
