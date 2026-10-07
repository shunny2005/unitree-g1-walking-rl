import numpy as np
import pytest

from evaluation.metrics import velocity_error_decomposition

CMD = np.array([0.3, 0.0])
T = 400


def _vel(bias=0.0, ripple_amp=0.0, period_steps=40):
    t = np.arange(T)
    vx = 0.3 + bias + ripple_amp * np.sin(2 * np.pi * t / period_steps)
    return np.stack([vx, np.zeros(T)], axis=1)


def test_perfect_tracking_has_zero_error_everywhere():
    d = velocity_error_decomposition(_vel(), CMD)
    assert d["instantaneous"] == pytest.approx(0.0, abs=1e-9)
    assert d["period_avg"] == pytest.approx(0.0, abs=1e-9)
    assert d["ripple_rms"] == pytest.approx(0.0, abs=1e-9)


def test_pure_bias_is_all_period_avg_error_and_no_ripple():
    d = velocity_error_decomposition(_vel(bias=0.05), CMD)
    assert d["period_avg"] == pytest.approx(0.05, abs=1e-9)
    assert d["ripple_rms"] == pytest.approx(0.0, abs=1e-9)


def test_pure_ripple_at_the_window_period_is_averaged_out_of_period_avg():
    d = velocity_error_decomposition(_vel(ripple_amp=0.08), CMD)
    assert d["period_avg"] == pytest.approx(0.0, abs=1e-9)          # a whole period averages to nothing
    # amp/sqrt(2) for whole periods; after skipping the start-up transient the sample is a partial period, so allow 2%
    assert d["ripple_rms"] == pytest.approx(0.08 / np.sqrt(2), rel=0.02)
    assert d["instantaneous"] > 0.05                                 # yet the instantaneous metric sees it


def test_ripple_raises_instantaneous_error_but_not_period_avg_error():
    smooth = velocity_error_decomposition(_vel(bias=0.03), CMD)
    rippled = velocity_error_decomposition(_vel(bias=0.03, ripple_amp=0.1), CMD)
    assert rippled["instantaneous"] > smooth["instantaneous"] + 0.02
    assert rippled["period_avg"] == pytest.approx(smooth["period_avg"], abs=1e-9)


def test_short_episode_returns_nan_instead_of_a_misleading_number():
    d = velocity_error_decomposition(_vel()[:50], CMD)
    assert all(np.isnan(v) for v in d.values())


def test_startup_transient_is_skipped():
    v = _vel()
    v[:10] = [5.0, 5.0]  # violent first steps
    assert velocity_error_decomposition(v, CMD)["instantaneous"] == pytest.approx(0.0, abs=1e-9)
