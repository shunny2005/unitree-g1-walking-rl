import math

import numpy as np
import pytest

from evaluation.banded import aggregate, aggregate_by_band, band_label, command_speed


def ep(speed, falls=False, ratio=0.9, yaw=10.0, straight=0.95, adherence=float("nan"), reliable=True):
    return {
        "command": [speed, 0.0],
        "termination_reason": "fell_height" if falls else None,
        "distance_ratio": ratio, "distance_ratio_reliable": reliable,
        "mean_vel_error": 0.04, "mean_period_avg_vel_error": 0.02, "vel_ripple_rms": 0.03,
        "yaw_drift_per_10s_deg": yaw, "straightness": straight, "clock_adherence": adherence,
    }


def test_command_speed_is_the_vector_norm():
    assert command_speed({"command": [0.3, 0.4]}) == pytest.approx(0.5)


def test_band_labels():
    assert band_label(0.0, 0.15) == "<0.15"
    assert band_label(0.15, 0.4) == "0.15-0.4"
    assert band_label(0.4, math.inf) == ">=0.4"


def test_episodes_land_in_the_right_bands_and_edges_belong_to_the_upper_band():
    eps = [ep(0.05), ep(0.15), ep(0.30), ep(0.40), ep(0.55)]
    b = aggregate_by_band(eps)
    assert b["all"]["n"] == 5
    assert (b["<0.15"]["n"], b["0.15-0.4"]["n"], b[">=0.4"]["n"]) == (1, 2, 2)


def test_empty_bands_are_omitted():
    assert "0.15-0.4" not in aggregate_by_band([ep(0.05), ep(0.5)])


def test_falls_and_reliable_only_distance_ratio():
    a = aggregate([ep(0.5, falls=True, ratio=5.0, reliable=False), ep(0.5, ratio=0.8), ep(0.5, ratio=1.0)])
    assert a["falls"] == 1 and a["n"] == 3
    assert a["dist_ratio"] == pytest.approx(0.9)          # the short failed episode's spurious 5.0 is excluded


def test_yaw_is_averaged_in_absolute_value():
    assert aggregate([ep(0.5, yaw=30.0), ep(0.5, yaw=-10.0)])["abs_yaw_per_10s_deg"] == pytest.approx(20.0)


def test_nan_metrics_do_not_poison_the_mean_and_an_all_nan_column_stays_nan():
    a = aggregate([ep(0.5, adherence=float("nan")), ep(0.5, adherence=float("nan"))])
    assert np.isnan(a["clock_adherence"])
    assert aggregate([ep(0.5, adherence=0.8), ep(0.5, adherence=float("nan"))])["clock_adherence"] == pytest.approx(0.8)
