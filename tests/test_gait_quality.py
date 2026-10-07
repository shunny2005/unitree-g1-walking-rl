"""evaluation.gait_quality.compute_metrics -- pure-function tests on a small,
hand-constructed synthetic rollout (no simulator needed), following this
project's convention (control/pd_controller.py, envs/gait.py) of testing
formulas independently of MuJoCo."""

import numpy as np
import pytest

from evaluation.gait_quality import compute_metrics

DT = 0.02
PDT = 0.002
DECIMATION = 10
N = 6  # control steps


def _base_raw(**overrides) -> dict:
    n = N
    raw = dict(
        command=np.array([0.3, 0.0]),
        control_dt=DT,
        physics_dt=PDT,
        steps=n,
        episode_length_s=n * DT,
        terminated=False,
        termination_reason=None,
        action=np.zeros((n, 12)),
        target=np.zeros((n, 12)),
        qpos=np.zeros((n, 12)),
        qvel=np.zeros((n, 12)),
        foot_z=np.full((n, 2), 0.03),
        foot_vel_z=np.zeros((n, 2)),
        # left airborne at steps 1-2 (touchdown at 3); right airborne at
        # steps 2-3 (touchdown at 4). Both start grounded (step 0).
        contact=np.array(
            [
                [True, True],
                [False, True],
                [False, False],
                [True, False],
                [True, True],
                [True, True],
            ]
        ),
        pelvis_xy=np.stack([np.linspace(0, 1.2, n), np.zeros(n)], axis=1),  # 1.2 m forward over the episode
        pelvis_z=np.full(n, 0.75),
        yaw=np.zeros(n),
        roll=np.zeros(n),
        pitch=np.zeros(n),
        ang_vel_z=np.zeros(n),
        substep_torque=np.zeros((n * DECIMATION, 12)),
        substep_qvel=np.zeros((n * DECIMATION, 12)),
        total_mass_kg=35.0,
    )
    raw.update(overrides)
    return raw


def test_swing_and_stance_durations_match_the_hand_constructed_contact_pattern():
    m = compute_metrics(_base_raw())
    # left: airborne steps 1-2 -> a 2-step (40ms) swing run
    assert m["swing_duration_mean_left_ms"] == pytest.approx(40.0)
    # right: airborne steps 2-3 -> a 2-step (40ms) swing run
    assert m["swing_duration_mean_right_ms"] == pytest.approx(40.0)
    assert m["left_liftoffs"] == 1
    assert m["right_liftoffs"] == 1


def test_double_support_and_both_airborne_fractions():
    m = compute_metrics(_base_raw())
    # both-grounded at steps 0, 4, 5 -> 3/6
    assert m["double_support_fraction"] == pytest.approx(3 / 6)
    # both-airborne at step 2 (left's swing steps 1-2 overlaps right's swing
    # steps 2-3 for exactly one step) -> 1/6
    assert m["both_airborne_fraction"] == pytest.approx(1 / 6)


def test_asymmetric_torque_produces_correctly_signed_asymmetry_ratio():
    torque = np.zeros((N * DECIMATION, 12))
    torque[:, :6] = 10.0  # left leg: constant 10 N*m
    torque[:, 6:] = 5.0  # right leg: constant 5 N*m
    m = compute_metrics(_base_raw(substep_torque=torque))
    assert m["torque_rms_left"] == pytest.approx(10.0)
    assert m["torque_rms_right"] == pytest.approx(5.0)
    # asym = (left - right) / (|left| + |right|) = 5/15
    assert m["torque_rms_asym"] == pytest.approx(5.0 / 15.0)


def test_mechanical_work_matches_hand_calculation_for_constant_power():
    # constant torque=2 N*m, constant qvel=3 rad/s on every joint/substep ->
    # power = 6 W/joint * 12 joints = 72 W, for N*DECIMATION substeps * PDT seconds
    torque = np.full((N * DECIMATION, 12), 2.0)
    qvel = np.full((N * DECIMATION, 12), 3.0)
    raw = _base_raw(substep_torque=torque, substep_qvel=qvel)
    m = compute_metrics(raw)
    expected_work = 72.0 * (N * DECIMATION) * PDT
    assert m["mech_abs_work_j"] == pytest.approx(expected_work)
    assert m["mech_positive_work_j"] == pytest.approx(expected_work)  # all positive here


def test_negative_power_counts_toward_absolute_but_not_positive_work():
    torque = np.full((N * DECIMATION, 12), 2.0)
    qvel = np.full((N * DECIMATION, 12), -3.0)  # torque and velocity opposed -> negative power
    raw = _base_raw(substep_torque=torque, substep_qvel=qvel)
    m = compute_metrics(raw)
    assert m["mech_positive_work_j"] == pytest.approx(0.0)
    expected_abs_work = 72.0 * (N * DECIMATION) * PDT
    assert m["mech_abs_work_j"] == pytest.approx(expected_abs_work)


def test_cost_of_transport_matches_hand_calculation():
    torque = np.full((N * DECIMATION, 12), 1.0)
    qvel = np.full((N * DECIMATION, 12), 1.0)
    raw = _base_raw(substep_torque=torque, substep_qvel=qvel, total_mass_kg=35.0)
    m = compute_metrics(raw)
    distance_m = 1.2  # from _base_raw's pelvis_xy
    abs_work = 12.0 * (N * DECIMATION) * PDT  # 12 joints * 1W each
    expected_cot = abs_work / (35.0 * 9.81 * distance_m)
    assert m["cost_of_transport"] == pytest.approx(expected_cot, rel=1e-6)


def test_action_delta_and_jerk_are_zero_for_a_constant_action():
    raw = _base_raw(action=np.tile(np.arange(12, dtype=float), (N, 1)))
    m = compute_metrics(raw)
    assert m["action_delta_rms"] == pytest.approx(0.0)
    assert m["action_jerk_rms"] == pytest.approx(0.0)


def test_action_delta_matches_hand_calculation_for_a_step_change():
    action = np.zeros((N, 12))
    action[N // 2 :] = 1.0  # a single step change halfway through, all 12 dims
    m = compute_metrics(_base_raw(action=action))
    # one nonzero delta of norm sqrt(12), the rest zero -> RMS over (N-1) deltas
    n_deltas = N - 1
    expected_rms = np.sqrt((12.0) / n_deltas)  # sum of squared norms = 12, divided by n_deltas, sqrt
    assert m["action_delta_rms"] == pytest.approx(expected_rms)
