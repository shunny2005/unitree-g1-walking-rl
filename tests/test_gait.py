import dataclasses

import numpy as np
import pytest

from config.loader import GaitClockConfig, load_config
from envs.g1_walk_env import G1WalkEnv
from envs.gait import (
    clock_features,
    contact_foot_vel_raw,
    contact_schedule_raw,
    leg_phases,
    stance_mask,
    swing_height_raw,
)

DT = 0.02


def test_phase_is_periodic_with_the_configured_period():
    steps_per_period = int(round(0.8 / DT))  # 40
    assert np.allclose(leg_phases(0, DT, 0.8, 0.5), [0.0, 0.5])
    assert np.allclose(leg_phases(steps_per_period, DT, 0.8, 0.5), leg_phases(0, DT, 0.8, 0.5))
    assert np.allclose(leg_phases(10, DT, 0.8, 0.5), [0.25, 0.75])


def test_right_leg_runs_half_a_cycle_behind_and_wraps():
    left, right = leg_phases(30, DT, 0.8, 0.5)  # left = 0.75 -> right wraps to 0.25
    assert left == pytest.approx(0.75)
    assert right == pytest.approx(0.25)


def test_clock_features_are_sin_cos_of_left_phase():
    assert np.allclose(clock_features(np.array([0.0, 0.5])), [0.0, 1.0])
    assert np.allclose(clock_features(np.array([0.25, 0.75])), [1.0, 0.0], atol=1e-6)
    f = clock_features(np.array([0.37, 0.87]))
    assert f.dtype == np.float32 and float(f @ f) == pytest.approx(1.0, abs=1e-6)


def test_stance_mask_uses_the_stance_fraction_threshold():
    assert np.array_equal(stance_mask(np.array([0.0, 0.54]), 0.55), [True, True])
    assert np.array_equal(stance_mask(np.array([0.55, 0.9]), 0.55), [False, False])


def test_only_one_foot_is_in_swing_at_a_time_with_half_cycle_offset():
    """Stance fraction 0.55 > 0.5 means a brief double-support overlap, never a double-swing."""
    for step in range(40):
        stance = stance_mask(leg_phases(step, DT, 0.8, 0.5), 0.55)
        assert stance.any(), f"both feet scheduled to swing at step {step}"


def test_contact_schedule_counts_matching_feet():
    stance = np.array([True, False])
    assert contact_schedule_raw(np.array([True, False]), stance) == 2.0
    assert contact_schedule_raw(np.array([True, True]), stance) == 1.0
    assert contact_schedule_raw(np.array([False, True]), stance) == 0.0


def test_never_lifting_a_foot_is_penalized_every_step_of_its_swing_window():
    """The property that makes this un-dodgeable, unlike a touchdown-only reward."""
    both_down = np.array([True, True])
    raws = [contact_schedule_raw(both_down, stance_mask(leg_phases(s, DT, 0.8, 0.5), 0.55)) for s in range(40)]
    assert min(raws) < 2.0            # some steps a foot should be airborne but is not
    assert sum(r < 2.0 for r in raws) >= 30


def test_swing_height_counts_only_airborne_feet():
    z = np.array([0.08, 0.05])
    assert swing_height_raw(z, np.array([False, False]), 0.08) == pytest.approx(0.03**2)
    assert swing_height_raw(z, np.array([True, True]), 0.08) == 0.0
    assert swing_height_raw(z, np.array([True, False]), 0.08) == pytest.approx(0.03**2)


def test_contact_foot_velocity_counts_only_grounded_feet():
    vel = np.array([[0.3, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert contact_foot_vel_raw(vel, np.array([True, False])) == pytest.approx(0.09)
    assert contact_foot_vel_raw(vel, np.array([False, False])) == 0.0
    assert contact_foot_vel_raw(vel, np.array([True, True])) == pytest.approx(0.09 + 2.0)


@pytest.fixture(scope="module")
def base_cfg():
    return load_config("config/exp0012_air_time.yaml")


def _with_clock(cfg, **weights):
    return dataclasses.replace(
        cfg,
        gait_clock=GaitClockConfig(enabled=True),
        reward_weights=dataclasses.replace(cfg.reward_weights, **weights),
    )


def test_observation_is_44_dims_by_default_and_46_with_the_clock(base_cfg):
    off, on = G1WalkEnv(base_cfg), G1WalkEnv(_with_clock(base_cfg))
    try:
        assert off.reset(seed=0)[0].shape == (44,)
        obs, _ = on.reset(seed=0)
        assert obs.shape == (46,) and on.observation_space.shape == (46,)
        assert np.allclose(obs[-2:], [0.0, 1.0])  # phase 0 -> sin 0, cos 1
    finally:
        off.close(); on.close()


def test_clock_entries_advance_with_the_step_count(base_cfg):
    env = G1WalkEnv(_with_clock(base_cfg))
    try:
        env.reset(seed=0)
        obs = None
        for _ in range(10):  # quarter period at 0.8 s / 0.02 s
            obs, *_ = env.step(np.zeros(12))
        assert np.allclose(obs[-2:], [1.0, 0.0], atol=1e-5)
    finally:
        env.close()


def test_default_config_reward_is_unchanged_by_the_new_terms(base_cfg):
    env = G1WalkEnv(base_cfg)
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(12))
        assert info["gait_contact_bonus"] == 0.0
        assert info["feet_swing_height_penalty"] == 0.0 and info["contact_foot_vel_penalty"] == 0.0
    finally:
        env.close()


def test_gait_terms_flow_into_the_reward_when_enabled(base_cfg):
    env = G1WalkEnv(_with_clock(base_cfg, gait_contact_bonus=0.18, contact_foot_vel_penalty=0.2))
    try:
        env.reset(seed=0)
        # the reset pose holds the feet ~7 mm above the floor; they land by step 2
        for _ in range(5):
            _, _, _, _, info = env.step(np.zeros(12))
        assert info["gait_contact_bonus"] > 0.0          # standing still matches part of the schedule
        assert info["contact_foot_vel_penalty"] <= 0.0
    finally:
        env.close()


def test_schedule_reward_without_the_clock_is_rejected(base_cfg):
    bad = dataclasses.replace(base_cfg, reward_weights=dataclasses.replace(base_cfg.reward_weights, gait_contact_bonus=0.18))
    with pytest.raises(ValueError, match="gait_clock"):
        G1WalkEnv(bad)


def test_run_episode_reports_straightness_and_clock_adherence(base_cfg):
    """A random policy falls fast, but the new per-episode keys must still be well formed."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from evaluation.evaluate import run_episode

    clocked = dataclasses.replace(base_cfg, gait_clock=GaitClockConfig(enabled=True))
    env = G1WalkEnv(clocked)
    model = PPO("MlpPolicy", DummyVecEnv([lambda: G1WalkEnv(clocked)]), device="cpu", n_steps=64, batch_size=32)
    try:
        e = run_episode(model, env, seed=1)
        assert 0.0 <= e["clock_adherence"] <= 1.0
        assert 0.0 <= e["straightness"] <= 1.0 + 1e-9 and e["path_length_m"] >= 0.0
    finally:
        env.close()


def test_clock_adherence_is_nan_when_the_gait_clock_is_off(base_cfg):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from evaluation.evaluate import run_episode

    env = G1WalkEnv(base_cfg)
    model = PPO("MlpPolicy", DummyVecEnv([lambda: G1WalkEnv(base_cfg)]), device="cpu", n_steps=64, batch_size=32)
    try:
        assert np.isnan(run_episode(model, env, seed=1)["clock_adherence"])
    finally:
        env.close()
