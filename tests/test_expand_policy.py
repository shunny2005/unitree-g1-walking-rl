import dataclasses

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config.loader import GaitClockConfig, load_config
from envs.g1_walk_env import G1WalkEnv
from training.expand_policy import load_expanded


def _ppo(cfg, seed=0):
    env = DummyVecEnv([lambda: G1WalkEnv(cfg)])
    return PPO(
        "MlpPolicy", env, n_steps=64, batch_size=32, seed=seed, device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])),
    )


@pytest.fixture(scope="module")
def cfgs():
    base = load_config("config/exp0012_air_time.yaml")
    return base, dataclasses.replace(base, gait_clock=GaitClockConfig(enabled=True))


def test_expanded_policy_computes_the_same_function_regardless_of_the_new_inputs(cfgs, tmp_path):
    base, clocked = cfgs
    old = _ppo(base, seed=1)
    ckpt = str(tmp_path / "old.zip")
    old.save(ckpt)

    new = _ppo(clocked, seed=2)  # different seed: its own init must be fully overwritten
    assert load_expanded(new, ckpt) == 2

    rng = np.random.default_rng(0)
    for _ in range(5):
        obs44 = rng.normal(size=(1, 44)).astype(np.float32)
        for clock in (np.zeros((1, 2)), rng.normal(size=(1, 2)) * 10):
            obs46 = np.concatenate([obs44, clock.astype(np.float32)], axis=1)
            a_old, _ = old.predict(obs44, deterministic=True)
            a_new, _ = new.predict(obs46, deterministic=True)
            assert np.allclose(a_old, a_new, atol=1e-6)
            v_old = old.policy.predict_values(torch.as_tensor(obs44))
            v_new = new.policy.predict_values(torch.as_tensor(obs46))
            assert torch.allclose(v_old, v_new, atol=1e-6)


def test_new_input_columns_are_zero_and_trainable(cfgs, tmp_path):
    base, clocked = cfgs
    old = _ppo(base)
    ckpt = str(tmp_path / "old.zip")
    old.save(ckpt)
    new = _ppo(clocked, seed=3)
    load_expanded(new, ckpt)
    first = new.policy.mlp_extractor.policy_net[0].weight
    assert first.shape[1] == 46
    assert torch.count_nonzero(first[:, 44:]) == 0
    assert first.requires_grad


def test_same_size_checkpoint_loads_unchanged(cfgs, tmp_path):
    base, _ = cfgs
    old = _ppo(base, seed=4)
    ckpt = str(tmp_path / "old.zip")
    old.save(ckpt)
    new = _ppo(base, seed=5)
    assert load_expanded(new, ckpt) == 0
    obs = np.random.default_rng(1).normal(size=(1, 44)).astype(np.float32)
    assert np.allclose(old.predict(obs, deterministic=True)[0], new.predict(obs, deterministic=True)[0])
