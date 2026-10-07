"""envs.reset.sample_command -- default behavior unchanged, and
zero_command_prob (EXP-0042 / F-0005 follow-up) actually mixes in exact
(0,0) commands at roughly the requested rate."""

import numpy as np
import pytest

from envs.reset import sample_command

VX_RANGE = (0.0, 0.6)
VY_RANGE = (-0.2, 0.2)


def test_default_zero_command_prob_never_forces_exact_zero():
    # With plain continuous uniform sampling, landing on exactly (0,0) is a
    # measure-zero event -- across many draws it should not happen even once.
    rng = np.random.default_rng(0)
    draws = [sample_command(rng, VX_RANGE, VY_RANGE) for _ in range(10_000)]
    assert not any(np.array_equal(d, np.zeros(2, dtype=np.float32)) for d in draws)


def test_zero_command_prob_one_always_returns_zero():
    rng = np.random.default_rng(0)
    for _ in range(50):
        cmd = sample_command(rng, VX_RANGE, VY_RANGE, zero_command_prob=1.0)
        assert cmd == pytest.approx(np.zeros(2))


def test_zero_command_prob_matches_requested_rate_approximately():
    rng = np.random.default_rng(0)
    n = 20_000
    p = 0.2
    draws = [sample_command(rng, VX_RANGE, VY_RANGE, zero_command_prob=p) for _ in range(n)]
    n_zero = sum(1 for d in draws if np.array_equal(d, np.zeros(2, dtype=np.float32)))
    assert n_zero / n == pytest.approx(p, abs=0.02)


def test_non_zero_draws_still_respect_the_configured_ranges():
    rng = np.random.default_rng(0)
    draws = [sample_command(rng, VX_RANGE, VY_RANGE, zero_command_prob=0.5) for _ in range(2000)]
    nonzero = [d for d in draws if not np.array_equal(d, np.zeros(2, dtype=np.float32))]
    assert nonzero  # sanity: some non-zero draws actually happened
    for d in nonzero:
        assert VX_RANGE[0] <= d[0] <= VX_RANGE[1]
        assert VY_RANGE[0] <= d[1] <= VY_RANGE[1]
