"""Periodic gait clock and the per-step gait reward quantities built on it
(R-0004, docs/RESEARCH_LOG.md). Pure numpy, independent of MuJoCo, so it is
unit-testable without a simulator -- same reason as control/pd_controller.py.

The clock and terms follow the official Unitree G1 recipe
(legged_gym/envs/g1/g1_env.py in unitreerobotics/unitree_rl_gym):
  phase       = (step * control_dt) % period / period
  left phase  = phase,   right phase = (phase + offset) % 1
  in stance   = leg phase < stance_fraction (0.55)
Each function returns a *raw* (unweighted) value; the env multiplies by the
configured weight, keeping these pure state-free functions.
"""

from __future__ import annotations

import numpy as np


def leg_phases(step: int, control_dt: float, period_s: float, offset: float) -> np.ndarray:
    """[left, right] phase in [0, 1) at control step `step`."""
    phase = (step * control_dt) % period_s / period_s
    return np.array([phase, (phase + offset) % 1.0])


def clock_features(phases: np.ndarray) -> np.ndarray:
    """[sin, cos] of the LEFT-foot phase -- the same two observation entries
    the Unitree G1 uses. The right foot is a fixed offset behind, so it adds
    no information."""
    angle = 2.0 * np.pi * phases[0]
    return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)


def stance_mask(phases: np.ndarray, stance_fraction: float) -> np.ndarray:
    """True where the schedule says the foot should be on the ground."""
    return phases < stance_fraction


def contact_schedule_raw(contact: np.ndarray, scheduled_stance: np.ndarray) -> float:
    """Number of feet (0, 1 or 2) whose contact state matches the schedule.
    Evaluated every step, so a foot that never lifts loses this reward
    continuously -- it cannot be dodged the way a touchdown-only reward can."""
    return float(np.sum(contact == scheduled_stance))


def swing_height_raw(foot_z: np.ndarray, contact: np.ndarray, target_m: float) -> float:
    """Sum over airborne feet of (foot height - target)^2. Grounded feet are
    not counted."""
    return float(np.sum(np.square(foot_z - target_m) * ~contact))


def contact_foot_vel_raw(foot_vel: np.ndarray, contact: np.ndarray) -> float:
    """Sum over grounded feet of |foot velocity|^2 (foot_vel is (2, 3)).
    Airborne feet are not counted. A foot that slides while in contact --
    the skating gait of F-0004 -- pays for it directly."""
    return float(np.sum(np.square(foot_vel) * contact[:, None]))
