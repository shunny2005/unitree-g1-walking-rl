"""Reset behavior. No domain randomization yet (challenge Phase 8 -- add
only after basic walking works, per CLAUDE.md). Reset is deterministic
given a seed: same keyframe base pose + default leg pose, zero velocity.
"""

from __future__ import annotations

import mujoco
import numpy as np

from control.joint_map import JointMap


def reset_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: JointMap,
    q_default: np.ndarray,
    stand_key_id: int,
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, stand_key_id)
    data.qpos[joint_map.qpos_adr] = q_default
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def sample_command(
    rng: np.random.Generator,
    vx_range: tuple[float, float],
    vy_range: tuple[float, float],
    zero_command_prob: float = 0.0,
) -> np.ndarray:
    # zero_command_prob (F-0005 follow-up, EXP-0042): with plain continuous
    # uniform sampling, an exact (0,0) command has ~0 probability, so the
    # zero-command case is never actually trained on even though vx_range
    # usually includes 0.0 as its lower bound. Mix in exact zero commands
    # explicitly so the policy is optimized for that case directly, instead
    # of relying on a reward-penalty patch to generalize to an input the
    # training distribution never produces.
    if zero_command_prob > 0.0 and rng.uniform() < zero_command_prob:
        return np.array([0.0, 0.0], dtype=np.float32)
    vx = rng.uniform(*vx_range)
    vy = rng.uniform(*vy_range)
    return np.array([vx, vy], dtype=np.float32)
