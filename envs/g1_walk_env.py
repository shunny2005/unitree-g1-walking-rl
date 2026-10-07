"""Gymnasium environment: Unitree G1, 12 leg joints RL-controlled via PD,
walking task with a commanded (vx, vy) velocity.

Vertical slice built on top of already-verified pieces:
- model/indexing verified in scripts/smoke_test_model.py
- control mapping verified in scripts/smoke_test_pd_hold.py

See docs/SIMULATION.md and docs/RL_PIPELINE.md for the full contract.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from config.loader import Config
from control.default_pose import UPPER_BODY_ORDER, UPPER_BODY_TARGETS, build_q_default
from control.joint_map import build_joint_map
from control.pd_controller import action_to_target, build_asymmetric_gains, gains_from_config, pd_torque
from envs.contacts import AirTimeTracker, GroundTimeTracker, both_feet_contact, verify_foot_geoms
from envs.domain_randomization import DomainRandomizer, PushScheduler
from envs.gait import (
    clock_features,
    contact_foot_vel_raw,
    contact_schedule_raw,
    leg_phases,
    stance_mask,
    swing_height_raw,
)
from envs.observations import base_frame_velocity, build_observation, obs_dim, projected_gravity, quat_to_yaw
from envs.reset import reset_state, sample_command
from envs.termination import check_termination
from rewards.walking_reward import compute_reward

REPO_ROOT = Path(__file__).resolve().parent.parent


class G1WalkEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(self, config: Config, render_mode: str | None = None):
        self.cfg = config
        self.render_mode = render_mode

        model_path = REPO_ROOT / config.model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        assert abs(self.model.opt.timestep - config.sim.physics_dt) < 1e-9, (
            f"model physics dt {self.model.opt.timestep} != config {config.sim.physics_dt}"
        )

        self.joint_map = build_joint_map(self.model)
        self.q_default = build_q_default()
        if config.pd_gains_left is not None:
            self.kp, self.kd = build_asymmetric_gains(config.pd_gains, config.pd_gains_left)
        else:
            self.kp, self.kd = gains_from_config(config.pd_gains)
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        assert self.pelvis_id >= 0 and self.stand_key_id >= 0
        self.upper_body_actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in UPPER_BODY_ORDER]
        )
        verify_foot_geoms(self.model)
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._air_time_tracker = AirTimeTracker()
        self._ground_time_tracker = GroundTimeTracker()
        self._domain_randomizer = DomainRandomizer(self.model, self.floor_geom_id)
        self._push_scheduler = PushScheduler()
        self._motor_strength = 1.0

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self._gait = config.gait_clock
        self._obs_dim = obs_dim(self._gait.enabled)
        scheduled_terms = set(config.reward_schedule)
        uses_schedule_reward = config.reward_weights.gait_contact_bonus != 0.0 or "gait_contact_bonus" in scheduled_terms
        if uses_schedule_reward and not self._gait.enabled:
            raise ValueError(
                "gait_contact_bonus rewards contact matching the gait clock, but gait_clock.enabled is False: "
                "the policy cannot follow a schedule it cannot observe"
            )
        self._foot_body_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        )
        assert np.all(self._foot_body_ids >= 0)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        self.max_control_steps = config.sim.episode_control_steps
        self._prev_action = np.zeros(12, dtype=np.float32)
        self._command = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._rng = np.random.default_rng()
        self._viewer = None
        self._curriculum_progress = 0.0

    def set_curriculum_progress(self, progress: float) -> None:
        """Called by training.curriculum_callback.CurriculumProgressCallback
        (via VecEnv.env_method), once per rollout, with progress =
        current_timestep / total_timesteps. Terms named in
        cfg.reward_schedule linearly ramp from their `start` value at
        progress=0 to `end` at progress>=end_fraction; every other reward
        weight is unaffected. Never called (progress stays 0.0, its
        __init__ default) for a config with no reward_schedule section, so
        this is a no-op for every existing config."""
        self._curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def _effective_reward_weights(self):
        if not self.cfg.reward_schedule:
            return self.cfg.reward_weights
        overrides = {}
        for name, term in self.cfg.reward_schedule.items():
            frac = min(1.0, self._curriculum_progress / term.end_fraction) if term.end_fraction > 0 else 1.0
            overrides[name] = term.start + (term.end - term.start) * frac
        return dataclasses.replace(self.cfg.reward_weights, **overrides)

    def _gait_clock_obs(self) -> np.ndarray | None:
        if not self._gait.enabled:
            return None
        phases = leg_phases(self._step_count, self.cfg.sim.control_dt, self._gait.period_s, self._gait.offset)
        return clock_features(phases)

    def _foot_velocities(self) -> np.ndarray:
        """World-frame linear velocity of each foot body, shape (2, 3)."""
        res = np.zeros(6)
        vel = np.zeros((2, 3))
        for i, body_id in enumerate(self._foot_body_ids):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), res, 0)
            vel[i] = res[3:]
        return vel

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomize physics (friction/mass) before reset_state's mj_forward
        # call, so forward kinematics/dynamics for this episode are computed
        # with this episode's parameters, not the previous episode's.
        episode_rand = self._domain_randomizer.randomize_episode(self._rng, self.cfg.domain_randomization)
        self._motor_strength = episode_rand.motor_strength
        self._push_scheduler.reset(self._rng, self.cfg.domain_randomization)

        reset_state(self.model, self.data, self.joint_map, self.q_default, self.stand_key_id)
        self._command = sample_command(
            self._rng, self.cfg.command.vx_range, self.cfg.command.vy_range, self.cfg.command.zero_command_prob
        )
        self._prev_action = np.zeros(12, dtype=np.float32)
        self._step_count = 0
        initial_contact = both_feet_contact(self.model, self.data, self.floor_geom_id)
        self._air_time_tracker.reset(initial_contact)
        self._ground_time_tracker.reset(initial_contact)
        self._yaw_at_reset = quat_to_yaw(self.data.xquat[self.pelvis_id])

        obs = build_observation(
            self.model, self.data, self.joint_map, self.q_default,
            self.pelvis_id, self._command, self._prev_action, self._gait_clock_obs(),
        )
        return obs, {"command": self._command.copy()}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        target = action_to_target(action, self.q_default, self.cfg.action.action_scale, self.joint_map.joint_range)

        self._push_scheduler.maybe_push(self.data, self.cfg.sim.control_dt, self._rng, self.cfg.domain_randomization)

        # Per-substep torque/velocity, captured for gait-quality instrumentation
        # (docs/GAIT_QUALITY_OBJECTIVE.md) -- exposed via `info` only, never fed
        # to reward/observation/training. Fixed-size (decimation, 12) buffers
        # pre-allocated so a substep that breaks early (numerical_invalid) still
        # returns a well-shaped array (trailing rows stay zero, matching "no
        # further torque was applied" -- physically correct, not a placeholder).
        decimation = self.cfg.sim.control_decimation
        substep_torque = np.zeros((decimation, 12))
        substep_qvel = np.zeros((decimation, 12))
        last_torque = np.zeros(12)
        for i in range(decimation):
            q = self.data.qpos[self.joint_map.qpos_adr]
            qd = self.data.qvel[self.joint_map.qvel_adr]
            last_torque = pd_torque(target, q, qd, self.kp, self.kd, forcerange=self.joint_map.actuator_forcerange)
            last_torque = last_torque * self._motor_strength  # domain randomization: per-episode motor strength
            substep_torque[i] = last_torque
            substep_qvel[i] = qd
            self.data.ctrl[self.joint_map.actuator_ids] = last_torque
            self.data.ctrl[self.upper_body_actuator_ids] = UPPER_BODY_TARGETS
            mujoco.mj_step(self.model, self.data)
            if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
                break

        self._step_count += 1

        pelvis_height = float(self.data.xpos[self.pelvis_id][2])
        grav = projected_gravity(self.model, self.data, self.pelvis_id)
        lin_vel, ang_vel = base_frame_velocity(self.model, self.data, self.pelvis_id)
        foot_contact = both_feet_contact(self.model, self.data, self.floor_geom_id)
        air_time_reward_raw = self._air_time_tracker.step(
            foot_contact, self.cfg.sim.control_dt, self.cfg.reward_weights.feet_air_time_threshold_s
        )
        stance_overrun_raw = self._ground_time_tracker.step(
            foot_contact, self.cfg.sim.control_dt, self.cfg.reward_weights.max_stance_s
        )
        current_yaw = quat_to_yaw(self.data.xquat[self.pelvis_id])
        heading_deviation_rad = float(
            np.arctan2(np.sin(current_yaw - self._yaw_at_reset), np.cos(current_yaw - self._yaw_at_reset))
        )

        gait_kwargs = {}
        if self._gait.enabled:
            phases = leg_phases(self._step_count, self.cfg.sim.control_dt, self._gait.period_s, self._gait.offset)
            gait_kwargs = dict(
                gait_contact_raw=contact_schedule_raw(foot_contact, stance_mask(phases, self._gait.stance_fraction)),
                swing_height_raw=swing_height_raw(
                    self.data.xpos[self._foot_body_ids][:, 2], foot_contact,
                    self.cfg.reward_weights.feet_swing_height_target_m,
                ),
                contact_foot_vel_raw=contact_foot_vel_raw(self._foot_velocities(), foot_contact),
            )

        term = check_termination(
            pelvis_height=pelvis_height,
            projected_gravity_z=float(grav[2]),
            qpos=self.data.qpos,
            qvel=self.data.qvel,
            cfg=self.cfg.termination,
        )

        if term.terminated and term.reason == "numerical_invalid":
            obs = np.zeros(self._obs_dim, dtype=np.float32)
            reward, reward_info = 0.0, {"total": 0.0}
        else:
            reward, reward_info = compute_reward(
                base_lin_vel_xy=lin_vel[:2],
                command_vel_xy=self._command,
                torque=last_torque,
                base_ang_vel_z=float(ang_vel[2]),
                action=action.astype(np.float64),
                prev_action=self._prev_action.astype(np.float64),
                air_time_reward_raw=air_time_reward_raw,
                stance_overrun_raw=stance_overrun_raw,
                heading_deviation_rad=heading_deviation_rad,
                weights=self._effective_reward_weights(),
                **gait_kwargs,
            )
            obs = build_observation(
                self.model, self.data, self.joint_map, self.q_default,
                self.pelvis_id, self._command, action.astype(np.float32), self._gait_clock_obs(),
            )

        self._prev_action = action.astype(np.float32)
        truncated = (not term.terminated) and self._step_count >= self.max_control_steps

        info: dict[str, Any] = dict(reward_info)
        info["termination_reason"] = term.reason
        info["pelvis_height"] = pelvis_height
        info["command"] = self._command.copy()
        info["base_lin_vel_xy"] = lin_vel[:2].copy()
        info["substep_torque"] = substep_torque
        info["substep_qvel"] = substep_qvel

        return obs, reward, term.terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
