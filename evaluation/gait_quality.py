"""Gait-quality instrumentation: fluidity, symmetry, geometry, and efficiency
metrics (groups B-E of docs/GAIT_QUALITY_OBJECTIVE.md). Kept separate from
evaluation/evaluate.py's task-performance metrics (group A, unchanged) so
neither file's purpose gets diluted.

Usage as a library:
    from evaluation.gait_quality import run_gait_quality_episode
    result = run_gait_quality_episode(model, env, seed=1000, forced_command=(0.35, 0.0))

See evaluation/evaluate_gait_quality.py for the command-grid CLI.
"""

from __future__ import annotations

import numpy as np

from envs.contacts import LEFT_FOOT_GEOMS, RIGHT_FOOT_GEOMS, foot_in_contact
from envs.observations import base_frame_velocity, projected_gravity, quat_to_yaw
from control.pd_controller import action_to_target

G = 9.81
_EPS = 1e-9


def run_gait_quality_episode(model, env, seed: int, forced_command: tuple | None = None) -> dict:
    """Run one deterministic episode (model.predict(..., deterministic=True)),
    optionally overriding the sampled command (same mechanism gait_diagnostic.py
    and the F-0005 zero-command tests use: set env._command after reset), and
    return raw per-step trajectories plus every group B-E metric.
    """
    obs, info = env.reset(seed=seed)
    if forced_command is not None:
        env._command = np.array(forced_command, dtype=np.float32)
    command = env._command.copy()

    floor_geom = env.floor_geom_id
    left_foot_body, right_foot_body = env._foot_body_ids

    actions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    foot_z: list[np.ndarray] = []
    foot_vel_z: list[np.ndarray] = []
    contact: list[np.ndarray] = []
    pelvis_xy: list[np.ndarray] = []
    pelvis_z: list[float] = []
    yaw: list[float] = []
    roll: list[float] = []
    pitch: list[float] = []
    ang_vel_z: list[float] = []
    substep_torque_chunks: list[np.ndarray] = []
    substep_qvel_chunks: list[np.ndarray] = []

    terminated = truncated = False
    termination_reason = None
    steps = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        target = action_to_target(
            np.asarray(action, dtype=np.float64), env.q_default, env.cfg.action.action_scale, env.joint_map.joint_range
        )
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        actions.append(np.asarray(action, dtype=np.float64))
        targets.append(target)
        qpos.append(env.data.qpos[env.joint_map.qpos_adr].copy())
        qvel.append(env.data.qvel[env.joint_map.qvel_adr].copy())
        foot_z.append(env.data.xpos[[left_foot_body, right_foot_body]][:, 2].copy())
        fv = env._foot_velocities()
        foot_vel_z.append(fv[:, 2].copy())
        contact.append(
            np.array(
                [foot_in_contact(env.data, LEFT_FOOT_GEOMS, floor_geom), foot_in_contact(env.data, RIGHT_FOOT_GEOMS, floor_geom)]
            )
        )
        pelvis_xy.append(env.data.xpos[env.pelvis_id][:2].copy())
        pelvis_z.append(float(env.data.xpos[env.pelvis_id][2]))
        yaw.append(quat_to_yaw(env.data.xquat[env.pelvis_id]))
        grav = projected_gravity(env.model, env.data, env.pelvis_id)
        roll.append(float(grav[0]))
        pitch.append(float(grav[1]))
        _, av = base_frame_velocity(env.model, env.data, env.pelvis_id)
        ang_vel_z.append(float(av[2]))
        substep_torque_chunks.append(info["substep_torque"])
        substep_qvel_chunks.append(info["substep_qvel"])
        termination_reason = info.get("termination_reason")

    raw = dict(
        command=command,
        control_dt=env.cfg.sim.control_dt,
        physics_dt=env.cfg.sim.physics_dt,
        steps=steps,
        episode_length_s=steps * env.cfg.sim.control_dt,
        terminated=terminated,
        termination_reason=termination_reason,
        action=np.array(actions),
        target=np.array(targets),
        qpos=np.array(qpos),
        qvel=np.array(qvel),
        foot_z=np.array(foot_z),
        foot_vel_z=np.array(foot_vel_z),
        contact=np.array(contact),
        pelvis_xy=np.array(pelvis_xy),
        pelvis_z=np.array(pelvis_z),
        yaw=np.array(yaw),
        roll=np.array(roll),
        pitch=np.array(pitch),
        ang_vel_z=np.array(ang_vel_z),
        substep_torque=np.concatenate(substep_torque_chunks, axis=0) if substep_torque_chunks else np.zeros((0, 12)),
        substep_qvel=np.concatenate(substep_qvel_chunks, axis=0) if substep_qvel_chunks else np.zeros((0, 12)),
        total_mass_kg=float(np.sum(env.model.body_mass)),
    )
    metrics = compute_metrics(raw)
    return {"raw": raw, "metrics": metrics}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _finite_diff(x: np.ndarray, dt: float, order: int = 1) -> np.ndarray:
    """order=1: velocity-of-x; order=2: acceleration-of-x (i.e. jerk if x is
    already a velocity). Uses np.diff, so output is one (or `order`) shorter."""
    d = x
    for _ in range(order):
        d = np.diff(d, axis=0) / dt
    return d


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True-runs of a 1D bool array as (start, end) index pairs
    (end exclusive), dropping any run touching either boundary (incomplete)."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    return [r for r in runs if r[0] > 0]  # drop a run starting at t=0 (unknown true start)


def _touchdown_indices(contact_col: np.ndarray) -> np.ndarray:
    """Control-step indices where a foot transitions airborne -> grounded."""
    return np.where((~contact_col[:-1]) & contact_col[1:])[0] + 1


def compute_metrics(raw: dict) -> dict:
    dt = raw["control_dt"]
    pdt = raw["physics_dt"]
    n = raw["steps"]
    m: dict = {}

    # ---- B. Fluidity / smoothness ----
    a = raw["action"]
    if n >= 2:
        da = np.diff(a, axis=0)
        m["action_delta_rms"] = _rms(np.linalg.norm(da, axis=1))
    else:
        m["action_delta_rms"] = float("nan")
    if n >= 3:
        djerk = a[2:] - 2 * a[1:-1] + a[:-2]
        m["action_jerk_rms"] = _rms(np.linalg.norm(djerk, axis=1))
    else:
        m["action_jerk_rms"] = float("nan")

    qvel = raw["qvel"]
    m["joint_vel_rms"] = _rms(qvel)
    joint_accel = _finite_diff(qvel, dt, order=1) if n >= 2 else np.zeros((0, 12))
    m["joint_accel_rms"] = _rms(joint_accel)
    m["joint_accel_peak"] = float(np.max(np.abs(joint_accel))) if joint_accel.size else float("nan")
    joint_jerk = _finite_diff(qvel, dt, order=2) if n >= 3 else np.zeros((0, 12))
    m["joint_jerk_rms"] = _rms(joint_jerk)

    target = raw["target"]
    if n >= 2:
        dtarget = np.linalg.norm(np.diff(target, axis=0), axis=1)
        m["target_delta_p95"] = float(np.percentile(dtarget, 95))
        m["target_delta_max"] = float(np.max(dtarget))
    else:
        m["target_delta_p95"] = m["target_delta_max"] = float("nan")

    pelvis_z = raw["pelvis_z"]
    if n >= 2:
        trend = np.polyfit(np.arange(n), pelvis_z, 1)
        detrended = pelvis_z - np.polyval(trend, np.arange(n))
        m["pelvis_z_osc_std"] = float(np.std(detrended))
        m["pelvis_z_osc_p2p"] = float(np.max(detrended) - np.min(detrended))
    else:
        m["pelvis_z_osc_std"] = m["pelvis_z_osc_p2p"] = float("nan")
    m["roll_osc_std"] = float(np.std(raw["roll"])) if n else float("nan")
    m["pitch_osc_std"] = float(np.std(raw["pitch"])) if n else float("nan")
    m["yaw_rate_osc_std"] = float(np.std(raw["ang_vel_z"])) if n else float("nan")

    if n >= 2:
        heading0 = raw["yaw"][0]
        lateral_dir = np.array([-np.sin(heading0), np.cos(heading0)])
        rel_vel = np.diff(raw["pelvis_xy"], axis=0) / dt
        lateral_vel = rel_vel @ lateral_dir
        m["lateral_sway_std"] = float(np.std(lateral_vel))
    else:
        m["lateral_sway_std"] = float("nan")

    contact = raw["contact"]
    foot_vel_z = raw["foot_vel_z"]
    touchdown_vel = [[], []]
    for foot in range(2):
        for idx in _touchdown_indices(contact[:, foot]):
            touchdown_vel[foot].append(float(foot_vel_z[idx - 1, foot]))
    m["touchdown_vel_left"] = float(np.mean(np.abs(touchdown_vel[0]))) if touchdown_vel[0] else float("nan")
    m["touchdown_vel_right"] = float(np.mean(np.abs(touchdown_vel[1]))) if touchdown_vel[1] else float("nan")

    # ---- D. Gait geometry (+ feeds C. Symmetry) ----
    foot_z = raw["foot_z"]
    swing_runs = [_runs(~contact[:, foot]) for foot in range(2)]
    stance_runs = [_runs(contact[:, foot]) for foot in range(2)]
    swing_dur = [[(e - s) * dt for s, e in runs] for runs in swing_runs]
    stance_dur = [[(e - s) * dt for s, e in runs] for runs in stance_runs]
    clearance = [[float(np.max(foot_z[s:e, foot])) for s, e in swing_runs[foot]] for foot in range(2)]
    avg_swing_h = [[float(np.mean(foot_z[s:e, foot])) for s, e in swing_runs[foot]] for foot in range(2)]

    touchdowns = [_touchdown_indices(contact[:, foot]) for foot in range(2)]
    step_freq = [
        float(1.0 / np.mean(np.diff(td)) / dt) if len(td) >= 2 else float("nan") for td in touchdowns
    ]
    pelvis_xy = raw["pelvis_xy"]
    stride_len = [
        [float(np.linalg.norm(pelvis_xy[b] - pelvis_xy[aidx])) for aidx, b in zip(td[:-1], td[1:])] for td in touchdowns
    ]
    # step length: forward pelvis displacement between a left touchdown and the next right touchdown (or vice versa)
    all_td = sorted([(idx, 0) for idx in touchdowns[0]] + [(idx, 1) for idx in touchdowns[1]])
    step_len = []
    for (idx_a, foot_a), (idx_b, foot_b) in zip(all_td[:-1], all_td[1:]):
        if foot_a != foot_b:
            step_len.append(float(np.linalg.norm(pelvis_xy[idx_b] - pelvis_xy[idx_a])))

    for label, foot in (("left", 0), ("right", 1)):
        m[f"swing_duration_mean_{label}_ms"] = float(np.mean(swing_dur[foot]) * 1000) if swing_dur[foot] else float("nan")
        m[f"stance_duration_mean_{label}_ms"] = float(np.mean(stance_dur[foot]) * 1000) if stance_dur[foot] else float("nan")
        m[f"peak_clearance_{label}_m"] = float(np.mean(clearance[foot])) if clearance[foot] else float("nan")
        m[f"avg_swing_height_{label}_m"] = float(np.mean(avg_swing_h[foot])) if avg_swing_h[foot] else float("nan")
        m[f"stride_length_{label}_m"] = float(np.mean(stride_len[foot])) if stride_len[foot] else float("nan")
        m[f"step_frequency_{label}_hz"] = step_freq[foot]
        m[f"contact_fraction_{label}"] = float(np.mean(contact[:, foot])) if n else float("nan")
        cycle_s = (1.0 / step_freq[foot]) if step_freq[foot] == step_freq[foot] and step_freq[foot] > 0 else float("nan")
        m[f"duty_factor_{label}"] = (
            (np.mean(stance_dur[foot]) / cycle_s) if stance_dur[foot] and cycle_s == cycle_s and cycle_s > 0 else float("nan")
        )

    m["step_length_mean_m"] = float(np.mean(step_len)) if step_len else float("nan")
    m["double_support_fraction"] = float(np.mean(contact[:, 0] & contact[:, 1])) if n else float("nan")
    m["both_airborne_fraction"] = float(np.mean(~contact[:, 0] & ~contact[:, 1])) if n else float("nan")
    m["left_liftoffs"] = int(len(swing_runs[0]))
    m["right_liftoffs"] = int(len(swing_runs[1]))

    # realized left/right phase offset (fraction of a full left-foot cycle
    # between a left touchdown and the nearest following right touchdown),
    # compared against the gait clock's configured 0.5-cycle offset
    if len(touchdowns[0]) >= 2 and len(touchdowns[1]) >= 1:
        left_period = np.mean(np.diff(touchdowns[0])) * dt
        offsets = []
        for lt in touchdowns[0]:
            after = touchdowns[1][touchdowns[1] > lt]
            if len(after):
                offsets.append(((after[0] - lt) * dt % left_period) / left_period if left_period > 0 else float("nan"))
        m["realized_phase_offset"] = float(np.mean(offsets)) if offsets else float("nan")
    else:
        m["realized_phase_offset"] = float("nan")

    # ---- C. Symmetry summary (asymmetry ratios for the paired metrics above) ----
    def asym_pair(left_key: str, right_key: str) -> float:
        left, right = m.get(left_key), m.get(right_key)
        if left is None or right is None or left != left or right != right:
            return float("nan")
        return float((left - right) / (abs(left) + abs(right) + _EPS))

    # (left key, right key, output name) -- left/right keys carry a unit
    # suffix (_ms, _m) that the "<metric>_asym" output name omits (the
    # asymmetry ratio itself is unit-independent).
    _asym_keys = (
        ("swing_duration_mean_left_ms", "swing_duration_mean_right_ms", "swing_duration_mean_asym"),
        ("stance_duration_mean_left_ms", "stance_duration_mean_right_ms", "stance_duration_mean_asym"),
        ("peak_clearance_left_m", "peak_clearance_right_m", "peak_clearance_asym"),
        ("avg_swing_height_left_m", "avg_swing_height_right_m", "avg_swing_height_asym"),
        ("stride_length_left_m", "stride_length_right_m", "stride_length_asym"),
        ("touchdown_vel_left", "touchdown_vel_right", "touchdown_vel_asym"),
    )
    for left_key, right_key, out_name in _asym_keys:
        m[out_name] = asym_pair(left_key, right_key)

    # per-leg torque RMS / action-magnitude RMS / joint-range (needs leg index split: first 6 = left, last 6 = right)
    torque = raw["substep_torque"]
    m["torque_rms_left"] = _rms(torque[:, :6])
    m["torque_rms_right"] = _rms(torque[:, 6:])
    m["torque_rms_asym"] = asym_pair("torque_rms_left", "torque_rms_right")
    m["action_mag_rms_left"] = _rms(np.linalg.norm(a[:, :6], axis=1)) if n else float("nan")
    m["action_mag_rms_right"] = _rms(np.linalg.norm(a[:, 6:], axis=1)) if n else float("nan")
    m["action_mag_rms_asym"] = asym_pair("action_mag_rms_left", "action_mag_rms_right")
    qpos = raw["qpos"]
    for i, group in enumerate(("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")):
        if n:
            m[f"range_{group}_left"] = float(np.max(qpos[:, i]) - np.min(qpos[:, i]))
            m[f"range_{group}_right"] = float(np.max(qpos[:, i + 6]) - np.min(qpos[:, i + 6]))

    # ---- E. Efficiency ----
    net_disp = pelvis_xy[-1] - pelvis_xy[0] if n else np.zeros(2)
    distance_m = float(np.linalg.norm(net_disp))
    m["distance_traveled_m"] = distance_m
    m["torque_rms"] = _rms(torque)
    m["torque_abs_mean"] = float(np.mean(np.abs(torque))) if torque.size else float("nan")
    m["torque_sq_integral"] = float(np.sum(np.square(torque)) * pdt)
    m["action_mag_rms"] = _rms(np.linalg.norm(a, axis=1)) if n else float("nan")

    sub_qvel = raw["substep_qvel"]
    power = torque * sub_qvel  # (n*decimation, 12), W per joint
    pos_work = float(np.sum(np.clip(power, 0, None)) * pdt)
    abs_work = float(np.sum(np.abs(power)) * pdt)
    m["mech_positive_work_j"] = pos_work
    m["mech_abs_work_j"] = abs_work
    m["work_per_meter_j"] = abs_work / distance_m if distance_m > 1e-3 else float("nan")
    m["cost_of_transport"] = (
        abs_work / (raw["total_mass_kg"] * G * distance_m) if distance_m > 1e-3 else float("nan")
    )

    return m
