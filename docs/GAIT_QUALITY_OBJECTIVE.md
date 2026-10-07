# Gait Quality Objective: Scorecard, Instrumentation, and Regression Gates

Status: living document, program start 2026-09-23. Owner: main agent, this session's sustained gait-quality research program.

## Purpose and scope

The canonical policy (`exp0043_zero_command_durability_check` @ 12M) walks, tracks commanded velocity well, rarely falls, and doesn't spin at zero command. It has never been optimized for how the walk *looks and costs*: stiffness, stride shape, left/right symmetry, or mechanical effort. This program's goal is to improve those properties **without materially degrading the task-performance properties already established** (ADR-001 through ADR-019).

"Fluid" and "efficient" are not aesthetic judgments here. Each is defined below as a small set of measured proxies, instrumented once, tracked every evaluation from now on. No single scalar score is computed by default -- the individual metrics stay visible so a regression in one dimension can't hide behind an improvement in another (this mirrors ADR-003's original reasoning for `dist_ratio`: don't let one aggregate number hide what's actually happening).

## What already exists vs. what's new here

Already measured, every evaluation, unchanged: fall rate, episode length, `dist_ratio`, velocity error (instantaneous, period-averaged, ripple), yaw drift, clock adherence, termination reasons (`evaluation/evaluate.py`, `evaluation/evaluate_banded.py`, `evaluation/metrics.py`). These remain the **task-performance regression gates** (group A below) -- nothing here replaces them.

New in this program: everything in groups B-F, implemented in `evaluation/gait_quality.py` (metrics) and `evaluation/evaluate_gait_quality.py` (CLI across a command grid), plus a small, purely-additive telemetry change to `envs/g1_walk_env.py`'s `step()` (exposes per-physics-substep torque and joint velocity in `info`, for accurate torque/work integrals instead of only the final substep of each control step -- training and reward are untouched by this; see "Instrumentation notes" below).

## A. Task performance (regression constraints, not optimization targets in this program)

Unchanged from existing tooling:
- fall rate, episode survival length, termination reasons
- commanded vs. actual forward/lateral velocity (`mean_vel_error`, `mean_period_avg_vel_error`, `mean_vel_ripple_rms`)
- `dist_ratio` (distance traveled / distance commanded, reliable-episode subset)
- yaw drift (mean abs deg/10s, mixed commands) and stand-still drift (dedicated zero-command test, F-0005's protocol)

Baseline values and exact regression tolerances are recorded in "Baseline and regression gates" below, measured before any experiment in this program runs.

## B. Fluidity / smoothness

All computed per control step (50Hz) unless noted; substep quantities (500Hz) are used for torque/work integrals specifically, see instrumentation notes.

**Action smoothness**
- `action_delta_rms` = RMS over the episode of `||a_t - a_(t-1)||` (L2 norm over the 12-dim action)
- `action_jerk_rms` = RMS of `||a_t - 2a_(t-1) + a_(t-2)||` (second difference, an action-space jerk proxy)

**Joint-motion smoothness** (computed from `qd` at control-step resolution -- 20ms; this is a real resolution limit, stated once here rather than caveated every table: jerk figures especially should be read as indicative, not exact, since true jerk is a 500Hz-physics-rate quantity being estimated from a 50Hz signal)
- RMS joint velocity (per joint and aggregate, rad/s)
- RMS joint acceleration (finite difference of `qd`, rad/s^2)
- RMS joint jerk (second finite difference of `qd`, rad/s^3) -- reported but explicitly flagged low-confidence per the resolution note above
- peak joint acceleration (max abs, single worst joint-step)
- count of "large discontinuous target changes": control steps where `||target_t - target_{t-1}||` (the PD position target, not the raw action) exceeds a threshold set from the baseline's own 95th percentile, not an arbitrary number

**Base motion quality** (pelvis body frame unless noted)
- vertical pelvis position oscillation: std and peak-to-peak of pelvis z over the episode, high-pass-ish by detrending against the episode's linear height trend (walking has a real small vertical bob; this isolates oscillation from any net height drift)
- roll oscillation: std of pelvis roll angle (from `projected_gravity`'s x-component, small-angle)
- pitch oscillation: std of pelvis pitch angle (from `projected_gravity`'s y-component)
- yaw-rate oscillation: std of `base_ang_vel_z` (distinct from the *mean* yaw-rate bias that drives heading drift -- this is the oscillatory part)
- lateral body sway: std of pelvis world-frame y-velocity component perpendicular to the commanded heading

**Impact/contact quality**
- foot touchdown vertical velocity: the foot body's world-frame z-velocity at the control step immediately before a contact transition from airborne to grounded (a direct, low-noise proxy for impact severity -- computed from data this project already reads every step, no new MuJoCo contact-force API needed)
- impact asymmetry: left vs. right touchdown velocity, mean and per-episode difference
- **Explicitly not measured**: raw contact force/impulse via MuJoCo's constraint-force API. Per the task's own caution ("do not introduce a metric too noisy or simulator-specific to interpret") and this project's own history of preferring low-noise, physically direct proxies (ADR-003's `dist_ratio`, R-0005's ripple/period-average split) -- touchdown velocity is a adequate, much lower-noise proxy for impact severity in a rigid-contact simulator where raw constraint forces are known to be spiky and step-size-dependent.

## C. Symmetry

Per-episode, split by left/right using `control/joint_map.py`'s existing group labeling (first 6 canonical-order joints = left leg, last 6 = right leg):

- peak foot clearance (max foot-body z during a swing, per foot)
- average swing foot height (mean foot-body z while airborne, per foot)
- swing duration (mean, per foot -- already partially computed in `diagnostics/gait_diagnostic.py`, generalized here)
- stance duration (mean, per foot)
- step length (forward pelvis displacement between successive opposite-foot touchdowns) and stride length (between successive same-foot touchdowns)
- step frequency (per foot, from touchdown-to-touchdown period)
- torque RMS (per leg, summed over that leg's 6 joints)
- action magnitude (RMS `||a||` restricted to that leg's 6 action dims)
- hip/knee/ankle trajectory range (max - min of `q` over the episode, per joint, both legs reported side by side)
- touchdown vertical velocity (per foot, from group B)
- contact duration (total time in contact / episode length, per foot)

All of these are reported as a compact left/right table (`{metric}_left`, `{metric}_right`, `{metric}_asym = (left - right) / (|left| + |right| + eps)`), not a single symmetry scalar, so the *direction and shape* of any asymmetry stays visible.

**Mirror-consistency testing (deferred, not implemented this cycle).** The task suggests checking whether the policy's outputs approximately mirror under mirrored left/right states and commands. This requires knowing, for certain, the sign convention of every one of the 12 joint axes in the actual G1 MuJoCo model (a roll joint's "positive" direction is not guaranteed to be the same physical direction on the left and right leg -- getting this wrong would silently produce a meaningless or actively misleading result, which is worse than not measuring it). This is deferred as a documented stretch item, not skipped out of laziness: before attempting it, the joint axis directions must be read directly from `assets/g1/g1_walk.xml`'s joint `axis` fields and cross-checked against a known symmetric pose, not assumed. Tracked as a follow-up; the behavioral left/right table above is the primary, lower-risk symmetry tool for this program.

## D. Gait geometry

Reported **as a function of commanded speed** (not one aggregate number), for the standard command grid (see "Command grid" below):
- stride length, step length
- cadence (steps/second, both feet)
- swing duration, stance duration
- double-support fraction (both feet grounded / episode length)
- foot clearance (peak, both feet pooled)
- duty factor (stance duration / full-cycle duration, per foot)
- left/right phase relationship (the two feet's actual phase offset, measured from touchdown timing, compared against the clock's configured 0.5-cycle offset -- this checks whether the policy's *realized* gait actually matches the clock's *scheduled* structure, which is a real open question, see "Known question" below)

**Known question to resolve in Family 1 (before any reward change):** the gait clock schedules each foot to be in stance for `stance_fraction=0.55` of an 0.8s cycle, i.e. swing for `0.45 * 0.8s = 360ms`. Measured swing duration in prior diagnostics has been 100-120ms -- about a third of the clock's own scheduled swing window, even though clock *adherence* (a per-step binary match) is measured at ~81-82%. These two facts are not necessarily in tension (adherence only requires "is the foot doing what the schedule says at this instant," which a foot that leaves the ground briefly and stance-locks early can still mostly satisfy), but the gap between "the clock schedules a 360ms swing" and "the policy executes a 120ms swing" is exactly the kind of quantitative discrepancy this program should explain before touching the reward -- it's a leading hypothesis for why the gait looks short-strided/choppy rather than fluid. See Family 1.

## E. Efficiency

Computed from **all** physics substeps in an episode (500Hz), not just the once-per-control-step torque already in `info` (see instrumentation notes -- the existing `torque_penalty` reward term and `mean_torque_cost` eval metric use only the final substep's torque per control step, which undercounts true effort; this program's efficiency metrics use the full substep resolution):

- RMS joint torque (aggregate and per-joint), N*m
- mean absolute torque, N*m
- torque-squared integral over the episode, `sum_t sum_i tau_i(t)^2 * physics_dt`
- action magnitude RMS (`||a_t||`, control-step resolution -- action, not torque, so this is a distinct, cheaper-to-compute proxy)
- positive mechanical work: `sum_t sum_i max(0, tau_i(t) * qd_i(t)) * physics_dt`, Joules
- absolute mechanical work: `sum_t sum_i |tau_i(t) * qd_i(t)| * physics_dt`, Joules
- work per meter traveled: `abs_work / distance_traveled_m`
- **cost of transport (COT)**: `abs_work / (total_mass_kg * 9.81 * distance_traveled_m)`, dimensionless. Explicitly labeled a **simulation/mechanical-work proxy, not a real electrical or battery cost of transport** -- it does not model motor electrical efficiency, gearbox losses, or regenerative braking, none of which this project's actuator model (ideal PD -> clipped torque) represents.

**Anti-exploit check, built in from day one**: every efficiency table is reported next to `dist_ratio` and velocity error from the *same* rollout. A checkpoint that reduces work-per-meter by moving less or tracking worse is not efficient by this program's definition -- it's a regression that happens to look good on one axis. Efficiency comparisons between two checkpoints are only meaningful at matched (or near-matched) `dist_ratio`.

## F. Robustness and generalization

Not the primary target of the early experiment families (task performance robustness already has a documented, separate track -- `exp0039_push_only_dr`/`exp0040_push_motor_dr`, ADR-016, not yet merged into the current canonical). This program's robustness work (Family 7) specifically asks: **does a gait-quality improvement survive the disturbances the robustness track already characterized**, and does it compose with them. Evaluated incrementally, one factor at a time (pushes first, since ADR-016 already found that's the cheap one), never as a first-experiment kitchen sink.

## Command grid

Every gait-quality evaluation in this program uses the same fixed grid of forced commands (not the continuous-random per-episode sampling used in training), each run for the standard 20s episode on a fixed seed set, so command-conditioned results are always comparable run to run:

| Bucket | Command (vx, vy) m/s | Why |
|---|---|---|
| stand still | (0.0, 0.0) | F-0005's regime; must stay a hard regression gate |
| slow | (0.15, 0.0) | historically the fall-prone regime (seeds 1004/1005, F-0002/EXP-0027/`exp0037`) |
| slow-2 | (0.2, 0.0) | second slow point, brackets the historically difficult region |
| medium | (0.35, 0.0) | mid-range, most of the training distribution's mass |
| high | (0.55, 0.0) | near the top of the trained range (0.6 m/s max) |
| lateral | (0.0, 0.18) | near-max sideways, trained but rarely evaluated directly |
| diagonal | (0.35, 0.15) | representative mixed command |

Each bucket is evaluated on a small fixed seed set (default: seeds 1000-1004, five seeds; expanded to the full 20-seed standard set or the 60/100-seed banded sets for any candidate being considered for promotion). Seeds 1004 and 1005 specifically are always included at the "slow" bucket, per the task's explicit instruction to include historically difficult cases rather than let them hide in an aggregate.

## Baseline and regression gates

**Rebaselined 2026-09-24 on the current canonical** (ADR-020, `exp0047_ankle_gain_more_zerocmd` @ 4M), 20 seeds (1000-1019) per command bucket, via `evaluation/evaluate_gait_quality.py`. Full data: `experiments/gait_quality/baseline_exp0047_20seed/gait_quality_by_bucket.json`. The original `exp0043` baseline (2026-09-23, `experiments/gait_quality/baseline_exp0043_20seed/`) is kept for history and is what EXP-0045/0046/0047 were actually measured against -- future experiments should use the table below.

**Gait-quality baseline by command bucket** (forced commands, 20 seeds each):

| Bucket | falls | dist (m) | torque_rms | work/m (J) | COT | swing L/R (ms) | clearance L/R (m) | action_jerk_rms | duty factor L/R | double-support |
|---|---|---|---|---|---|---|---|---|---|---|
| stand_still | 0/20 | 0.12 | 6.7 | 15550* | 47.5* | 97/125 | 0.056/0.051 | 1.293 | 0.73/0.66 | 0.400 |
| slow (0.15) | 0/20 | 3.04 | 6.8 | 615 | 1.88 | 115/126 | 0.061/0.056 | 1.417 | 0.70/0.65 | 0.360 |
| slow2 (0.2) | 0/20 | 3.99 | 7.0 | 498 | 1.52 | 106/131 | 0.060/0.058 | 1.464 | 0.70/0.64 | 0.350 |
| medium (0.35) | 0/20 | 6.99 | 7.6 | 338 | 1.03 | 106/135 | 0.063/0.063 | 1.624 | 0.69/0.63 | 0.333 |
| high (0.55) | 0/20 | 10.53 | 9.1 | 322 | 0.98 | 90/101 | 0.064/0.062 | 1.897 | 0.69/0.63 | 0.331 |
| lateral (vy=0.18) | **0/20** (was 11/20) | 2.31 | 7.2 | 767 | 2.35 | 55/82 | 0.053/0.045 | 1.284 | 0.84/0.68 | 0.519 |
| diagonal | 0/20 | 7.49 | 7.9 | 333 | 1.02 | 117/123 | 0.069/0.059 | 1.674 | 0.68/0.65 | 0.335 |

\* stand-still `work/m`/`COT` remain near-meaningless (tiny denominator) -- not a gate, reported for completeness.

**F-0006 update:** the lateral bucket's fall rate dropped from 11/20 (55%) to 0/20 on the identical original seeds, an unplanned side effect of EXP-0047 (not its target). **Confirmed on a fresh, disjoint 60-seed set at 6/60 (10%)** -- a large, real improvement (~82% relative reduction) but not the complete fix the 20-seed number alone suggested; the regression-gate table below uses this fresh-seed rate, not the original-seed 0/20. The underlying asymmetric gait pattern is still present (`duty_factor_left` 0.84 vs `duty_factor_right` 0.68, similar shape to the original finding) -- the robot survives that pattern much more often than before, not a redesigned symmetric gait. See F-0006's "Unplanned resolution" entry in `docs/FAILURE_ANALYSIS.md`.

**Hard gates** (unchanged in kind, values updated to the new floor; a candidate that violates any of these on the forward-walking buckets is rejected regardless of other improvements):
- falls: 0 on every forward-walking bucket (unchanged)
- `dist_ratio` (standard 20-seed, reliable subset): >= 0.94 (baseline 0.970, 3-point tolerance)
- stand-still drift (F-0005 test): <= 3.0 deg/20s (1.5x baseline's 2.0) and never past 5.1 deg/20s (the historical `exp0042` floor, still the hard ceiling)
- gait must remain alternating: liftoff count per foot >= 40/episode and both-feet-grounded fraction <= 0.60 on any forward-walking bucket (baseline 0.33-0.40)

**Soft gates:**
- yaw drift (mixed commands, 100-seed set): <= 10.8 deg/10s (1.3x baseline's 8.33, unchanged since `exp0047`'s own yaw tied `exp0043`'s)
- clock adherence: >= 72% (10 points below baseline's ~82.7%)
- `lateral` bucket falls: no more than 10/60 on a fresh seed set (the current, still-imperfect floor after F-0006's incidental improvement -- 11/20 (55%) originally, now 6/60 (10%) on fresh seeds) is *not yet at 0*, so this stays a soft gate; a candidate that pushes it materially higher (e.g. back past 20/60) is flagged, not auto-rejected

**Task-performance baseline (group A, unchanged tooling), current canonical:** standard 20-seed eval: 0/20 falls, `dist_ratio` 0.970 (reliable subset, 10/20 episodes -- see IMPLEMENTATION.md section 1 for why), yaw drift 5.5 deg/10s. Fresh 100-seed validation: `dist_ratio` 0.971, yaw drift 8.33 deg/10s, 0/100 falls. Zero-command drift 2.0 deg/20s (best recorded in this project).

<details>
<summary>Historical: original <code>exp0043</code> baseline (2026-09-23), superseded by the table above</summary>

| Bucket | falls | dist (m) | torque_rms | work/m (J) | COT | swing L/R (ms) | clearance L/R (m) | action_jerk_rms | duty factor L/R | double-support |
|---|---|---|---|---|---|---|---|---|---|---|
| stand_still | 0/20 | 0.17 | 6.7 | 10060 | 30.8 | 102/111 | 0.056/0.049 | 1.349 | 0.72/0.68 | 0.417 |
| slow (0.15) | 0/20 | 2.94 | 6.7 | 613 | 1.87 | 103/119 | 0.058/0.053 | 1.431 | -- | -- |
| slow2 (0.2) | 0/20 | 4.02 | 6.8 | 469 | 1.43 | 103/120 | 0.060/0.055 | 1.498 | -- | -- |
| medium (0.35) | 0/20 | 6.91 | 7.5 | 336 | 1.03 | 103/115 | 0.062/0.058 | 1.636 | 0.70/0.63 | 0.340 |
| high (0.55) | 0/20 | 10.28 | 8.8 | 319 | 0.97 | 75/86 | 0.059/0.058 | 1.884 | 0.72/0.63 | 0.359 |
| lateral (vy=0.18) | **11/20** | 1.76 | 7.4 | 844 | 2.58 | 43/79 | 0.051/0.048 | 1.328 | 0.90/0.67 | 0.554 |
| diagonal | 0/20 | 7.18 | 7.7 | 332 | 1.01 | 102/96 | 0.065/0.055 | 1.661 | -- | -- |

This is the table EXP-0045/0046/0047's regression checks used. The `exp0047` rebaseline above supersedes it -- most notably, the `lateral` bucket's 11/20 falls dropped to 0/20 on the identical 20 seeds (an unplanned side effect of EXP-0047, see F-0006's "Unplanned resolution" in `docs/FAILURE_ANALYSIS.md`).

</details>

Every experiment in this program reports its full metrics against the current table above, not just whichever numbers happened to improve.

## Instrumentation notes

- `envs/g1_walk_env.py`'s `step()` now also returns, in `info`: `substep_torque` (shape `(decimation, 12)`, N*m, post motor-strength-scaling, matching exactly what's applied to the actuators) and `substep_qvel` (shape `(decimation, 12)`, rad/s, joint velocity at the start of each substep). This is the only change to environment code in this program's setup phase. It does not touch `reward`, `observation`, `action`, `termination`, or any training-visible quantity -- `info` is not consumed by PPO/SB3 training, only by evaluation and diagnostic scripts, so this is provably a no-op for training and for every existing config/test (verified: `tests/test_env_contract.py` and the reward/observation tests don't inspect these new keys and continue to pass unchanged).
- Joint acceleration/jerk are finite-differenced from `qd` at 20ms (control-step) resolution. True jerk lives at the 500Hz physics rate; this is a deliberate, documented approximation, not an oversight.
- Total robot mass for COT: `sum(model.body_mass)` (excludes the massless world body by construction in MuJoCo).

## What this document does not do

It does not collapse groups B-F into one scalar. It does not define training reward terms -- that happens per-experiment, in `experiments/EXPERIMENT_LOG.md`, only after a specific measured deficiency motivates one (per this project's standing ADR-005 discipline and this task's explicit anti-reward-soup instruction). This document defines what "better" means well enough to run that loop honestly.
