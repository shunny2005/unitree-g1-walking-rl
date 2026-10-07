# Research Log

Append concise, decision-relevant research here. Prefer primary sources.

## Seed research to re-verify at project start

### R-0001 — Unitree framework landscape
Question: Does the challenge-linked Unitree repository train G1 directly in MuJoCo?

Sources:
- https://github.com/unitreerobotics/unitree_rl_gym
- https://github.com/unitreerobotics/unitree_rl_mjlab
- https://github.com/mujocolab/mjlab

Finding at scaffold creation:
- `unitree_rl_gym` documents training in its Gym/Isaac-Gym environment and MuJoCo as the Sim2Sim deployment stage.
- `unitree_rl_mjlab` documents MuJoCo as its training physics backend.
- `mjlab` documents NVIDIA GPU as a training requirement and macOS as evaluation-only.

Implication:
Do not assume the challenge-linked repository is the best direct MuJoCo-training path. Detect the actual machine and make a documented framework decision.

Confidence:
High from official project READMEs, but re-verify because repositories change.

### R-0002 — Claude Code project structure
Question: How should this repository expose persistent instructions and subagents to Claude Code?

Sources:
- https://code.claude.com/docs/
- https://code.claude.com/docs/en/best-practices
- https://code.claude.com/docs/fr/claude-directory

Finding:
- project `CLAUDE.md` is loaded at session start;
- custom project subagents live under `.claude/agents/*.md`;
- commands/skills can package reusable prompts;
- subagents are most useful for isolated/parallel work rather than trivial edits.

Implication:
This scaffold uses `CLAUDE.md`, custom subagents, and reusable locomotion commands.

Confidence:
High from official Claude Code documentation.

### R-0003 — Re-verification of Unitree/mjlab training backends + local sibling evidence (2026-09-18)
Question: Is R-0001's finding still current, and does this specific machine change the decision?

Sources:
- https://github.com/unitreerobotics/unitree_rl_gym (README, doc/setup_en.md, deploy/deploy_mujoco/) — last push 2025-07-25
- https://github.com/unitreerobotics/unitree_rl_mjlab and https://github.com/mujocolab/mjlab (README, installation docs) — both pushed within 24h of this check
- https://github.com/DLR-RM/stable-baselines3 issue #914 (MPS support)
- Local: `../standing-rl` and `../walking-rl` (sibling projects, same machine, same G1 robot, already working)
- Local machine audit: `uname -a`, `sysctl hw.memsize`/`hw.ncpu`, `system_profiler SPDisplaysDataType` -> Apple M5 Pro, arm64, no NVIDIA GPU, 24GB RAM, 15 cores, macOS 26.5.2, system Python 3.14.6.

Finding:
- `unitree_rl_gym` trains via Isaac Gym (Nvidia+Linux); MuJoCo is Sim2Sim validation only. No macOS/CPU training path.
- `mjlab`/`unitree_rl_mjlab` requires an NVIDIA GPU for training per its own README ("macOS is supported for evaluation only"); uses MuJoCo Warp, not MJX-on-CPU. This is current, actively-maintained behavior, not stale docs.
- Direct MuJoCo + Gymnasium + stable-baselines3 has a real CPU/macOS-arm64 story. SB3 does not reliably use Apple MPS (open upstream issue) — must set `device="cpu"` explicitly.
- Sibling local projects `standing-rl` and `walking-rl` (same team, same G1 robot) already independently built and run exactly this stack: Python 3.12 venv (not 3.14 — wheels lag), `mujoco>=3.1`, `gymnasium>=0.29`, `stable-baselines3>=2.3`, `torch>=2.2`, `device: cpu`, vendored G1 MJCF from `mujoco_menagerie` (commit `da76818e269b82289eba39808e2fb91d679d6994`) with leg actuators converted from `<position>` to `<motor>` so a custom per-joint-group PD controller can drive them. Validated numeric starting points found there: `physics_dt=0.002`, `control_decimation=10` (50 Hz control), leg PD gains (hip 100/2, knee 150/4, ankle 40/2), `action_scale=0.25`, PPO `n_steps=512, batch_size=256, net=[256,256], device=cpu`.

Implication:
Confirms ADR-001 (direct MuJoCo + Gymnasium + stable-baselines3, Python 3.12). These sibling numeric values are used only as informed, cited starting points for our own from-scratch walking environment/config — not copied wholesale, since the challenge's own environment/reward must be understood and owned here, and the task (walking, velocity tracking) differs from standing-rl's task (holding a fixed pose).

Confidence:
High — verified against live upstream READMEs/commit activity today, cross-checked against two independently-working local implementations on the identical machine.

---

### R-0004 -- How proven humanoid-walking setups get real stepping (2026-09-21)
Question: This project's gait is stuck at a 60 ms median swing with about 2.5 cm of foot lift, every from-scratch attempt to add stepping terms has collapsed or skated (EXP-0022/0024/0032/0033), and its only working recipe is warm-starting from an already-walking policy. What do reference humanoid-locomotion implementations do that this one does not?

Sources:
- Official Unitree G1 config and environment, read verbatim: https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_config.py and https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_env.py
- Humanoid-Gym (Gu et al., 2024), reward and observation tables: https://ar5iv.labs.arxiv.org/html/2404.05695 (the PDF did not parse; HTML version used)
- Asymmetric actor-critic for legged locomotion (search results only, not read in depth): https://arxiv.org/pdf/2505.20619 and https://arxiv.org/html/2405.10830v1

Finding (verbatim from code, not summaries):
1. Both references give the policy an explicit **gait clock**. Unitree G1: `phase = (episode_step * dt) % 0.8 / 0.8`, left leg phase = `phase`, right leg = `(phase + 0.5) % 1`, and `sin(2*pi*phase)`, `cos(2*pi*phase)` are appended to the observation. Humanoid-Gym also feeds a clock input.
2. Both reward **contact matching the clock**. Unitree `_reward_contact`: per foot, +1 when `contact == (leg_phase < 0.55)` (stance for 55% of the cycle), weight 0.18. Humanoid-Gym: a periodic contact-pattern reward against a stance mask, weight 1.0. Because it is evaluated every step, a foot that never lifts loses reward continuously; it cannot be dodged the way a touchdown-only reward can.
3. Unitree adds two terms this project lacks: `feet_swing_height` (weight -20, squared error of foot height from 0.08 m while not in contact) and `contact_no_vel` (weight -0.2, squared foot velocity while in contact -- this penalizes skating directly). It also sets `feet_air_time` to 0.0, i.e. it does not use a touchdown air-time reward at all.
4. Both use an **asymmetric critic**: the critic sees privileged state (base linear velocity; in Humanoid-Gym also friction, mass, push force) that the actor does not. Humanoid-Gym also stacks 15 observation frames. Unitree uses a small recurrent (LSTM) actor.
5. Foot height in Unitree's code is the `ankle_roll_link` body z, the same quantity this project's gait diagnostic already reports (canonical: rest about 3.3 cm, peak 5.7 cm; the reference targets 8 cm).

Implication: this project's stepping problem is structural, not a weight-tuning problem. Nothing external tells the policy *when* to step, and its only stepping reward (touchdown air-time) can be avoided by not stepping, which is exactly the skating outcome seen in F-0004 and EXP-0032/0033. The reference recipes solve this with a clock plus a per-step contact-schedule reward. Adding the clock changes the observation size, so the existing checkpoints cannot be loaded directly and a warm start needs a function-preserving expansion of the first layer. The asymmetric critic and frame stacking are also worth having but are separate, larger changes.

Confidence / remaining uncertainty: high that these components exist and how they are defined (read from the reference source). Unknown whether they transfer to this setup: the references train with thousands of parallel GPU envs and 10,000 iterations, this project uses 12 CPU envs, and Humanoid-Gym reports no ablation of the clock or contact reward. Unitree's schedule also makes the robot step in place at zero command, which may cost tracking near zero speed. Treat as a hypothesis to test, not a known fix.

---

### R-0005 -- What the clock policy's velocity error actually is, and how the reference tracks velocity (2026-09-21)
Question: The clock-driven policy (`exp0036`) walks with a genuine stride but has 2.4x the canonical's velocity error (0.085 vs 0.036 m/s), and it looked like it overshoots near-zero commands. What is the error made of, and how do reference implementations handle tracking with a periodic gait and zero commands?

Sources:
- Humanoid-Gym environment, read verbatim via the GitHub API: `humanoid/envs/custom/humanoid_env.py` in https://github.com/roboterax/humanoid-gym (`_get_phase`, `_get_gait_phase`, `compute_observations`, `_reward_feet_contact_number`, `_reward_track_vel_hard`, `_reward_low_speed`)
- Search results on mode-conditioned periodic rewards (stand / step in place / walk): https://arxiv.org/pdf/2505.20619 and https://lacuna.tiptreesystems.com/work/revisiting-reward-design-and-evaluation-for-robust-humanoid-standing-and-walking/wrk_d60d75a6e47909214537b95eb0d2d35e (search snippets only; the page fetch was rate-limited and not read)
- This project's own measurement (`evaluation/metrics.py`, 12 seeds and then the 20-seed eval).

Finding:
1. **No command gating in the reference.** Humanoid-Gym's phase is `episode_step * dt / cycle_time` and the stance mask and contact reward depend only on that (with a double-support window where `|sin(2*pi*phase)| < 0.1`); nothing scales or zeroes the clock by command magnitude. Unitree G1 is the same (verbatim in R-0004). So the reference also steps to the clock at zero command; there is no gating recipe to copy.
2. **A much sharper velocity reward.** Humanoid-Gym's `_reward_track_vel_hard` is `exp(-10 * ||v_cmd - v||)` (plus a yaw term). This project's is `exp(-||e||^2 / 0.25^2)`. At 0.10 m/s error they give 0.37 and 0.85; at 0.05, 0.61 and 0.96. The project's term is nearly flat where the clock policy operates. Humanoid-Gym's contact reward is also asymmetric: `+1` on a match, `-0.3` on a mismatch, averaged over feet.
3. **The error is mostly within-stride ripple, not steady mis-tracking** (measured here). Instantaneous error 0.084 for `exp0036` = period-averaged error 0.046 plus rms ripple 0.077, versus 0.040 = 0.032 + 0.027 for the canonical (12 seeds; 20-seed eval: 0.046 + 0.075 vs 0.025 + 0.026). A longer, slower stride oscillates the body velocity more. By contrast `exp0031`'s 0.061 is mostly steady bias (0.056 period-averaged, ripple 0.024): the old metric could not tell these apart.
4. **The apparent near-zero-command overshoot was not real.** For commands under 0.15 m/s, `exp0036`'s mean speed minus command is +0.002 m/s (canonical -0.022). The earlier `dist_ratio` of 1.14 there came from 7 episodes with a tiny expected distance.

Implication: (a) evaluation should report the period-averaged error and the ripple separately -- added as `mean_period_avg_vel_error` / `mean_vel_ripple_rms` and to `results/compare.py`; (b) the honest gap between the clock policy and the canonical is a period-averaged error of 0.046 vs 0.025 plus more ripple, not a 2.4x tracking failure; (c) a sharper velocity reward is the reference-backed lever for closing it, at the risk that reducing ripple costs stride (ripple is partly intrinsic to a longer stride).

Confidence / remaining uncertainty: high on the reference code and on the measurement (deterministic, 12 and 20 seeds, and the metric is unit-tested). Whether the sharper reward can reduce ripple without shortening the stride is unknown -- that is the experiment. The two arXiv/site sources were not read in depth.

---

### R-0006 -- Why the clock policy trails at high speed, and what the references do about yaw (2026-09-21)
Question: `exp0037` matches the canonical on tracking and adds a real stride, but its `dist_ratio` at commanded speeds of 0.4 m/s and above is 0.825 vs 0.900. Is that undershoot, sideways drift, or turning -- and what do reference implementations do about yaw?

Sources:
- Humanoid-Gym `humanoid_env.py` and `humanoid_config.py`, read verbatim via the GitHub API (https://github.com/roboterax/humanoid-gym): `_reward_track_vel_hard`, `_reward_tracking_ang_vel`, and the `scales` block.
- Unitree G1 config (R-0004): `tracking_ang_vel` 0.5, `ang_vel_xy` -0.05.
- This project's own measurements on 60 fresh seeds (2000-2059), reproduced by the new `evaluation/evaluate_banded.py`.

Finding:
1. **The high-speed gap is heading, not speed.** In the >=0.4 m/s band (26 episodes), `exp0037` runs 3% below the commanded speed with zero mean lateral velocity and path length 0.967 of expected (canonical 0.989); but heading drift is 43-44 deg/10 s (canonical 33), straightness 0.851 (0.909) and `dist_ratio` 0.825 (0.899). At low and mid speeds `exp0037` is straighter than the canonical (4.5 vs 17.8 and 12.9 vs 18.9 deg/10 s).
2. **The yaw signal is mostly in-stride oscillation, not drift.** Instantaneous yaw rate averages 0.57 rad/s in magnitude in the high band (canonical 0.37) with a steady episode-mean bias of only 0.076 rad/s (canonical 0.056) -- about 8x smaller. Accumulated heading deviation is 1.12 rad^2 per step in the high band vs 0.11 below 0.4 m/s (canonical 0.79 and 0.23); the in-stride swing itself contributes only about 0.008 rad^2.
3. **The references track yaw *rate*, and it is a live command.** Humanoid-Gym's `track_vel_hard` is `(exp(-10*|lin err|) + exp(-10*|yaw-rate err|))/2 - 0.2*(lin err + yaw-rate err)`, plus a Gaussian `tracking_ang_vel` (scale 1.1) and `track_vel_hard` scale 0.5; Unitree uses `tracking_ang_vel` 0.5. Neither penalizes accumulated heading: yaw is a commanded rate that whoever issues commands closes the loop on, so open-loop straight-line drift is not something the references optimize. This project's open-loop drift metric is stricter than the reference interface.

Implication: applying the reference's instantaneous yaw-rate reward here would mostly punish the gait's own yaw swing (at 0.57 rad/s, `exp(-10*|w|)` is about 0.003, so almost no gradient) and would not touch the 0.076 rad/s bias that accumulates. The lever has to act on the cumulative component -- ADR-010's lesson (a cumulative penalty for a cumulative failure). The existing `heading_deviation_penalty` does, and it is dominated by drift, not swing.

Confidence / remaining uncertainty: high on the reference code and the measurements (deterministic, 60 fresh seeds, tooling unit-tested and reproducing the ad-hoc numbers exactly). The high band has 26 episodes, so its numbers carry noticeable sampling error. Whether a heading penalty can straighten the high-speed walk without costing tracking or stride is the experiment; earlier heading fine-tunes on the canonical (EXP-0027/0028/0031) gave mixed results.

---

### R-0007 -- Which domain-randomization factor actually causes the canonical's falls, and how does the reference push robots? (2026-09-22)
Question: under the full four-factor domain randomization used by `exp0021`/`exp0029` (friction, mass, motor strength, pushes), the canonical `exp0038` falls in 50 of 60 fresh episodes -- worse than its predecessor's own robustness numbers suggested was necessary. Which factor causes that, and is this project's push recipe unusually harsh compared to a reference?

Sources:
- This project's own ablation: each of the four `envs/domain_randomization.py` factors enabled alone against `exp0038`, 60 fresh seeds (2000-2059), the other three held at their nominal (inert) values.
- Official Unitree G1 config and the shared `LeggedRobotCfg`, read verbatim via the GitHub API: `legged_gym/envs/g1/g1_config.py` and `legged_gym/envs/base/legged_robot.py`/`legged_robot_config.py` in https://github.com/unitreerobotics/unitree_rl_gym (`domain_rand` class, `_push_robots`).

Finding:
1. **The four factors are not equally responsible.** Friction alone: 0/60 falls, `dist_ratio` unchanged. Mass alone: 2/60. Motor strength alone (x[0.7, 1.0] torque multiplier): 22/60, `vel_err` roughly triples. Pushes alone (this project's existing range, 0.1-0.5 m/s every 3-6 s): 35/60, the single largest factor. All four together: 50/60 -- worse than any one factor, so they compound rather than simply add.
2. **This project's push range is already gentler than the reference's.** Unitree G1's `domain_rand.max_push_vel_xy = 1.5` m/s (the shared base class defaults to 1.0), applied to *every* environment every `push_interval_s` (G1: 5 s) by directly setting root linear velocity to a value drawn independently per axis from `[-max_vel, max_vel]` (`_push_robots`, verbatim). This project's own range (0.1-0.5 m/s magnitude, random angle, 3-6 s per-environment interval) is smaller in every respect. Push training is also a default-on ingredient of the reference recipe (`push_robots = True`), not an optional add-on.
3. **The implication is exposure, not magnitude.** A push range smaller than the reference's standard default still causes more damage than any other factor, on a policy that has never experienced one. This is consistent with this lineage's repeatedly-observed fragility to any unfamiliar perturbation (ADR-011 follow-up, EXP-0027/0028).

Implication: full four-factor domain randomization (the recipe `exp0021`/`exp0029` used) is not the minimal fix. A push-only fine-tune, at this project's existing (already-conservative) range, is the natural next experiment -- it targets the dominant factor in isolation and should cost much less clean-condition performance than randomizing all four, since friction and mass need no correction at all and motor-strength robustness can be added separately if still needed after pushes are addressed.

Confidence / remaining uncertainty: high on the ablation (deterministic, 60 seeds, reproducible) and on the reference code. Not yet known whether push-only training recovers as much robustness as the full recipe, or whether motor-strength weakening needs its own fix once pushes are handled.

---

## Template for new entries

### R-XXXX — Title
Question:

Sources:
- 

Finding:

Implication:

Confidence / remaining uncertainty:
