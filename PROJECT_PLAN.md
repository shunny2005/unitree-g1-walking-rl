# Project Plan

Status: living document. Claude should rewrite this as evidence changes.

**Progress as of 2026-09-23**: everything below (Phases 0-9) is historical -- the challenge's minimum bar was exceeded around EXP-0009 and every phase through 9 was completed long ago. See `PROJECT_STATE.md` for the current, up-to-date status (canonical checkpoint, latest experiment, next steps) -- this file is kept for the phase structure/history, not as the live status source. **A new phase, Phase 10 (gait quality), started 2026-09-23** -- see its own section below, added at the end rather than woven into the historical phases 0-9 (which remain an accurate record of how the project reached its current canonical).

**Progress as of 2026-09-18** (historical, phases 0-7 as they stood then): Phases 0-7 complete, through six Phase-7 iterations (see PROJECT_STATE.md, docs/DECISIONS.md ADR-001 through ADR-005, experiments/EXPERIMENT_LOG.md EXP-0001 through EXP-0011). Three reward/PPO-config changes were tried and rejected on evidence (EXP-0006 `target_kl` -> degenerate near-motionless policy; EXP-0007 lower `alive_bonus` -> no improvement; EXP-0008 both combined -> much worse) before EXP-0009 isolated the real fix: the original config simply needed more training. A user-reported concern then led to a new diagnostic finding the resulting policy drifted 95 degrees per 10s and took chattery steps -- invisible to every metric used until then; a first fix (`ang_vel_penalty=1.0`, EXP-0010) repeated the same degenerate-stillness mistake, caught immediately and corrected by measuring the actual gait's yaw-rate cost before recalibrating (EXP-0011). **Canonical baseline is now `exp0011_gait_quality_v2`'s evaluated-and-selected ~10M-timestep checkpoint**: 18.3s mean survival, 0.671 distance ratio, 2.99m distance traveled (best yet), 2/20 falls (best yet), and yaw drift cut from 95 to 11.6 degrees/10s in a controlled test. Phase 8 (robustness/domain randomization) not started -- correctly deferred until basic walking was confirmed, which it now is more convincingly than ever; not required for the challenge's minimum bar. Phase 9 (deliverables) is finished and updated in docs/DELIVERABLES.md. Remaining open item: stepping is still small/chattery, only partially improved (F-0003 second half).

## Success criteria from the challenge
- Unitree G1 runs in simulation.
- Observation and action spaces are explicit and explainable.
- Reward begins minimal and evolves only from observed failures.
- PPO rollout -> update loop runs end to end through a maintained trainer.
- Training curves and episode-length/related learning evidence are recorded.
- Deterministic play/evaluation exists.
- Final writeup and video/visual attempt can be produced.
- Robustness/domain randomization comes after basic behavior works.

## Phase 0 — Ground truth and machine audit
Deliverables:
- challenge requirements summarized in `PROJECT_STATE.md`;
- hardware/software audit;
- current repo contents understood;
- open questions listed;
- research tasks delegated where useful.

Exit test:
- we know what "done" means and what machine constraints matter.

## Phase 1 — Architecture/framework decision
Compare:
- official `unitree_rl_gym`;
- official `unitree_rl_mjlab`;
- direct MuJoCo + Gymnasium + maintained PPO library;
- Berkeley Humanoid only as a reference unless research justifies more.

Decision dimensions:
- actual MuJoCo training support;
- hardware requirements;
- challenge alignment;
- implementation transparency;
- setup risk;
- training throughput;
- evaluation/visualization support.

Deliverable:
- decision record in `docs/DECISIONS.md`.

Exit test:
- one implementation route selected with evidence and a fallback route.

## Phase 2 — MuJoCo vertical slice
Implement the smallest possible simulator loop:
- locate/load G1 model assets;
- render if available;
- reset to known pose;
- advance physics safely;
- inspect `qpos`, `qvel`, controls, base pose;
- validate finite values and dimensions.

Tests:
- model loads;
- N steps complete without NaNs;
- reset is repeatable;
- joint/control indexing is explicit.

## Phase 3 — Control and environment contract
Define:
- controlled DoFs;
- action semantics;
- default pose;
- action scale;
- PD/control mapping if used;
- physics dt;
- control dt/decimation;
- observation vector and each field;
- termination vs truncation;
- reset behavior;
- Gymnasium API contract if applicable.

Tests:
- observation shape/dtype/bounds;
- action shape/dtype/bounds;
- deterministic reset with seed;
- random action rollout;
- zero-action rollout;
- obvious invalid-state checks.

## Phase 4 — Minimal reward
Start with:
- velocity tracking;
- survival/alive signal if justified;
- torque/effort penalty.

Instrument every reward component separately.

Tests:
- reward finite;
- signs/scales make sense on hand-constructed states;
- falling terminates as intended;
- no single penalty numerically dominates by accident.

## Phase 5 — PPO smoke pipeline
Integrate maintained PPO implementation.

Goal is not walking yet.
Goal is proving:
environment -> rollout -> GAE/value internals -> optimizer updates -> checkpoint -> reload -> deterministic evaluation.

Smoke budget:
tiny and deliberately cheap.

Tests:
- learning runs for several updates;
- no NaNs;
- losses/entropy/value stats are logged;
- checkpoint reload reproduces evaluation behavior closely enough.

## Phase 6 — Baseline learning run
Create one canonical baseline config.
Run enough to see whether learning signal exists within available compute.

Track:
- return;
- episode length;
- velocity error;
- reward components;
- base height/orientation;
- termination reasons;
- torque/action saturation.

Freeze this baseline before experimentation.

## Phase 7 — Iterative experiments
Loop:
1. diagnose one failure;
2. form one hypothesis;
3. change one conceptual variable where practical;
4. run bounded experiment;
5. evaluate on the same deterministic protocol;
6. log result;
7. keep/revert based on evidence.

Common classes of investigation:
- control dt/decimation;
- PD gains/action scale;
- observation normalization;
- reward scaling;
- reset state;
- command curriculum;
- PPO batch/rollout settings;
- termination thresholds.

Do not add complexity just because another repository has it.

## Phase 8 — Robustness
Only after baseline behavior clearly learns.

Candidate randomization:
- friction;
- body mass/inertia within justified ranges;
- motor strength;
- pushes;
- latency/noise only if relevant.

Evaluate clean and randomized environments separately.

## Phase 9 — Deliverables
Produce:
- concise approach writeup;
- observation/action definitions;
- reward equation/terms and rationale;
- training/evaluation methodology;
- reward and episode-length curves;
- experiment summary;
- limitations;
- video or visual rollout;
- exact reproduction commands.

Run final independent audit before completion.

## Phase 10 — Gait quality (fluidity, efficiency, symmetry, robustness)
Started 2026-09-23, user-directed. Everything above (Phases 0-9) targeted "does it walk, track commands, and survive" -- this phase targets "does it walk *well*": smoothly, efficiently, symmetrically, and robustly, without giving up what Phases 0-9 established.

Deliverables (in progress):
- a measurable scorecard, not vague aesthetic judgment: `docs/GAIT_QUALITY_OBJECTIVE.md` (task-performance regression gates + fluidity/symmetry/geometry/efficiency metric groups, none collapsed into one scalar);
- instrumentation: `evaluation/gait_quality.py` (metrics), `evaluation/evaluate_gait_quality.py` (command-grid CLI), a purely-additive substep-torque/velocity telemetry hook in `envs/g1_walk_env.py`;
- a measured baseline on the canonical, by command bucket, not one aggregate number (EXP-0044);
- regression gates set from that measurement, not invented;
- a bounded, evidence-driven experiment loop (profile before touching rewards; one conceptual change per experiment; Pareto comparison, not a single blended score; every experiment pre-registered and logged in `experiments/EXPERIMENT_LOG.md` whether it succeeds or not).

Exit test: no fixed exit condition by design (an open-ended improvement program, not a phase with a clean "done"). Stops per the program's own stopping rules: a real blocker, a compute/storage constraint, several consecutive well-motivated experiments failing to improve any Pareto metric with research suggesting a larger redesign is needed, or a clearly documented plateau -- see the final research-style summary produced at that point (best baseline, best result, improvements, rejected experiments, remaining limitations, confidence, recommended next three experiments, reproduction commands).

Two significant findings already surfaced by the baseline measurement alone (before any training in this phase): F-0006 (55% fall rate on a pure-lateral command, `docs/FAILURE_ANALYSIS.md`) and F-0007 (realized swing duration far short of the gait clock's own schedule, with an evidence-based control-rate-limitation lead). See `PROJECT_STATE.md` "Latest experiment" for current status.
