# Gait Quality Research Program: Summary (EXP-0044 through EXP-0049)

Status: end-of-session summary, 2026-09-24. This program is not "finished" -- the scorecard, instrumentation, and open questions all remain active -- but this session's bounded-experiment line has reached its planned stop condition. See "Recommended next experiments" for what a future session should pick up.

## What this program was

User-directed sustained program to improve the canonical walking policy's fluidity, efficiency, symmetry, and robustness without regressing the task-performance properties nine prior canonical updates (ADR-001 through ADR-019) had already established. Full objective/scorecard: `docs/GAIT_QUALITY_OBJECTIVE.md`. Full experiment records: `experiments/EXPERIMENT_LOG.md` EXP-0044 through EXP-0049. Full failure/finding records: `docs/FAILURE_ANALYSIS.md` F-0006, F-0007.

## Best starting baseline: `exp0043_zero_command_durability_check` @ 12M

The task-performance canonical this program inherited, unchanged by it in isolation -- measured before any gait-quality training happened (EXP-0044):

| Metric | Value |
|---|---|
| Standard 20-seed `dist_ratio` | 0.961 |
| Standard 20-seed yaw drift | 6.3 deg/10s |
| Banded 100-seed `dist_ratio` | 0.974 |
| Banded 100-seed yaw drift | 8.33 deg/10s |
| Banded 100-seed falls | 0/100 |
| Zero-command drift (F-0005 test) | 2.1 deg/20s |
| Clock adherence | ~81% |
| Medium-bucket swing duration (L/R) | 102.6 / 115.3 ms |
| Medium-bucket foot clearance (L/R) | 0.062 / 0.058 m |
| **Lateral-command falls (found by this program, F-0006)** | **11/20 (55%)** |
| **Left-foot swing ceiling vs right's (found by this program, F-0007)** | **140ms vs 180ms max, 1127 vs 1080 swings sampled** |

The last two rows are the program's own baseline-measurement discoveries, not pre-existing knowledge -- neither had ever been measured before this program's command-grid evaluation existed.

## Best resulting checkpoint: `exp0047_ankle_gain_more_zerocmd` @ 4M (canonical, ADR-020)

Promoted after combining two single-lever findings (EXP-0045, EXP-0046) that each failed alone: `ankle_pitch`/`ankle_roll` PD gain raised (kp 40->60, kd 2->3) plus `command.zero_command_prob` doubled (0.15->0.30) as a direct countermeasure for the zero-command destabilization the gain change alone caused.

| Metric | `exp0043` (before) | `exp0047` (after) | Change |
|---|---|---|---|
| Standard 20-seed `dist_ratio` | 0.961 | 0.970 | +0.9 pp |
| Standard 20-seed yaw drift | 6.3 | 5.5 deg/10s | -13% |
| Banded 100-seed `dist_ratio` | 0.974 | 0.971 | -0.3 pp (tied) |
| Banded 100-seed yaw drift | 8.33 | 8.33 deg/10s | tied |
| Banded 100-seed falls | 0/100 | 0/100 | tied |
| Banded 60-seed `dist_ratio` / falls | -- | 0.980 / 1/60 | -- |
| Zero-command drift | 2.1 | **2.0 deg/20s** | best in project |
| Clock adherence | ~81% | **~82.7%** | +1.7 pp |
| Medium-bucket swing duration, right foot | 115.3 | **135.3 ms** | **+17.3%** |
| Medium-bucket foot clearance (L/R) | 0.062 / 0.058 | 0.063 / 0.063 | right caught up to left |
| Lateral-command falls (fresh 60-seed) | 11/20 = 55% (orig. 20 seeds) | 6/60 = 10% | -82% relative (incidental, see below) |

**Every task-performance metric this project has tracked since ADR-003 matched or beat the previous canonical.** This is a genuine Pareto improvement, not a documented trade-off.

## Improvements, quantified

- **Swing duration (the program's primary target, F-0007):** right foot +17.3% (115.3 -> 135.3ms, medium-bucket). Left foot moved much less (+2.6%, essentially within noise) -- an asymmetric improvement, not a symmetric one.
- **Foot clearance symmetry:** right foot's clearance rose from 0.058m to 0.063m, matching the left foot's 0.062-0.063m for the first time -- the asymmetry didn't just shrink on one axis, it closed on this one.
- **Zero-command drift:** improved to the best level recorded anywhere in this project (2.0 deg/20s), monotonically across all 4 training checkpoints -- not just avoided a regression, genuinely better.
- **Lateral-command robustness (F-0006, incidental):** 55% -> 10% fall rate, confirmed on a fresh seed set disjoint from discovery. This was not EXP-0047's target and its mechanism is not confirmed (plausibly the ankle-gain increase and/or the doubled zero-command exposure providing general stability margin, but not isolated by ablation).
- **Every other task-performance axis (tracking, general heading, falls, clock adherence):** matched or modestly improved.

## Rejected experiments (kept on record, not wasted effort)

| Experiment | What it tried | Result | Why rejected |
|---|---|---|---|
| **EXP-0045** | Ankle gain alone (kp 40->60), reward unchanged | Right-foot swing +13.9%, but zero-command drift hit 8.1 deg/20s (vs. 5.1 hard gate) at 3 of 4 checkpoints | Destabilized a previously-hard-won property as an unpredicted side effect |
| **EXP-0046** | `feet_air_time_bonus` weight alone (3.0->5.0), gains unchanged | Zero-command stayed stable, but swing duration barely moved (best +8.7%) | Didn't achieve its purpose; useful as a clean isolation of EXP-0045's mechanism (confirmed control-specific, not general fine-tuning risk) |
| **EXP-0048** | Asymmetric left-ankle gain (kp 60->90), right unchanged | Left-foot swing ceiling moved 140->180ms (matching right's), but yaw drift hit 20-28 deg/10s (vs. 10.8 soft gate) and `dist_ratio` fell to 0.81-0.85 | Confirmed the left foot is control-rate-limited (a real, valuable finding) but the specific fix destabilizes heading control severely |
| **EXP-0049** | Graded version of EXP-0048 (left kp=75, half the step) | Heading cost scaled down (yaw ~10.0) but swing-duration benefit vanished too (102-107ms, flat vs. baseline) | No usable middle ground found between these two tested points; closes this specific lever's line of investigation for this session |

Each of these produced real, usable evidence even though none was promoted -- EXP-0045+0046 together isolated the zero-command destabilization to the control-mechanics change specifically; EXP-0048+0049 together established that the left foot's swing-duration benefit behaves like a threshold effect while the heading-control cost behaves continuously, ruling out a simple gain-magnitude search as the fix.

## Remaining limitations

- **F-0006 is substantially improved, not closed.** 10% fall rate on pure-lateral commands (fresh seeds) is much better than 55%, but not zero, and the underlying asymmetric gait pattern under that command is unchanged (`duty_factor_left` 0.84 vs `duty_factor_right` 0.68) -- the robot survives it more often, the pattern itself wasn't fixed. The fix was incidental; its actual mechanism (ankle gain? zero-command exposure? both?) has not been isolated.
- **F-0007 is substantially improved, not closed.** Right-foot swing duration up 17.3%; left foot's response across three separate experiments (control, reward, combined, plus the two asymmetric-gain probes) remains disproportionately smaller, confirmed control-rate-limited in principle (EXP-0048) but with no usable fix found yet. The gait clock's left-phase-leads structural convention is the leading hypothesis for *why the symmetric-trained policy never found this capability on its own*, and remains completely untested.
- **PD gains are now part of the canonical's identity, a first for this lineage.** Any future fine-tune inherits `ankle_pitch`/`ankle_roll` kp=60/kd=3; reverting would need re-deriving compatible weights, not a config swap.
- **Swing duration is still well under a human-scale stride** (120-135ms vs. a plausible 200-300ms target), and still well under the gait clock's own 360ms-scheduled window.
- **The fixed 0.8s gait period, regardless of commanded speed, remains untested as a lever** -- F-0007's evidence (duty factor exceeding the clock's own configured stance fraction at every speed) is a concrete, unaddressed reason to suspect it may still be a binding constraint even after this program's improvements.
- **Domain-randomization robustness (`exp0039`/`exp0040`, push/motor-strength) has not been rebuilt on `exp0047`** -- still built on the much older `exp0038`, and whether it composes with any of this program's changes is unknown.
- **Every other limitation carried forward unchanged from before this program**: single training seed throughout the entire project; sim-only/CPU-only; open-loop heading; no upper-body control; the unexplained seeds-1004/1005 low-command fall; no sim-to-real work.

## Confidence

**VERIFIED** (directly measured, on seeds disjoint from selection, multiple independent sets):
- `exp0047`'s improvement over `exp0043` on every tracked task-performance metric (100+ seed validation, banded and standard sets)
- F-0006's improvement from 55% to 10% (confirmed on a fresh 60-seed set, not just the original discovery seeds)
- The left foot's swing-duration ceiling is control-rate-limited (EXP-0048's direct counter-example: raising gain moved a ceiling that had never moved in 1127+ prior samples)
- The asymmetric-gain heading-control cost scales down with a smaller step, while the swing-duration benefit does not scale proportionally (EXP-0049 vs. EXP-0048 head-to-head comparison)
- Model physical symmetry between legs (mass, inertia, actuator forcerange, sagittal joint ranges all directly inspected and confirmed identical)

**INFERRED** (strongly supported by evidence but not directly, independently confirmed):
- The swing-duration benefit is a genuine threshold effect rather than a smooth function of gain -- based on exactly two tested points (kp=75, kp=90); a finer sweep would strengthen or complicate this
- The gait clock's left-phase-leads convention is the root structural cause of the left/right asymmetry -- inferred by elimination (physical symmetry confirmed, consistency across 4+ independently-trained checkpoints never reversed) but never directly tested (e.g. by swapping the clock's phase reference)
- F-0006's mechanism (why lateral-command stability improved) -- plausible but not isolated by ablation; could be the ankle gain, the zero-command exposure, or their interaction

**UNVERIFIED** (not tested at all, a real gap):
- Whether any of this program's findings replicate across a different training seed -- every experiment in this entire project, including all of this program's, used seed 0 only
- Whether `exp0047`'s improvements are durable under further continued training, the way `exp0043` was specifically stress-tested for the zero-command property (`exp0047` has not had an equivalent "continue the same trajectory and re-check" experiment)
- Whether the lateral-command falls-worse-at-smaller-gain-step observation (EXP-0049) is a real effect or 5-seed noise

## Recommended next three experiments (ordered by expected information value)

1. **The clock-structure hypothesis, as a from-scratch comparison.** This is the single highest-value unanswered question in the entire program: does swapping which foot the gait clock references (making the right foot's phase the primary `clock_features` signal instead of the left's) flip which foot ends up with the shorter swing? A clean answer would either confirm the leading structural hypothesis (redirecting all future work toward a clock redesign) or rule it out entirely (redirecting toward yet another mechanism). Must be a from-scratch comparison, not a fine-tune -- warm-starting an already-converged symmetric-clock policy onto a swapped-meaning observation input would confound any result. Expensive (hours, not the ~10-minute bounded-experiment cadence this session used) and therefore deliberately not started this session.
2. **Recalibrate the heading-control reward terms for an asymmetric-gain regime, then retry EXP-0048's gain change.** EXP-0049 established that the heading-control cost of an asymmetric gain scales predictably (unlike the swing benefit) -- this makes it the more tractable lever of the two. Per this project's own ADR-005 discipline, the existing heading terms were calibrated entirely under symmetric-leg dynamics and have never been re-measured under an asymmetric one; a properly recalibrated heading penalty might let EXP-0048's real swing-duration gain be kept without its regression.
3. **Isolate F-0006's actual mechanism and confirm its durability.** Run the lateral-command test on `exp0047`-with-only-the-ankle-gain-change and `exp0047`-with-only-the-zero-command-exposure-change (two cheap, already-buildable ablation configs) to find out which change is actually responsible for the incidental fix, then re-check the fall rate on a larger, fresh seed set (e.g. 100+ seeds, matching the rigor already applied to task-performance validation) before treating 10% as a stable number.

A fourth, standing item that qualifies every finding above: a training-seed replication check (repeat `exp0047`'s recipe from a different random seed) would tell us how much of this program's specific numbers are one-draw noise versus a reliably reproducible recipe -- not run this session given the added compute, but worth flagging as the item that would most change how confidently any of the above should be read.

## Reproduction

```bash
# Canonical (exp0047): full 11-stage lineage in IMPLEMENTATION.md section 11.
# The gait-quality-specific stage (stage 11):
PYTHONPATH=. python -m training.train --n-envs 12 --seed 0 \
  --run-name exp0047_ankle_gain_more_zerocmd --timesteps 4000000 --checkpoint-freq 1000000 \
  --config config/exp0047_ankle_gain_more_zerocmd.yaml \
  --init-from experiments/runs/exp0043_zero_command_durability_check/best_checkpoint_12M.zip \
  --init-learning-rate 1e-5

# Gait-quality command-grid evaluation (any checkpoint):
PYTHONPATH=. python -m evaluation.evaluate_gait_quality \
  --model experiments/runs/exp0047_ankle_gain_more_zerocmd/best_checkpoint_4M.zip \
  --config config/exp0047_ankle_gain_more_zerocmd.yaml \
  --seeds 1000-1019 --out-dir experiments/gait_quality/<name>

# Rejected-experiment configs, for anyone continuing this line:
# config/exp0045_ankle_gain.yaml            (control lever alone)
# config/exp0046_air_time_weight.yaml       (reward lever alone)
# config/exp0048_left_ankle_gain.yaml       (asymmetric gain, kp=90)
# config/exp0049_left_ankle_gain_graded.yaml (asymmetric gain, kp=75)
```

Full test suite (115 tests, including 12 new this program -- 8 for `evaluation/gait_quality.py`, 4 for per-leg PD gains): `PYTHONPATH=. python -m pytest tests/ -q`.
