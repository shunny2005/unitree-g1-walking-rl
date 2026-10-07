# Project Snapshot: Unitree G1 Walking Policy (for external review)

This is a self-contained status report, written to hand to someone (or another AI) with no prior context on the project, so they can give an outside opinion on where things stand and what to do next.

## What this is

A reinforcement-learning walking controller for the Unitree G1 humanoid robot, trained entirely in simulation (MuJoCo physics, no real hardware). It's a from-scratch implementation of the Wisconsin Humanoids locomotion new-member challenge: build the whole pipeline (physics → observation → policy → control → reward → training → evaluation) rather than just running someone else's stack. It's a solo side project, trained on a CPU-only Apple laptop (Apple M5 Pro, no GPU), not a cluster or cloud job — every training run has been a few minutes to tens of minutes of wall-clock time.

Stack: MuJoCo (physics) + Gymnasium (environment interface) + stable-baselines3's PPO implementation (the RL algorithm itself is not custom — the project's own work is the environment, the control mapping, the reward function, and the evaluation/experiment discipline around them).

## What the walking actually looks like right now

Given a target walking speed (0 to 0.6 m/s forward, up to ±0.2 m/s sideways, fixed for the whole episode), the robot:

- **Walks a real, alternating gait** — both feet take turns lifting and planting, not a shuffle or a foot-drag. Median time a foot spends in the air per step is about 120ms, both feet are on the ground simultaneously about 37% of the time.
- **Follows a fixed internal rhythm**: it steps on a roughly 0.8-second cycle (left and right feet offset by half a cycle) about 80-82% of the time, regardless of how fast it's told to walk. This rhythm isn't imposed by the controller — it's a clock signal fed into the policy's input, and the policy learned to follow it because doing so is rewarded.
- **Tracks the commanded speed reasonably well**: on a standard evaluation set, it covers about 96-99% of the distance a perfect speed-tracker would (this number moves around a bit depending on exactly which seeds/commands are tested — see numbers below).
- **Walks fairly straight**: heading drifts on the order of 6-13 degrees per 10 seconds under a real walking command (varies by test set and command speed — it drifts more at higher commanded speeds).
- **Stands still without spinning when told to stand still** — this was a real bug until recently (see "Notable bug found and fixed" below); it's now fixed and holds up under a lot of scrutiny.
- **Rarely falls** in the conditions it's actually tested under: 0 falls in 100 held-out evaluation episodes on the current best checkpoint, and 0 falls across roughly 400 evaluation episodes total collected while validating the two most recent fixes.
- **Has NOT been trained or tested against physical disturbances** (pushes, uneven ground, carrying extra weight) in its current best form — see "Known limitations" below. An earlier experimental branch did add push-robustness, but hasn't been merged into the current best checkpoint yet.
- **Doesn't swing its arms or move its torso** — only the 12 leg joints are under RL control; the waist and arms are held rigid in a fixed pose. This is a known simplification, not an oversight.
- **Stride length is short of a natural human gait** — a 100-120ms swing phase is well under a more human-scale 200-300ms stride. The gait looks more like a brisk, somewhat stiff-legged walk than a natural one.

In short: **it's a real, stable, straight-line walking gait that tracks a commanded speed and doesn't fall under the conditions it's been tested in — but it's a "clean room" gait**, never tested against pushes, uneven terrain, or a second training seed, with a noticeably short stride.

## Current best model, quantified

Evaluated on 100 held-out random seeds it was never trained or selected on:

| Metric | Value |
|---|---|
| Falls | 0 / 100 |
| Distance actually covered vs. commanded (ratio) | 0.974 |
| Velocity tracking error (steady-state) | ~0.013 m/s |
| Heading drift under normal walking | 8.3 deg per 10s (mixed commands) |
| Heading drift when told to stand still | 2.1 deg per 20s (down from 22.5 before the fix below) |
| Gait-clock adherence (feet on the ground/air on schedule) | ~81% |
| Median swing (foot-in-air) duration | 120 ms |

## How it got here (brief history, most important parts)

The policy is the result of a chain of ~10 sequential fine-tuning runs (not one big training run), each one fixing one specific, measured problem found by directly testing the previous checkpoint (not by guessing):

1. A minimal reward (track velocity + stay alive + don't waste torque) trained from scratch, at a deliberately low learning rate — found this reaches better tracking than the "obvious" learning rate.
2. Fine-tuned in a longer-stride reward on top of that, warm-started (not from scratch — from-scratch attempts to reward longer steps kept exploiting the reward, e.g. by sliding on the ground instead of lifting the foot).
3. Added a "gait clock" input (a repeating rhythm signal, copied from Unitree's own reference implementation) plus a reward for following it — this is what got the robot from a shuffle to a real alternating gait.
4. Sharpened the velocity-tracking reward once it was clear the clock policy's tracking error was mostly harmless within-stride wobble, not real mis-tracking.
5. Added a penalty for accumulated heading drift (not just instantaneous spin) once it was clear the robot could have zero average rotation rate while still slowly curving off course.
6. **Notable bug found and fixed**: deliberately testing a command the robot was never trained on exactly — "stand perfectly still" — found it would spin in place, 19-26 degrees over 20 seconds, completely undetected by every prior evaluation (because evaluation always happened to test some nonzero speed). The first fix (just penalizing it harder) worked initially but decayed with more training. The fix that actually held up: literally including "stand still" as a real, recurring command during training (15% of episodes), not just as something to be penalized in the abstract. This was then stress-tested by training 6M steps further specifically to check whether the fix would decay the way the first attempt did — it didn't; it kept improving instead, down to 2.1 degrees.

Every experiment along the way is logged with a stated hypothesis, what was measured, and the decision made — including the failed attempts (a "reward the robot ignored," a policy that learned to stand still and farm survival reward, one that learned to skate on a planted foot to avoid a poorly-designed reward term, etc.). Nothing here is a single lucky run; it's an evidence-driven iteration loop, and about 10-20% of experiments didn't pan out and were logged as such rather than discarded.

## Known limitations / open questions (unsolved, honestly stated)

- **Single random seed.** Every training run used the same seed. It's never been checked how much a different seed would change the result — the current numbers could be somewhat lucky or somewhat unlucky, unknown which.
- **No physical robustness in the current best model.** It's never been pushed, walked on uneven ground, or had its mass/friction/motor-strength varied during training. A side experiment found that robustness to *pushes specifically* is nearly free to add (small fine-tune, no measurable cost), while robustness to *motor-strength variation* costs real tracking precision — but neither has been merged into the current best checkpoint yet.
- **Fixed stepping rhythm.** The 0.8-second stepping period doesn't change with commanded speed — a real human (and the reference implementation this project follows) would take faster, shorter steps to go faster rather than keeping the same rhythm. Untried.
- **Short stride.** 100-120ms swing vs. a more natural 200-300ms. There's a known, untried lever for this (a config knob that was deliberately not pushed further because past experiments showed it trades away tracking precision).
- **Open-loop heading.** The robot is trained to walk straight from wherever it started, not to actively steer toward a commanded heading. Real quadruped/humanoid RL controllers usually take a yaw-rate command instead; this project's interface is stricter/simpler.
- **A specific, unexplained low-speed fall.** Two particular random seeds (out of the standard 20-seed test set) have caused falls in several past checkpoints at a specific low commanded speed. The current model happens not to fall on them, but nobody has found *why* those two seeds are hard.
- **A left/right foot-height asymmetry** was noticed while chasing the "stands still but spins" bug — the left foot consistently lifts a bit higher than the right during standing. It's a plausible root cause of some of the drift issues, but was never directly corrected (only compensated for indirectly).
- **No hardware transfer.** Simulation only. Sim-to-real gap (actuator dynamics, sensor noise, latency, etc.) has not been addressed at all.

## What would be useful feedback

Given all of the above, the things this project would most benefit from outside perspective on:
1. Is there a smarter way to close the "single seed" and "no physical robustness in the merged model" gaps than just running more of the same recipe (more seeds, then re-run the push-robustness fine-tune on the current checkpoint)?
2. Is the fixed-period gait clock a real limitation worth fixing (speed-dependent stepping frequency), or is this a minor cosmetic gap not worth the engineering effort relative to the open robustness/generalization gaps?
3. Any known failure modes or blind spots in this kind of foot-height/gait-clock-based reward design that this project hasn't hit yet but should proactively test for?
4. General sanity check: does the "chain of small, measured fine-tunes" approach (vs. one big training run, or a curriculum) look sound, or is there a known better practice being missed here?
