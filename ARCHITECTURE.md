# How This Walking Policy Works, Layer by Layer

A plain-English tour of the system, from physics up to training. For exact formulas, weights, and the full experimental history, see [IMPLEMENTATION.md](IMPLEMENTATION.md) — this file is the map, that one is the territory.

Current canonical: `exp0043_zero_command_durability_check` @ 12M steps.

## The one-paragraph version

A physics simulation (MuJoCo) holds a model of the Unitree G1 robot. Every 20ms, a neural network looks at the robot's current state and outputs 12 numbers (one per leg joint). Those numbers get turned into motor torques by a PD controller. The robot moves, we score how well it's walking, and that score trains the network via PPO. Repeat millions of times. That's it — everything below is detail on each link in that chain.

```
 physics state  →  observation  →  policy network  →  action  →  PD control  →  torques  →  physics step
      ▲                                                                                          │
      └──────────────────────────────── reward + termination check ◄────────────────────────────┘
```

## Layer 1 — The robot and the physics engine

**What it is:** a MuJoCo model of the real Unitree G1 humanoid (`assets/g1/`, vendored from Google DeepMind's `mujoco_menagerie`, unmodified upstream). MuJoCo simulates rigid-body physics — gravity, contact, joint limits — at 500Hz (a physics step every 2ms).

**Why this layer exists:** you can't train a walking policy without something to walk. This is the ground truth the rest of the system is built on top of.

**One local modification:** the 12 leg joints were converted from MuJoCo's built-in position actuators to raw torque actuators (`scripts/build_walk_model.py`), so this project's own PD controller (Layer 2) is what's actually driving the legs, not MuJoCo's.

## Layer 2 — Control: turning "intent" into torque

**What it is:** the policy doesn't directly command torques. It outputs a small offset (in [-1, 1]) per joint, which gets scaled and added to a default standing pose to get a *target* joint angle. A PD (proportional-derivative) controller then computes the torque needed to drive each joint toward that target:

```
torque = kp * (target_angle - current_angle) - kd * current_velocity
```

This runs every physics step (500Hz), but the policy only picks new targets every 10th step (50Hz) — the "control decimation." Between policy decisions, the PD controller keeps chasing the same target.

**Why this layer exists:** asking a neural network to output raw torques directly is a much harder learning problem (jittery, unstable) than asking it to output *where the joint should go* and letting a simple, well-understood feedback loop handle the low-level physics. This is standard practice in legged-robot RL, not a project-specific choice.

**Scope:** only the 12 leg joints are controlled this way. The waist and arms (17 joints) are frozen at a fixed standing pose — not part of the learning problem.

## Layer 3 — Observation: what the network is allowed to see

**What it is:** a 46-number snapshot of the robot's state, fed to the network every 20ms:

| What | How many numbers |
|---|---|
| Leg joint angles (relative to default pose) | 12 |
| Leg joint velocities | 12 |
| Body rotation rate | 3 |
| Which way is "down" (from the body's own perspective) | 3 |
| Commanded walking speed (forward, sideways) | 2 |
| What the network did last step | 12 |
| A repeating "clock" telling it what phase of the stride it should be in | 2 |

**Why this layer exists:** this is the interface between physics and intelligence — everything the policy knows about the world has to come through here. It's deliberately restricted to what a real robot could plausibly sense (no "cheating" with information like absolute position or velocity that's hard to measure on hardware) — see the note on linear velocity below.

**The gait clock (2 numbers) is the newest and most impactful addition:** without it, nothing tells the policy *when* it should be lifting a foot versus planting it — it has to invent its own rhythm from scratch, and early attempts at this produced a shuffling, barely-lifting gait. The clock is a simple repeating signal (period 0.8s, left/right feet offset by half a cycle) borrowed directly from Unitree's own reference implementation. It doesn't force a gait — it's just a metronome the reward layer can reference.

**What's deliberately left out:** the robot's absolute forward/sideways speed. That's genuinely hard to measure on a real robot without extra sensors, so it's used only in the reward (as a training-time-only "cheat" to score the walk) — never given to the network itself. This matters: it means the policy has to learn to walk well *without* directly knowing its own speed, the same constraint a real deployment would have.

## Layer 4 — Commands, reset, and termination

**Commands:** once per episode, the robot is given a target walking velocity (a forward speed 0–0.6 m/s, a sideways speed ±0.2 m/s) that stays fixed for the whole 20-second episode. 15% of the time, the command is exactly "stand still" (zero) — this is deliberate (see Layer 8) and not just an edge of the random range.

**Reset:** every episode starts from the same standing pose, zero velocity. Deterministic given a seed, so evaluation is reproducible.

**Termination — the episode ends early (counts as a failure) if:**
- the robot's pelvis drops below 0.5m (it fell down), or
- the robot tips more than ~72° (it's tipping over), or
- the physics produces garbage numbers (a bug, not a real failure mode encountered in practice)

Otherwise the episode runs the full 20 seconds (1000 control steps) and "truncates" normally.

**Why this layer exists:** this is what turns an open-ended physics simulation into a well-posed learning problem — a clear start state, a clear signal for "you failed," and a clear time horizon.

## Layer 5 — Reward: defining "good"

**What it is:** a single number, computed every control step, that PPO is trying to maximize in total over an episode. It's a sum of about a dozen terms, added one at a time over the project's history, each one added *only* after a specific, measured failure motivated it (not speculatively):

- **Track the commanded velocity** (two terms — a smooth one and a sharper one for precision) — the core objective.
- **Stay alive** — a small constant bonus every step it hasn't fallen.
- **Don't waste torque** — a tiny penalty on motor effort.
- **Don't spin** — penalizes rotating in place.
- **Don't twitch** — penalizes jerky, step-to-step changes in action.
- **Actually take steps, and follow the clock** — rewards feet lifting for a real amount of time, at a real height, in the rhythm the clock suggests, and penalizes a foot sliding while it's on the ground.
- **Walk straight** — penalizes the robot's heading drifting away from where it started, accumulated over the whole episode (not just instantaneous spin — a robot can have zero angular velocity on average while still slowly curving off course, and this term is specifically about catching that).

**Why one term at a time, not all at once from the start:** every term this project tried to add speculatively (without first measuring the failure it was meant to fix) either did nothing (too weak) or broke something else (too strong, or created an exploit — e.g. a policy that keeps a foot glued to the ground to dodge a reward it doesn't like). The working recipe: watch the robot, name what's actually wrong, measure how big that problem is in reward terms, then add a term sized to that measurement.

## Layer 6 — The policy network and PPO training loop

**The network:** a small feedforward network (two hidden layers of 256 neurons), taking the 46-number observation and outputting 12 numbers (the action). There's a second, identical-shaped network (the "value network" or "critic") that only exists during training — it predicts how much future reward to expect, which is what makes PPO's learning signal much less noisy than reward alone.

**The training loop (this is what stable-baselines3, a well-tested open-source library, does — not custom code):**
1. Run the current policy in 12 copies of the environment in parallel, collecting ~12,000 steps of experience.
2. Use the value network to estimate, for each step, "was this action better or worse than expected?" (this is called GAE — generalized advantage estimation).
3. Nudge the network's weights slightly to make good-than-expected actions more likely and bad-than-expected ones less likely — but only a *small*, clipped nudge per update, which is what makes PPO stable rather than prone to wild swings.
4. Repeat, tens of thousands of times, saving a checkpoint periodically.

**Why not write PPO from scratch:** it's a well-understood, thoroughly-tested algorithm; reimplementing it would spend effort on infrastructure instead of on the actual robot-specific problem (the environment, the reward, the control mapping) — where this project's real work is.

**Warm-starting:** rather than training one enormous run from scratch, the canonical policy is the result of a *sequence* of ten training runs, each one fine-tuning the previous checkpoint at a low learning rate to fix one specific, measured issue (add a stride reward, add the gait clock, sharpen tracking, fix a heading drift, fix a zero-command spin, confirm that fix holds up). This is much more stable than trying to get everything right in one from-scratch run — see `IMPLEMENTATION.md` section 8 for the full chain.

## Layer 7 — Evaluation

**What it is:** a completely separate pass from training — no randomness in action selection (the network's best guess only, not exploration), fixed seeds, and metrics that go beyond raw reward:

- **Distance ratio**: how far the robot actually walked versus how far the command asked for. Catches a policy that stands still and farms survival reward without ever addressing whether it's walking — reward and episode-length alone can't tell you that.
- **Yaw drift**: how much the heading wanders over the episode.
- **Velocity error**, split into steady mis-tracking versus normal within-stride wobble (a walking gait naturally oscillates a bit; the two are different problems).
- **Falls**, and — recently — a dedicated test that forces the command to exactly zero and checks whether the robot stands still without spinning (this caught a real bug that ordinary evaluation never would have, since it never happened to test that exact input).

**Why a separate layer at all:** reward is what the training algorithm optimizes, but it's not the same thing as "is this actually a good walk" — a reward function is always an imperfect proxy, and this project found real gaps between the two (a policy discovered a way to score well by skating instead of stepping, one stood almost still, one span in place at a specific untested input) that only showed up once evaluation looked past the aggregate reward number.

## Layer 8 — Extras: domain randomization and the zero-command fix

Two additions sit on top of the core pipeline above, each addressing a specific measured gap rather than being part of the baseline recipe:

- **Domain randomization** (off in the canonical config, available as an alternative checkpoint): randomizing friction, mass, motor strength, and adding random pushes during training, so the policy is exposed to more than one exact physics setup. An ablation found pushes matter far more than the other three factors, and fixing just that is nearly free; adding motor-strength robustness too costs real tracking precision — a genuine trade-off, not a free win.
- **Zero-command training** (`command.zero_command_prob`, active in the canonical): a small but real bug was found where, if asked to stand perfectly still, the robot would slowly spin instead — because "stand still" was never actually practiced during training, only walking at various speeds. The fix wasn't to add a bigger penalty for spinning (tried first, didn't hold up) but to literally include "stand still" as one of the commands practiced during training, 15% of the time. That turned out to work, and — importantly — kept working under further training rather than fading out, which was checked directly rather than assumed.

## Where to go for more

- **The exact numbers** (reward weights, PD gains, PPO hyperparameters, per-experiment results): [IMPLEMENTATION.md](IMPLEMENTATION.md)
- **Why each design decision was made, and what was rejected**: `docs/DECISIONS.md`
- **Every bug found and how it was diagnosed**: `docs/FAILURE_ANALYSIS.md`
- **The full experimental history**, one entry per training run with hypothesis/results/decision: `experiments/EXPERIMENT_LOG.md`
- **External research that shaped a decision** (e.g. reading Unitree's own reference code): `docs/RESEARCH_LOG.md`
