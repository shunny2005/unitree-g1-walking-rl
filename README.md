# Unitree G1 Walking Policy (MuJoCo + PPO)

A reinforcement-learning walking controller for the Unitree G1 humanoid, trained entirely in simulation on a CPU-only laptop: **MuJoCo** physics, a custom **Gymnasium** environment, and **stable-baselines3 PPO**. The environment, control mapping, reward, evaluation, and experiment tracking are written from scratch; only PPO itself comes from the library.

The policy controls the 12 leg joints through a PD law, tracks a commanded planar velocity (forward 0-0.6 m/s, lateral ±0.2 m/s), follows a periodic gait clock, and stands still without drifting when commanded to zero.

## Where to start

| If you want... | Read |
|---|---|
| The system explained layer by layer, in plain English | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Exact formulas, reward weights, PPO settings, training lineage, results | [IMPLEMENTATION.md](IMPLEMENTATION.md) |
| Current status, open problems, next steps | [PROJECT_STATE.md](PROJECT_STATE.md) |
| Why each design decision was made (and what was rejected) | [docs/DECISIONS.md](docs/DECISIONS.md) |
| Every failure found and how it was diagnosed | [docs/FAILURE_ANALYSIS.md](docs/FAILURE_ANALYSIS.md) |
| The full experiment-by-experiment history | [experiments/EXPERIMENT_LOG.md](experiments/EXPERIMENT_LOG.md) |
| The gait-quality (fluidity/symmetry/efficiency) program | [docs/GAIT_QUALITY_OBJECTIVE.md](docs/GAIT_QUALITY_OBJECTIVE.md), [docs/GAIT_QUALITY_PROGRAM_SUMMARY.md](docs/GAIT_QUALITY_PROGRAM_SUMMARY.md) |

## Current best policy

`experiments/runs/exp0047_ankle_gain_more_zerocmd/best_checkpoint_4M.zip`, config `config/exp0047_ankle_gain_more_zerocmd.yaml` (the end of an 11-stage warm-start fine-tuning lineage, ~66M effective training steps, a few hours of CPU time).

On 100 held-out evaluation seeds: 0 falls, distance ratio 0.971, yaw drift 8.3 deg/10 s, zero-command drift 2.0 deg/20 s, ~83% gait-clock adherence. Rollout videos and training curves live alongside each checkpoint in `experiments/runs/<name>/`.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/ -q

# Evaluate the current best policy (20 fixed seeds)
PYTHONPATH=. python -m evaluation.evaluate \
  --model experiments/runs/exp0047_ankle_gain_more_zerocmd/best_checkpoint_4M.zip \
  --config config/exp0047_ankle_gain_more_zerocmd.yaml

# Watch it walk in the interactive MuJoCo viewer
# (macOS requires mjpython for the viewer; Linux/Windows can use plain python)
PYTHONPATH=. mjpython -m scripts.play_policy
```

Full reproduction commands for the whole training lineage are in [IMPLEMENTATION.md](IMPLEMENTATION.md) section 11.

## Repository layout

```
config/        one YAML per experiment; config/loader.py is the typed schema
control/       joint indexing, PD controller, default pose
envs/          Gymnasium environment, observations, contacts, gait clock, termination, reset
rewards/       all reward terms, returned separately
training/      PPO entry point, vectorized envs, checkpoint expansion
evaluation/    deterministic evaluation, speed-banded evaluation, gait-quality metrics, video
diagnostics/   controlled-gait diagnostic plots
experiments/   experiment log and per-run artifacts (configs, eval JSON, curves, videos, best checkpoints)
docs/          decisions, failure analyses, research log, objective/summary docs
tests/         unit and contract tests
assets/g1/     vendored Unitree G1 MuJoCo model (see below)
```

## Honest limits

Simulation only (no hardware transfer), CPU only, and **a single training seed throughout** -- so seed-to-seed variance is unmeasured. Known open issues (a ~10% fall rate under pure-lateral commands, a left/right swing-duration asymmetry, a fixed 0.8 s gait period) are documented in [PROJECT_STATE.md](PROJECT_STATE.md) and [IMPLEMENTATION.md](IMPLEMENTATION.md) section 13.

## Third-party assets and license

`assets/g1/` is the Unitree G1 model vendored, unmodified, from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (see `assets/g1/PROVENANCE.md` and the upstream `assets/g1/LICENSE`); the derived walking model `assets/g1/g1_walk.xml` is generated from it by `scripts/build_walk_model.py`. The project's own code does not yet carry a license file -- add one before reusing it.

## Acknowledgements

Reward structure and gait-clock design follow ideas from the official [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) and [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) implementations; see `docs/RESEARCH_LOG.md` for what was borrowed, what was measured, and what was rejected.
