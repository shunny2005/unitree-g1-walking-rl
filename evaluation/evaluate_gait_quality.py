"""Gait-quality evaluation across the fixed command grid
(docs/GAIT_QUALITY_OBJECTIVE.md). Separate from evaluate.py/evaluate_banded.py
(task performance, group A, unchanged) -- this covers groups B-E (fluidity,
symmetry, geometry, efficiency).

Usage:
    python -m evaluation.evaluate_gait_quality --model <ckpt.zip> --config <cfg.yaml> --out-dir experiments/gait_quality/baseline_exp0043
    python -m evaluation.evaluate_gait_quality --model <ckpt.zip> --config <cfg.yaml> --seeds 1000-1019 --out-dir ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config.loader import load_config
from envs.g1_walk_env import G1WalkEnv
from evaluation.gait_quality import run_gait_quality_episode

REPO_ROOT = Path(__file__).resolve().parent.parent

# (bucket name, (vx, vy)) -- see docs/GAIT_QUALITY_OBJECTIVE.md "Command grid"
COMMAND_GRID: list[tuple[str, tuple[float, float]]] = [
    ("stand_still", (0.0, 0.0)),
    ("slow", (0.15, 0.0)),
    ("slow2", (0.2, 0.0)),
    ("medium", (0.35, 0.0)),
    ("high", (0.55, 0.0)),
    ("lateral", (0.0, 0.18)),
    ("diagonal", (0.35, 0.15)),
]

DEFAULT_SEEDS = [1000, 1001, 1002, 1003, 1004]  # 1004 is a historically difficult seed (F-0002/EXP-0027)


def _mean_finite(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seeds", default=None, help="e.g. '1000-1019' or '1000,1002,1004'; default: 1000-1004")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.seeds is None:
        seeds = DEFAULT_SEEDS
    elif "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    cfg = load_config(args.config)
    env = G1WalkEnv(cfg)
    model = PPO.load(args.model, device=cfg.ppo.device)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}
    bucket_summaries: dict[str, dict] = {}

    for bucket, command in COMMAND_GRID:
        episodes = []
        for seed in seeds:
            result = run_gait_quality_episode(model, env, seed=seed, forced_command=command)
            episodes.append(
                {
                    "seed": seed,
                    "metrics": result["metrics"],
                    "terminated": bool(result["raw"]["terminated"]),
                    "termination_reason": result["raw"]["termination_reason"],
                    "episode_length_s": result["raw"]["episode_length_s"],
                }
            )
        all_results[bucket] = episodes

        metric_names = episodes[0]["metrics"].keys()
        summary = {
            name: _mean_finite([e["metrics"][name] for e in episodes if isinstance(e["metrics"][name], (int, float))])
            for name in metric_names
        }
        summary["n_episodes"] = len(episodes)
        summary["n_falls"] = sum(1 for e in episodes if e["terminated"])
        bucket_summaries[bucket] = summary

        falls = summary["n_falls"]
        print(
            f"{bucket:12s} cmd={command}  falls={falls}/{len(seeds)}  "
            f"dist={summary['distance_traveled_m']:.2f}m  "
            f"torque_rms={summary['torque_rms']:.1f}  work/m={summary['work_per_meter_j']:.1f}J  "
            f"COT={summary['cost_of_transport']:.2f}  "
            f"swing(L/R)={summary['swing_duration_mean_left_ms']:.0f}/{summary['swing_duration_mean_right_ms']:.0f}ms  "
            f"clearance(L/R)={summary['peak_clearance_left_m']:.3f}/{summary['peak_clearance_right_m']:.3f}m  "
            f"action_jerk={summary['action_jerk_rms']:.3f}"
        )

    (out_dir / "gait_quality_by_bucket.json").write_text(
        json.dumps({"seeds": seeds, "buckets": bucket_summaries}, indent=2)
    )
    (out_dir / "gait_quality_episodes.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "episodes": {
                    bucket: [{k: v for k, v in e.items()} for e in episodes] for bucket, episodes in all_results.items()
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir}/gait_quality_by_bucket.json and gait_quality_episodes.json")
    env.close()


if __name__ == "__main__":
    main()
