"""Speed-banded evaluation on a seed range, defaulting to seeds 2000-2059
(disjoint from training and from the standard 1000-1019 eval seeds), so a
comparison is not decided by one lucky or unlucky 20-episode subsample --
the canonical itself fell 0/20 on the standard seeds but 1/60 on these.

Usage:
    python -m evaluation.evaluate_banded --model <ckpt.zip> --config <cfg.yaml> [--n-seeds 60]
"""

from __future__ import annotations

import argparse
import json

from stable_baselines3 import PPO

from config.loader import load_config
from envs.g1_walk_env import G1WalkEnv
from evaluation.banded import aggregate_by_band
from evaluation.evaluate import run_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--n-seeds", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of the table")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = G1WalkEnv(cfg)
    model = PPO.load(args.model, device=cfg.ppo.device)
    episodes = [run_episode(model, env, s) for s in range(args.seed_start, args.seed_start + args.n_seeds)]
    bands = aggregate_by_band(episodes)
    if args.json:
        print(json.dumps(bands, indent=2))
    else:
        cols = ["n", "falls", "dist_ratio", "vel_err", "vel_avg_err", "vel_ripple", "abs_yaw_per_10s_deg", "straightness", "clock_adherence"]
        print(f"{'band':9s} " + " ".join(f"{c[:11]:>11s}" for c in cols))
        for name, b in bands.items():
            print(f"{name:9s} " + " ".join(f"{b[c]:11.3f}" if c not in ('n', 'falls') else f"{b[c]:11d}" for c in cols))
    env.close()


if __name__ == "__main__":
    main()
