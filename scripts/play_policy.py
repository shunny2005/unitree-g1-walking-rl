"""Interactive MuJoCo viewer: watch a trained policy walk in real time.

Opens the native MuJoCo viewer window (mujoco.viewer.launch_passive, the
same mechanism envs/g1_walk_env.py's own render() uses) and steps the
policy through it in real time, looping episodes until the window is
closed or the process is interrupted.

Usage:
    python -m scripts.play_policy
    python -m scripts.play_policy --model <ckpt.zip> --config <cfg.yaml> --vx 0.3 --vy 0.0
"""

from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO

from config.loader import load_config
from envs.g1_walk_env import G1WalkEnv

DEFAULT_MODEL = "experiments/runs/exp0047_ankle_gain_more_zerocmd/best_checkpoint_4M.zip"
DEFAULT_CONFIG = "config/exp0047_ankle_gain_more_zerocmd.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=1002)
    parser.add_argument("--vx", type=float, default=None, help="override commanded forward velocity")
    parser.add_argument("--vy", type=float, default=None, help="override commanded lateral velocity")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = G1WalkEnv(cfg, render_mode="human")
    model = PPO.load(args.model, device=cfg.ppo.device)

    print(f"Loaded {args.model}\nconfig: {args.config}\nClose the viewer window or Ctrl+C here to stop.")

    seed = args.seed
    try:
        while True:
            obs, info = env.reset(seed=seed)
            if args.vx is not None or args.vy is not None:
                cmd = env._command.copy()
                if args.vx is not None:
                    cmd[0] = args.vx
                if args.vy is not None:
                    cmd[1] = args.vy
                env._command = cmd
            print(f"episode seed={seed} command={env._command}")
            env.render()
            terminated = truncated = False
            while not (terminated or truncated):
                t0 = time.time()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                elapsed = time.time() - t0
                sleep_s = cfg.sim.control_dt - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
            print(f"episode ended: terminated={terminated} truncated={truncated} reason={info.get('termination_reason')}")
            seed += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
