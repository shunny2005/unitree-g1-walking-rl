"""Aggregate per-episode results by commanded-speed band. The all-episodes
mean hid the structure of every recent comparison: the clock policy's
overall `dist_ratio` looked better while its high-speed band was worse
(R-0005), and the later diagnosis of that gap (R-0006) needed heading drift,
straightness and speed bias broken out by band. Pure functions over
`evaluation.evaluate.run_episode` dicts, so they are unit-testable.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_EDGES = (0.0, 0.15, 0.4, math.inf)  # m/s: near-zero / mid / high commanded speed


def band_label(lo: float, hi: float) -> str:
    return f"<{hi:g}" if lo <= 0 else (f">={lo:g}" if math.isinf(hi) else f"{lo:g}-{hi:g}")


def command_speed(episode: dict) -> float:
    return float(math.hypot(*episode["command"]))


def _mean(values) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(episodes: list[dict]) -> dict:
    """Summary of one group of episodes. `dist_ratio` is over episodes long
    enough to be reliable (docs/DECISIONS.md ADR-003 addendum); falls count
    every episode that terminated."""
    reliable = [e["distance_ratio"] for e in episodes if e["distance_ratio_reliable"]]
    return {
        "n": len(episodes),
        "falls": sum(e["termination_reason"] is not None for e in episodes),
        "dist_ratio": _mean(reliable),
        "vel_err": _mean(e["mean_vel_error"] for e in episodes),
        "vel_avg_err": _mean(e["mean_period_avg_vel_error"] for e in episodes),
        "vel_ripple": _mean(e["vel_ripple_rms"] for e in episodes),
        "abs_yaw_per_10s_deg": _mean(abs(e["yaw_drift_per_10s_deg"]) for e in episodes),
        "straightness": _mean(e["straightness"] for e in episodes),
        "clock_adherence": _mean(e["clock_adherence"] for e in episodes),
    }


def aggregate_by_band(episodes: list[dict], edges=DEFAULT_EDGES) -> dict[str, dict]:
    """{'all': ..., '<0.15': ..., '0.15-0.4': ..., '>=0.4': ...}; empty bands are omitted."""
    out = {"all": aggregate(episodes)}
    for lo, hi in zip(edges[:-1], edges[1:]):
        group = [e for e in episodes if lo <= command_speed(e) < hi]
        if group:
            out[band_label(lo, hi)] = aggregate(group)
    return out
