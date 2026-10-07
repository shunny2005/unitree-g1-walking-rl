"""Velocity-error decomposition. `mean_vel_error` (the mean instantaneous
||v_cmd - v||) mixes two different things: steady mis-tracking and the
within-stride oscillation of body velocity that any walking gait has. A
longer, slower stride oscillates the velocity more, so the instantaneous
number can rise even while the robot tracks the command about as well
over a stride (R-0005, docs/RESEARCH_LOG.md: the clock policy's error was
0.084 = 0.046 period-averaged error + 0.077 rms ripple, versus the
canonical's 0.040 = 0.032 + 0.027). Reporting both keeps a gait-rhythm
change from being misread as a tracking failure -- or hidden by one.
"""

from __future__ import annotations

import numpy as np

# One gait-clock period (0.8 s at 50 Hz), used for every policy, clocked or not,
# so the numbers are comparable across runs.
DEFAULT_WINDOW_STEPS = 40
DEFAULT_SKIP_STEPS = 25  # start-up transient after reset


def velocity_error_decomposition(
    vel_xy: np.ndarray,
    command_xy: np.ndarray,
    window: int = DEFAULT_WINDOW_STEPS,
    skip: int = DEFAULT_SKIP_STEPS,
) -> dict[str, float]:
    """vel_xy: (T, 2) base-frame velocity history; command_xy: (2,).

    Returns
      instantaneous: mean ||v_t - cmd||
      period_avg:    mean ||movavg_window(v)_t - cmd||  (steady tracking, ripple averaged out)
      ripple_rms:    rms ||v_t - movavg_window(v)_t||   (within-stride oscillation)
    or NaNs if the episode is too short to fill one window after the skip.
    """
    v = np.asarray(vel_xy, dtype=float)[skip:]
    if len(v) < window:
        return {"instantaneous": float("nan"), "period_avg": float("nan"), "ripple_rms": float("nan")}
    cmd = np.asarray(command_xy, dtype=float)
    kernel = np.ones(window) / window
    moving = np.stack([np.convolve(v[:, j], kernel, mode="valid") for j in range(v.shape[1])], axis=1)
    return {
        "instantaneous": float(np.mean(np.linalg.norm(v - cmd, axis=1))),
        "period_avg": float(np.mean(np.linalg.norm(moving - cmd, axis=1))),
        "ripple_rms": float(np.sqrt(np.mean(np.sum((v[window - 1 :] - moving) ** 2, axis=1)))),
    }
