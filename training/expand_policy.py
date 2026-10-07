"""Warm-start a PPO policy whose observation is LARGER than the checkpoint's
(e.g. adding the 2 gait-clock entries, R-0004, to a 44-input checkpoint).

The project's only working recipe for introducing new stepping terms is to
fine-tune an already-walking policy (EXP-0016 through EXP-0026; every
from-scratch attempt collapsed or skated). Adding observation entries
breaks that recipe unless the old weights can still be loaded, so this
copies every parameter and zero-pads the first-layer input columns for the
new entries. With zero weights on the new inputs the expanded network computes
exactly the same function as the original until training changes them --
the new entries must be appended LAST (envs/observations.build_observation).
"""

from __future__ import annotations

import torch
from stable_baselines3 import PPO


def load_expanded(new_model: PPO, old_checkpoint: str) -> int:
    """Copy `old_checkpoint`'s weights into `new_model`, zero-padding input
    columns where `new_model`'s observation is larger. Returns the number of
    zero-padded input columns. Raises if any other shape differs."""
    old_model = PPO.load(old_checkpoint, device="cpu")
    old_state = old_model.policy.state_dict()
    new_state = new_model.policy.state_dict()
    if set(old_state) != set(new_state):
        raise ValueError(f"parameter names differ: {set(old_state) ^ set(new_state)}")

    padded_cols = 0
    with torch.no_grad():
        for name, new_t in new_state.items():
            old_t = old_state[name]
            if old_t.shape == new_t.shape:
                new_t.copy_(old_t)
            elif old_t.ndim == 2 and old_t.shape[0] == new_t.shape[0] and old_t.shape[1] < new_t.shape[1]:
                new_t.zero_()
                new_t[:, : old_t.shape[1]] = old_t
                padded_cols = max(padded_cols, new_t.shape[1] - old_t.shape[1])
            else:
                raise ValueError(f"cannot expand {name}: {tuple(old_t.shape)} -> {tuple(new_t.shape)}")
    new_model.policy.load_state_dict(new_state)
    return padded_cols
