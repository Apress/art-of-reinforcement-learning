"""Utility functions"""
from typing import Iterable
import numpy as np


def linear_schedule(begin_value, end_value, begin_t, end_t=None, decay_steps=None):
    """Linear schedule, used for exploration epsilon in DQN agents."""

    decay_steps = decay_steps if end_t is None else end_t - begin_t

    def step(t):
        """Implements a linear transition from a begin to an end value."""
        frac = min(max(t - begin_t, 0), decay_steps) / decay_steps
        return (1 - frac) * begin_value + frac * end_value

    return step


def split_indices_into_bins(
    bin_size: int,
    max_indices: int,
    min_indices: int = 0,
    shuffle: bool = False,
) -> Iterable[int]:
    """Split indices to small bins."""

    # Split indices into 'bins' with bin_size.
    indices = np.arange(min_indices, max_indices)

    if shuffle:
        np.random.shuffle(indices)

    if max_indices <= bin_size:
        return [indices]

    indices_list = []
    for i in range(0, len(indices), bin_size):
        indices_list.append(indices[i : i + bin_size])  # noqa: E203

    # Make sure the last one has the same 'bin_size'.
    if len(indices_list[-1]) != bin_size:
        indices_list[-1] = indices[-bin_size:]

    return indices_list
