"""Utility functions"""


def linear_schedule(begin_value, end_value, begin_t, end_t=None, decay_steps=None):
    """Linear schedule, used for exploration epsilon in DQN agents."""

    decay_steps = decay_steps if end_t is None else end_t - begin_t

    def step(t):
        """Implements a linear transition from a begin to an end value."""
        frac = min(max(t - begin_t, 0), decay_steps) / decay_steps
        return (1 - frac) * begin_value + frac * end_value

    return step
