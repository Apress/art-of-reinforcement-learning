"""Utility functions"""
from typing import Iterable
import numpy as np


def print_policy(env, policy):
    """Make it human readable."""
    results = {}
    for s in range(policy.shape[0]):  # For every state
        if not env.is_terminal_state(s):
            legal_actions = env.get_legal_actions(s)
            a_star = argmax_over_legal_actions(policy[s], legal_actions)
            state_name = env.get_state_name(s)
            results[state_name] = env.get_action_name(a_star)
    print(results)


def print_state_value(env, V, num_decimals=2):
    """Make it human readable."""
    results = {}
    for s in range(V.shape[0]):  # For every state
        state_name = env.get_state_name(s)
        results[state_name] = round(V[s], num_decimals)
    print(results)


def print_state_action_value(env, Q, num_decimals=2):
    """Make it human readable."""
    results = {}
    for s in range(Q.shape[0]):  # For every state
        state_name = env.get_state_name(s)
        results[state_name] = {}

        for a in env.get_legal_actions(s):  # For every legal action
            action = env.get_action_name(a)
            results[state_name][action] = round(Q[s, a], num_decimals)
    print(results)


def compute_vstar_from_qstar(env, Q):
    """Compute the optimal state value function from optimal state-action value function."""
    results = np.zeros(Q.shape[0])
    for s in range(Q.shape[0]):  # For every state
        legal_actions = env.get_legal_actions(s)
        a_star = argmax_over_legal_actions(Q[s], legal_actions)
        results[s] = Q[s, a_star]

    return results


def argmax_over_legal_actions(q, legal_actions):
    """Since not every action is legal in a state, the standard numpy.argmax() will fail for some case.
    For example, if the values are negative for legal actions, and 0 for illegal actions,
    then the standard numpy.argmax() will select one from those illegal actions instead of legal action.

    This custom argmax makes sure we only select the ones over legal actions.
    """
    assert len(q.shape) == 1

    action_dim = q.shape[0]

    # Create a mask to skip illegal actions.
    # example: mask = [0, 1, 0] means index=1 is skipped during argmax() operation.

    # By default all actions are masked as illegal.
    mask = [1] * action_dim
    for action in legal_actions:
        mask[action] = 0

    # Create a masked array to making illegal actions invalid.
    mask_q = np.ma.masked_array(q, mask=mask)

    max_indices = np.argwhere(mask_q == np.amax(mask_q))

    # return mask_arr.argmax()

    # Break ties if have multiple maximum values.
    max_indices = max_indices.flatten().tolist()

    return np.random.choice(max_indices)


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
        # raise ValueError(
        #     f'Expect max_indices to be greater than bin_size, got {max_indices} and {bin_size}'
        # )
        return [indices]

    indices_list = []
    for i in range(0, len(indices), bin_size):
        indices_list.append(indices[i : i + bin_size])  # noqa: E203

    # Make sure the last one has the same 'bin_size'.
    if len(indices_list[-1]) != bin_size:
        indices_list[-1] = indices[-bin_size:]

    # Sanity checks
    # for item in indices_list:
    #     assert len(item) == bin_size

    return indices_list
