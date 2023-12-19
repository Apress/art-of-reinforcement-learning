"""Utility functions"""
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
    print('\n')


def print_state_value(env, V, num_decimals=2):
    """Make it human readable."""
    results = {}
    for s in range(V.shape[0]):  # For every state
        state_name = env.get_state_name(s)
        results[state_name] = round(V[s], num_decimals)

    print(results)
    print('\n')


def print_state_action_value(env, Q, num_decimals=2):
    """Make it human readable."""
    results = {}
    for s in range(Q.shape[0]):  # For every state
        s_name = env.get_state_name(s)
        for a in env.get_legal_actions(s):  # For every legal action
            action = env.get_action_name(a)
            results[f'{s_name}-{action}'] = round(Q[s, a], num_decimals)

    print(results)
    print('\n')


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

    num_actions = q.shape[0]

    # By default all actions are masked as illegal.
    mask = [1] * num_actions
    for action in legal_actions:
        mask[action] = 0

    # Create a masked array to making illegal actions invalid.
    mask_q = np.ma.masked_array(q, mask=mask)

    max_indices = np.argwhere(mask_q == np.amax(mask_q))

    # Break ties if have multiple maximum values.
    max_indices = max_indices.flatten().tolist()

    return np.random.choice(max_indices)
