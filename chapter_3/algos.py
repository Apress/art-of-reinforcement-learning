"""DP algorithms for solving MDP tasks."""
import numpy as np
import utils


def create_initial_state_value(env):
    """Returns a 1D numpy.ndarray contains zeros as initial state-value for each states."""
    return np.zeros(env.num_states)


def create_initial_state_action_value(env):
    """Returns a 2D numpy.ndarray with shape [num_states, num_actions] contains zeros as initial state-value for each state-action pairs."""
    return np.zeros((env.num_states, env.num_actions))


def create_random_policy(env):
    """Returns a 2D numpy.ndarray with shape [num_states, num_actions] contains uniform random policy distribution across all legal actions for each states."""
    random_policy = np.zeros((env.num_states, env.num_actions))
    for state in env.get_states():
        legal_actions = env.get_legal_actions(state)
        if legal_actions:
            for action in legal_actions:  # For every legal action
                random_policy[state, action] = 1 / len(legal_actions)
    return random_policy


def create_empty_policy(env):
    """Returns a 2D numpy.ndarray with shape [num_states, num_actions] contains a template for a deterministic policy, with zero probabilities for each state-action pairs."""
    return np.zeros((env.num_states, env.num_actions))


def compute_mrp_state_value(env, discount, delta_threshold=1e-5):
    """
    Given a initial state-value, using dynamic programming to
     compute the state-value function for MRP.

    Args:
        env: a MRP environment.
        discount: discount factor, must be 0 <= discount <= 1.
        delta_threshold: the threshold determining the accuracy of the estimation.

    Returns:
        estimated state value function for the input MRP environment.

    """
    assert 0.0 <= discount <= 1.0

    # Initialize state value function to be all zeros for all states.
    V = create_initial_state_value(env)

    n = 0
    while True:
        delta = 0
        for state in env.get_states():
            old_v = V[state]
            new_v = 0

            reward, successor_states = env.transition_from_state(state)
            new_v += reward

            # Note one state-action might have multiple successor states with different transition probability
            for transition_probs, state_tp1 in successor_states:
                # Weight by the transition probability
                new_v += discount * transition_probs * V[state_tp1]

            V[state] = new_v
            delta = max(delta, abs(old_v - new_v))

        if delta < delta_threshold:
            break
        n += 1
    return V


def policy_evaluation(env, policy, discount, delta_threshold=1e-5):
    """
    Given a policy, and state value function, using dynamic programming to
     estimate the state-value function for this policy.

    Args:
        env: a MDP environment.
        policy: policy we want to evaluate.
        V: state value function for the input policy.
        discount: discount factor, must be 0 <= discount <= 1.
        delta_threshold: the threshold determining the accuracy of the estimation.

    Returns:
        estimated state value function for the input policy.

    """
    assert 0.0 <= discount <= 1.0

    count = 0

    # Initialize state value function to be all zeros for all states.
    V = create_initial_state_value(env)

    while True:
        delta = 0
        for state in env.get_states():
            old_v = V[state]
            new_v = 0
            for action in env.get_legal_actions(state):  # For every legal action
                g = 0
                pi_prob = policy[state, action]

                reward, successor_states = env.transition_from_state_action(state, action)
                g += reward

                # Note one state-action might have multiple successor states with different transition probability
                for transition_probs, state_tp1 in successor_states:
                    # Weight by the transition probabilityironment
                    g += discount * transition_probs * V[state_tp1]

                # Weight by the probability of selecting this action when following the policy
                new_v += pi_prob * g
            V[state] = new_v
            delta = max(delta, abs(old_v - new_v))

        count += 1
        if delta < delta_threshold:
            break

    return V


def policy_improvement(env, V, discount):
    """
    Given estimated state-value function,
    using dynamic programming to compute an improve deterministic policy.

    Args:
        env: a MDP environment.
        V: estimated state value function.
        discount: discount factor, must be 0 <= discount <= 1.

    Returns:
        new_policy: the improved deterministic policy.
        Q: the state-action value function for the policy.
    """
    assert 0.0 <= discount <= 1.0

    # Initialize state-action value with all values set to zero.
    Q = create_initial_state_action_value(env)

    # Step 1: Compute state-action value function.
    for state in env.get_states():
        for action in env.get_legal_actions(state):  # For every legal action
            new_v = 0

            reward, successor_states = env.transition_from_state_action(state, action)
            new_v += reward

            # Note one state-action might have multiple successor states with different transition probability
            for transition_probs, state_tp1 in successor_states:
                # Weight by the transition probabilityironment
                new_v += discount * transition_probs * V[state_tp1]

            Q[state, action] = new_v

    # Step 2: Compute an improved deterministic policy.
    new_policy = create_empty_policy(env)
    for state in env.get_states():
        # Select the best action among the legal actions.
        legal_actions = env.get_legal_actions(state)
        best_action = action = utils.argmax_over_legal_actions(Q[state, :], legal_actions)
        # Set the probability to 1.0 for the best action.
        new_policy[state, best_action] = 1.0

    return new_policy, Q


def policy_iteration(env, discount, delta_threshold=1e-5):
    """
    Given a arbitrary policy and state-value function, using dynamic programming to
    find a optimal policy along with optimal state value function V*,
    and optimal state-action value function Q*.

    Args:
        env: a MDP environment.
        discount: discount factor, must be 0 <= discount <= 1.
        delta_threshold: the threshold determining the accuracy
            of the estimation for policy evaluation, default 1e-5.

    Returns:
        pi: the optimal policy.
        V: the optimal state value function.
        Q: the optimal state-action value function.
    """
    count = 0

    # Initialize a random policy.
    policy = create_random_policy(env)
    while True:
        # Reset on every iteration.
        is_policy_stable = True

        V = policy_evaluation(env, policy, discount, delta_threshold)
        new_policy, Q = policy_improvement(env, V, discount)

        count += 1

        # Check if old policy is the same as new policy
        for state in env.get_states():
            best_action = np.argmax(Q[state, :])
            if new_policy[state, best_action] != policy[state, best_action]:
                is_policy_stable = False

        # Replace old policy with new policy
        policy = new_policy

        if is_policy_stable:
            break

    return policy, V, Q


def value_iteration(env, discount, delta_threshold=1e-5):
    """
    Given a MDP environment, using dynamic programming to
    find a optimal policy along with optimal state value function V*.

    Args:
        env: a MDP environment.
        discount: discount factor, must be 0 <= discount <= 1.
        delta_threshold: the threshold determining the accuracy
            of the estimation for policy evaluation, default 1e-5.

    Returns:
        pi: the optimal policy.
        V: the optimal state value function.

    """
    assert 0.0 <= discount <= 1.0

    count = 0

    # Initialize state value function filled with zeros.
    V = create_initial_state_value(env)

    while True:
        delta = 0
        for state in env.get_states():
            old_v = V[state]
            # Store the expected returns for each actions.
            estimated_returns = []
            for action in env.get_legal_actions(state):
                value_for_action = 0

                reward, successor_states = env.transition_from_state_action(state, action)
                value_for_action += reward

                # Note one state-action might have multiple successor states with different transition probability
                for transition_probs, state_tp1 in successor_states:
                    # Weight by the transition probability
                    value_for_action += discount * transition_probs * V[state_tp1]

                estimated_returns.append(value_for_action)

            # Use the maximum expected returns across all actions as state value.
            V[state] = max(estimated_returns)
            delta = max(delta, abs(old_v - V[state]))

        count += 1
        if delta < delta_threshold:
            break

    # Step 2: Compute an optimal policy from optimal state value function.
    optimal_policy = create_empty_policy(env)
    for state in env.get_states():
        # Store the expected returns for each actions.
        estimated_returns = {}
        for action in env.get_legal_actions(state):  # For every legal action
            value_for_action = 0

            reward, successor_states = env.transition_from_state_action(state, action)
            value_for_action += reward

            # Note one state-action might have multiple successor states with different transition probability
            for transition_probs, state_tp1 in successor_states:
                # Weight by the transition probability
                value_for_action += discount * transition_probs * V[state_tp1]

            estimated_returns[action] = value_for_action

        # Get the best action a based on the q(s, a) values, notice the action is the key in the dict estimated_returns.
        best_action = max(estimated_returns, key=estimated_returns.get)

        # Set the probability to 1.0 for the best action.
        optimal_policy[state, best_action] = 1.0

    return optimal_policy, V
