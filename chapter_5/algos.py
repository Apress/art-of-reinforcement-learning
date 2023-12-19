"""Temporal Difference learning algorithms to do policy evaluation and control."""
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


def compute_optimal_policy_from_optimal_q(env, Q):
    """Compute a (optimal) deterministic policy based on the optimal state-action value function."""
    optimal_policy = np.zeros_like(Q)
    for s in range(Q.shape[0]):  # For every state
        legal_actions = env.get_legal_actions(s)
        best_action = utils.argmax_over_legal_actions(Q[s], legal_actions)
        # Set the probability to 1.0 for the best action.
        optimal_policy[s, best_action] = 1.0
    return optimal_policy


def compute_returns(rewards, discount):
    """Compute returns for every time step in the episode trajectory.

    Args:
        rewards: a list of rewards from an episode.
        discount: discount factor, must be 0 <= discount <= 1.

    Returns:
        returns: return for every single time step in the episode trajectory.
    """
    assert 0.0 <= discount <= 1.0

    returns = []
    G_t = 0
    # We do it backwards so it's more efficient and easier to implement.
    for t in reversed(range(len(rewards))):
        G_t = rewards[t] + discount * G_t
        returns.append(G_t)
    returns.reverse()

    return returns


def sample_action_from_policy(policy, state):
    """Sample an action for a specific state when following the given policy."""
    probs = policy[state, :]
    action = np.random.choice(np.arange(probs.shape[0]), replace=True, p=probs)
    return action


def create_e_greedy_policy(env):
    """Returns a e-greedy policy wrapped around the state-action value function,
    which is much simpler compared how we update the policy in Monte Carlo control.
    """

    def act(Q, state, epsilon):
        """Give a state and exploration rate epsilon, returns the action by following the e-greedy policy."""
        action = None
        legal_actions = env.get_legal_actions(state)
        if np.random.rand() < epsilon:
            action = np.random.choice(legal_actions)
        else:
            action = utils.argmax_over_legal_actions(Q[state, :], legal_actions)
        return action

    return act


def td0_policy_evaluation(env, policy, discount, learning_rate, num_updates):
    """TD(0) policy evaluation for state value function.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        policy: the policy that we want to evaluation.
        discount: discount factor, must be 0 <= discount <= 1.
        learning_rate: the learning rate when update step size
        num_updates: number of updates to the value function.

    Returns:
        V: the estimated state value function for the input policy.
    """
    assert 0.0 <= discount <= 1.0
    assert isinstance(num_updates, int)

    # Initialize state value function
    V = np.zeros(env.num_states)

    i = 0

    state = env.reset()

    while i < num_updates:
        # Sample an action for state when following the policy.
        action = sample_action_from_policy(policy, state)

        # Take the action in the environment and observe successor state and reward.
        state_tp1, reward, done = env.step(action)

        V[state] += learning_rate * (reward + discount * V[state_tp1] - V[state])

        state = state_tp1
        # For episodic environment only.
        if done:
            state = env.reset()

        i += 1

    return V


def td0_policy_evaluation_qpi(env, policy, discount, learning_rate, num_updates):
    """TD(0) policy evaluation for state-action value function.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        policy: the policy that we want to evaluation.
        discount: discount factor, must be 0 <= discount <= 1.
        learning_rate: the learning rate when update step size
        num_updates: number of updates to the value function.

    Returns:
        Q: the estimated state-action value function for the input policy.
    """
    assert 0.0 <= discount <= 1.0
    assert isinstance(num_updates, int)

    # Initialize state value function
    Q = np.zeros((env.num_states, env.num_actions))

    i = 0

    state = env.reset()
    # Sample an action for state when following the policy.
    action = sample_action_from_policy(policy, state)
    while i < num_updates:
        # Take the action in the environment and observe successor state and reward.
        state_tp1, reward, done = env.step(action)

        action_tp1 = sample_action_from_policy(policy, state_tp1)

        Q[state, action] += learning_rate * (reward + discount * Q[state_tp1, action_tp1] - Q[state, action])

        state = state_tp1
        action = action_tp1
        # For episodic environment only.
        if done:
            state = env.reset()
            # Sample an action for state when following the policy.
            action = sample_action_from_policy(policy, state)

        i += 1

    return Q


def sarsa(env, discount, begin_epsilon, end_epsilon, learning_rate, num_updates):
    """SARSA on-policy Temporal Difference control.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        discount: discount factor, must be 0 <= discount <= 1.
        begin_epsilon: initial exploration rate for the e-greedy policy, must be 0 <= begin_epsilon < 1.
        end_epsilon: the final exploration rate for the e-greedy policy, must be 0 <= end_epsilon < 1.
        learning_rate: the learning rate when update step size
        num_updates: number of updates to the value function.

    Returns:
        policy: the final policy (possible optimal policy) after run control for num_updates.
        Q: the estimated state-action value function for the output policy.
    """

    assert 0.0 <= discount <= 1.0
    assert 0.0 <= begin_epsilon <= 1.0
    assert 0.0 <= end_epsilon <= 1.0
    assert isinstance(num_updates, int)

    # Initialize state-action value function
    Q = np.zeros((env.num_states, env.num_actions))

    # Create an e-greedy policy derived from state-action value function
    e_greedy_policy = create_e_greedy_policy(env)
    t = 0

    # Create a linear decay function for the exploration rate epsilon.
    decay_fn = utils.linear_schedule(begin_epsilon, end_epsilon, 0, num_updates)
    epsilon = begin_epsilon

    state = env.reset()
    # Sample an action for state when following the e-greedy policy.
    action = e_greedy_policy(Q, state, epsilon)

    while t < num_updates:
        # Take the action in the environment and observe successor state and reward.
        state_tp1, reward, done = env.step(action)
        # Sample an action for next state but not make the move in the environment yet.
        action_tp1 = e_greedy_policy(Q, state_tp1, epsilon)
        Q[state, action] += learning_rate * (reward + discount * Q[state_tp1, action_tp1] - Q[state, action])

        state = state_tp1
        action = action_tp1

        # For episodic environment only.
        if done:
            state = env.reset()
            # Sample an action for state when following the e-greedy policy.
            action = e_greedy_policy(Q, state, epsilon)

        t += 1
        epsilon = decay_fn(t)

    # Compute a optimal policy from optimal state-action value function
    optimal_policy = compute_optimal_policy_from_optimal_q(env, Q)
    return optimal_policy, Q


def n_step_sarsa(
    env,
    discount,
    begin_epsilon,
    end_epsilon,
    learning_rate,
    n_steps,
    num_updates,
):
    """n-step SARSA on-policy Temporal Difference control.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        discount: discount factor, must be 0 <= discount <= 1.
        begin_epsilon: initial exploration rate for the e-greedy policy, must be 0 <= begin_epsilon < 1.
        end_epsilon: the final exploration rate for the e-greedy policy, must be 0 <= end_epsilon < 1.
        learning_rate: the learning rate when update step size.
        n_steps: number of steps for the n-step return.
        num_updates: number of updates to the value function.

    Returns:
        policy: the final policy (possible optimal policy) after run control for num_updates.
        Q: the estimated state-action value function for the output policy.
    """

    assert 0.0 <= discount <= 1.0
    assert 0.0 <= begin_epsilon <= 1.0
    assert 0.0 <= end_epsilon <= 1.0
    assert n_steps >= 1
    assert isinstance(num_updates, int)

    # Initialize state-action value function
    Q = np.zeros((env.num_states, env.num_actions))

    # Create an e-greedy policy derived from state-action value function
    e_greedy_policy = create_e_greedy_policy(env)
    t = 0

    # Create a linear decay function for the exploration rate epsilon.
    decay_fn = utils.linear_schedule(begin_epsilon, end_epsilon, 0, num_updates)
    epsilon = begin_epsilon

    # Store N tuples of transition
    transitions = []

    state = env.reset()
    # Sample an action for state when following the e-greedy policy.
    action = e_greedy_policy(Q, state, epsilon)

    while t < num_updates:
        # Take the action in the environment and observe successor state and reward.
        state_tp1, reward, done = env.step(action)
        # Sample an action for next state but not make the move in the environment yet.
        action_tp1 = e_greedy_policy(Q, state_tp1, epsilon)

        transitions.append((state, action, reward, state_tp1, action_tp1))

        while len(transitions) == n_steps or done and len(transitions) > 0:
            # Unpack list of tuples into separate lists.
            states, actions, rewards, states_tp1, actions_tp1 = map(list, zip(*transitions))
            n = len(rewards)

            G = 0
            for i in reversed(range(len(rewards))):
                G = rewards[i] + discount * G

            s_0 = states[0]
            a_0 = actions[0]
            s_n = states_tp1[-1]
            a_n = actions_tp1[-1]

            td_target = G + discount**n * Q[s_n, a_n]
            Q[s_0, a_0] += learning_rate * (td_target - Q[s_0, a_0])

            # Remove the first item.
            transitions.pop(0)
            t += 1
            epsilon = decay_fn(t)

        state = state_tp1
        action = action_tp1

        # For episodic environment only.
        if done:
            state = env.reset()
            # Sample an action for state when following the e-greedy policy.
            action = e_greedy_policy(Q, state, epsilon)
            transitions = []

    # Compute a optimal policy from optimal state-action value function
    optimal_policy = compute_optimal_policy_from_optimal_q(env, Q)
    return optimal_policy, Q


def q_learning(env, discount, begin_epsilon, end_epsilon, learning_rate, num_updates):
    """Q-learning off-policy algorithm.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        discount: discount factor, must be 0 <= discount <= 1.
        begin_epsilon: initial exploration rate for the e-greedy policy, must be 0 <= begin_epsilon < 1.
        end_epsilon: the final exploration rate for the e-greedy policy, must be 0 <= end_epsilon < 1.
        learning_rate: the learning rate when update step size
        num_updates: number of updates to the value function.

    Returns:
        policy: the optimal policy based on the estimated (possible optimal) after run the search for num_updates.
        Q: the estimated (possible optimal) state-action value function.
    """

    assert 0.0 <= discount <= 1.0
    assert 0.0 <= begin_epsilon <= 1.0
    assert 0.0 <= end_epsilon <= 1.0
    assert isinstance(num_updates, int)

    # Initialize state-action value function
    Q = np.zeros((env.num_states, env.num_actions))

    # Create an e-greedy policy derived from state-action value function
    e_greedy_policy = create_e_greedy_policy(env)
    t = 0

    # Create a linear decay function for the exploration rate epsilon.
    decay_fn = utils.linear_schedule(begin_epsilon, end_epsilon, 0, num_updates)
    epsilon = begin_epsilon

    state = env.reset()
    while t < num_updates:
        # Sample an action for state when following the e-greedy policy.
        action = e_greedy_policy(Q, state, epsilon)

        # Take the action in the environment and observe successor state and reward.
        state_tp1, reward, done = env.step(action)

        # Only get the maximum q value among legal actions.
        legal_actions = env.get_legal_actions(state_tp1)
        best_action_tp1 = utils.argmax_over_legal_actions(Q[state_tp1], legal_actions)

        Q[state, action] += learning_rate * (reward + discount * Q[state_tp1, best_action_tp1] - Q[state, action])

        state = state_tp1

        # For episodic environment only.
        if done:
            state = env.reset()

        t += 1
        epsilon = decay_fn(t)

    # Compute a optimal policy from optimal state-action value function
    optimal_policy = compute_optimal_policy_from_optimal_q(env, Q)
    return optimal_policy, Q


def double_q_learning(env, discount, begin_epsilon, end_epsilon, learning_rate, num_updates):
    """Double Q-learning off-policy algorithm.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        discount: discount factor, must be 0 <= discount <= 1.
        begin_epsilon: initial exploration rate for the e-greedy policy, must be 0 <= begin_epsilon < 1.
        end_epsilon: the final exploration rate for the e-greedy policy, must be 0 <= end_epsilon < 1.
        learning_rate: the learning rate when update step size
        num_updates: number of updates to the value function.

    Returns:
        policy: the optimal policy based on the estimated (possible optimal) after run the search for num_updates.
        Q: the estimated (possible optimal) state-action value function.
    """

    assert 0.0 <= discount <= 1.0
    assert 0.0 <= begin_epsilon <= 1.0
    assert 0.0 <= end_epsilon <= 1.0
    assert isinstance(num_updates, int)

    # Initialize two state-action value functions Q1, Q2
    Q1 = np.zeros((env.num_states, env.num_actions))
    Q2 = np.zeros((env.num_states, env.num_actions))

    # Create an e-greedy policy derived from state-action value function
    e_greedy_policy = create_e_greedy_policy(env)
    t = 0

    # Create a linear decay function for the exploration rate epsilon.
    decay_fn = utils.linear_schedule(begin_epsilon, end_epsilon, 0, num_updates)
    epsilon = begin_epsilon

    state = env.reset()
    while t < num_updates:
        # Merger two state-action value functions Q1, Q2
        merged_Q = np.mean(np.array([Q1, Q2]), axis=0)
        # merged_Q = np.max(Q1, axis=1, keepdims=True) + Q2

        # Sample an action for state when following the e-greedy policy.
        action = e_greedy_policy(merged_Q, state, epsilon)

        # Take the action in the environment and observe successor state and reward.
        state_tp1, reward, done = env.step(action)

        # With 0.5 probability update Q1, and 1-0.5 probability update Q2.
        # Notice how it choses the A_t+1 for S_t+1.

        legal_actions = env.get_legal_actions(state_tp1)
        if np.random.rand() < 0.5:
            # best_action_tp1 = np.argmax(Q1[state_tp1, :])

            best_action_tp1 = utils.argmax_over_legal_actions(Q1[state_tp1], legal_actions)

            Q1[state, action] += learning_rate * (reward + discount * Q2[state_tp1, best_action_tp1] - Q1[state, action])
        else:
            # best_action_tp1 = np.argmax(Q2[state_tp1, :])
            best_action_tp1 = utils.argmax_over_legal_actions(Q2[state_tp1], legal_actions)
            Q2[state, action] += learning_rate * (reward + discount * Q1[state_tp1, best_action_tp1] - Q2[state, action])

        state = state_tp1

        # For episodic environment only.
        if done:
            state = env.reset()

        t += 1
        epsilon = decay_fn(t)

    # Merger two state-action value functions Q1, Q2
    merged_Q = np.mean(np.array([Q1, Q2]), axis=0)
    # merged_Q = np.max(Q1, axis=1, keepdims=True) + Q2

    # Compute a optimal policy from optimal state-action value function
    optimal_policy = compute_optimal_policy_from_optimal_q(env, merged_Q)
    return optimal_policy, merged_Q


def n_step_q_learning(
    env,
    discount,
    begin_epsilon,
    end_epsilon,
    learning_rate,
    n_steps,
    num_updates,
):
    """n-step Q-learning off-policy algorithm.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        discount: discount factor, must be 0 <= discount <= 1.
        begin_epsilon: initial exploration rate for the e-greedy policy, must be 0 <= begin_epsilon < 1.
        end_epsilon: the final exploration rate for the e-greedy policy, must be 0 <= end_epsilon < 1.
        learning_rate: the learning rate when update step size.
        n_steps: number of steps for the n-step return.
        num_updates: number of updates to the value function.

    Returns:
        policy: the optimal policy based on the estimated (possible optimal) after run the search for num_updates.
        Q: the estimated (possible optimal) state-action value function.
    """

    assert 0.0 <= discount <= 1.0
    assert 0.0 <= begin_epsilon <= 1.0
    assert 0.0 <= end_epsilon <= 1.0
    assert n_steps >= 1
    assert isinstance(num_updates, int)

    # Initialize state-action value function
    Q = np.zeros((env.num_states, env.num_actions))

    # Create an e-greedy policy derived from state-action value function
    e_greedy_policy = create_e_greedy_policy(env)
    t = 0

    # Create a linear decay function for the exploration rate epsilon.
    decay_fn = utils.linear_schedule(begin_epsilon, end_epsilon, 0, num_updates)
    epsilon = begin_epsilon

    # Store N tuples of transition
    transitions = []

    state = env.reset()
    while t < num_updates:
        # Sample an action for state when following the e-greedy policy.
        action = e_greedy_policy(Q, state, epsilon)

        # Take the action in the environment and observe successor state and reward.
        state_tp1, reward, done = env.step(action)

        transitions.append((state, action, reward, state_tp1))

        while len(transitions) == n_steps or done and len(transitions) > 0:
            # Unpack list of tuples into separate lists.
            states, actions, rewards, states_tp1 = map(list, zip(*transitions))
            n = len(rewards)

            G = 0
            for i in reversed(range(len(rewards))):
                G = rewards[i] + discount * G

            s_0 = states[0]
            a_0 = actions[0]
            s_n = states_tp1[-1]

            # Only get the maximum q value among legal actions.
            legal_actions = env.get_legal_actions(s_n)
            best_action_tp1 = utils.argmax_over_legal_actions(Q[s_n], legal_actions)

            td_target = G + discount**n * Q[s_n, best_action_tp1]
            Q[s_0, a_0] += learning_rate * (td_target - Q[s_0, a_0])

            # Remove the first item.
            transitions.pop(0)
            t += 1
            epsilon = decay_fn(t)

        state = state_tp1

        # For episodic environment only.
        if done:
            state = env.reset()
            # Sample an action for state when following the e-greedy policy.
            action = e_greedy_policy(Q, state, epsilon)
            transitions = []

    # Compute a optimal policy from optimal state-action value function
    optimal_policy = compute_optimal_policy_from_optimal_q(env, Q)
    return optimal_policy, Q
