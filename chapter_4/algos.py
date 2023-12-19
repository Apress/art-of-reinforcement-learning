"""Monte Carlo methods and Temporal Difference learning algorithms to do policy evaluation and control."""
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


def compute_optimal_policy_from_optimal_q(Q):
    """Compute a (optimal) deterministic policy based on the optimal state-action value function."""
    optimal_policy = np.zeros_like(Q)
    for s in range(Q.shape[0]):  # For every state
        best_action = np.argmax(Q[s, :])
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


def mc_policy_evaluation(env, policy, discount, num_episodes, first_visit=True):
    """Run Monte Carlo policy evaluation for state value function.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        policy: the policy that we want to evaluation.
        discount: discount factor, must be 0 <= discount <= 1.
        num_episodes: number of episodes to run.
        first_visit: use first-visit MC, default on.

    Returns:
        V: the estimated state value function for the input policy after run evaluation for num_episodes.
    """
    assert 0.0 <= discount <= 1.0
    assert isinstance(num_episodes, int)

    # Initialize
    N = np.zeros(env.num_states)  # counter for visits number
    V = np.zeros(env.num_states)  # state value function
    G = np.zeros(env.num_states)  # total returns

    for _ in range(num_episodes):
        # Sample an episode trajectory using the given policy.
        episode = []
        state = env.reset()
        while True:
            # Sample an action for state when following the policy.
            action = sample_action_from_policy(policy, state)
            # Take the action in the environment and observe successor state and reward.
            state_tp1, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = state_tp1
            if done:
                break

        # Unpack list of tuples into separate lists.
        states, _, rewards = map(list, zip(*episode))
        # Compute returns for every time step in the episode.
        returns = compute_returns(rewards, discount)

        # Loop over all state in the episode.
        for t, state in enumerate(states):
            G_t = returns[t]
            # Check if this is the first time state visited in the episode.
            if first_visit and state in states[:t]:
                continue

            N[state] += 1
            G[state] += G_t
            V[state] = G[state] / N[state]

    return V


def incremental_mc_policy_evaluation(env, policy, discount, num_episodes, first_visit=False):
    """Run incremental Monte Carlo policy evaluation for state value function.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        policy: the policy that we want to evaluation.
        discount: discount factor, must be 0 <= discount <= 1.
        num_episodes: number of episodes to run.
        first_visit: use first-visit MC, default off.

    Returns:
        V: the estimated state value function for the input policy after run evaluation for num_episodes.
    """
    assert 0.0 <= discount <= 1.0
    assert isinstance(num_episodes, int)

    # Initialize
    N = np.zeros(env.num_states)  # counter for visits number
    V = np.zeros(env.num_states)  # state value function

    for _ in range(num_episodes):
        # Sample an episode trajectory using the given policy.
        episode = []
        state = env.reset()
        while True:
            # Sample an action for state when following the policy.
            action = sample_action_from_policy(policy, state)
            # Take the action in the environment and observe successor state and reward.
            state_tp1, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = state_tp1
            if done:
                break

        # Unpack list of tuples into separate lists.
        states, _, rewards = map(list, zip(*episode))
        # Compute returns for every time step in the episode.
        returns = compute_returns(rewards, discount)

        # Loop over all state in the episode.
        for t, state in enumerate(states):
            G_t = returns[t]
            # Check if this is the first time state visited in the episode.
            if first_visit and state in states[:t]:
                continue

            N[state] += 1
            V[state] += (G_t - V[state]) / N[state]

    return V


def incremental_mc_policy_evaluation_qpi(env, policy, discount, num_episodes, first_visit=False):
    """Run incremental Monte Carlo policy evaluation for state-action value function.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        policy: the policy that we want to evaluation.
        discount: discount factor, must be 0 <= discount <= 1.
        num_episodes: number of episodes to run.
        first_visit: use first-visit MC, default off.

    Returns:
        Q: the estimated state-action value function for the input policy after run evaluation for num_episodes.
    """
    assert 0.0 <= discount <= 1.0
    assert isinstance(num_episodes, int)

    # Initialize
    N = np.zeros((env.num_states, env.num_actions))  # counter for visits number
    Q = np.zeros((env.num_states, env.num_actions))  # state-action value function

    for _ in range(num_episodes):
        # Sample an episode trajectory using the given policy.
        episode = []
        state = env.reset()
        while True:
            # Sample an action for state when following the policy.
            action = sample_action_from_policy(policy, state)
            # Take the action in the environment and observe successor state and reward.
            state_tp1, reward, done = env.step(action)
            episode.append(((state, action), reward))
            state = state_tp1
            if done:
                break

        # Unpack list of tuples into separate lists.
        state_actions, rewards = map(list, zip(*episode))
        # Compute returns for every time step in the episode.
        returns = compute_returns(rewards, discount)

        # Loop over all state-action pairs in the episode.
        for t, state_action_pair in enumerate(state_actions):
            G_t = returns[t]
            # Check if this is the first time state visited in the episode.
            if first_visit and state_action_pair in state_actions[:t]:
                continue

            state, action = state_action_pair
            N[state, action] += 1
            Q[state, action] += (G_t - Q[state, action]) / N[state, action]

    return Q


def mc_policy_control(env, discount, epsilon, num_episodes, first_visit=True):
    """Run Monte Carlo policy improvement (control) with e-greedy policy using incremental method.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        discount: discount factor, must be 0 <= discount <= 1.
        epsilon: initial exploration rate for the e-greedy policy, must be 0 <= epsilon <= 1.
        num_episodes: number of episodes to run.
        first_visit: use first-visit MC, default on.

    Returns:
        policy: the final policy (possible optimal policy) after run control for num_episodes.
        Q: the estimated state-action value function for the output policy.
    """
    assert 0.0 <= discount <= 1.0
    assert 0.0 <= epsilon <= 1.0
    assert isinstance(num_episodes, int)

    # Initialize
    N = np.zeros((env.num_states, env.num_actions))  # counter for visits number
    Q = np.zeros((env.num_states, env.num_actions))  # state-action value function

    policy = create_random_policy(env)
    i = 0

    while i < num_episodes:
        # Sample an episode trajectory using the given policy.
        episode = []
        state = env.reset()
        while True:
            # Sample an action for state when following the policy.
            action = sample_action_from_policy(policy, state)

            # Take the action in the environment and observe successor state and reward.
            state_tp1, reward, done = env.step(action)

            episode.append(((state, action), reward))
            state = state_tp1
            if done:
                break

        # Unpack list of tuples into separate lists.
        state_actions, rewards = map(list, zip(*episode))

        # Policy evaluation.
        # Compute returns for every time step in the episode.
        returns = compute_returns(rewards, discount)
        # Loop over all state-action pairs in the episode.
        for t, state_action_pair in enumerate(state_actions):
            G_t = returns[t]
            # Check if this is the first time state visited in the episode.
            if first_visit and state_action_pair in state_actions[:t]:
                continue

            state, action = state_action_pair
            N[state, action] += 1
            Q[state, action] += (G_t - Q[state, action]) / N[state, action]

        # Policy improvement.
        for state_action_pair in state_actions:
            state, action = state_action_pair
            legal_actions = env.get_legal_actions(state)
            num_legal_actions = len(legal_actions)

            best_action = utils.argmax_over_legal_actions(Q[state], legal_actions)
            for action in legal_actions:
                if action == best_action:
                    policy[state, action] = 1 - epsilon + epsilon / num_legal_actions
                else:
                    policy[state, action] = epsilon / num_legal_actions

        i += 1
        epsilon = 1 / i

    return policy, Q
