"""
Sample code for using double Q-learning to solve Casino MDP example.
"""

from envs.casino_mdp import CasinoMDPEnv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_initial_state_action_value_function(env):
    """Return initial state-action value function, notice using numpy.array seems to fail in this case since lots of actions are illegal in state A, B."""
    # Q = np.zeros((env.num_states, env.num_actions))
    Q = {}
    for s in env.get_states():
        Q[s] = {}
        for a in env.get_legal_actions(s):
            Q[s][a] = 0

    return Q


def e_greedy_policy(Q, state, epsilon):
    """Epsilon greedy policy according to state-action value function."""
    legal_actions = list(Q[state].keys())
    if np.random.rand() < epsilon:
        return np.random.choice(legal_actions)
    # Notice the action is the key in the Q[state] dict.
    return max(Q[state], key=Q[state].get)


def q_learning(env, discount, epsilon, learning_rate, num_runs, num_episodes):
    """Q-learning off-policy algorithm.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        discount: discount factor, must be 0 <= discount <= 1.
        epsilon: epsilon for the e-greedy policy, must be 0 <= epsilon < 1.
        learning_rate: the learning rate when update step size
        num_runs: number of trail runs.
        num_episodes: number of episodes to run, per trail run.

    Returns:
        num_go_lefts: a 1D numpy.ndarray record how many times the go left action have been selected in each episode. Notice, the go left action could only be selected once for each single episode.
    """

    assert 0.0 <= discount <= 1.0
    assert 0.0 <= epsilon <= 1.0
    assert isinstance(num_runs, int)
    assert isinstance(num_episodes, int)

    num_go_rights = np.zeros((num_runs, num_episodes))

    for i in range(num_runs):
        # Initialize state-action value function
        Q = create_initial_state_action_value_function(env)

        for j in range(num_episodes):
            state = env.reset()

            while True:
                # Sample an action for state when following the e-greedy policy.
                action = e_greedy_policy(Q, state, epsilon)

                if env.is_start_state(state):
                    if env.is_go_right(action):
                        num_go_rights[i, j] = 1

                # Take the action in the environment and observe successor state and reward.
                state_tp1, reward, done = env.step(action)

                Q[state][action] += learning_rate * (
                    reward + discount * np.max(list(Q[state_tp1].values())) - Q[state][action]
                )

                state = state_tp1
                if done:
                    break

    return num_go_rights


def double_q_learning(env, discount, epsilon, learning_rate, num_runs, num_episodes):
    """Double Q-learning off-policy algorithm.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        discount: discount factor, must be 0 <= discount <= 1.
        epsilon: epsilon for the e-greedy policy, must be 0 <= epsilon < 1.
        learning_rate: the learning rate when update step size
        num_runs: number of trail runs.
        num_episodes: number of episodes to run, per trail run.

    Returns:
        policy: the optimal policy based on the estimated (possible optimal) after run the search for num_updates.
        Q: the estimated (possible optimal) state-action value function.
    """

    assert 0.0 <= discount <= 1.0
    assert 0.0 <= epsilon <= 1.0
    assert isinstance(num_runs, int)
    assert isinstance(num_episodes, int)

    num_go_rights = np.zeros((num_runs, num_episodes))

    for i in range(num_runs):
        # Initialize two state-action value functions Q1, Q2
        Q1 = create_initial_state_action_value_function(env)
        Q2 = create_initial_state_action_value_function(env)

        for j in range(num_episodes):
            state = env.reset()
            while True:
                # Sample an action for state when following the e-greedy policy using Q1.
                action = e_greedy_policy(Q1, state, epsilon)

                if env.is_start_state(state):
                    if env.is_go_right(action):
                        num_go_rights[i, j] = 1

                # Take the action in the environment and observe successor state and reward.
                state_tp1, reward, done = env.step(action)

                # With 0.5 probability update Q1, and 1-0.5 probability update Q2.
                # Notice how it choses the A_t+1 for S_t+1.
                if np.random.rand() < 0.5:
                    best_action_tp1 = max(Q1[state_tp1], key=Q1[state_tp1].get)
                    Q1[state][action] += learning_rate * (
                        reward + discount * Q2[state_tp1][best_action_tp1] - Q1[state][action]
                    )
                else:
                    best_action_tp1 = max(Q2[state_tp1], key=Q2[state_tp1].get)
                    Q2[state][action] += learning_rate * (
                        reward + discount * Q1[state_tp1][best_action_tp1] - Q2[state][action]
                    )

                state = state_tp1
                if done:
                    break

    return num_go_rights


if __name__ == '__main__':
    print('Running Q-learning vs. Double Q-learning on Casino MDP, this may take few minutes...')

    # fig5_8
    discount = 1.0
    epsilon = 0.1
    learning_rate = 0.02
    num_runs = 1000
    num_episodes = 500

    q_go_right = q_learning(CasinoMDPEnv(), discount, epsilon, learning_rate, num_runs, num_episodes)

    double_q_go_right = double_q_learning(CasinoMDPEnv(), discount, epsilon, learning_rate, num_runs, num_episodes)

    q_go_right = q_go_right.mean(axis=0)
    double_q_go_right = double_q_go_right.mean(axis=0)

    episodes = np.arange(num_episodes)

    df = pd.DataFrame(
        data=np.column_stack((episodes, q_go_right, double_q_go_right)),
        columns=['episode', 'q_go_right', 'double_q_go_right'],
    )

    plt.figure(figsize=(12, 8))

    plt.plot(
        'episode',
        'q_go_right',
        data=df,
        color='red',
        linewidth=1,
        label='Q-learning',
    )
    plt.plot(
        'episode',
        'double_q_go_right',
        data=df,
        color='green',
        linewidth=1,
        label='Double Q-learning',
    )
    plt.plot(
        episodes,
        np.ones(num_episodes) * 0.05,
        color='black',
        linewidth=1,
        linestyle='--',
        label='Optimal',
    )
    plt.xlabel('Episodes', fontsize=16, labelpad=20)
    plt.ylabel('% of go right \n in center hall', fontsize=16)
    plt.legend(fontsize=14)

    plt.tight_layout()

    plt.show()
