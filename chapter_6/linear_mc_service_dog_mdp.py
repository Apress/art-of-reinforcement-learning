from absl import app
from absl import flags
import logging

import numpy as np
from envs.dog_mdp import DogMDPEnv
import utils

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_float('discount', 0.9, 'Discount rate.')

flags.DEFINE_float('exploration_epsilon_begin_value', 1.0, 'Begin value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_end_value', 0.05, 'End (decayed) value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_decay_step', 5000, 'Total steps to decay value of the exploration rate.')
flags.DEFINE_multi_integer(
    'list_train_episodes',
    [100, 1000, 10000],
    'A list of number of training episodes to run for different experiments.',
)
flags.DEFINE_integer('num_runs', 10, 'Number of independent runs for the experiment.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')


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


def one_hot_vector(index, num_items):
    assert 0 <= index < num_items

    onehot = np.zeros(num_items, dtype=np.float64)
    onehot[index] = 1.0
    return onehot


def apply_e_greedy_policy(q_values, epsilon, legal_actions, random_state):
    """Apply epsilon-greedy policy on Q values."""
    assert len(q_values.shape) == 1, 'Assume q_values is a vector.'
    if random_state.rand() < epsilon:
        a_t = random_state.choice(legal_actions)
    else:
        a_t = utils.argmax_over_legal_actions(q_values, legal_actions)
    return a_t


class LinearMCAgent:
    """Monte Carlo agent with linear value function approximation."""

    def __init__(
        self,
        num_states,
        num_actions,
        discount,
        learning_rate,
        random_state,
        exploration_epsilon_schedule,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.learning_rate = learning_rate

        self.w = np.zeros((self.num_states + self.num_actions))

        self.random_state = random_state

        self.exploration_epsilon_schedule = exploration_epsilon_schedule

        # Counters and statistics
        self.step_t = -1

    def q_function(self, state, action):
        x = self.extract_features(state, action)
        return np.inner(x, self.w)

    def act(self, observation, legal_actions):
        """Given an environment observation, returns an action
        according to the epsilon-greedy policy."""
        self.step_t += 1
        a_t = self.choose_action(observation, self.exploration_epsilon, legal_actions)
        return a_t

    def update(self, state_actions, rewards, first_visit=False):
        """Give an episode sequence in the format of lists of tuples (s, a) and (r), go through the sequence and update the weights."""

        returns = compute_returns(rewards, self.discount)
        # Loop over all state-action pairs in the episode.
        for t, state_action_pair in enumerate(state_actions):
            G_t = returns[t]
            # Check if this is the first time state visited in the episode.
            if first_visit and state_action_pair in state_actions[:t]:
                continue

            state, action = state_action_pair

            x = self.extract_features(state, action)

            self.w += self.learning_rate * (G_t - self.q_function(state, action)) * x

    def extract_features(self, state, action):
        onehot_s = one_hot_vector(state, self.num_states)
        onehot_a = one_hot_vector(action, self.num_actions)
        x = np.concatenate((onehot_s, onehot_a), axis=0)
        return x

    def choose_action(self, observation, epsilon, legal_actions):
        """Given an environment observation and exploration rate,
        returns an action according to the epsilon-greedy policy."""
        q_t = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            q_sa = self.q_function(observation, action)
            q_t[action] = q_sa
        return apply_e_greedy_policy(q_t, epsilon, legal_actions, self.random_state)

    @property
    def exploration_epsilon(self):
        """Call external schedule function"""
        return self.exploration_epsilon_schedule(self.step_t)


def get_state_values(q_func, env):
    Q = np.zeros((env.num_states, env.num_actions))
    V = np.zeros(env.num_states)
    for s in env.get_states():
        for a in env.get_actions():
            Q[s, a] = q_func(s, a)

    for s in range(V.shape[0]):  # For every state
        legal_actions = env.get_legal_actions(s)
        a_star = utils.argmax_over_legal_actions(Q[s], legal_actions)
        V[s] = Q[s, a_star]

    return V


def run_train_episodes(env, agent, num_train_episodes):
    """Run training for num_train_episodes."""

    for _ in range(num_train_episodes):
        episode_sequence = []
        s_t = env.reset()
        while True:
            a_t = agent.act(s_t, env.get_legal_actions(s_t))

            # Take the action in the environment and observe successor state and reward.
            s_tp1, r_t, done = env.step(a_t)
            episode_sequence.append(((s_t, a_t), r_t))

            s_t = s_tp1
            if done:
                legal_actions = env.get_legal_actions(s_t)
                a_t = np.random.choice(legal_actions)
                episode_sequence.append(((s_t, a_t), 0))

                # Unpack list of tuples into separate lists.
                state_actions, rewards = map(list, zip(*episode_sequence))
                agent.update(state_actions, rewards)
                break

    return get_state_values(agent.q_function, env)


def main(argv):
    """Trains Monte Carlo agent with linear VFA on service dog MDP environment."""
    del argv

    random_state = np.random.RandomState(FLAGS.seed)

    def environment_builder():
        return DogMDPEnv()

    # Create training and evaluation environments
    train_env = environment_builder()

    num_actions = train_env.num_actions
    num_states = train_env.num_states

    logging.info('Environment: %s', 'Service Dog MDP')
    logging.info('Action space: %s', num_actions)
    logging.info('Observation space: %s', num_states)

    # Create training and evaluation agent instances
    train_agent = LinearMCAgent(
        num_actions=num_actions,
        num_states=num_states,
        discount=FLAGS.discount,
        learning_rate=FLAGS.learning_rate,
        random_state=random_state,
        exploration_epsilon_schedule=utils.linear_schedule(
            begin_t=0,
            decay_steps=FLAGS.exploration_epsilon_decay_step,
            begin_value=FLAGS.exploration_epsilon_begin_value,
            end_value=FLAGS.exploration_epsilon_end_value,
        ),
    )

    for num_episodes in FLAGS.list_train_episodes:
        state_values = []
        for run in range(FLAGS.num_runs):
            v = run_train_episodes(train_env, train_agent, num_episodes)
            state_values.append(v)
        avg_state_values = np.array(state_values).mean(axis=0)
        logging.info(f'Number of train episodes: {num_episodes}')
        utils.print_state_value(train_env, avg_state_values)


if __name__ == '__main__':
    app.run(main)
