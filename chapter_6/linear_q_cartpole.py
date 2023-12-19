"""Example code of using Q-learning with Linear Value Function Approximation to solve Cart Pole problem."""
from typing import NamedTuple, Optional
import collections

from absl import app
from absl import flags
import logging

import gym
import numpy as np

import tiles3 as tc
import utils
import trackers as trackers_lib
import csv_writer


FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.0025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('exploration_epsilon_begin_value', 1.0, 'Begin value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_end_value', 0.05, 'End (decayed) value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_decay_step', 100000, 'Total steps to decay value of the exploration rate.')
flags.DEFINE_integer('num_iterations', 50, 'Number iterations to run.')
flags.DEFINE_integer('num_train_steps', int(2e4), 'Number training environment steps to run per iteration.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_string('results_csv_path', '', 'Path for CSV log file.')


class Transition(NamedTuple):
    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    r_t: Optional[float]
    s_tp1: Optional[np.ndarray]
    done_tp1: Optional[bool]


def create_tile_coding_instance(state_dim, num_tilings, num_tiles):
    """Setup a tile coding runtime environment, returns the shape of weights w, and a function to extract features.
    Notice only works for CartPole.
    """

    # limits of the state space
    cart_pos_min = -4.8
    cart_pos_max = 4.8
    cart_vel_min = -10
    cart_vel_max = 10
    pole_ang_min = -0.25
    pole_ang_max = 0.25
    pole_angvel_min = -10
    pole_angvel_max = 10

    cart_pos_scale = num_tiles / (cart_pos_max - cart_pos_min)
    cart_vel_scale = num_tiles / (cart_vel_max - cart_vel_min)
    pole_ang_scale = num_tiles / (pole_ang_max - pole_ang_min)
    pole_angvel_scale = num_tiles / (pole_angvel_max - pole_angvel_min)

    weights_shape = num_tiles * num_tiles * num_tilings * state_dim
    tc_hash_table = tc.IHT(weights_shape)

    def extract_features(state, action):
        """Create feature vector x using tile coding."""
        cart_position, cart_velocity, pole_ang, pole_angvel = state
        active_tile_index = tc.tiles(
            tc_hash_table,
            num_tilings,
            [
                cart_pos_scale * cart_position,
                cart_vel_scale * cart_velocity,
                pole_ang_scale * pole_ang,
                pole_angvel_scale * pole_angvel,
            ],
            [action],
        )

        x = np.zeros(weights_shape)
        x[active_tile_index] = 1.0

        return x

    return weights_shape, extract_features


def apply_e_greedy_policy(q_values, epsilon, random_state):
    """Apply epsilon-greedy policy on Q values."""
    assert len(q_values.shape) == 1, 'Assume q_values is a vector.'
    num_actions = q_values.shape[0]
    if random_state.rand() < epsilon:
        a_t = random_state.randint(0, num_actions)
    else:
        a_t = np.argmax(q_values)
    return a_t


class LinearQAgent:
    """Q-learning agent with linear value function approximation."""

    def __init__(
        self,
        state_dim,
        num_tilings,
        num_tiles,
        num_actions,
        learning_rate,
        random_state,
        discount,
        exploration_epsilon_schedule,
    ):
        (
            self.weights_shape,
            self.extract_features,
        ) = create_tile_coding_instance(state_dim, num_tilings, num_tiles)
        self.w = np.zeros(self.weights_shape)

        self.num_actions = num_actions
        self.learning_rate = learning_rate

        self.random_state = random_state
        self.discount = discount
        self.exploration_epsilon_schedule = exploration_epsilon_schedule

        # Counters and statistics
        self.step_t = -1

    def q_function(self, state, action):
        x = self.extract_features(state, action)
        return np.inner(x, self.w)

    def act(self, observation):
        """Given an environment observation, returns an action
        according to the epsilon-greedy policy."""
        self.step_t += 1
        a_t = self.choose_action(observation)
        return a_t

    def update(self, transition):
        """Give an transition (s, a, r, s'), update the weights."""

        s_t = transition.s_t
        a_t = transition.a_t
        r_t = transition.r_t
        s_tp1 = transition.s_tp1
        done_tp1 = transition.done_tp1

        discount_tp1 = 0.0 if done_tp1 else self.discount

        q_tp1 = []
        for action in range(self.num_actions):
            q_tp1.append(self.q_function(s_tp1, action))
        max_q_tp1 = np.max(q_tp1)

        x = self.extract_features(s_t, a_t)
        self.w += self.learning_rate * (r_t + discount_tp1 * max_q_tp1 - self.q_function(s_t, a_t)) * x

    def choose_action(self, observation):
        """Given an environment observation and exploration rate,
        returns an action according to the epsilon-greedy policy."""
        q_t = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            q_sa = self.q_function(observation, action)
            q_t[action] = q_sa
        return apply_e_greedy_policy(q_t, self.exploration_epsilon, self.random_state)

    @property
    def exploration_epsilon(self):
        """Call external schedule function"""
        return self.exploration_epsilon_schedule(self.step_t)


def run_train_steps(env, agent, num_train_steps):
    """Run training for num_train_steps."""
    trackers = trackers_lib.make_default_trackers()
    s_t = env.reset()
    for _ in range(num_train_steps):
        a_t = agent.act(s_t)

        # Take the action in the environment and observe successor state and reward.
        s_tp1, r_t, done, _ = env.step(a_t)

        for tracker in trackers:
            tracker.step(r_t, done)

        agent.update(Transition(s_t=s_t, a_t=a_t, r_t=r_t, s_tp1=s_tp1, done_tp1=done))

        s_t = s_tp1
        if done:
            s_t = env.reset()

    return trackers_lib.generate_statistics(trackers)


def main(argv):
    """Trains tabular Q-learning agent with linear VFA."""
    del argv

    random_state = np.random.RandomState(FLAGS.seed)

    writer = None
    if FLAGS.results_csv_path:
        writer = csv_writer.CsvWriter(FLAGS.results_csv_path)

    def environment_builder():
        env = gym.make('CartPole-v1')
        env.seed(random_state.randint(1, 2**32))
        return env

    # Create training and evaluation environments
    train_env = environment_builder()

    num_actions = train_env.action_space.n
    state_dim = train_env.observation_space.shape

    logging.info('Environment: %s', train_env.spec.id)
    logging.info('Action space: %s', num_actions)
    logging.info('Observation space: %s', state_dim)

    # Create training and evaluation agent instances
    train_agent = LinearQAgent(
        num_actions=num_actions,
        state_dim=state_dim[0],
        num_tilings=8,
        num_tiles=16,
        learning_rate=FLAGS.learning_rate,
        random_state=random_state,
        discount=FLAGS.discount,
        exploration_epsilon_schedule=utils.linear_schedule(
            begin_t=0,
            decay_steps=FLAGS.exploration_epsilon_decay_step,
            begin_value=FLAGS.exploration_epsilon_begin_value,
            end_value=FLAGS.exploration_epsilon_end_value,
        ),
    )

    # Start to run training and evaluation iterations
    for iteration in range(1, FLAGS.num_iterations + 1):
        # Run trains steps
        train_stats = run_train_steps(train_env, train_agent, FLAGS.num_train_steps)

        # Logging
        log_output = [
            ('iteration', iteration, '%3d'),
            ('step', iteration * FLAGS.num_train_steps, '%5d'),
            ('train_step_rate', train_stats['step_rate'], '%4.0f'),
            ('train_episode_return', train_stats['mean_episode_return'], '% 2.2f'),
            ('train_num_episodes', train_stats['num_episodes'], '%3d'),
            (
                'train_exploration_epsilon',
                train_agent.exploration_epsilon,
                '%.3f',
            ),
        ]
        log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
        logging.info(log_output_str)

        if writer:
            writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))


if __name__ == '__main__':
    app.run(main)
