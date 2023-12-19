import collections
import os
from absl import app
from absl import flags
import logging

import torch
from torch import nn
import gym
import numpy as np

import utils
import trackers as trackers_lib
import csv_writer
import replay as replay_lib

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Classic control task name, like CartPole-v1, MountainCar-v0.',
)

flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')

flags.DEFINE_float(
    'exploration_epsilon_begin_value',
    1.0,
    'Begin value of the exploration rate in e-greedy policy.',
)
flags.DEFINE_float(
    'exploration_epsilon_end_value',
    0.01,
    'End (decayed) value of the exploration rate in e-greedy policy.',
)
flags.DEFINE_float(
    'exploration_epsilon_decay_step',
    100000,
    'Total steps to decay value of the exploration rate.',
)
flags.DEFINE_float(
    'eval_exploration_epsilon_rate',
    0.05,
    'The exploration rate in e-greedy policy for evaluation actor only.',
)

flags.DEFINE_integer('num_iterations', 50, 'Number iterations to run.')
flags.DEFINE_integer(
    'num_train_steps',
    int(2e4),
    'Number training environment steps to run per iteration.',
)
flags.DEFINE_integer(
    'num_eval_steps',
    int(2e4),
    'Number evaluation environment steps to run per iteration.',
)
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'Directory to store checkpoint file, default empty means do not create checkpoint.',
)
flags.DEFINE_string('results_csv_path', '', 'Path for CSV log file.')


class DqnMlpNet(nn.Module):
    """MLP DQN network."""

    def __init__(self, state_dim: int, action_dim: int):
        """
        Args:
            state_dim: the dimension of the input vector to the neural network
            action_dim: the number of units for the output liner layer
        """
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given state, return state-action value for all possible actions"""

        q_values = self.body(x)  # [batch_size, action_dim]
        return q_values


def apply_e_greedy_policy(q_values, epsilon, random_state):
    """Apply epsilon-greedy policy on Q values."""
    assert len(q_values.shape) == 2, 'Assume q_values is 2D.'
    action_dim = q_values.shape[1]
    if random_state.rand() < epsilon:
        a_t = random_state.randint(0, action_dim)
    else:
        a_t = torch.argmax(q_values, dim=1).cpu().item()
    return a_t


class NaiveDeepQAgent:
    """Naive deep Q learning agent for interacting with the environment and do learning."""

    def __init__(
        self,
        network,
        optimizer,
        random_state,
        exploration_epsilon_schedule,
        discount,
        device,
    ):
        self.device = device
        self.random_state = random_state

        self.network = network.to(device=self.device)
        self.optimizer = optimizer
        self.discount = discount
        self.exploration_epsilon_schedule = exploration_epsilon_schedule

        # Counters and statistics
        self.step_t = -1
        self.update_t = 0

    def act(self, observation):
        """Given an environment observation, returns an action.
        Also (conditionally) update network parameters."""
        self.step_t += 1
        a_t = self.choose_action(observation, self.exploration_epsilon)

        return a_t

    def update(self, transitions):
        self.optimizer.zero_grad()
        loss = self.calc_loss(transitions)

        # Compute gradients
        loss.backward()

        # Update parameters
        self.optimizer.step()

    @torch.no_grad()
    def choose_action(self, observation, epsilon):
        """Given an environment observation and exploration rate,
        returns an action according to the epsilon-greedy policy."""
        s_t = torch.from_numpy(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        q_t = self.network(s_t)
        return apply_e_greedy_policy(q_t, epsilon, self.random_state)

    def calc_loss(self, transitions):
        """Calculate loss for a given batch of transitions."""
        s_t = torch.from_numpy(transitions.s_t[None, ...]).to(device=self.device, dtype=torch.float32)  # [1, state_shape]
        a_t = torch.tensor(transitions.a_t).to(device=self.device, dtype=torch.int64)  # [1]
        r_t = torch.tensor(transitions.r_t).to(device=self.device, dtype=torch.float32)  # [1]
        s_tp1 = torch.from_numpy(transitions.s_tp1[None, ...]).to(device=self.device, dtype=torch.float32)  # [1, state_shape]
        done_tp1 = torch.tensor(transitions.done_tp1).to(device=self.device, dtype=torch.bool)  # [1]

        discount_tp1 = (~done_tp1).float() * self.discount

        # Compute predicted q values for s_t
        q_t = self.network(s_t)  # [1, action_dim]

        # Compute predicted q values for s_tp1 using target network, then compute the TD target
        with torch.no_grad():
            q_tp1 = self.network(s_tp1)  # [1, action_dim]
            target_t = r_t + discount_tp1 * torch.max(q_tp1, dim=1)[0]

        # Compute loss which is 0.5 * square(td_errors)
        qa_t = q_t[torch.arange(0, s_t.shape[0]), a_t]
        td_error = target_t - qa_t

        loss = 0.5 * td_error**2

        # Averaging over batch dimension
        loss = torch.mean(loss, dim=0)
        return loss

    @property
    def exploration_epsilon(self):
        """Call external schedule function"""
        return self.exploration_epsilon_schedule(self.step_t)


class EpsilonGreedyActor:
    """Epsilon-greedy actor for evaluation only."""

    def __init__(self, network, exploration_epsilon, random_state, device):
        self.device = device
        self.network = network.to(device=device)
        self.exploration_epsilon = exploration_epsilon
        self.random_state = random_state

    def act(self, observation):
        """Given an environment observation, returns an action."""
        return self.choose_action(observation)

    @torch.no_grad()
    def choose_action(self, observation):
        """Given an environment observation and exploration rate,
        returns an action according to the epsilon-greedy policy."""
        s_t = torch.tensor(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        q_t = self.network(s_t)
        return apply_e_greedy_policy(q_t, self.exploration_epsilon, self.random_state)


def run_train_loop(env, agent, num_train_steps):
    """Run training for some steps."""
    trackers = trackers_lib.make_default_trackers()

    s_t = env.reset()
    for _ in range(num_train_steps):
        a_t = agent.act(s_t)

        # Take the action in the environment and observe successor state and reward.
        s_tp1, r_t, done, info = env.step(a_t)

        for tracker in trackers:
            tracker.step(r_t, done)

        transition = replay_lib.Transition(s_t=s_t, a_t=a_t, r_t=r_t, s_tp1=s_tp1, done_tp1=done)
        agent.update(transition)

        s_t = s_tp1
        if done:
            s_t = env.reset()

    return trackers_lib.generate_statistics(trackers)


def run_evaluation_loop(env, agent, num_eval_steps):
    """Run evaluation for some steps."""
    trackers = trackers_lib.make_default_trackers()

    s_t = env.reset()
    for _ in range(num_eval_steps):
        a_t = agent.act(s_t)
        s_tp1, r_t, done, _ = env.step(a_t)

        for tracker in trackers:
            tracker.step(r_t, done)

        s_t = s_tp1

        if done:
            s_t = env.reset()

    return trackers_lib.generate_statistics(trackers)


def main(argv):
    """Trains naive deep Q learning agent.

    For every iteration, the code does these in sequence:
        1. Run train agent for num_train_steps and periodically update network parameters
        2. Run evaluation agent for num_eval_steps on a separate evaluation environment
        3. Logging statistics to a csv file
        4. Create checkpoint file

    """
    del argv

    random_state = np.random.RandomState(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Runs naive deep Q learning on {device}')

    writer = None
    if FLAGS.results_csv_path:
        writer = csv_writer.CsvWriter(FLAGS.results_csv_path)

    def environment_builder():
        env = gym.make(FLAGS.environment_name)
        env.seed(random_state.randint(1, 2**32))
        return env

    # Create training and evaluation environments
    train_env = environment_builder()
    eval_env = environment_builder()

    action_dim = train_env.action_space.n
    state_dim = train_env.observation_space.shape

    logging.info('Environment: %s', train_env.spec.id)
    logging.info('Action space: %s', action_dim)
    logging.info('Observation space: %s', state_dim)

    # Initialize network and optimizer
    network = DqnMlpNet(state_dim=state_dim[0], action_dim=action_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate)

    # Create training and evaluation agent instances
    train_agent = NaiveDeepQAgent(
        network=network,
        optimizer=optimizer,
        random_state=random_state,
        exploration_epsilon_schedule=utils.linear_schedule(
            begin_t=0,
            decay_steps=FLAGS.exploration_epsilon_decay_step,
            begin_value=FLAGS.exploration_epsilon_begin_value,
            end_value=FLAGS.exploration_epsilon_end_value,
        ),
        discount=FLAGS.discount,
        device=device,
    )

    eval_agent = EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon_rate,
        random_state=random_state,
        device=device,
    )

    # Start to run training and evaluation iterations
    for iteration in range(1, FLAGS.num_iterations + 1):
        # Run trains steps
        network.train()
        train_stats = run_train_loop(train_env, train_agent, FLAGS.num_train_steps)

        # Run evaluation steps
        network.eval()
        eval_stats = run_evaluation_loop(eval_env, eval_agent, FLAGS.num_eval_steps)

        # Logging
        log_output = [
            ('iteration', iteration, '%3d'),
            ('step', iteration * FLAGS.num_train_steps, '%5d'),
            ('train_step_rate', train_stats['step_rate'], '%4.0f'),
            ('train_episode_return', train_stats['mean_episode_return'], '% 2.2f'),
            ('train_num_episodes', train_stats['num_episodes'], '%3d'),
            ('eval_episode_return', eval_stats['mean_episode_return'], '% 2.2f'),
            ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
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

        # Create checkpoint files
        if FLAGS.checkpoint_dir and os.path.exists(FLAGS.checkpoint_dir):
            torch.save(
                {
                    'network': network.state_dict(),
                },
                os.path.join(FLAGS.checkpoint_dir, f'{FLAGS.environment_name}_iteration_{iteration}.ckpt'),
            )

    if writer:
        writer.close()


if __name__ == '__main__':
    app.run(main)
