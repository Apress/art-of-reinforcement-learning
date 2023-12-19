from typing import Callable, Tuple, Optional
import math
import collections
import os

from absl import app
from absl import flags
import logging

import torch
from torch import nn

import numpy as np

import utils
import trackers as trackers_lib
import csv_writer
import gym_env_processor
import replay as replay_lib

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'Pong',
    'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.',
)
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Skip frames by number of action repeats.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_bool(
    'compress_state',
    True,
    'Compress state images when store in experience replay.',
)
flags.DEFINE_integer('replay_capacity', int(1e6), 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 50000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 32, 'Sample batch size when do learning.')

flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')

flags.DEFINE_float(
    'exploration_epsilon_begin_value',
    1.0,
    'Begin value of the exploration rate in e-greedy policy.',
)
flags.DEFINE_float(
    'exploration_epsilon_end_value',
    0.1,
    'End (decayed) value of the exploration rate in e-greedy policy.',
)
flags.DEFINE_float(
    'exploration_epsilon_decay_step',
    int(1e6),
    'Total steps or frames (after frame skip) to decay value of the exploration rate.',
)
flags.DEFINE_float(
    'eval_exploration_epsilon_rate',
    0.01,
    'The exploration rate in e-greedy policy for evaluation actor only.',
)

flags.DEFINE_integer('num_iterations', 100, 'Number iterations to run.')
flags.DEFINE_integer(
    'num_train_steps',
    int(1e6 / 4),
    'Number training environment steps or frames (after frame skip) to run per iteration.',
)
flags.DEFINE_integer(
    'num_eval_steps',
    int(2e5),
    'Number evaluation environment steps or frames (after frame skip) to run per iteration.',
)
flags.DEFINE_integer(
    'update_interval',
    4,
    'The frequency (measured in number of frames seen by the agent after frame skip) to sample a batch transitions from replay to update the network parameters.',
)

flags.DEFINE_integer(
    'target_network_update_interval',
    2500,
    'The frequency (measured in number of network parameter updates) to update target network parameters.',
)
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'Directory to store checkpoint file, default empty means do not create checkpoint.',
)
flags.DEFINE_string('results_csv_path', '', 'Path for CSV log file.')


def calc_conv2d_output(h_w, kernel_size: int = 1, stride: int = 1, pad: int = 0, dilation: int = 1):
    """Takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, Tuple):
        kernel_size = (kernel_size, kernel_size)

    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return (h, w)


def initialize_weights(net: nn.Module) -> None:
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(net, nn.Module)

    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

            if module.bias is not None:
                nn.init.zeros_(module.bias)


class DqnConvNet(nn.Module):
    """Conv2d DQN network."""

    def __init__(self, state_dim: tuple, action_dim: int):
        """
        Args:
            state_dim: the shape of the input image to the neural network
            action_dim: the number of units for the output liner layer
        """
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')

        super().__init__()

        # Compute the output shape of final conv2d layer
        c, h, w = state_dim
        h, w = calc_conv2d_output((h, w), 8, 4)
        h, w = calc_conv2d_output((h, w), 4, 2)
        h, w = calc_conv2d_output((h, w), 3, 1)
        conv2d_out_size = 64 * h * w

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv2d_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given state, return state-action value for all possible actions"""
        x = x.float() / 255.0

        q_values = self.net(x)  # [batch_size, action_dim]
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


class DQNAgent:
    """DQN agent for interacting with the environment and do learning."""

    def __init__(
        self,
        network,
        target_network,
        optimizer,
        random_state,
        replay,
        exploration_epsilon_schedule,
        update_interval,
        target_network_update_interval,
        min_replay_size,
        batch_size,
        discount,
        device,
    ):
        self.device = device
        self.random_state = random_state

        self.network = network.to(device=self.device)
        self.optimizer = optimizer

        self.target_network = target_network.to(device=self.device)
        self.update_target_network()
        # Disable autograd for target network
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.replay = replay
        self.batch_size = batch_size
        self.discount = discount
        self.exploration_epsilon_schedule = exploration_epsilon_schedule
        self.min_replay_size = min_replay_size
        self.update_interval = update_interval
        self.target_network_update_interval = target_network_update_interval

        # Counters and statistics
        self.step_t = -1
        self.update_t = 0
        self.target_update_t = 0

    def act(self, observation):
        """Given an environment observation, returns an action.
        Also (conditionally) update network parameters."""
        self.step_t += 1
        a_t = self.choose_action(observation, self.exploration_epsilon)

        # Return if replay is not ready
        if self.replay.size < self.min_replay_size:
            return a_t

        # Update network parameters
        if self.step_t % self.update_interval == 0:
            transitions = self.replay.sample(self.batch_size)
            self.update(transitions)
            self.update_t += 1

            if self.update_t % self.target_network_update_interval == 0:
                self.update_target_network()
                self.target_update_t += 1

        return a_t

    def save_transition(self, transition):
        self.replay.add(transition)

    @torch.no_grad()
    def choose_action(self, observation, epsilon):
        """Given an environment observation and exploration rate,
        returns an action according to the epsilon-greedy policy."""
        s_t = torch.from_numpy(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        q_t = self.network(s_t)
        return apply_e_greedy_policy(q_t, epsilon, self.random_state)

    def update(self, transitions):
        self.optimizer.zero_grad()
        loss = self.calc_loss(transitions)
        loss.backward()

        # if self.grad_error_bound > 0:
        #     torch.nn.utils.clip_grad.clip_grad_value_(self.network.parameters(), self.grad_error_bound)
        self.optimizer.step()

    def calc_loss(self, transitions):
        """Calculate loss for a given batch of transitions."""
        s_t = torch.from_numpy(transitions.s_t).to(device=self.device, dtype=torch.float32)  # [batch_size, state_shape]
        a_t = torch.from_numpy(transitions.a_t).to(device=self.device, dtype=torch.int64)  # [batch_size]
        r_t = torch.from_numpy(transitions.r_t).to(device=self.device, dtype=torch.float32)  # [batch_size]
        s_tp1 = torch.from_numpy(transitions.s_tp1).to(device=self.device, dtype=torch.float32)  # [batch_size, state_shape]
        done_tp1 = torch.from_numpy(transitions.done_tp1).to(device=self.device, dtype=torch.bool)  # [batch_size]

        discount_tp1 = (~done_tp1).float() * self.discount

        # Compute predicted q values for s_t
        q_t = self.network(s_t)  # [batch_size, action_dim]

        # Compute predicted q values for s_tp1 using target network, then compute the TD target
        with torch.no_grad():
            q_tp1 = self.target_network(s_tp1)  # [batch_size, action_dim]
            target_t = r_t + discount_tp1 * torch.max(q_tp1, dim=1)[0]

        # Compute loss which is 0.5 * square(td_errors)
        qa_t = q_t[torch.arange(0, s_t.shape[0]), a_t]
        td_error = target_t - qa_t

        loss = 0.5 * td_error**2

        # Averaging over batch dimension
        loss = torch.mean(loss, dim=0)
        return loss

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

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
    t = 0

    while t < num_train_steps:
        s_t = env.reset()
        done = False
        loss_life = False

        while True:  # For each step in the current episode
            a_t = agent.act(s_t)

            # Take the action in the environment and observe successor state and reward.
            s_tp1, r_t, done, info = env.step(a_t)
            t += 1

            # Only keep track of non-clipped/unscaled raw reward when collecting statistics
            raw_reward = r_t
            if 'raw_reward' in info and isinstance(info['raw_reward'], (float, int)):
                raw_reward = info['raw_reward']

            # For Atari games, check if treat loss a life as a soft-terminal state
            loss_life = False
            if 'loss_life' in info and isinstance(info['loss_life'], bool):
                loss_life = info['loss_life']

            for tracker in trackers:
                tracker.step(raw_reward, done)

            agent.save_transition(
                replay_lib.Transition(
                    s_t=s_t,
                    a_t=a_t,
                    r_t=r_t,
                    s_tp1=s_tp1,
                    done_tp1=done or loss_life,
                )
            )

            s_t = s_tp1
            if done:
                break

    return trackers_lib.generate_statistics(trackers)


def run_evaluation_loop(env, agent, num_eval_steps):
    """Run evaluation for some steps."""
    trackers = trackers_lib.make_default_trackers()
    t = 0

    while t < num_eval_steps:
        s_t = env.reset()

        while True:
            a_t = agent.act(s_t)
            s_tp1, r_t, done, info = env.step(a_t)
            t += 1

            # Only keep track of non-clipped/unscaled raw reward when collecting statistics
            raw_reward = r_t
            if 'raw_reward' in info and isinstance(info['raw_reward'], (float, int)):
                raw_reward = info['raw_reward']

            for tracker in trackers:
                tracker.step(raw_reward, done)

            s_t = s_tp1
            if done:
                break

    return trackers_lib.generate_statistics(trackers)


def main(argv):
    """Trains DQN agent.

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

    logging.info(f'Runs DQN agent on {device}')

    writer = None
    if FLAGS.results_csv_path:
        writer = csv_writer.CsvWriter(FLAGS.results_csv_path)

    def environment_builder():
        return gym_env_processor.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            seed=random_state.randint(1, 2**32),
            max_noop_steps=30,
            terminal_on_life_loss=True,
        )

    # Create training and evaluation environments
    train_env = environment_builder()
    eval_env = environment_builder()

    action_dim = train_env.action_space.n
    state_dim = train_env.observation_space.shape

    logging.info('Environment: %s', train_env.spec.id)
    logging.info('Action space: %s', action_dim)
    logging.info('Observation space: %s', state_dim)

    # Initialize network and optimizer
    network = DqnConvNet(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate)
    target_network = DqnConvNet(state_dim=state_dim, action_dim=action_dim)

    if FLAGS.compress_state:

        def encoder(transition):
            return transition._replace(
                s_t=replay_lib.compress_array(transition.s_t),
                s_tp1=replay_lib.compress_array(transition.s_tp1),
            )

        def decoder(transition):
            return transition._replace(
                s_t=replay_lib.uncompress_array(transition.s_t),
                s_tp1=replay_lib.uncompress_array(transition.s_tp1),
            )

    else:
        encoder = None
        decoder = None

    train_agent = DQNAgent(
        network=network,
        target_network=target_network,
        optimizer=optimizer,
        random_state=random_state,
        replay=replay_lib.UniformReplay(
            FLAGS.replay_capacity,
            replay_lib.TransitionStructure,
            random_state,
            encoder,
            decoder,
        ),
        exploration_epsilon_schedule=utils.linear_schedule(
            begin_t=FLAGS.min_replay_size,
            decay_steps=FLAGS.exploration_epsilon_decay_step,
            begin_value=FLAGS.exploration_epsilon_begin_value,
            end_value=FLAGS.exploration_epsilon_end_value,
        ),
        update_interval=FLAGS.update_interval,
        target_network_update_interval=FLAGS.target_network_update_interval,
        min_replay_size=FLAGS.min_replay_size,
        batch_size=FLAGS.batch_size,
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
        # Run training steps
        network.train()
        train_stats = run_train_loop(train_env, train_agent, FLAGS.num_train_steps)

        # Run evaluation steps
        network.eval()
        eval_stats = run_evaluation_loop(eval_env, eval_agent, FLAGS.num_eval_steps)

        # Logging
        log_output = [
            ('iteration', iteration, '%3d'),
            (
                'step',
                iteration * FLAGS.num_train_steps * FLAGS.environment_frame_skip,
                '%5d',
            ),
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
                f'{FLAGS.checkpoint_dir}/{FLAGS.environment_name}_seed_{FLAGS.seed}_iteration_{iteration}.ckpt',
            )

    if writer:
        writer.close()


if __name__ == '__main__':
    app.run(main)
