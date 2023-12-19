import collections
from typing import Tuple
import os

from absl import app
from absl import flags
import logging

import math
import torch
from torch import nn
import numpy as np

from ppo import PPOAgent, PolicyGreedyActor, run_train_loop, run_evaluation_loop
import utils
import csv_writer
import gym_env_processor

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
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 1.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_integer('batch_size', 64, 'Sample batch size when do learning.')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate for policy network.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('gae_lambda', 0.95, 'Lambda for the GAE general advantage estimator.')
flags.DEFINE_float('entropy_coef', 0.025, 'Coefficient for the entropy loss.')
flags.DEFINE_float('value_coef', 0.5, 'Coefficient for the state-value loss.')
flags.DEFINE_integer('sequence_length', 128, 'Collect N transitions before update parameters.')
flags.DEFINE_integer(
    'num_epochs',
    4,
    'Number of update parameters epochs to run when collected a sequence of transitions.',
)
flags.DEFINE_float(
    'clip_epsilon_begin_value',
    0.2,
    'Begin clip epsilon in the PPO surrogate objective function.',
)
flags.DEFINE_float(
    'clip_epsilon_end_value',
    0.1,
    'Final clip epsilon in the PPO surrogate objective function.',
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
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ActorCriticConvNet(nn.Module):
    """Conv2d policy network with baseline head."""

    def __init__(self, state_dim: int, action_dim: int):
        """
        Args:
            state_dim: the dimension of the input vector to the neural network
            action_dim: the number of units for the output liner layer
        """
        super().__init__()

        # Compute the output shape of final conv2d layer
        c, h, w = state_dim
        h, w = calc_conv2d_output((h, w), 8, 4)
        h, w = calc_conv2d_output((h, w), 4, 2)
        h, w = calc_conv2d_output((h, w), 3, 1)
        conv2d_out_size = 64 * h * w

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv2d_out_size, 512),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(512, action_dim)

        self.value_head = nn.Linear(512, 1)

        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given state, returns raw probabilities logits for all possible actions
        and the predicted state value."""
        x = x.float() / 255.0
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict state-value
        value = self.value_head(features)
        return pi_logits, value


def main(argv):
    """Trains PPO agent with entropy loss.

    For every iteration, the code does these in sequence:
        1. Run train agent for num_train_episodes and periodically update network parameters
        2. Run evaluation agent for num_eval_episodes on a separate evaluation environment
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
    logging.info(f'Runs PPO agent on {device}')

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
    policy_network = ActorCriticConvNet(state_dim=state_dim, action_dim=action_dim)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.learning_rate)

    # Create training and evaluation agent instances
    train_agent = PPOAgent(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        discount=FLAGS.discount,
        gae_lambda=FLAGS.gae_lambda,
        value_coef=FLAGS.value_coef,
        entropy_coef=FLAGS.entropy_coef,
        sequence_length=FLAGS.sequence_length,
        num_epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size,
        clip_epsilon_schedule=utils.linear_schedule(
            begin_t=0,
            decay_steps=FLAGS.num_iterations * FLAGS.num_train_steps,
            begin_value=FLAGS.clip_epsilon_begin_value,
            end_value=FLAGS.clip_epsilon_end_value,
        ),
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=device,
    )

    eval_agent = PolicyGreedyActor(
        policy_network=policy_network,
        device=device,
    )

    # Start to run training and evaluation iterations
    for iteration in range(1, FLAGS.num_iterations + 1):
        # Run training steps
        policy_network.train()
        train_stats = run_train_loop(train_env, train_agent, FLAGS.num_train_steps)

        # Run evaluation steps
        policy_network.eval()
        eval_stats = run_evaluation_loop(eval_env, eval_agent, FLAGS.num_eval_steps)

        # Logging
        log_output = [
            ('iteration', iteration, '%3d'),
            (
                'step',
                iteration * FLAGS.num_train_steps * FLAGS.environment_frame_skip,
                '%5d',
            ),
            ('train_step_rate', train_stats['step_rate'], '% 2.2f'),
            ('train_episode_return', train_stats['mean_episode_return'], '% 2.2f'),
            ('train_num_episodes', train_stats['num_episodes'], '%3d'),
            ('eval_episode_return', eval_stats['mean_episode_return'], '% 2.2f'),
            ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
        ]
        log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
        logging.info(log_output_str)

        if writer:
            writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))

        # Create checkpoint files
        if FLAGS.checkpoint_dir and os.path.exists(FLAGS.checkpoint_dir):
            torch.save(
                {
                    'policy_network': policy_network.state_dict(),
                },
                os.path.join(FLAGS.checkpoint_dir, f'{FLAGS.environment_name}_iteration_{iteration}.ckpt'),
            )

    if writer:
        writer.close()


if __name__ == '__main__':
    app.run(main)
