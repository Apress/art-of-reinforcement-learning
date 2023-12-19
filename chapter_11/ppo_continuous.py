import collections
from typing import Tuple
import os

from absl import app
from absl import flags
import logging

import torch
from torch import nn
import numpy as np

from ppo import ContinuousPPOAgent, ContinuousPolicyGreedyActor, run_train_loop, run_evaluation_loop

import utils
import csv_writer
import gym_env_processor


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'Humanoid-v4',
    'Classic robotic control task name, like Ant-v4, Humanoid-v4.',
)
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('policy_lr', 0.0002, 'Learning rate for actor (policy) network.')
flags.DEFINE_float('value_lr', 0.0003, 'Learning rate for critic (baseline) network.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('gae_lambda', 0.95, 'Lambda for the GAE general advantage estimator.')
flags.DEFINE_float('entropy_coef', 0.1, 'Coefficient for the entropy loss.')
flags.DEFINE_integer('sequence_length', 2048, 'Collect N transitions before update parameters.')
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
    0.02,
    'Final clip epsilon in the PPO surrogate objective function.',
)

flags.DEFINE_integer('hidden_size', 64, 'Number of hidden units in the linear layer.')

flags.DEFINE_integer('num_iterations', 20, 'Number iterations to run.')
flags.DEFINE_integer(
    'num_train_steps',
    int(1e5),
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


class GaussianActorMlpNet(nn.Module):
    """Gaussian Actor MLP network for continuous action space."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(hidden_size, action_dim)
        self.std_head = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Given raw state x, predict the mean and sigma of probability density function."""
        features = self.body(x)

        pi_mu = self.mu_head(features)
        pi_sigma = torch.exp(self.std_head(features))

        return pi_mu, pi_sigma


class GaussianCriticMlpNet(nn.Module):
    """Gaussian Critic MLP network for continuous action space."""

    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state x, predict the state-value."""

        value = self.net(x)

        return value


def main(argv):
    """Trains PPO agent with entropy loss.

    For every iteration, the code does these in sequence:
        1. Run train agent for num_train_episodes and periodically update network parameters
        2. Run evaluation agent for num_eval_episodes on a separate evaluation environment
        3. Logging statistics to a csv file
        4. Create checkpoint file

    """
    del argv

    torch.manual_seed(FLAGS.seed)
    random_state = np.random.RandomState(FLAGS.seed)
    device = 'cpu'

    logging.info(f'Runs PPO agent on {device}')

    writer = None
    if FLAGS.results_csv_path:
        writer = csv_writer.CsvWriter(FLAGS.results_csv_path)

    def environment_builder():
        return gym_env_processor.create_continuous_environment(
            env_name=FLAGS.environment_name,
            seed=random_state.randint(1, 2**32),
        )

    # Create training and evaluation environments
    train_env = environment_builder()
    eval_env = environment_builder()

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    logging.info('Environment: %s', train_env.spec.id)
    logging.info('Action space: %s', action_dim)
    logging.info('Observation space: %s', state_dim)

    # Initialize network and optimizer
    policy_network = GaussianActorMlpNet(state_dim=state_dim, action_dim=action_dim, hidden_size=FLAGS.hidden_size)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.policy_lr)

    value_network = GaussianCriticMlpNet(state_dim=state_dim, hidden_size=FLAGS.hidden_size)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=FLAGS.value_lr)

    # Create training and evaluation agent instances
    train_agent = ContinuousPPOAgent(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        value_network=value_network,
        value_optimizer=value_optimizer,
        discount=FLAGS.discount,
        gae_lambda=FLAGS.gae_lambda,
        entropy_coef=FLAGS.entropy_coef,
        sequence_length=FLAGS.sequence_length,
        num_epochs=FLAGS.num_epochs,
        batch_size=int(FLAGS.sequence_length / 4),
        clip_epsilon_schedule=utils.linear_schedule(
            begin_t=0,
            decay_steps=(FLAGS.num_iterations * FLAGS.num_train_steps),
            begin_value=FLAGS.clip_epsilon_begin_value,
            end_value=FLAGS.clip_epsilon_end_value,
        ),
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=device,
    )

    eval_agent = ContinuousPolicyGreedyActor(policy_network=policy_network, device=device)

    # Start to run training iterations
    for iteration in range(1, FLAGS.num_iterations + 1):
        # Run training steps
        policy_network.train()
        value_network.train()
        train_stats = run_train_loop(train_env, train_agent, FLAGS.num_train_steps)

        # Run training steps
        policy_network.eval()
        eval_stats = run_evaluation_loop(eval_env, eval_agent, FLAGS.num_eval_steps)

        # Logging
        log_output = [
            ('iteration', iteration, '%3d'),
            ('step', iteration * FLAGS.num_train_steps, '%5d'),
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
                    'value_network': value_network.state_dict(),
                },
                os.path.join(FLAGS.checkpoint_dir, f'{FLAGS.environment_name}_iteration_{iteration}.ckpt'),
            )

    if writer:
        writer.close()


if __name__ == '__main__':
    app.run(main)
