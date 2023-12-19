import collections
from typing import Tuple
import os

from absl import app
from absl import flags
import logging

import torch
from torch import nn
import gym
import numpy as np

from ppo import PPOAgent, PolicyGreedyActor, run_train_loop, run_evaluation_loop

import utils
import csv_writer


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Classic control task name, like CartPole-v1, MountainCar-v0.',
)
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate for policy network.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('gae_lambda', 0.95, 'Lambda for the GAE general advantage estimator.')
flags.DEFINE_float('value_coef', 0.5, 'Coefficient for the state value loss.')
flags.DEFINE_float('entropy_coef', 0.01, 'Coefficient for the entropy loss.')
flags.DEFINE_integer('sequence_length', 64, 'Collect N transitions before update parameters.')
flags.DEFINE_integer(
    'num_epochs',
    4,
    'Number of update parameters epochs to run when collected a sequence of transitions.',
)
flags.DEFINE_float(
    'clip_epsilon_begin_value',
    0.1,
    'Begin clip epsilon in the PPO surrogate objective function.',
)
flags.DEFINE_float(
    'clip_epsilon_end_value',
    0.1,
    'Final clip epsilon in the PPO surrogate objective function.',
)

flags.DEFINE_integer('batch_size', 64, 'Batch size when run epochs to update parameters.')

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


class ActorCriticMlpNet(nn.Module):
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
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given state, returns raw probabilities logits for all possible actions ans estimated state value."""

        features = self.body(x)
        pi_logits = self.policy_head(features)
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

    torch.manual_seed(FLAGS.seed)
    random_state = np.random.RandomState(FLAGS.seed)
    device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f'Runs PPO agent on {device}')

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
    state_dim = train_env.observation_space.shape[0]

    logging.info('Environment: %s', train_env.spec.id)
    logging.info('Action space: %s', action_dim)
    logging.info('Observation space: %s', state_dim)

    # Initialize networks
    policy_network = ActorCriticMlpNet(state_dim=state_dim, action_dim=action_dim)
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
            decay_steps=(FLAGS.num_iterations * FLAGS.num_train_steps),
            begin_value=FLAGS.clip_epsilon_begin_value,
            end_value=FLAGS.clip_epsilon_end_value,
        ),
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=device,
    )

    eval_agent = PolicyGreedyActor(policy_network=policy_network, device=device)

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
                },
                os.path.join(FLAGS.checkpoint_dir, f'{FLAGS.environment_name}_iteration_{iteration}.ckpt'),
            )

    if writer:
        writer.close()


if __name__ == '__main__':
    app.run(main)
