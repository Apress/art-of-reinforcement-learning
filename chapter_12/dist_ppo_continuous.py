import os

# This forces OpenMP to use 1 single thread, which is needed to
# prevent contention between multiple process.
os.environ['OMP_NUM_THREADS'] = '1'
# Tell numpy to only use one core. If we don't do this, each process may
# try to use all of the cores and the resulting contention may result
# in no speedup over the serial version. Note that if numpy is using
# OpenBLAS, then you need to set OPENBLAS_NUM_THREADS=1, and you
# probably need to do it from the command line (so it happens before
# numpy is imported).
os.environ['MKL_NUM_THREADS'] = '1'

import collections
from typing import Tuple
import pickle

import multiprocessing as mp

from absl import app
from absl import flags
import logging

import torch
from torch import nn
import numpy as np

from ppo import (
    ContinuousPPOLeanerAgent,
    ContinuousPPOActor,
    ContinuousPolicyGreedyActor,
    run_actor_loop,
    run_learner_loop,
    run_evaluation_loop,
)

import utils
import csv_writer
import trackers as trackers_lib
import gym_env_processor


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'Humanoid-v4',
    'Classic robotic control task name, like Ant-v4, Humanoid-v4.',
)
flags.DEFINE_integer('num_actors', 4, 'Number of actor processes to run.')
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
flags.DEFINE_string(
    'results_csv_path',
    '',
    'Path for CSV log file.',
)


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
    """Trains distributed PPO agent on classic robotic control tasks.

    For every iteration, the code does these in sequence:
        1. Run train agent for num_train_episodes and periodically update network parameters
        2. Run evaluation agent for num_eval_episodes on a separate evaluation environment
        3. Logging statistics to a csv file
        4. Create checkpoint file

    """
    torch.manual_seed(FLAGS.seed)
    random_state = np.random.RandomState(FLAGS.seed)

    logging.info('Runs distributed PPO agent')

    writer = None
    if FLAGS.results_csv_path:
        writer = csv_writer.CsvWriter(FLAGS.results_csv_path)

    def environment_builder():
        return gym_env_processor.create_continuous_environment(
            env_name=FLAGS.environment_name,
            seed=random_state.randint(1, 2**32),
        )

    # Create training and evaluation environments
    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    logging.info('Environment: %s', eval_env.spec.id)
    logging.info('Action space: %s', action_dim)
    logging.info('Observation space: %s', state_dim)

    # Initialize network
    def create_policy_network():
        return GaussianActorMlpNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=FLAGS.hidden_size,
        )

    def create_value_network():
        return GaussianCriticMlpNet(
            state_dim=state_dim,
            hidden_size=FLAGS.hidden_size,
        )

    policy_network = create_policy_network()
    value_network = create_value_network()

    eval_policy_network = create_policy_network()

    # Create train agent instances
    train_agent = ContinuousPPOLeanerAgent(
        policy_network=policy_network,
        policy_lr=FLAGS.policy_lr,
        value_network=value_network,
        value_lr=FLAGS.value_lr,
        discount=FLAGS.discount,
        gae_lambda=FLAGS.gae_lambda,
        entropy_coef=FLAGS.entropy_coef,
        num_epochs=FLAGS.num_epochs,
        batch_size=int((FLAGS.sequence_length * FLAGS.num_epochs) / 4),
        clip_epsilon_schedule=utils.linear_schedule(
            begin_t=0,
            decay_steps=FLAGS.num_iterations * (FLAGS.num_train_steps / FLAGS.sequence_length),
            begin_value=FLAGS.clip_epsilon_begin_value,
            end_value=FLAGS.clip_epsilon_end_value,
        ),
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    eval_agent = ContinuousPolicyGreedyActor(
        policy_network=eval_policy_network,
        device='cpu',
    )

    eval_trackers = trackers_lib.make_default_trackers()

    queue = mp.Queue()

    # Create shared objects so all actor processes can access them
    manager = mp.Manager()

    # Store serialized actor instances in a shared memory, so we can persist the actor's internal state
    # Like using the same environment across iterations
    serialized_actors = manager.list(
        [
            pickle.dumps(
                ContinuousPPOActor(
                    i,
                    create_policy_network(),
                    'cpu',
                    environment_builder(),
                    None,
                )
            )
            for i in range(FLAGS.num_actors)
        ]
    )

    # Set start conditions for each actor
    actor_conditions = manager.list([False for i in range(FLAGS.num_actors)])

    # Store actor statistics
    actor_statistics = manager.list([None for i in range(FLAGS.num_actors)])

    # Store latest parameters for the policy network
    shared_params = manager.dict({'policy': train_agent.get_policy_state_dict()})

    # Count number of done actors
    counter = mp.Value('i', 0)

    # Start to run training iterations
    for iteration in range(1, FLAGS.num_iterations + 1):
        # On each iteration, reset start conditions for each actor
        with counter.get_lock():
            counter.value = 0

        for i in range(FLAGS.num_actors):
            actor_conditions[i] = True

        # Start actors
        processes = []
        for i in range(FLAGS.num_actors):
            p = mp.Process(
                target=run_actor_loop,
                args=(
                    FLAGS.seed,
                    i,
                    FLAGS.num_train_steps,
                    FLAGS.sequence_length,
                    serialized_actors,
                    actor_conditions,
                    actor_statistics,
                    shared_params,
                    queue,
                    counter,
                ),
            )
            p.start()
            processes.append(p)

        # Run learner loop on the main process
        policy_network.train()
        run_learner_loop(
            agent=train_agent,
            num_actors=len(processes),
            actor_conditions=actor_conditions,
            shared_params=shared_params,
            queue=queue,
            counter=counter,
        )

        # Logging
        # Average statistics over actors
        mean_train_step_rate = np.mean([stats['step_rate'] for stats in actor_statistics]).item()
        mean_train_episode_return = np.mean([stats['mean_episode_return'] for stats in actor_statistics]).item()
        mean_train_num_episodes = np.mean([stats['num_episodes'] for stats in actor_statistics]).item()

        # Run evaluation steps
        eval_policy_network.load_state_dict(train_agent.get_policy_state_dict())
        eval_policy_network.eval()
        eval_stats = run_evaluation_loop(
            env=eval_env,
            agent=eval_agent,
            num_eval_steps=FLAGS.num_eval_steps,
            trackers=eval_trackers,
        )
        log_output = [
            ('iteration', iteration, '%3d'),
            ('step', iteration * FLAGS.num_train_steps, '%5d'),
            ('train_step_rate', mean_train_step_rate, '%2.2f'),
            ('train_episode_return', mean_train_episode_return, '%2.2f'),
            ('train_num_episodes', mean_train_num_episodes, '%3d'),
            (
                'eval_episode_return',
                eval_stats['mean_episode_return'],
                '% 2.2f',
            ),
            ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
        ]
        log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
        logging.info(log_output_str)

        if writer:
            writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))

        for p in processes:
            p.join()

        # Create checkpoint files
        if FLAGS.checkpoint_dir and os.path.exists(FLAGS.checkpoint_dir):
            torch.save(
                {
                    'policy_network': policy_network.state_dict(),
                    'value_network': value_network.state_dict(),
                },
                os.path.join(FLAGS.checkpoint_dir, f'{FLAGS.environment_name}_iteration_{iteration}.ckpt'),
            )

    queue.close()

    if writer:
        writer.close()


if __name__ == '__main__':
    # Set multiprocessing start mode
    mp.set_start_method('spawn')
    app.run(main)
