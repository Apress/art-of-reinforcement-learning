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
from torch.nn import functional as F
import numpy as np
import math

from rnd_ppo import (
    RNDPPOLeanerAgent,
    PPOActor,
    PolicyGreedyActor,
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
    'MontezumaRevenge',
    'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.',
)
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Skip frames by number of action repeats.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_integer('num_actors', 32, 'Number of actor processes to run.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for policy network.')
flags.DEFINE_float('rnd_learning_rate', 0.0001, 'Learning rate for policy network.')
flags.DEFINE_float('ext_discount', 0.999, 'Discount rate for extrinsic reward.')
flags.DEFINE_float('int_discount', 0.99, 'Discount rate for intrinsic reward.')

flags.DEFINE_float('gae_lambda', 0.95, 'Lambda for the GAE general advantage estimator.')
flags.DEFINE_float('value_coef', 0.5, 'Coefficient for the state value loss.')
flags.DEFINE_float('entropy_coef', 0.001, 'Coefficient for the entropy loss.')
flags.DEFINE_integer(
    'rnd_random_obs_steps',
    50,
    'Collect N transitions to update the observation normalization statistics.',
)
flags.DEFINE_integer('sequence_length', 128, 'Collect N transitions before update parameters.')
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

flags.DEFINE_integer('num_iterations', 78, 'Number iterations to run.')
flags.DEFINE_integer(
    'num_train_steps',
    128 * 500,
    'Number training environment steps or frames (after frame skip) to run per iteration.',
)
flags.DEFINE_integer(
    'num_eval_steps',
    45000,
    'Number evaluation environment steps or frames (after frame skip) to run per iteration.',
)
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_string(
    'checkpoint_dir',
    'checkpoints/',
    'Directory to store checkpoint file, default empty means do not create checkpoint.',
)
flags.DEFINE_string(
    'results_csv_path',
    '',
    'Path for CSV log file.',
)


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


class RndConvNet(nn.Module):
    """RND Conv2d network.

    From the paper "Exploration by Random Network Distillation"
    https://arxiv.org/abs/1810.12894
    """

    def __init__(self, state_dim: int, is_target: bool = False, latent_dim: int = 512) -> None:
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            is_target: if True, use one single linear layer at the head, default False.
            latent_dim: the embedding latent dimension, default 256.
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
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        if is_target:
            self.head = nn.Linear(conv2d_out_size, latent_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(conv2d_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, latent_dim),
            )

        # Initialize weights.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                module.bias.data.zero_()

        if is_target:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state x, returns the feature embedding."""
        # RND normalizes state using a running mean and std instead of divide by 255.
        x = self.body(x)
        return self.head(x)


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
            nn.Linear(conv2d_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 448),
            nn.ReLU(),
        )

        self.extra_policy_fc = nn.Linear(448, 448)
        self.extra_value_fc = nn.Linear(448, 448)

        self.policy_head = nn.Linear(448, action_dim)
        self.ext_value_head = nn.Linear(448, 1)
        self.int_value_head = nn.Linear(448, 1)

        for layer in self.body.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

        for layer in [self.extra_policy_fc, self.extra_value_fc]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(0.1))
            layer.bias.data.zero_()

        for layer in [self.policy_head, self.ext_value_head, self.int_value_head]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(0.01))
            layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given state, returns raw probabilities logits for all possible actions
        and the predicted state value."""
        x = x.float() / 255.0
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_features = features + F.relu(self.extra_policy_fc(features))
        pi_logits = self.policy_head(pi_features)

        # Predict state-value
        value_features = features + F.relu(self.extra_value_fc(features))
        ext_value = self.ext_value_head(value_features)
        int_value = self.int_value_head(value_features)
        return pi_logits, ext_value, int_value


def main(argv):
    """Trains PPO agent with entropy loss.

    For every iteration, the code does these in sequence:
        1. Run train agent for num_train_episodes and periodically update network parameters
        2. Run evaluation agent for num_eval_episodes on a separate evaluation environment
        3. Logging statistics to a csv file
        4. Create checkpoint file

    """
    random_state = np.random.RandomState(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logging.info('Runs distributed PPO agent')

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
            terminal_on_life_loss=False,
            clip_reward=True,
            sticky_action=True,
            max_noop_steps=0,
            max_episode_steps=18000,  # max episode frames before apply frame skip, which is 4500 x 4
        )

    # Create evaluation environment
    eval_env = environment_builder()

    action_dim = eval_env.action_space.n
    state_dim = eval_env.observation_space.shape

    logging.info('Environment: %s', eval_env.spec.id)
    logging.info('Action space: %s', action_dim)
    logging.info('Observation space: %s', state_dim)

    # Initialize network
    def create_policy_network():
        return ActorCriticConvNet(state_dim=state_dim, action_dim=action_dim)

    policy_network = create_policy_network()
    eval_policy_network = create_policy_network()

    rnd_state_dim = (1,) + state_dim[1:]
    rnd_target_network = RndConvNet(state_dim=rnd_state_dim, is_target=True)
    rnd_predictor_network = RndConvNet(state_dim=rnd_state_dim, is_target=False)

    # Create train agent instances
    train_agent = RNDPPOLeanerAgent(
        policy_network=policy_network,
        learning_rate=FLAGS.learning_rate,
        rnd_target_network=rnd_target_network,
        rnd_predictor_network=rnd_predictor_network,
        rnd_learning_rate=FLAGS.rnd_learning_rate,
        ext_discount=FLAGS.ext_discount,
        int_discount=FLAGS.int_discount,
        rnd_experience_proportion=min(
            1.0, 32 / FLAGS.num_actors
        ),  # for actors <=32, this is set to 1.0, for actors > 32, it should be 32/num_actors.
        gae_lambda=FLAGS.gae_lambda,
        value_coef=FLAGS.value_coef,
        entropy_coef=FLAGS.entropy_coef,
        num_epochs=FLAGS.num_epochs,
        batch_size=int((FLAGS.sequence_length * FLAGS.num_actors) / 4),
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

    eval_agent = PolicyGreedyActor(policy_network=eval_policy_network, device='cpu')
    eval_trackers = trackers_lib.make_default_trackers()

    queue = mp.Queue()

    # Create shared objects so all actor processes can access them
    manager = mp.Manager()

    # Store serialized actor instances in a shared memory, so we can persist the actor's internal state
    # Like using the same environment across iterations
    serialized_actors = manager.list(
        [
            pickle.dumps(
                PPOActor(
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

    # Warm up to update RND observation normalizer statistics
    train_agent.initialize_obs_rms(env=environment_builder(), num_steps=int(FLAGS.num_actors * FLAGS.rnd_random_obs_steps))

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
        mean_train_episode_visited_rooms = np.mean([stats['mean_episode_visited_rooms'] for stats in actor_statistics]).item()
        mean_train_num_episodes = np.mean([stats['num_episodes'] for stats in actor_statistics]).item()

        # Run evaluation steps
        eval_policy_network.load_state_dict(train_agent.get_policy_state_dict())
        eval_stats = run_evaluation_loop(
            env=eval_env,
            agent=eval_agent,
            num_eval_steps=FLAGS.num_eval_steps,
            trackers=eval_trackers,
        )
        log_output = [
            ('iteration', iteration, '%3d'),
            ('step', iteration * FLAGS.num_train_steps * FLAGS.environment_frame_skip, '%5d'),
            ('total_step', iteration * FLAGS.num_train_steps * FLAGS.environment_frame_skip * FLAGS.num_actors, '%5d'),
            ('train_step_rate', mean_train_step_rate, '%2.2f'),
            ('train_episode_return', mean_train_episode_return, '%2.2f'),
            ('train_episode_visited_rooms', mean_train_episode_visited_rooms, '%2.2f'),
            ('train_num_episodes', mean_train_num_episodes, '%3d'),
            (
                'eval_episode_return',
                eval_stats['mean_episode_return'],
                '% 2.2f',
            ),
            ('eval_episode_visited_rooms', eval_stats['mean_episode_visited_rooms'], '%2.2f'),
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
                    'rnd_target_network': rnd_target_network.state_dict(),
                    'rnd_predictor_network': rnd_predictor_network.state_dict(),
                },
                f'{FLAGS.checkpoint_dir}/{eval_env.spec.id}_iteration_{iteration}.ckpt',
            )

    queue.close()

    if writer:
        writer.close()


if __name__ == '__main__':
    # Set multiprocessing start mode
    mp.set_start_method('spawn')
    app.run(main)
