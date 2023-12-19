"""Tests trained PPO agent on Atari MontezumaRevenge."""

from absl import app
from absl import flags
import logging

from typing import Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from rnd_ppo import PolicyGreedyActor

import gym_env_processor

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'MontezumaRevenge',
    'Atari name without NoFrameskip and version.',
)
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Skip frames by number of action repeats.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_integer('max_episode_steps', 18000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_string(
    'load_checkpoint_file',
    './checkpoints/MontezumaRevengeNoFrameskip-v4_seed2_iteration_77.ckpt',
    'Load a specific checkpoint file.',
)
flags.DEFINE_string('recording_video_dir', 'recordings', 'Path for recording a video of agent self-play.')


def calc_conv2d_output(h_w, kernel_size: int = 1, stride: int = 1, pad: int = 0, dilation: int = 1):
    """Takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, Tuple):
        kernel_size = (kernel_size, kernel_size)

    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return (h, w)


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
    """Tests PPO agent."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def environment_builder():
        return gym_env_processor.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            seed=random_state.randint(1, 2**32),
            terminal_on_life_loss=False,
            clip_reward=False,
            sticky_action=True,
            max_noop_steps=0,
            max_episode_steps=FLAGS.max_episode_steps,
        )

    # Create evaluation environment
    eval_env = environment_builder()

    action_dim = eval_env.action_space.n
    state_dim = eval_env.observation_space.shape

    logging.info('Environment: %s', eval_env.spec.id)
    logging.info('Action space: %s', action_dim)
    logging.info('Observation space: %s', state_dim)

    # Initialize network
    policy_network = ActorCriticConvNet(state_dim=state_dim, action_dim=action_dim)

    if FLAGS.load_checkpoint_file:
        loaded_state = torch.load(FLAGS.load_checkpoint_file, map_location=torch.device('cpu'))
        policy_network.load_state_dict(loaded_state['policy_network'])

    policy_network.eval()

    # Create evaluation agent instance
    eval_agent = PolicyGreedyActor(
        policy_network=policy_network,
        device=runtime_device,
    )

    gym_env_processor.play_and_record_video(agent=eval_agent, env=eval_env, save_dir=FLAGS.recording_video_dir)


if __name__ == '__main__':
    app.run(main)
