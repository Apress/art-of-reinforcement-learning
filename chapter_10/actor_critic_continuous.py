import collections
from typing import Tuple
import os

from absl import app
from absl import flags
import logging

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np

import trackers as trackers_lib
import csv_writer
import gym_env_processor

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'Humanoid-v4',
    'Classic robotic control task name, like Ant-v4, Humanoid-v4.',
)
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('policy_lr', 0.0002, 'Learning rate for actor (policy) network.')
flags.DEFINE_float('value_lr', 0.0003, 'Learning rate for critic (baseline) network.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('gae_lambda', 0.95, 'Lambda for the GAE general advantage estimator.')
flags.DEFINE_float('entropy_coef', 0.1, 'Coefficient for the entropy loss.')
flags.DEFINE_integer('sequence_length', 2048, 'Collect N transitions before update parameters.')
flags.DEFINE_integer('hidden_size', 64, 'Number of hidden units in the linear layer.')
flags.DEFINE_integer('num_iterations', 50, 'Number iterations to run.')
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


class ContinuousActorCriticAgent:
    """Actor-Critic agent for interacting with the environment and do learning."""

    def __init__(
        self,
        policy_network,
        policy_optimizer,
        value_network,
        value_optimizer,
        discount,
        gae_lambda,
        entropy_coef,
        sequence_length,
        clip_grad,
        max_grad_norm,
        device,
    ):
        self.device = device

        self.policy_network = policy_network.to(device=self.device)
        self.policy_optimizer = policy_optimizer
        self.value_network = value_network.to(device=self.device)
        self.value_optimizer = value_optimizer
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

        self.sequence_length = sequence_length
        self.sequence = []

        self.clip_grad = clip_grad
        self.max_grad_norm = max_grad_norm

        # Counters and statistics
        self.step_t = -1
        self.update_t = 0

    def act(self, observation, reward, done):
        """Given an environment observation, returns an action.
        Also (conditionally) update network parameters."""
        self.step_t += 1

        a_t = self.choose_action(observation)

        self.sequence.append((observation, a_t, reward, done))

        if len(self.sequence) >= self.sequence_length:
            self.update()

            del self.sequence[:]

        return a_t

    def update(self):
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        transitions = self.get_transitions_from_sequence(self.sequence)

        # Unpack list of tuples into separate lists
        s_t, a_t, return_t, advantage_t = map(list, zip(*transitions))

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self.device, dtype=torch.float32)
        # Actions are continuous values
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self.device, dtype=torch.float32)
        return_t = torch.from_numpy(np.stack(return_t, axis=0)).to(device=self.device, dtype=torch.float32)
        advantage_t = torch.from_numpy(np.stack(advantage_t, axis=0)).to(device=self.device, dtype=torch.float32)

        # Given past states, get predicted action probabilities and state value
        pi_mu_t, pi_sigma_t = self.policy_network(s_t)
        pi_m = Normal(pi_mu_t, pi_sigma_t)
        pi_logprob_a_t = pi_m.log_prob(a_t).sum(axis=-1)
        entropy_loss = pi_m.entropy()

        v_t = self.value_network(s_t)

        policy_loss = advantage_t.detach() * pi_logprob_a_t

        value_loss = 0.5 * torch.square(return_t - v_t.squeeze(-1))

        # Averaging over batch dimension
        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)
        value_loss = torch.mean(value_loss)

        # Negative sign to indicate we want to maximize the policy gradient objective function
        policy_loss = -(policy_loss + self.entropy_coef * entropy_loss)

        # Compute gradients
        policy_loss.backward()
        value_loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

            torch.nn.utils.clip_grad_norm_(
                self.value_network.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

        # Update parameters
        self.policy_optimizer.step()
        self.value_optimizer.step()

        self.update_t += 1

    def get_transitions_from_sequence(self, sequence):
        # Unpack list of tuples into separate lists.
        (observations, actions, rewards, dones) = map(list, zip(*sequence))

        # Get predicted state values
        with torch.no_grad():
            states = torch.from_numpy(np.stack(observations, axis=0)).to(device=self.device, dtype=torch.float32)
            values = self.value_network(states).squeeze(-1).cpu().numpy()

        # Our transitions in self.sequence is actually mismatched.
        # For example, the reward is one step behind, and there's not successor states
        # so we need to offset this
        s_t = observations[:-1]
        a_t = actions[:-1]
        v_t = values[:-1]
        r_t = rewards[1:]

        v_tp1 = values[1:]
        done_tp1 = dones[1:]

        # Compute returns and advantages
        return_t, advantage_t = self.compute_returns_and_advantages(v_t, r_t, v_tp1, done_tp1)

        # Zip multiple lists into list of tuples
        transitions = list(zip(s_t, a_t, return_t, advantage_t))

        return transitions

    def compute_returns_and_advantages(self, v_t, r_t, v_tp1, done_tp1):
        """This GAE returns and advantages was introduced in chapter 11 when we introduced PPO"""
        v_t = torch.from_numpy(np.stack(v_t, axis=0)).to(device=self.device, dtype=torch.float32)
        r_t = torch.from_numpy(np.stack(r_t, axis=0)).to(device=self.device, dtype=torch.float32)
        v_tp1 = torch.from_numpy(np.stack(v_tp1, axis=0)).to(device=self.device, dtype=torch.float32)
        done_tp1 = torch.from_numpy(np.stack(done_tp1, axis=0)).to(device=self.device, dtype=torch.bool)

        discount_tp1 = (~done_tp1).float() * self.discount

        lambda_ = torch.ones_like(discount_tp1) * self.gae_lambda  # If scalar, make into vector.

        delta_t = r_t + discount_tp1 * v_tp1 - v_t

        advantage_t = torch.zeros_like(delta_t, dtype=torch.float32)

        gae_t = 0
        for i in reversed(range(len(delta_t))):
            gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
            advantage_t[i] = gae_t

        return_t = advantage_t + v_t

        advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)

        return return_t.cpu().numpy(), advantage_t.cpu().numpy()

    @torch.no_grad()
    def choose_action(self, observation):
        """Given an environment observation, returns an action according to the policy."""
        s_t = torch.from_numpy(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        pi_mu, pi_sigma = self.policy_network(s_t)
        pi_m = Normal(pi_mu, pi_sigma)
        a_t = pi_m.sample()

        return a_t.squeeze(0).cpu().numpy()


class ContinuousPolicyGreedyActor:
    """Policy greedy actor for evaluation only."""

    def __init__(self, policy_network, device):
        self.device = device
        self.policy_network = policy_network.to(device=device)

    def act(self, observation):
        """Given an environment observation, returns an action."""
        return self.choose_action(observation)

    @torch.no_grad()
    def choose_action(self, observation):
        """Given an environment observation, returns an action according to the policy."""
        s_t = torch.tensor(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        pi_mu, pi_sigma = self.policy_network(s_t)
        pi_m = Normal(pi_mu, pi_sigma)
        a_t = pi_m.sample()
        return a_t.squeeze(0).cpu().numpy()


def run_train_loop(env, agent, num_train_steps):
    """Run training for some steps."""
    trackers = trackers_lib.make_default_trackers()

    t = 0

    while t < num_train_steps:
        s_t = env.reset()
        r_t = 0
        done = False

        while True:
            a_t = agent.act(s_t, r_t, done)

            # Take the action in the environment and observe successor state and reward.
            s_t, r_t, done, info = env.step(a_t)

            t += 1

            # Only keep track of non-clipped/unscaled raw reward when collecting statistics
            raw_reward = r_t
            if 'raw_reward' in info and isinstance(info['raw_reward'], (float, int)):
                raw_reward = info['raw_reward']

            for tracker in trackers:
                tracker.step(raw_reward, done)

            if done:
                # Send the final transition to the agent before reset
                unsued_a_t = agent.act(s_t, r_t, done)  # noqa: F841
                break

    return trackers_lib.generate_statistics(trackers)


def run_evaluation_loop(env, agent, num_eval_steps):
    """Run evaluation for some steps."""
    trackers = trackers_lib.make_default_trackers()

    s_t = env.reset()
    for _ in range(num_eval_steps):
        a_t = agent.act(s_t)
        s_t, r_t, done, _ = env.step(a_t)

        for tracker in trackers:
            tracker.step(r_t, done)

        if done:
            s_t = env.reset()

    return trackers_lib.generate_statistics(trackers)


def main(argv):
    """Trains Actor-Critic agent on robotic control tasks.

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

    device = 'cpu'

    logging.info(f'Runs Actor-Critic agent on {device}')

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
    policy_network = GaussianActorMlpNet(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=FLAGS.hidden_size,
    )
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.policy_lr)

    value_network = GaussianCriticMlpNet(state_dim=state_dim, hidden_size=FLAGS.hidden_size)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=FLAGS.value_lr)

    # Create training and evaluation agent instances
    train_agent = ContinuousActorCriticAgent(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        value_network=value_network,
        value_optimizer=value_optimizer,
        discount=FLAGS.discount,
        gae_lambda=FLAGS.gae_lambda,
        entropy_coef=FLAGS.entropy_coef,
        sequence_length=FLAGS.sequence_length,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=device,
    )

    eval_agent = ContinuousPolicyGreedyActor(
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
