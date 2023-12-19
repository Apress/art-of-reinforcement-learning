import collections
import os

from absl import app
from absl import flags
import logging

import torch
from torch import nn
from torch.distributions import Categorical
import gym
import numpy as np

import trackers as trackers_lib
import csv_writer

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Classic control task name, like CartPole-v1, MountainCar-v0.',
)

flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')

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


class PolicyMlpNet(nn.Module):
    """MLP policy network."""

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
        """Given state, returns raw probabilities logits for all possible actions"""

        pi_logits = self.body(x)  # [batch_size, action_dim]
        return pi_logits


class ReinforceAgent:
    """REINFORCE agent for interacting with the environment and do learning."""

    def __init__(
        self,
        policy_network,
        optimizer,
        discount,
        device,
    ):
        self.device = device

        self.policy_network = policy_network.to(device=self.device)
        self.optimizer = optimizer
        self.discount = discount

        # Counters and statistics
        self.step_t = -1
        self.update_t = 0

    def act(self, observation):
        """Given an environment observation, returns an action.
        Also (conditionally) update policy_network parameters."""
        self.step_t += 1
        return self.choose_action(observation)

    def update(self, episode_sequence):
        # Unpack list of tuples into separate lists.
        observations, actions, rewards = map(list, zip(*episode_sequence))

        T = len(rewards)

        self.optimizer.zero_grad()

        # Accumulate gradients over time steps t=t, t+1, ..., T-1
        for t in range(T):
            s_t = torch.from_numpy(observations[t][None, ...]).to(device=self.device, dtype=torch.float32)  # [1, state_shape]
            a_t = torch.tensor(actions[t]).to(device=self.device, dtype=torch.int64)  # [1]

            # Calculate returns from t=t, t+1, ..., T-1, notice we're doing it backwards
            g_t = 0
            for i in reversed(range(t, T)):
                g_t = rewards[i] + self.discount * g_t

            pi_logits = self.policy_network(s_t)
            m = Categorical(logits=pi_logits)
            logprob_a_t = m.log_prob(a_t)

            # Negative sign to indicate we want to maximize the policy gradient objective function
            policy_loss = -(self.discount**t * g_t * logprob_a_t)

            policy_loss = torch.mean(policy_loss, dim=0)
            policy_loss.backward()

        self.optimizer.step()

    @torch.no_grad()
    def choose_action(self, observation):
        """Given an environment observation, returns an action according to the policy."""
        s_t = torch.from_numpy(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        pi_logits = self.policy_network(s_t)
        a_t = Categorical(logits=pi_logits).sample()
        return a_t.cpu().item()


class PolicyGreedyActor:
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
        pi_logits = self.policy_network(s_t)
        pi_probs = torch.softmax(pi_logits, dim=-1)
        a_t = torch.argmax(pi_probs, dim=-1)
        return a_t.cpu().item()


def run_train_loop(env, agent, num_train_steps):
    """Run training for some steps."""
    trackers = trackers_lib.make_default_trackers()

    episode_sequence = []
    s_t = env.reset()
    for _ in range(num_train_steps):
        a_t = agent.act(s_t)

        # Take the action in the environment and observe successor state and reward.
        s_tp1, r_t, done, info = env.step(a_t)

        for tracker in trackers:
            tracker.step(r_t, done)

        episode_sequence.append((s_t, a_t, r_t))

        s_t = s_tp1
        if done:
            agent.update(episode_sequence)
            del episode_sequence[:]
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
    """Trains REINFORCE agent.

    For every iteration, the code does these in sequence:
        1. Run train agent for num_train_episodes and periodically update policy_network parameters
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

    logging.info(f'Runs REINFORCE agent on {device}')

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

    # Initialize policy_network and optimizer
    policy_network = PolicyMlpNet(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.learning_rate)

    # Create training and evaluation agent instances
    train_agent = ReinforceAgent(
        policy_network=policy_network,
        optimizer=optimizer,
        discount=FLAGS.discount,
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
