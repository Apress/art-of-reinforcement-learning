"""Implements PPO algorithm"""
import torch
from torch.distributions import Categorical, Normal
import numpy as np

import utils
import trackers as trackers_lib


class PPOAgent:
    def __init__(
        self,
        policy_network,
        policy_optimizer,
        discount,
        gae_lambda,
        value_coef,
        entropy_coef,
        sequence_length,
        num_epochs,
        batch_size,
        clip_epsilon_schedule,
        clip_grad,
        max_grad_norm,
        device,
    ):
        self.device = device

        self.policy_network = policy_network.to(device=self.device)
        self.policy_optimizer = policy_optimizer

        self.discount = discount
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.clip_epsilon_schedule = clip_epsilon_schedule

        self.clip_grad = clip_grad
        self.max_grad_norm = max_grad_norm

        self.sequence_length = sequence_length

        self.sequence = []

        # Counters and statistics
        self.step_t = 0
        self.update_t = 0

    def act(self, observation, reward, done):
        self.step_t += 1

        a_t, logprob_a_t, v_t = self.choose_action(observation)

        self.sequence.append((observation, a_t, logprob_a_t, v_t, reward, done))

        if len(self.sequence) >= self.sequence_length:
            self.update()

        return a_t

    def update(self):
        transitions = self.get_transitions_from_sequences(self.sequence)

        # Run M epochs to update network parameters
        for _ in range(self.num_epochs):
            # Split sequence into batches
            batch_indices = utils.split_indices_into_bins(self.batch_size, len(transitions), shuffle=True)

            for indices in batch_indices:
                mini_batch = [transitions[i] for i in indices]
                self.update_policy_net(mini_batch)
                self.update_t += 1

        # Remove old transitions
        del self.sequence[:]

    def get_transitions_from_sequences(self, sequence):
        (observations, actions, logprob_actions, values, rewards, dones) = map(list, zip(*sequence))

        # Our transitions in self.sequence is actually mismatched.
        # For example, the reward is one step behind, and there's not successor states
        # so we need to offset this
        s_t = observations[:-1]
        a_t = actions[:-1]
        logprob_a_t = logprob_actions[:-1]
        v_t = values[:-1]
        r_t = rewards[1:]

        v_tp1 = values[1:]
        done_tp1 = dones[1:]

        # Compute returns and advantages
        return_t, advantage_t = self.compute_returns_and_advantages(v_t, r_t, v_tp1, done_tp1)

        # Zip multiple lists into list of tuples
        sequence = list(zip(s_t, a_t, logprob_a_t, return_t, advantage_t))

        return sequence

    def compute_returns_and_advantages(self, v_t, r_t, v_tp1, done_tp1):
        v_t = np.stack(v_t, axis=0)
        r_t = np.stack(r_t, axis=0)
        v_tp1 = np.stack(v_tp1, axis=0)
        done_tp1 = np.stack(done_tp1, axis=0)

        discount_tp1 = (~done_tp1).astype(np.float32) * self.discount

        lambda_ = np.ones_like(discount_tp1) * self.gae_lambda  # If scalar, make into vector.

        delta_t = r_t + discount_tp1 * v_tp1 - v_t

        advantage_t = np.zeros_like(delta_t, dtype=np.float32)

        gae_t = 0
        for i in reversed(range(len(delta_t))):
            gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
            advantage_t[i] = gae_t

        return_t = advantage_t + v_t

        advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)

        return return_t, advantage_t

    def update_policy_net(self, mini_batch):
        self.policy_optimizer.zero_grad()

        # Unpack list of tuples into separate lists
        s_t, a_t, logprob_a_t, return_t, advantage_t = map(list, zip(*mini_batch))

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self.device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self.device, dtype=torch.int64)  # Actions are discrete
        behavior_logprob_a_t = torch.from_numpy(np.stack(logprob_a_t, axis=0)).to(device=self.device, dtype=torch.float32)
        return_t = torch.from_numpy(np.stack(return_t, axis=0)).to(device=self.device, dtype=torch.float32)
        advantage_t = torch.from_numpy(np.stack(advantage_t, axis=0)).to(device=self.device, dtype=torch.float32)

        # Given past states, get predicted action probabilities and state value
        pi_logits_t, v_t = self.policy_network(s_t)
        pi_m = Categorical(logits=pi_logits_t)
        pi_logprob_a_t = pi_m.log_prob(a_t)
        entropy_loss = pi_m.entropy()

        # Compute clipped surrogate objective
        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t.detach())
        clipped_ratio = torch.clamp(ratio, min=1.0 - self.clip_epsilon, max=1.0 + self.clip_epsilon)

        # clipped_ratio = torch.where(advantage_t > 0, 1.0 + self.clip_epsilon, 1.0 - self.clip_epsilon)

        policy_loss = torch.min(ratio * advantage_t.detach(), clipped_ratio * advantage_t.detach())

        # Compute state value loss
        value_loss = 0.5 * torch.square(return_t - v_t.squeeze(-1))

        # Averaging over batch dimension
        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)
        value_loss = torch.mean(value_loss)

        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + self.entropy_coef * entropy_loss) + self.value_coef * value_loss

        loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

        self.policy_optimizer.step()

    @torch.no_grad()
    def choose_action(self, observation):
        """Given an environment observation, returns an action according to the policy."""
        s_t = torch.from_numpy(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        pi_logits, value = self.policy_network(s_t)
        pi_m = Categorical(logits=pi_logits)
        a_t = pi_m.sample()
        pi_logprob_a_t = pi_m.log_prob(a_t)

        return (
            a_t.squeeze(0).cpu().item(),
            pi_logprob_a_t.squeeze(0).cpu().item(),
            value.squeeze(0).cpu().item(),
        )

    @property
    def clip_epsilon(self):
        """Call external clip epsilon scheduler"""
        return self.clip_epsilon_schedule(self.step_t)


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
        pi_logits, _ = self.policy_network(s_t)
        pi_probs = torch.softmax(pi_logits, dim=-1)
        a_t = torch.argmax(pi_probs, dim=-1)
        return a_t.cpu().item()


class ContinuousPPOAgent(PPOAgent):
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
        num_epochs,
        batch_size,
        clip_epsilon_schedule,
        clip_grad,
        max_grad_norm,
        device,
    ):
        super().__init__(
            policy_network=policy_network,
            policy_optimizer=policy_optimizer,
            discount=discount,
            gae_lambda=gae_lambda,
            value_coef=0,  # separate networks
            entropy_coef=entropy_coef,
            sequence_length=sequence_length,
            num_epochs=num_epochs,
            batch_size=batch_size,
            clip_epsilon_schedule=clip_epsilon_schedule,
            clip_grad=clip_grad,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        self.value_network = value_network.to(device=self.device)
        self.value_optimizer = value_optimizer

    def update(self):
        transitions = self.get_transitions_from_sequences(self.sequence)

        # Run M epochs to update network parameters
        for _ in range(self.num_epochs):
            # Split sequence into batches
            batch_indices = utils.split_indices_into_bins(self.batch_size, len(transitions), shuffle=True)

            for indices in batch_indices:
                mini_batch = [transitions[i] for i in indices]
                self.update_policy_net(mini_batch)
                self.update_value_net(mini_batch)
                self.update_t += 1

        # Remove old transitions
        del self.sequence[:]

    def update_policy_net(self, mini_batch):
        # Unpack list of tuples into separate lists
        s_t, a_t, logprob_a_t, _, advantage_t = map(list, zip(*mini_batch))

        self.policy_optimizer.zero_grad()

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self.device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(
            device=self.device, dtype=torch.float32
        )  # Actions are continuous values
        behavior_logprob_a_t = torch.from_numpy(np.stack(logprob_a_t, axis=0)).to(device=self.device, dtype=torch.float32)
        advantage_t = torch.from_numpy(np.stack(advantage_t, axis=0)).to(device=self.device, dtype=torch.float32)

        # Given past states, get predicted action probabilities and state value
        pi_mu_t, pi_sigma_t = self.policy_network(s_t)
        pi_m = Normal(pi_mu_t, pi_sigma_t)
        pi_logprob_a_t = pi_m.log_prob(a_t).sum(axis=-1)
        entropy_loss = pi_m.entropy()

        # Compute clipped surrogate objective
        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t.detach())
        clipped_ratio = torch.clamp(ratio, min=1.0 - self.clip_epsilon, max=1.0 + self.clip_epsilon)

        # clipped_ratio = torch.where(advantage_t > 0, 1.0 + self.clip_epsilon, 1.0 - self.clip_epsilon)

        policy_loss = torch.min(ratio * advantage_t.detach(), clipped_ratio * advantage_t.detach())

        # Averaging over batch dimension
        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)

        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        policy_loss = -(policy_loss + self.entropy_coef * entropy_loss)

        # Compute gradients
        policy_loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

        # Update parameters
        self.policy_optimizer.step()

    def update_value_net(self, mini_batch):
        # Unpack list of tuples into separate lists
        s_t, _, _, return_t, _ = map(list, zip(*mini_batch))

        self.value_optimizer.zero_grad()

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self.device, dtype=torch.float32)
        return_t = torch.from_numpy(np.stack(return_t, axis=0)).to(device=self.device, dtype=torch.float32)

        # Given past states, get predicted state value
        v_t = self.value_network(s_t)

        # Compute state value loss
        value_loss = 0.5 * torch.square(return_t - v_t.squeeze(-1))

        # Averaging over batch dimension
        value_loss = torch.mean(value_loss)

        # Compute gradients
        value_loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.value_network.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

        # Update parameters
        self.value_optimizer.step()

    @torch.no_grad()
    def choose_action(self, observation):
        """Given an environment observation, returns an action according to the policy."""
        s_t = torch.tensor(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        pi_mu, pi_sigma = self.policy_network(s_t)
        value = self.value_network(s_t)
        pi_m = Normal(pi_mu, pi_sigma)
        a_t = pi_m.sample()
        pi_logprob_a_t = pi_m.log_prob(a_t).sum(axis=-1)

        return (
            a_t.squeeze(0).cpu().numpy(),
            pi_logprob_a_t.squeeze(0).cpu().numpy(),
            value.squeeze(0).cpu().item(),
        )


class ContinuousPolicyGreedyActor(PolicyGreedyActor):
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
        loss_life = False

        while True:
            a_t = agent.act(s_t, r_t, done or loss_life)

            # Take the action in the environment and observe successor state and reward.
            s_t, r_t, done, info = env.step(a_t)
            t += 1

            # Only keep track of non-clipped/unscaled raw reward when collecting statistics
            raw_reward = r_t
            if 'raw_reward' in info and isinstance(info['raw_reward'], (float, int)):
                raw_reward = info['raw_reward']

            # Atari might use soft-terminal on loss life
            loss_life = False
            if 'loss_life' in info and info['loss_life'] is True:
                loss_life = True

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
        s_t, r_t, done, info = env.step(a_t)

        # Only keep track of non-clipped/unscaled raw reward when collecting statistics
        raw_reward = r_t
        if 'raw_reward' in info and isinstance(info['raw_reward'], (float, int)):
            raw_reward = info['raw_reward']

        for tracker in trackers:
            tracker.step(raw_reward, done)

        if done:
            s_t = env.reset()

    return trackers_lib.generate_statistics(trackers)
