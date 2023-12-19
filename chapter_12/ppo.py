"""
Implements distributed PPO algorithm.
Note currently only supports running on a single machine.
"""
import numpy as np
import pickle

import torch
from torch.distributions import Categorical, Normal

import utils
import trackers as trackers_lib


class PPOLeanerAgent:
    def __init__(
        self,
        policy_network,
        policy_lr,
        value_network,
        value_lr,
        discount,
        gae_lambda,
        entropy_coef,
        num_epochs,
        batch_size,
        clip_epsilon_schedule,
        clip_grad,
        max_grad_norm,
        device,
    ):
        self.device = device

        self.policy_network = policy_network.to(device=self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)
        self.value_network = value_network.to(device=self.device)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=value_lr)

        self.policy_network.train()
        self.value_network.train()

        self.discount = discount
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.clip_epsilon_schedule = clip_epsilon_schedule

        self.clip_grad = clip_grad
        self.max_grad_norm = max_grad_norm

        # Counters and statistics
        self.step_t = 0
        self.update_t = 0

    def update(self, sequence_lists):
        self.step_t += 1

        transitions = self.get_transitions_from_sequences(sequence_lists)

        # Run M epochs to update network parameters
        for _ in range(self.num_epochs):
            # Split sequence into batches
            batch_indices = utils.split_indices_into_bins(self.batch_size, len(transitions), shuffle=True)

            for indices in batch_indices:
                mini_batch = [transitions[i] for i in indices]
                self.update_policy_net(mini_batch)
                self.update_value_net(mini_batch)
                self.update_t += 1

    @torch.no_grad()
    def get_transitions_from_sequences(self, sequence_lists):
        transitions = []

        for sequence in sequence_lists:
            (observations, actions, logprob_actions, rewards, dones) = map(list, zip(*sequence))

            # Get predicted state values
            # In case of using separate nets for policy and value functions,
            # the actors only have policy net, so no values available when they made the decision
            states = torch.from_numpy(np.stack(observations, axis=0)).to(device=self.device, dtype=torch.float32)
            values = self.value_network(states).squeeze(-1).cpu().numpy()

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
            zipped_sequence = list(zip(s_t, a_t, logprob_a_t, return_t, advantage_t))

            transitions.extend(zipped_sequence)

        return transitions

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
        s_t, a_t, logprob_a_t, _, advantage_t = map(list, zip(*mini_batch))

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self.device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self.device, dtype=torch.int64)
        behavior_logprob_a_t = torch.from_numpy(np.stack(logprob_a_t, axis=0)).to(device=self.device, dtype=torch.float32)
        advantage_t = torch.from_numpy(np.stack(advantage_t, axis=0)).to(device=self.device, dtype=torch.float32)

        # Given past states, get predicted action probabilities
        pi_logits_t = self.policy_network(s_t)
        pi_m = Categorical(logits=pi_logits_t)
        pi_logprob_a_t = pi_m.log_prob(a_t)
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

        policy_loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

        self.policy_optimizer.step()

    def update_value_net(self, mini_batch):
        self.value_optimizer.zero_grad()

        # Unpack list of tuples into separate lists
        s_t, _, _, return_t, _ = map(list, zip(*mini_batch))

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self.device, dtype=torch.float32)
        return_t = torch.from_numpy(np.stack(return_t, axis=0)).to(device=self.device, dtype=torch.float32)

        v_t = self.value_network(s_t)

        value_loss = 0.5 * torch.square(return_t - v_t.squeeze(-1))

        # Averaging over batch dimension
        value_loss = torch.mean(value_loss)

        value_loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.value_network.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

        self.value_optimizer.step()

    @property
    def clip_epsilon(self):
        """Call external clip epsilon scheduler"""
        return self.clip_epsilon_schedule(self.step_t)

    def get_policy_state_dict(self):
        # To keep things consistent, we move the parameters to CPU
        return {k: v.cpu() for k, v in self.policy_network.state_dict().items()}

    def get_value_state_dict(self):
        # To keep things consistent, we move the parameters to CPU
        return {k: v.cpu() for k, v in self.value_network.state_dict().items()}


class ContinuousPPOLeanerAgent(PPOLeanerAgent):
    """PPO learner agent for robotic control tasks"""

    def update_policy_net(self, mini_batch):
        # Unpack list of tuples into separate lists
        s_t, a_t, logprob_a_t, _, advantage_t = map(list, zip(*mini_batch))

        self.policy_optimizer.zero_grad()

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self.device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self.device, dtype=torch.float32)
        behavior_logprob_a_t = torch.from_numpy(np.stack(logprob_a_t, axis=0)).to(device=self.device, dtype=torch.float32)
        advantage_t = torch.from_numpy(np.stack(advantage_t, axis=0)).to(device=self.device, dtype=torch.float32)

        pi_mu_t, pi_sigma_t = self.policy_network(s_t)
        pi_m = Normal(pi_mu_t, pi_sigma_t)
        pi_logprob_a_t = pi_m.log_prob(a_t).sum(axis=-1)
        entropy_loss = pi_m.entropy()

        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t.detach())
        clipped_ratio = torch.clamp(ratio, min=1.0 - self.clip_epsilon, max=1.0 + self.clip_epsilon)

        # clipped_ratio = torch.where(advantage_t > 0, 1.0 + self.clip_epsilon, 1.0 - self.clip_epsilon)

        policy_loss = torch.min(ratio * advantage_t.detach(), clipped_ratio * advantage_t.detach())

        # Averaging over batch dimension
        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)

        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        policy_loss = -(policy_loss + self.entropy_coef * entropy_loss)

        policy_loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

        self.policy_optimizer.step()


class PPOActor:
    """PPO actor for distributed training architecture."""

    def __init__(
        self,
        id,
        policy_network,
        device,
        env,
        trackers,
    ):
        self.id = id
        self.device = device
        self.policy_network = policy_network.to(device=self.device)
        self.env = env
        self.env_name = env.spec.id
        self.trackers = trackers

        self.state = None
        self.reward = None
        self.done = None
        self.loss_life = None

        self.step_t = 0
        self.num_episodes = 0

    def reset_trackers(self):
        trackers_lib.reset_trackers(self.trackers)

    def get_stats(self):
        return trackers_lib.generate_statistics(self.trackers)

    def reset_env(self):
        self.state = self.env.reset()
        self.reward = 0
        self.done = False
        self.loss_life = False

    def run_steps(self, num_steps):
        sequence = []

        if self.state is None:
            self.reset_env()

        while len(sequence) < num_steps:
            action, logprob = self.choose_action(self.state)
            sequence.append((self.state, action, logprob, self.reward, self.done or self.loss_life))

            s_tp1, r_t, done, info = self.env.step(action)
            self.step_t += 1

            # Only keep track of non-clipped/unscaled raw reward when collecting statistics
            raw_reward = r_t
            if 'raw_reward' in info and isinstance(info['raw_reward'], (float, int)):
                raw_reward = info['raw_reward']

            # For Atari games, check if treat loss a life as a soft-terminal state
            self.loss_life = False
            if 'loss_life' in info and isinstance(info['loss_life'], bool):
                self.loss_life = info['loss_life']

            for tracker in self.trackers:
                tracker.step(raw_reward, done)

            # We need to store the outputs from the environment,
            # in case the last "run_steps()" call ends in the middle of an episode,
            # and we want to continue from where we left in the next "run_steps()" call
            self.state = s_tp1
            self.reward = r_t
            self.done = done

            if done:
                action, logprob = self.choose_action(self.state)
                sequence.append((self.state, action, logprob, self.reward, self.done))
                self.reset_env()
                self.num_episodes += 1

        return sequence

    def update_policy_params(self, state_dict):
        # The stat dict from shared params is always on CPU
        if self.device != 'cpu':
            state_dict = {k: v.to(device=self.device) for k, v in state_dict.items()}

        self.policy_network.load_state_dict(state_dict)

    @torch.no_grad()
    def choose_action(self, observation):
        """Given an environment observation, returns an action according to the policy."""
        s_t = torch.from_numpy(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        pi_logits, _ = self.policy_network(s_t)
        pi_m = Categorical(logits=pi_logits)
        a_t = pi_m.sample()
        pi_logprob_a_t = pi_m.log_prob(a_t)

        return (
            a_t.squeeze(0).cpu().item(),
            pi_logprob_a_t.squeeze(0).cpu().item(),
        )


class ContinuousPPOActor(PPOActor):
    """PPO actor for distributed training architecture."""

    @torch.no_grad()
    def choose_action(self, observation):
        """Given an environment observation, returns an action according to the policy."""
        s_t = torch.from_numpy(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        pi_mu, pi_sigma = self.policy_network(s_t)
        pi_m = Normal(pi_mu, pi_sigma)
        a_t = pi_m.sample()
        pi_logprob_a_t = pi_m.log_prob(a_t).sum(axis=-1)

        return (
            a_t.squeeze(0).cpu().numpy(),
            pi_logprob_a_t.squeeze(0).cpu().numpy(),
        )


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


class ContinuousPolicyGreedyActor(PolicyGreedyActor):
    """Policy greedy actor for robotic control tasks, evaluation only."""

    @torch.no_grad()
    def choose_action(self, observation):
        """Given an environment observation, returns an action according to the policy."""
        s_t = torch.tensor(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        pi_mu, pi_sigma = self.policy_network(s_t)
        pi_m = Normal(pi_mu, pi_sigma)
        a_t = pi_m.sample()
        return a_t.squeeze(0).cpu().numpy()


def run_actor_loop(
    seed,
    id,
    num_train_steps,
    sequence_length,
    serialized_actors,
    actor_conditions,
    actor_statistics,
    shared_params,
    queue,
    counter,
):
    """Run actor loop for one iteration, which consists of num_train_steps."""
    # Temporally suppress DeprecationWarning
    import warnings

    warnings.filterwarnings('ignore', category=DeprecationWarning)

    torch.manual_seed(int(seed + id))

    actor = pickle.loads(serialized_actors[id])
    actor.reset_env()

    # A little hack to fix  "TypeError: cannot pickle '_thread.lock' object"
    # when using trackers with Tensorboard
    # So we create new trackers after initialized the process, and set the tracker steps
    actor.trackers = trackers_lib.make_default_trackers(f'{actor.env_name}-PPO-seed{seed}-actor{actor.id}')
    trackers_lib.set_tracker_steps(actor.trackers, actor.step_t, actor.num_episodes)

    actor.reset_trackers()

    num_sequence = int(num_train_steps / sequence_length)

    i = 0
    while i < num_sequence:
        if actor_conditions[id]:
            # To stay on-policy by always use latest policy to generate samples
            actor.update_policy_params(shared_params['policy'])
            # Run some steps to get a sequence of N transitions
            sequence = actor.run_steps(sequence_length)

            queue.put((id, sequence))
            queue.put((id, None))

            i += 1
            actor_conditions[id] = False

    # Update actor statistics
    actor_statistics[id] = actor.get_stats()

    actor.trackers = None

    # Persist actor internal states
    serialized_actors[id] = pickle.dumps(actor)

    with counter.get_lock():
        counter.value += 1


def run_learner_loop(
    agent,
    num_actors,
    actor_conditions,
    shared_params,
    queue,
    counter,
):
    """Runs learner for one iteration, which will end when all actors have finished their work."""
    # Wait for all actors finished one iteration
    while counter.value < num_actors:
        sequences = []
        c = 0

        while c < num_actors:
            id, data = queue.get()
            if data is not None:
                sequences.append(data)
            else:
                c += 1

        assert len(sequences) == num_actors
        agent.update(sequences)

        shared_params['policy'] = agent.get_policy_state_dict()

        # On every parameters update, reset start conditions for each actor
        for i in range(num_actors):
            actor_conditions[i] = True


def run_evaluation_loop(env, agent, num_eval_steps, trackers):
    """Run evaluation for some steps."""

    # Temporally suppress DeprecationWarning
    import warnings

    warnings.filterwarnings('ignore', category=DeprecationWarning)

    trackers_lib.reset_trackers(trackers)

    t = 0
    while t < num_eval_steps:
        s_t = env.reset()

        while True:
            a_t = agent.act(s_t)
            s_t, r_t, done, info = env.step(a_t)
            t += 1

            # Only keep track of non-clipped/unscaled raw reward when collecting statistics
            raw_reward = r_t
            if 'raw_reward' in info and isinstance(info['raw_reward'], (float, int)):
                raw_reward = info['raw_reward']

            for tracker in trackers:
                tracker.step(raw_reward, done)

            if done:
                break

    return trackers_lib.generate_statistics(trackers)
