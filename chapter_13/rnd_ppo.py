"""
Implements distributed PPO algorithm with RND module for curiosity-driven exploration.
Note currently only supports running on a single machine.
"""
import random
import numpy as np
import pickle
import torch
from torch.distributions import Categorical

import utils
import trackers as trackers_lib
from normalizer import RunningMeanStd, TorchRunningMeanStd


class RNDPPOLeanerAgent:
    def __init__(
        self,
        policy_network,
        learning_rate,
        rnd_target_network,
        rnd_predictor_network,
        rnd_learning_rate,
        ext_discount,
        int_discount,
        rnd_experience_proportion,
        gae_lambda,
        value_coef,
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
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.rnd_target_network = rnd_target_network.to(device=self.device)
        self.rnd_predictor_network = rnd_predictor_network.to(device=self.device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd_predictor_network.parameters(), lr=rnd_learning_rate)

        self.policy_network.train()
        self.rnd_predictor_network.train()

        # RND target is fixed
        self.rnd_target_network.eval()
        # Disable autograd for RND target networks.
        for p in self.rnd_target_network.parameters():
            p.requires_grad = False

        self.rnd_experience_proportion = rnd_experience_proportion

        self.ext_discount = ext_discount
        self.int_discount = int_discount
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.clip_epsilon_schedule = clip_epsilon_schedule

        self.clip_grad = clip_grad
        self.max_grad_norm = max_grad_norm

        # Accumulate running statistics to calculate mean and std online
        self.int_reward_rms = RunningMeanStd(shape=(1,))
        self.rnd_obs_rms = (
            TorchRunningMeanStd(shape=(1, 84, 84), device=self.device)
            if self.device != 'cpu'
            else RunningMeanStd(shape=(1, 84, 84))
        )  # channel first and we only normalize one frame

        # Counters and statistics
        self.step_t = -1
        self.update_t = 0

    def initialize_obs_rms(self, env, num_steps):
        random_obs = []
        env.reset()
        for i in range(num_steps):
            a_t = random.choice(range(0, env.action_space.n))
            s_t, _, done, _ = env.step(a_t)

            # RND networks only takes in one frame
            random_obs.append(s_t[-1:, :, :])

            if done:
                env.reset()

        self.normalize_rnd_obs(random_obs, True)

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
                self.update_rnd_predictor_net(mini_batch)

                self.update_t += 1

    @torch.no_grad()
    def get_transitions_from_sequences(self, sequence_lists):
        transitions = []

        for sequence in sequence_lists:
            (observations, actions, logprob_actions, ext_values, int_values, rewards, dones) = map(list, zip(*sequence))

            s_t = observations[:-1]
            a_t = actions[:-1]
            logprob_a_t = logprob_actions[:-1]
            ext_v_t = ext_values[:-1]
            ext_r_t = rewards[1:]

            ext_v_tp1 = ext_values[1:]
            done_tp1 = dones[1:]

            int_v_t = int_values[:-1]
            int_v_tp1 = int_values[1:]

            # Compute extrinsic returns and advantages
            (ext_return_t, ext_advantage_t) = self.compute_returns_and_advantages(
                ext_v_t, ext_r_t, ext_v_tp1, done_tp1, self.ext_discount
            )

            # Get observation for RND, note we only need last frame
            rnd_s_t = [s[-1:, ...] for s in s_t]

            # Compute intrinsic rewards
            int_r_t = self.compute_int_reward(rnd_s_t)

            # Compute intrinsic returns and advantages
            (int_return_t, int_advantage_t) = self.compute_returns_and_advantages(
                int_v_t,
                int_r_t,
                int_v_tp1,
                np.zeros_like(done_tp1),  # No dones for intrinsic reward.
                self.int_discount,
            )

            # Zip multiple lists into list of tuples
            zipped_sequence = list(
                zip(s_t, a_t, logprob_a_t, ext_return_t, ext_advantage_t, rnd_s_t, int_return_t, int_advantage_t)
            )

            transitions.extend(zipped_sequence)

        return transitions

    @torch.no_grad()
    def compute_int_reward(self, rnd_s_t):
        normed_s_t = self.normalize_rnd_obs(rnd_s_t)

        # normed_s_t = torch.from_numpy(np.stack(rnd_s_t, axis=0)).to(device=self.device, dtype=torch.float32)
        normed_s_t = normed_s_t.to(device=self.device, dtype=torch.float32)

        pred = self.rnd_predictor_network(normed_s_t)
        target = self.rnd_target_network(normed_s_t)

        int_r_t = torch.square(pred - target).mean(dim=1).detach().cpu().numpy()

        # Normalize intrinsic reward
        normed_int_r_t = self.normalize_int_rewards(int_r_t)

        return normed_int_r_t

    def compute_returns_and_advantages(self, v_t, r_t, v_tp1, done_tp1, discount):
        v_t = np.stack(v_t, axis=0)
        r_t = np.stack(r_t, axis=0)
        v_tp1 = np.stack(v_tp1, axis=0)
        done_tp1 = np.stack(done_tp1, axis=0)

        discount_tp1 = (~done_tp1).astype(np.float32) * discount

        lambda_ = np.ones_like(discount_tp1) * self.gae_lambda  # If scalar, make into vector.

        delta_t = r_t + discount_tp1 * v_tp1 - v_t

        advantage_t = np.zeros_like(delta_t, dtype=np.float32)

        gae_t = 0
        for i in reversed(range(len(delta_t))):
            gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
            advantage_t[i] = gae_t

        return_t = advantage_t + v_t

        # advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)

        return return_t, advantage_t

    def update_rnd_predictor_net(self, samples):
        self.rnd_optimizer.zero_grad()

        # Unpack list of tuples into separate lists
        _, _, _, _, _, rnd_s_t, _, _ = map(list, zip(*samples))

        normed_s_t = self.normalize_rnd_obs(rnd_s_t, True)
        # normed_s_t = torch.from_numpy(np.stack(normed_s_t, axis=0)).to(device=self.device, dtype=torch.float32)
        normed_s_t = normed_s_t.to(device=self.device, dtype=torch.float32)

        pred_t = self.rnd_predictor_network(normed_s_t)
        with torch.no_grad():
            target_t = self.rnd_target_network(normed_s_t)

        rnd_loss = torch.square(pred_t - target_t).mean(dim=1)

        # Proportion of experience used for train RND predictor
        if self.rnd_experience_proportion < 1:
            mask = torch.rand(rnd_loss.size())
            mask = torch.where(mask < self.rnd_experience_proportion, 1.0, 0.0).to(device=self.device, dtype=torch.float32)
            rnd_loss = rnd_loss * mask

        # Averaging over batch dimension
        rnd_loss = torch.mean(rnd_loss)

        # Compute gradients
        rnd_loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.rnd_predictor_network.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

        # Update parameters
        self.rnd_optimizer.step()

    def update_policy_net(self, mini_batch):
        self.policy_optimizer.zero_grad()

        # Unpack list of tuples into separate lists
        (s_t, a_t, logprob_a_t, ext_return_t, ext_advantage_t, _, int_return_t, int_advantage_t) = map(list, zip(*mini_batch))

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self.device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self.device, dtype=torch.int64)
        behavior_logprob_a_t = torch.from_numpy(np.stack(logprob_a_t, axis=0)).to(device=self.device, dtype=torch.float32)
        ext_return_t = torch.from_numpy(np.stack(ext_return_t, axis=0)).to(device=self.device, dtype=torch.float32)
        ext_advantage_t = torch.from_numpy(np.stack(ext_advantage_t, axis=0)).to(device=self.device, dtype=torch.float32)
        int_return_t = torch.from_numpy(np.stack(int_return_t, axis=0)).to(device=self.device, dtype=torch.float32)
        int_advantage_t = torch.from_numpy(np.stack(int_advantage_t, axis=0)).to(device=self.device, dtype=torch.float32)

        pi_logits_t, ext_v_t, int_v_t = self.policy_network(s_t)

        pi_m = Categorical(logits=pi_logits_t)
        pi_logprob_a_t = pi_m.log_prob(a_t)
        entropy_loss = pi_m.entropy()

        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t)

        # Combine extrinsic and intrinsic advantages together
        advantage_t = 2.0 * ext_advantage_t + 1.0 * int_advantage_t
        clipped_ratio = torch.clamp(ratio, min=1.0 - self.clip_epsilon, max=1.0 + self.clip_epsilon)

        policy_loss = torch.min(ratio * advantage_t.detach(), clipped_ratio * advantage_t.detach())

        ext_v_loss = 0.5 * torch.square(ext_return_t - ext_v_t.squeeze(-1))
        int_v_loss = 0.5 * torch.square(int_return_t - int_v_t.squeeze(-1))

        value_loss = ext_v_loss + int_v_loss

        # Averaging over batch dimension
        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)
        value_loss = torch.mean(value_loss)

        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + self.entropy_coef * entropy_loss) + self.value_coef * value_loss

        # Compute gradients
        loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

        # Update parameters
        self.policy_optimizer.step()

    @torch.no_grad()
    def normalize_rnd_obs(self, rnd_obs_list, update_stats=False):
        # GPU could be much faster
        if isinstance(self.rnd_obs_rms, TorchRunningMeanStd):
            tacked_obs = torch.from_numpy(np.stack(rnd_obs_list, axis=0)).to(device=self.device, dtype=torch.float32)
            if update_stats:
                self.rnd_obs_rms.update(tacked_obs)

            normed_obs = self.rnd_obs_rms.normalize(tacked_obs)
            normed_obs = normed_obs.clamp(-5, 5)

            return normed_obs
        else:
            normed_frames = []
            for obs in rnd_obs_list:
                if update_stats:
                    self.rnd_obs_rms.update(obs)
                normed_obs = self.rnd_obs_rms.normalize(obs)
                normed_obs = normed_obs.clip(-5, 5)
                normed_frames.append(normed_obs)

            return torch.from_numpy(np.stack(normed_frames, axis=0)).to(dtype=torch.float32)

    def normalize_int_rewards(self, int_rewards):
        """Compute returns then normalize the intrinsic reward based on these returns"""

        # From https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/ppo_agent.py#L257
        intrinsic_returns = []
        rewems = 0
        for t in reversed(range(len(int_rewards))):
            rewems = rewems * self.int_discount + int_rewards[t]
            intrinsic_returns.insert(0, rewems)
        self.int_reward_rms.update(np.ravel(intrinsic_returns).reshape(-1, 1))

        normed_int_rewards = int_rewards / np.sqrt(self.int_reward_rms.var + 1e-8)

        return normed_int_rewards.tolist()

    @property
    def clip_epsilon(self):
        """Call external clip epsilon scheduler"""
        return self.clip_epsilon_schedule(self.step_t)

    def get_policy_state_dict(self):
        # To keep things consistent, we move the parameters to CPU
        return {k: v.cpu() for k, v in self.policy_network.state_dict().items()}


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

        # We need to store the outputs from the environment,
        # in case the last "run_steps()" call ends in the middle of an episode,
        # and we want to continue from where we left in the next "run_steps()" call
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
        self.done = True
        self.loss_life = False

    def run_steps(self, num_steps):
        sequence = []

        if self.state is None:
            self.reset_env()

        while len(sequence) < num_steps:
            action, logprob, ext_value, int_value = self.choose_action(self.state)
            sequence.append((self.state, action, logprob, ext_value, int_value, self.reward, self.done or self.loss_life))

            s_tp1, r_t, done, info = self.env.step(action)
            self.step_t += 1

            # Only keep track of non-clipped/unscaled raw reward when collecting statistics
            raw_reward = r_t
            if 'raw_reward' in info and isinstance(info['raw_reward'], (float, int)):
                raw_reward = info['raw_reward']

            # For Atari games, check if treat loss a life as a soft-terminal state
            self.loss_life = False
            if 'loss_life' in info and info['loss_life'] is True:
                self.loss_life = True

            for tracker in self.trackers:
                tracker.step(raw_reward, done, info)

            self.state = s_tp1
            self.reward = r_t
            self.done = done

            if done:
                action, logprob, ext_value, int_value = self.choose_action(self.state)
                sequence.append((self.state, action, logprob, ext_value, int_value, self.reward, self.done))
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
        pi_logits, ext_value, int_value = self.policy_network(s_t)
        pi_m = Categorical(logits=pi_logits)
        a_t = pi_m.sample()
        pi_logprob_a_t = pi_m.log_prob(a_t)

        return (
            a_t.squeeze(0).cpu().item(),
            pi_logprob_a_t.squeeze(0).cpu().item(),
            ext_value.squeeze(0).cpu().item(),
            int_value.squeeze(0).cpu().item(),
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
        """Given an environment observation and exploration rate,
        returns an action according to the epsilon-greedy policy."""
        s_t = torch.tensor(observation[None, ...]).to(device=self.device, dtype=torch.float32)
        pi_logits, _, _ = self.policy_network(s_t)
        pi_m = Categorical(logits=pi_logits)
        a_t = pi_m.sample()
        return a_t.cpu().item()


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
    actor.trackers = trackers_lib.make_default_trackers(f'{actor.env_name}-RNDPPO-seed{seed}-actor{actor.id}')
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
                tracker.step(raw_reward, done, info)

            if done:
                break

    return trackers_lib.generate_statistics(trackers)
