"""Trackers to collect statistics during training or evaluation."""

import collections
import timeit
import numpy as np


class EpisodeTracker:
    """Tracks episode return and other statistics."""

    def __init__(self):
        self._num_steps_since_reset = None
        self._episode_returns = None
        self._episode_steps = None
        self._current_episode_rewards = None
        self._current_episode_step = None

    def step(self, reward, done) -> None:
        """Accumulates statistics from timestep."""
        self._current_episode_rewards.append(reward)

        self._num_steps_since_reset += 1
        self._current_episode_step += 1

        if done:
            self._episode_returns.append(sum(self._current_episode_rewards))
            self._episode_steps.append(self._current_episode_step)
            self._current_episode_rewards = []
            self._current_episode_step = 0

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0
        self._episode_returns = []
        self._episode_steps = []
        self._current_episode_step = 0
        self._current_episode_rewards = []

    def get(self):
        """Aggregates statistics and returns as a dictionary.
        Here the convention is `episode_return` is set to `current_episode_return`
        if a full episode has not been encountered. Otherwise it is set to
        `mean_episode_return` which is the mean return of complete episodes only. If
        no steps have been taken at all, `episode_return` is set to `NaN`.
        Returns:
          A dictionary of aggregated statistics.
        """
        if len(self._episode_returns) > 0:
            mean_episode_return = np.array(self._episode_returns).mean()
        else:
            if self._num_steps_since_reset > 0:
                current_episode_return = sum(self._current_episode_rewards)
            else:
                mean_episode_return = current_episode_return = np.nan
            mean_episode_return = current_episode_return

        return {
            'mean_episode_return': mean_episode_return,
            'num_episodes': len(self._episode_returns),
            'current_episode_step': self._current_episode_step,
            'num_steps_since_reset': self._num_steps_since_reset,
        }


class StepRateTracker:
    """Tracks step rate, number of steps taken and duration since last reset."""

    def __init__(self):
        self._num_steps_since_reset = None
        self._start = None

    def step(self, reward, done) -> None:
        del (reward, done)
        self._num_steps_since_reset += 1

    def reset(self) -> None:
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def get(self):
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            'step_rate': step_rate,
            'num_steps': self._num_steps_since_reset,
            'duration': duration,
        }


def make_default_trackers():
    trackers = [
        EpisodeTracker(),
        StepRateTracker(),
    ]

    reset_trackers(trackers)

    return trackers


def reset_trackers(trackers):
    for tracker in trackers:
        tracker.reset()


def generate_statistics(trackers):
    """Generates statistics from a sequence of timestep and actions."""
    # Merge all statistics dictionaries into one.
    statistics_dicts = (tracker.get() for tracker in trackers)
    return dict(collections.ChainMap(*statistics_dicts))
