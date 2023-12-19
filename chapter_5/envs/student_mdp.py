"""Class for student MDP environment in the David Silver RL course using DP algorithms.

Course slides link at:
https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf
"""

from typing import NamedTuple
import random


class StateSpace(NamedTuple):
    facebook: int = 0
    class1: int = 1
    class2: int = 2
    class3: int = 3
    sleep: int = 4


class ActionSpace(NamedTuple):
    facebook: int = 0
    study: int = 1
    quit: int = 2
    sleep: int = 3
    pub: int = 4


class StudentMDPEnv:

    """A very simple environment for student MDP example in the David Silver RL course."""

    def __init__(self):
        self.state_space = StateSpace()
        self.action_space = ActionSpace()

        # Get meaningful names in the format of {id: name}
        self.state_names = dict((v, k) for k, v in self.state_space._asdict().items())
        self.action_names = dict((v, k) for k, v in self.action_space._asdict().items())

        self.states = list(self.state_names.keys())
        self.actions = list(self.action_names.keys())

        self.num_states = len(self.states)
        self.num_actions = len(self.actions)

        # The dynamics function is a dict with state as key.
        # For each state, there's a sub-dict with legal actions as key and tuples of (immediate_reward, [(transition_probability, state_tp1)]).
        self.dynamics = {
            self.state_space.facebook: {
                self.action_space.facebook: (
                    -1,
                    [(1, self.state_space.facebook)],
                ),
                self.action_space.quit: (0, [(1, self.state_space.class1)]),
            },
            self.state_space.class1: {
                self.action_space.facebook: (
                    -1,
                    [(1, self.state_space.facebook)],
                ),
                self.action_space.study: (-2, [(1, self.state_space.class2)]),
            },
            self.state_space.class2: {
                self.action_space.study: (-2, [(1, self.state_space.class3)]),
                self.action_space.sleep: (0, [(1, self.state_space.sleep)]),
            },
            self.state_space.class3: {
                self.action_space.study: (10, [(1, self.state_space.sleep)]),
                self.action_space.pub: (
                    1,
                    [  # stochastic transition for pub action when in class 3
                        (0.2, self.state_space.class1),
                        (0.4, self.state_space.class2),
                        (0.4, self.state_space.class3),
                    ],
                ),
            },
            self.state_space.sleep: {  # just a place holder for terminal state
                self.action_space.sleep: (0, [(1, self.state_space.sleep)]),
            },
        }

        self.terminal_state = self.state_space.sleep

        self.current_state = None
        self.steps = 0

    def reset(self):
        """Reset the environment and returns the initial state of the environment."""
        self.current_state = self.state_space.class1
        self.steps = 0

        return self.current_state

    def step(self, action):
        """Make a move in the environment.

        Args:
            action: the action agent chose.

        Returns:
            state_tp1: next state of the environment.
            reward: the immediate reward from R(s, a, s').
            done: true if task is terminated, false otherwise.
        """

        if self.current_state is None or self.is_done():
            raise RuntimeError('Call reset before continue')
        if action not in self.get_legal_actions(self.current_state):
            raise ValueError(f'Invalid action {action}')

        reward, state_transitions = self.dynamics[self.current_state][action]
        # Randomly transition into one of the possible next state, if it's stochastic environment.
        # for deterministic environment, there's only one next state.
        transition = random.choice(state_transitions)

        _, state_tp1 = transition
        self.current_state = state_tp1
        self.steps += 1

        return state_tp1, reward, self.is_done()

    def sample_action(self):
        """Sample a action from all the available actions for current state."""
        legal_actions = self.get_legal_actions(self.current_state)
        action = random.choice(legal_actions)
        return action

    def transition_from_state_action(self, state, action):
        """For DP algorithms only.

        Given a state and action, use the dynamics function to
        transition into all possible next states."""
        reward, state_transitions = self.dynamics[state][action]
        return reward, state_transitions

    def get_states(self):
        """Returns all states in the state space."""
        return self.states

    def get_actions(self):
        """Returns all actions in the action space."""
        return self.actions

    def get_legal_actions(self, state):
        """Returns all legal actions for a given state."""
        return list(self.dynamics[state].keys())

    def get_state_name(self, state):
        """Returns the state name for a given state id."""
        return self.state_names[state]

    def get_action_name(self, action):
        """Returns the action name for a given action id."""
        return self.action_names[action]

    def is_terminal_state(self, state):
        """Returns true if the given state is terminal state, false otherwise."""
        return state == self.terminal_state

    def is_done(self):
        """Returns true if the current state is terminal state, false otherwise."""
        return self.current_state == self.terminal_state
