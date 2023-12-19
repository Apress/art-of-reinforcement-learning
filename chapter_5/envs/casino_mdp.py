"""
Environment class for Casino Slot Machines MDP example.
"""
from typing import NamedTuple
import random
import numpy as np


class StateSpace(NamedTuple):
    center_hall: int = 0
    left: int = 1
    right: int = 2
    end: int = 3


class ActionSpace(NamedTuple):
    go_left: int = 0
    go_right: int = 1
    play_a: int = 2
    play_b: int = 3
    play_c: int = 4
    play_d: int = 5
    play_e: int = 6
    play_f: int = 7
    play_g: int = 8
    play_h: int = 9
    play_i: int = 10
    play_j: int = 11
    play_k: int = 12


class CasinoMDPEnv:

    """A very simple environment for Casino Slot Machines MDP example,
    where the dynamics function is unknown to the agent."""

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
            self.state_space.center_hall: {
                self.action_space.go_left: (0, [(1, self.state_space.left)]),
                self.action_space.go_right: (0, [(1, self.state_space.right)]),
            },
            self.state_space.left: {  # We use additional logic in step() to deice the rewards when select each action in state LEFT.
                self.action_space.play_a: (0, [(1, self.state_space.end)]),
                self.action_space.play_b: (0, [(1, self.state_space.end)]),
                self.action_space.play_c: (0, [(1, self.state_space.end)]),
                self.action_space.play_d: (0, [(1, self.state_space.end)]),
                self.action_space.play_e: (0, [(1, self.state_space.end)]),
                self.action_space.play_f: (0, [(1, self.state_space.end)]),
                self.action_space.play_g: (0, [(1, self.state_space.end)]),
                self.action_space.play_h: (0, [(1, self.state_space.end)]),
                self.action_space.play_i: (0, [(1, self.state_space.end)]),
                self.action_space.play_j: (0, [(1, self.state_space.end)]),
                self.action_space.play_k: (0, [(1, self.state_space.end)]),
            },
            self.state_space.right: {  # We use additional logic in step() to deice the rewards when select each action in state RIGHT.
                self.action_space.play_a: (0, [(1, self.state_space.end)]),
                self.action_space.play_b: (0, [(1, self.state_space.end)]),
                self.action_space.play_c: (0, [(1, self.state_space.end)]),
                self.action_space.play_d: (0, [(1, self.state_space.end)]),
                self.action_space.play_e: (0, [(1, self.state_space.end)]),
                self.action_space.play_f: (0, [(1, self.state_space.end)]),
                self.action_space.play_g: (0, [(1, self.state_space.end)]),
                self.action_space.play_h: (0, [(1, self.state_space.end)]),
                self.action_space.play_i: (0, [(1, self.state_space.end)]),
                self.action_space.play_j: (0, [(1, self.state_space.end)]),
                self.action_space.play_k: (0, [(1, self.state_space.end)]),
            },
            self.state_space.end: {
                self.action_space.go_left: (0, [(1, self.state_space.end)]),
                self.action_space.go_right: (0, [(1, self.state_space.end)]),
            },
        }

        self.terminal_state = self.state_space.end

        self.current_state = None
        self.steps = 0

    def reset(self):
        """Reset the environment and returns the initial state of the environment."""
        self.current_state = self.state_space.center_hall
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

        if self.current_state == self.state_space.right and action not in (
            self.action_space.go_left,
            self.action_space.go_right,
        ):
            reward = np.random.normal(loc=-1.0, scale=5.0)
        elif self.current_state == self.state_space.left and action not in (
            self.action_space.go_left,
            self.action_space.go_right,
        ):
            reward = np.random.normal(loc=0, scale=0.1)

        self.current_state = state_tp1
        self.steps += 1
        return state_tp1, reward, self.is_done()

    def sample_action(self):
        """Sample a action from all the available actions for current state."""
        legal_actions = self.get_legal_actions(self.current_state)
        random_action = random.choice(legal_actions)
        return random_action

    def get_states(self):
        """Returns all states in the state space."""
        return self.states

    def get_actions(self):
        """Returns all actions in the action space."""
        return self.actions

    def get_legal_actions(self, state):
        """Returns all legal actions for given state."""
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

    def is_start_state(self, state):
        return state == self.state_space.center_hall

    def is_go_left(self, action):
        return action == self.action_space.go_left

    def is_go_right(self, action):
        return action == self.action_space.go_right
