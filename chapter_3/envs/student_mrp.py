"""Class for student MRP environment in the David Silver RL course using DP algorithms.

Course slides link at:
https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf
"""

from typing import NamedTuple


class StateSpace(NamedTuple):
    class1: int = 0
    class2: int = 1
    class3: int = 2
    pub: int = 3
    passed: int = 4
    facebook: int = 5
    sleep: int = 6


class StudentMRPEnv:

    """A very simple environment for the student MRP example in the David Silver RL course."""

    def __init__(self):
        self.state_space = StateSpace()

        # Get meaningful names in the format of {id: name}
        self.state_names = dict((v, k) for k, v in self.state_space._asdict().items())

        self.states = list(self.state_names.keys())

        self.num_states = len(self.states)

        # The dynamics function is a dict with state as key, a array for possible next state.
        # where each entry in the array is a tuple of (immediate_reward, [(transition_probability, state_tp1)]
        self.dynamics = {
            self.state_space.class1: (
                -2,
                [
                    (0.5, self.state_space.class2),
                    (0.5, self.state_space.facebook),
                ],
            ),
            self.state_space.class2: (
                -2,
                [
                    (0.8, self.state_space.class3),
                    (0.2, self.state_space.sleep),
                ],
            ),
            self.state_space.class3: (
                -2,
                [
                    (0.4, self.state_space.pub),
                    (0.6, self.state_space.passed),
                ],
            ),
            self.state_space.pub: (
                1,
                [
                    (0.2, self.state_space.class1),
                    (0.4, self.state_space.class2),
                    (0.4, self.state_space.class3),
                ],
            ),
            self.state_space.passed: (
                10,
                [(1.0, self.state_space.sleep)],
            ),
            self.state_space.facebook: (
                -1,
                [
                    (0.1, self.state_space.class1),
                    (0.9, self.state_space.facebook),
                ],
            ),
            self.state_space.sleep: (
                0,
                [(1.0, self.state_space.sleep)],
            ),  # just a place holder for terminal state
        }

    def get_states(self):
        """Returns all states in the state space."""
        return self.states

    def transition_from_state(self, state):
        """Given a state, use the dynamics function to
        transition into all possible next states."""
        reward, state_transitions = self.dynamics[state]
        return reward, state_transitions

    def get_state_name(self, state):
        """Returns the state name for a given state id."""
        return self.state_names[state]

    def is_terminal_state(self, state):
        """Returns true if the given state is terminal state, false otherwise."""
        return state == self.state_space.sleep
