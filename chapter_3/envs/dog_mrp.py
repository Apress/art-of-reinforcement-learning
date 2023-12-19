"""Class for service dog MRP environment."""

from typing import NamedTuple


class StateSpace(NamedTuple):
    room1: int = 0
    room2: int = 1
    room3: int = 2
    outside: int = 3
    found_item: int = 4
    end: int = 5


class DogMRPEnv:

    """A very simple environment for the service dog MRP example."""

    def __init__(self):
        self.state_space = StateSpace()

        # Get meaningful names in the format of {id: name}
        self.state_names = dict((v, k) for k, v in self.state_space._asdict().items())

        self.states = list(self.state_names.keys())
        self.num_states = len(list(self.state_names.keys()))

        # The dynamics function is a dict with state as key, a array for possible next state.
        # where each entry in the array is a tuple of (immediate_reward, [(transition_probability, state_tp1)]
        self.dynamics = {
            self.state_space.room1: (
                -1,
                [
                    (0.2, self.state_space.room1),
                    (0.8, self.state_space.room2),
                ],
            ),
            self.state_space.room2: (
                -1,
                [
                    (0.2, self.state_space.room1),
                    (0.4, self.state_space.room3),
                    (0.4, self.state_space.outside),
                ],
            ),
            self.state_space.room3: (
                -1,
                [
                    (0.2, self.state_space.room2),
                    (0.8, self.state_space.found_item),
                ],
            ),
            self.state_space.outside: (
                1,
                [
                    (0.2, self.state_space.room2),
                    (0.8, self.state_space.outside),
                ],
            ),
            self.state_space.found_item: (
                10,
                [
                    (1.0, self.state_space.end),
                ],
            ),
            self.state_space.end: (
                0,
                [  # just a place holder for terminal state
                    (1.0, self.state_space.end),
                ],
            ),
        }

    def get_states(self):
        """Returns all states in the state space."""
        return self.states

    def get_legal_actions(self, state):
        """Returns all legal actions for a given state."""
        return list(self.dynamics[state].keys())

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
        return state == self.state_space.end
