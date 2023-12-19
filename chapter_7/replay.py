# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Replay components for DQN-type agents."""
from typing import Any, Callable, Iterable, Optional, Sequence, NamedTuple, Tuple, TypeVar

import numpy as np
import snappy

CompressedArray = Tuple[bytes, Tuple, np.dtype]

# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar('ReplayStructure', bound=Tuple[Any, ...])


class Transition(NamedTuple):
    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    r_t: Optional[float]
    s_tp1: Optional[np.ndarray]
    done_tp1: Optional[float]


TransitionStructure = Transition(s_t=None, a_t=None, r_t=None, s_tp1=None, done_tp1=None)


def compress_array(array: np.ndarray) -> CompressedArray:
    """Compresses a numpy array with snappy."""
    return snappy.compress(array), array.shape, array.dtype


def uncompress_array(compressed: CompressedArray) -> np.ndarray:
    """Uncompresses a numpy array with snappy given its shape and dtype."""
    compressed_array, shape, dtype = compressed
    byte_string = snappy.uncompress(compressed_array)
    return np.frombuffer(byte_string, dtype=dtype).reshape(shape)


class UniformReplay:
    """Uniform replay, with circular buffer storage for flat named tuples."""

    def __init__(
        self,
        capacity: int,
        structure: ReplayStructure,
        random_state: np.random.RandomState,
        encoder: Optional[Callable[[ReplayStructure], Any]] = None,
        decoder: Optional[Callable[[Any], ReplayStructure]] = None,
    ):
        if capacity <= 0:
            raise ValueError(f'Expect capacity to be a positive integer, got {capacity}')
        self._structure = structure
        self._capacity = capacity
        self._random_state = random_state
        self._encoder = encoder or (lambda s: s)
        self._decoder = decoder or (lambda s: s)

        self._storage = [None] * capacity
        self._num_added = 0

    def add(self, item: ReplayStructure) -> None:
        """Adds single item to replay."""

        item_id = self._num_added % self._capacity
        self._storage[item_id] = self._encoder(item)
        self._num_added += 1

    def get(self, indices: Sequence[int]) -> Iterable[ReplayStructure]:
        """Retrieves items by IDs."""
        return [self._decoder(self._storage[i]) for i in indices]

    def sample(self, size: int) -> ReplayStructure:
        """Samples batch of items from replay uniformly, with replacement."""
        if self.size < size:
            raise RuntimeError(f'Replay only have {self.size} samples, got sample size {size}')

        indices = self._random_state.randint(self.size, size=size)
        samples = self.get(indices)
        transposed = zip(*samples)
        stacked = [np.stack(xs, axis=0) for xs in transposed]
        return type(self._structure)(*stacked)  # pytype: disable=not-callable

    @property
    def size(self) -> int:
        """Number of items currently contained in the replay."""
        return min(self._num_added, self._capacity)

    @property
    def capacity(self) -> int:
        """Total capacity of replay (max number of items stored at any one time)."""
        return self._capacity
