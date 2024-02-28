# Copyright (c) 2023 Michael Hu
# All rights reserved.
"""MCTS implementation for AlphaZero.

The positions are evaluated from the current player (or to move) perspective.

        A           Black to move

    B       C       White to move

  D   E             Black to move

For example, in the above two-player, zero-sum games search tree. 'A' is the root node,
and when the game is in state corresponding to node 'A', it's black's turn to move.
However the children nodes of 'A' are evaluated from white player's perspective.
So if we select the best child for node 'A', without further consideration,
we'd be actually selecting the best child for white player, which is not what we want.

Let's look at an simplified example where we don't consider number of visits and total values,
just the raw evaluation scores, if the evaluated scores (from white's perspective)
for 'B' and 'C' are 0.8 and 0.3 respectively. Then according to these results,
the best child of 'A' max(0.8, 0.3) is 'B', however this is done from white player's perspective.
But node 'A' represents black's turn to move, so we need to select the best child from black player's perspective,
which should be 'C' - the worst move for white, thus a best move for black.

One way to resolve this issue is to always switching the signs of the child node's Q values when we select the best child.

For example:
    ucb_scores = -node.child_Q() + node.child_U()

In this case, a max(-0.8, -0.3) will give us the correct results for black player when we select the best child for node 'A'.

"""

import copy
import math
from typing import Callable, Tuple, Mapping, Iterable, Any
import numpy as np

from alpha_zero.envs.base import BoardGameEnv


class Node:
    """Node in the MCTS search tree."""

    def __init__(
        self, to_play: int, prior: float = None, move: int = None, parent: Any = None
    ) -> None:
        """
        Args:
            to_play: the id of the current player.
            prior: a prior probability of the node for a specific action, could be empty in case of root node.
            move: the action associated with the prior probability.
            parent: the parent node, could be `None` if this is the root node.
        """

        self.to_play = to_play
        self.prior = prior
        self.move = move
        self.parent = parent
        self.is_expanded = False

        self.N = 0  # number of visits
        self.W = 0.0  # total action values

        self.children: Mapping[int, Node] = {}

        # number of virtual losses on this node, only used in 'parallel_uct_search'
        self.losses_applied = 0

    def child_U(self, c_puct_base: float, c_puct_init: float) -> np.ndarray:
        """Returns a 1D numpy.array contains prior score for all child."""
        pb_c = math.log((1.0 + self.N + c_puct_base) / c_puct_base) + c_puct_init
        return pb_c * np.array(
            [
                child.prior * (math.sqrt(self.N) / (1 + child.N))
                for child in self.children.values()
            ],
            dtype=np.float32,
        )

    def child_Q(self) -> np.ndarray:
        """Returns a 1D numpy.array contains mean action value for all child."""
        return np.array([child.Q for child in self.children.values()], dtype=np.float32)

    @property
    def child_N(self) -> np.ndarray:
        """Returns a 1D numpy.array contains visits count for all child."""
        return np.array([child.N for child in self.children.values()], dtype=np.int32)

    @property
    def Q(self) -> float:
        """Returns the mean action value Q(s, a)."""
        if self.N > 0:
            return self.W / self.N
        else:
            return 0.0

    @property
    def has_parent(self) -> bool:
        return isinstance(self.parent, Node)


def best_child(
    node: Node, legal_actions: np.ndarray, c_puct_base: float, c_puct_init: float
) -> Node:
    """Returns best child node with maximum action value Q plus an upper confidence bound U.

    Args:
        node: the current node in the search tree.
        legal_actions: a 1D bool numpy.array mask for all actions,
            where `1` represents legal move and `0` represents illegal move.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.

    Returns:
        The best child node.

    Raises:
        ValueError:
            if the node instance itself is a leaf node.
    """
    if not node.is_expanded:
        raise ValueError("Expand leaf node first.")

    # The child Q value is evaluated from the opponent perspective.
    # when we select the best child for node, we want to do so from node.to_play's perspective,
    # so we always switch the sign for node.child_Q values, this is required since we're talking about two-player, zero-sum games.
    ucb_scores = -node.child_Q() + node.child_U(c_puct_base, c_puct_init)

    # Exclude illegal actions, note in some cases, the max ucb_scores may be zero.
    ucb_scores = np.where(legal_actions, ucb_scores, -1000)

    move = np.argmax(ucb_scores)

    assert legal_actions[move]

    return node.children[move]


def expand(node: Node, prior_prob: np.ndarray, child_to_play: int) -> None:
    """Expand all actions, including illegal actions.

    Args:
        node: current leaf node in the search tree.
        prior_prob: 1D numpy.array contains prior probabilities of the state for all actions.
        child_to_play: the player id for children nodes.

    Raises:
        ValueError:
            if node instance already expanded.
            if input argument `prior` is not a valid 1D float numpy.array.
    """
    if node.is_expanded:
        raise RuntimeError("Node already expanded.")
    if (
        not isinstance(prior_prob, np.ndarray)
        or len(prior_prob.shape) != 1
        or prior_prob.dtype not in (np.float32, np.float64)
    ):
        raise ValueError(
            f"Expect `prior_prob` to be a 1D float numpy.array, got {prior_prob}"
        )

    for action in range(0, prior_prob.shape[0]):
        child = Node(
            to_play=child_to_play, prior=prior_prob[action], move=action, parent=node
        )
        node.children[action] = child

    node.is_expanded = True


def backup(node: Node, value: float) -> None:
    """Update statistics of the this node and all traversed parent nodes.

    Args:
        node: current leaf node in the search tree.
        value: the evaluation value.

    Raises:
        ValueError:
            if input argument `value` is not float data type.
    """

    if not isinstance(value, float):
        raise ValueError(f"Expect `value` to be a float type, got {type(value)}")

    while isinstance(node, Node):
        node.N += 1
        node.W += value
        node = node.parent
        value = -value


def add_dirichlet_noise(
    node: Node, legal_actions: np.ndarray, eps: float = 0.25, alpha: float = 0.03
) -> None:
    """Add dirichlet noise to a given node.

    Args:
        node: the root node we want to add noise to.
        legal_actions: a 1D bool numpy.array mask for all actions,
            where `1` represents legal move and `0` represents illegal move.
        eps: epsilon constant to weight the priors vs. dirichlet noise.
        alpha: parameter of the dirichlet noise distribution.

    Raises:
        ValueError:
            if input argument `node` is not expanded.
            if input argument `eps` or `alpha` is not float type
                or not in the range of [0.0, 1.0].
    """

    if not isinstance(node, Node) or not node.is_expanded:
        raise ValueError("Expect `node` to be expanded")
    if not isinstance(eps, float) or not 0.0 <= eps <= 1.0:
        raise ValueError(
            f"Expect `eps` to be a float in the range [0.0, 1.0], got {eps}"
        )
    if not isinstance(alpha, float) or not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"Expect `alpha` to be a float in the range [0.0, 1.0], got {alpha}"
        )

    alphas = np.ones_like(legal_actions) * alpha
    noise = legal_actions * np.random.dirichlet(alphas)

    for a, n in enumerate(noise):
        node.children[a].prior = node.children[a].prior * (1 - eps) + n * eps


def generate_search_policy(visit_counts: np.ndarray, temperature: float) -> np.ndarray:
    """Returns a policy action probabilities after MCTS search,
    proportional to its exponentialted visit count.

    Args:
        visit_counts: the visit number of the children nodes for the root node of the search tree.
        temperature: a parameter controls the level of exploration.

    Returns:
        a 1D numpy.array contains the action probabilities after MCTS search.

    Raises:
        ValueError:
            if node instance not expanded.
            if input argument `temperature` is not float type or not in range [0.0, 1.0].
    """
    if not isinstance(temperature, float) or not 0 <= temperature <= 1.0:
        raise ValueError(
            f"Expect `temperature` to be float type in the range [0.0, 1.0], got {temperature}"
        )

    if temperature > 0.0:
        # Simple hack to avoid overflow when doing power operation over large numbers
        exp = max(1.0, min(5.0, 1.0 / temperature))
        visit_counts = np.power(visit_counts, exp)

    assert np.all(visit_counts >= 0) and not np.any(np.isnan(visit_counts))
    pi_probs = visit_counts
    sums = np.sum(pi_probs)
    if sums > 0:
        pi_probs /= sums

    return pi_probs


def uct_search(
    env: BoardGameEnv,
    eval_func: Callable[
        [np.ndarray, bool], Tuple[Iterable[np.ndarray], Iterable[float]]
    ],
    root_node: Node,
    c_puct_base: float,
    c_puct_init: float,
    num_simulations: int = 800,
    root_noise: bool = False,
    warm_up: bool = False,
    deterministic: bool = False,
) -> Tuple[int, np.ndarray, float, float, Node]:
    """Single-threaded Upper Confidence Bound (UCB) for Trees (UCT) search without any rollout.

    It follows the following general UCT search algorithm, except here we don't do rollout.
    ```
    function UCTSEARCH(r,m)
    i←1
    for i ≤ m do
        n ← select(r)
        n ← expand(n)
        ∆ ← rollout(n)
        backup(n,∆)
    end for
    return end function
    ```

    Args:
        env: a gym like custom BoardGameEnv environment.
        eval_func: a evaluation function when called returns the
            action probabilities and winning probability from
            current player's perspective.
        root_node: root node of the search tree, this comes from reuse sub-tree.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        num_simulations: number of simulations to run, default 800.
        root_noise: whether add dirichlet noise to root node to encourage exploration, default off.
        warm_up: if true, use temperature 1.0 to generate play policy, other wise use 0.1, default off.
        deterministic: after the MCTS search, choose the child node with most visits number to play in the game,
            instead of sample through a probability distribution, default off.


    Returns:
         tuple contains:
             a integer indicate the sampled action to play in the environment.
             a 1D numpy.array search policy action probabilities from the MCTS search result.
             a float indicate the root node value
             a float indicate the best child value
             a Node instance represent subtree of this MCTS search, which can be used as next root node for MCTS search.

     Raises:
         ValueError:
             if input argument `env` is not valid BoardGameEnv instance.
             if input argument `num_simulations` is not a positive integer.
         RuntimeError:
             if the game is over.
    """
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f"Expect `env` to be a valid BoardGameEnv instance, got {env}")
    if not 1 <= num_simulations:
        raise ValueError(
            f"Expect `num_simulations` to a positive integer, got {num_simulations}"
        )
    if env.is_game_over():
        raise RuntimeError("Game is over.")

    # Create root node
    if root_node is None:
        root_node = Node(to_play=env.to_play, parent=None)
        prior_prob, value = eval_func(env.observation(), False)
        expand(root_node, prior_prob, env.opponent_player)
        backup(root_node, value)

    root_legal_actions = env.legal_actions

    # Add dirichlet noise to the prior probabilities to root node.
    if root_noise:
        add_dirichlet_noise(root_node, root_legal_actions)

    while root_node.N < num_simulations:
        node = root_node

        # Make sure do not touch the actual environment.
        sim_env = copy.deepcopy(env)
        obs = sim_env.observation()
        done = sim_env.is_game_over

        # Phase 1 - Select
        # Select best child node until one of the following is true:
        # - reach a leaf node.
        # - game is over.
        while node.is_expanded:
            node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init)
            # Make move on the simulation environment.
            obs, reward, done, _ = sim_env.step(node.move)
            if done:
                break

        # assert node.to_play == sim_env.to_play

        # Special case - If game is over, using the actual reward from the game to update statistics
        if done:
            # The reward is for the last player who made the move won/loss the game.
            # So when backing up value from node represents current player, we add a minus sign
            assert node.to_play != sim_env.last_player
            backup(node, -reward)
            continue

        # Phase 2 - Expand and evaluation
        prior_prob, value = eval_func(obs, False)
        # Children nodes are evaluated from opponent player's perspective.
        expand(node, prior_prob, sim_env.opponent_player)

        # Phase 3 - Backup statistics
        backup(node, value)

    # Play - generate action probability from the root node.
    search_pi = generate_search_policy(root_node.child_N, 1.0 if warm_up else 0.1)

    move = None
    next_root_node = None
    best_child_Q = 0.0

    if deterministic:
        # Choose the child with most visit count.
        move = np.argmax(root_node.child_N)
    else:
        # Sample an action
        # Prevent the agent to select pass move during opening moves
        while (
            move is None
            or (warm_up and env.has_pass_move and move == env.pass_move)
            or root_legal_actions[move] != 1
        ):
            move = np.random.choice(np.arange(search_pi.shape[0]), p=search_pi)

    next_root_node = root_node.children[move]
    next_root_node.parent = None
    # Child value is computed from opponent's perspective, so we switch the sign
    best_child_Q = -next_root_node.Q

    return (move, search_pi, root_node.Q, best_child_Q, next_root_node)


def add_virtual_loss(node: Node) -> None:
    """Propagate a virtual loss to the traversed path.

    Args:
        node: current leaf node in the search tree.
    """
    # This is a loss for both players in the traversed path,
    # since we want to avoid multiple threads to select the same path.
    # However since we'll switching the sign for child_Q when selecting best child,
    # here we use +1 instead of -1.
    vloss = +1
    while isinstance(node, Node):
        node.losses_applied += 1
        node.W += vloss
        node = node.parent


def revert_virtual_loss(node: Node) -> None:
    """Undo virtual loss to the traversed path.

    Args:
        node: current leaf node in the search tree.
    """

    vloss = -1
    while isinstance(node, Node):
        if node.losses_applied > 0:
            node.losses_applied -= 1
            node.W += vloss
        node = node.parent


def parallel_uct_search(
    env: BoardGameEnv,
    eval_func: Callable[
        [np.ndarray, bool], Tuple[Iterable[np.ndarray], Iterable[float]]
    ],
    root_node: Node,
    c_puct_base: float,
    c_puct_init: float,
    num_simulations: int = 800,
    num_parallel: int = 8,
    root_noise: bool = False,
    warm_up: bool = False,
    deterministic: bool = False,
) -> Tuple[int, np.ndarray, float, float, Node]:
    """Single-threaded Upper Confidence Bound (UCB) for Trees (UCT) search without any rollout.

    This implementation uses tree parallel search and batched evaluation.

    It follows the following general UCT search algorithm, except here we don't do rollout.
    ```
    function UCTSEARCH(r,m)
      i←1
      for i ≤ m do
          n ← select(r)
          n ← expand(n)
          ∆ ← rollout(n)
          backup(n,∆)
      end for
      return end function
    ```

    Args:
        env: a gym like custom BoardGameEnv environment.
        eval_func: a evaluation function when called returns the
            action probabilities and winning probability from
            current player's perspective.
        root_node: root node of the search tree, this comes from reuse sub-tree.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        temperature: a parameter controls the level of exploration
            when generate policy action probabilities after MCTS search.
        num_simulations: number of simulations to run, default 800.
        num_parallel: Number of parallel leaves for MCTS search. This is also the batch size for neural network evaluation.
        root_noise: whether add dirichlet noise to root node to encourage exploration,
            default off.
        warm_up: if true, use temperature 1.0 to generate play policy, other wise use 0.1, default off.
        deterministic: after the MCTS search, choose the child node with most visits number to play in the game,
            instead of sample through a probability distribution, default off.


    Returns:
        tuple contains:
            a integer indicate the sampled action to play in the environment.
            a 1D numpy.array search policy action probabilities from the MCTS search result.
            a float indicate the root node value
            a float indicate the best child value
            a Node instance represent subtree of this MCTS search, which can be used as next root node for MCTS search.

    Raises:
        ValueError:
            if input argument `env` is not valid BoardGameEnv instance.
            if input argument `num_simulations` is not a positive integer.
        RuntimeError:
            if the game is over.
    """
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f"Expect `env` to be a valid BoardGameEnv instance, got {env}")
    if not 1 <= num_simulations:
        raise ValueError(
            f"Expect `num_simulations` to a positive integer, got {num_simulations}"
        )
    if env.is_game_over():
        raise RuntimeError("Game is over.")

    # Create root node
    if root_node is None:
        root_node = Node(to_play=env.to_play, parent=None)
        prior_prob, value = eval_func(env.observation(), False)
        expand(root_node, prior_prob, env.opponent_player)
        backup(root_node, value)

    root_legal_actions = env.legal_actions

    # Add dirichlet noise to the prior probabilities to root node.
    if root_noise:
        add_dirichlet_noise(root_node, root_legal_actions)

    while root_node.N < num_simulations + num_parallel:
        leaves = []
        failsafe = 0

        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            # This is necessary as when a game is over no leaf is added to leaves,
            # as we use the actual game results to update statistic
            failsafe += 1
            node = root_node

            # Make sure do not touch the actual environment.
            sim_env = copy.deepcopy(env)
            obs = sim_env.observation()
            done = sim_env.is_game_over

            # Phase 1 - Select
            # Select best child node until one of the following is true:
            # - reach a leaf node.
            # - game is over.
            while node.is_expanded:
                node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init)
                # Make move on the simulation environment.
                obs, reward, done, _ = sim_env.step(node.move)
                if done:
                    break

            # assert node.to_play == sim_env.to_play

            # Special case - If game is over, using the actual reward from the game to update statistics.
            if done:
                # The reward is for the last player who made the move won/loss the game.
                # So when backing up value from node represents current player, we add a minus sign
                assert node.to_play != sim_env.last_player
                backup(node, -reward)
                continue
            else:
                add_virtual_loss(node)
                leaves.append((node, obs, sim_env.opponent_player))

        if leaves:
            batched_nodes, batched_obs, batched_opponent_player = map(
                list, zip(*leaves)
            )
            prior_probs, values = eval_func(np.stack(batched_obs, axis=0), True)

            for leaf, prior_prob, value, opponent_player in zip(
                batched_nodes, prior_probs, values, batched_opponent_player
            ):
                revert_virtual_loss(leaf)

                # If a node was picked multiple times (despite virtual losses), we shouldn't
                # expand it more than once.
                if leaf.is_expanded:
                    continue

                # Children nodes are evaluated from opponent player's perspective.
                expand(leaf, prior_prob, opponent_player)
                backup(leaf, value)

    # Play - generate action probability from the root node.
    search_pi = generate_search_policy(root_node.child_N, 1.0 if warm_up else 0.1)

    move = None
    next_root_node = None
    best_child_Q = 0.0

    if deterministic:
        # Choose the child with most visit count.
        move = np.argmax(root_node.child_N)
    else:
        # Sample an action
        # Prevent the agent to select pass move during opening moves
        while (
            move is None
            or (warm_up and env.has_pass_move and move == env.pass_move)
            or root_legal_actions[move] != 1
        ):
            move = np.random.choice(np.arange(search_pi.shape[0]), p=search_pi)

    next_root_node = root_node.children[move]
    next_root_node.parent = None
    # Child value is computed from opponent's perspective, so we switch the sign
    best_child_Q = -next_root_node.Q

    return (move, search_pi, root_node.Q, best_child_Q, next_root_node)
