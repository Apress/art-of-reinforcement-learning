def compute_returns(rewards, discount):
    """Given a list of reward signals and the discount rate,
    compute discounted returns.

    G_0 = = R_0 + γ R_1 + γ^2 R_2 + ... + γ^{T-1} R_{T-1}

    Args:
        rewards: a list of rewards from an episode trajectory.
        discount: discount factor, must be 0 <= discount <= 1.

    Returns:
        return G_0 the discounted returns.

    """
    assert 0.0 <= discount <= 1.0

    G_t = 0
    # We do it backwards so it's more efficient and easier to implement.
    for t in reversed(range(len(rewards))):
        G_t = rewards[t] + discount * G_t

    return G_t


if __name__ == '__main__':
    discount = 0.9

    cases = [
        [-1, -1, -1, 10],
        [-1, 10],
        [-1, 1, -1, -1, 10],
        [+1] * 10000,
    ]

    for i, rewards in enumerate(cases):
        G = compute_returns(rewards, discount)

        print(f'Discounted return for case {i+1} is: {G}')
