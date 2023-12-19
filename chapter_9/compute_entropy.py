import numpy as np

probs_dist1 = np.array([1, 0, 0, 0])

probs_dist2 = np.array([0.6, 0.4, 0, 0])

probs_dist3 = np.array([0.25, 0.25, 0.25, 0.25])


def compute_entropy(probs_dist):
    N = probs_dist.shape[0]

    entropy = 0
    for i in range(N):
        if probs_dist[i] > 0:
            entropy += probs_dist[i] * np.log(probs_dist[i])

    return -entropy


def compute_entropy_and_print_results(probs_dist):
    entropy = compute_entropy(probs_dist)
    print(f'Entropy for probability distribution {probs_dist} is: {entropy}')
    print('/n')


compute_entropy_and_print_results(probs_dist1)
compute_entropy_and_print_results(probs_dist2)
compute_entropy_and_print_results(probs_dist3)
