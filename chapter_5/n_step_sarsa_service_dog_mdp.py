"""
Sample code of using SARSA algorithm with N-step return to solve the service dog MDP example.
"""

from envs.dog_mdp import DogMDPEnv
import algos
import utils

import numpy as np


if __name__ == '__main__':
    print('Running N-step SARSA on Service Dog MDP, this may take few minutes...')

    # Table 5.4
    discount = 0.9
    begin_epsilon = 1.0
    end_epsilon = 0.01
    learning_rate = 0.01
    num_updates = 1000
    num_runs = 100
    env = DogMDPEnv()

    results_Q = {}
    for n_steps in [1, 2, 3, 4, 5]:
        results_Q[n_steps] = []
        for _ in range(num_runs):
            policy, Q = algos.n_step_sarsa(
                env,
                discount,
                begin_epsilon,
                end_epsilon,
                learning_rate,
                n_steps,
                num_updates,
            )
            results_Q[n_steps].append(Q)

    for key in results_Q.keys():
        Q = np.array(results_Q[key]).mean(axis=0)
        V = utils.compute_vstar_from_qstar(env, Q)
        print(f'State value function (n-steps={key}):')
        utils.print_state_value(env, V)
