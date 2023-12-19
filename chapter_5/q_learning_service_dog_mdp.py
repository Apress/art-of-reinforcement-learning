"""
Sample code of using Q-learning algorithm to solve the service dog MDP example.
"""

from envs.dog_mdp import DogMDPEnv
import numpy as np
import algos
import utils


if __name__ == '__main__':
    print('Running Q-learning on Service Dog MDP, this may take few minutes...')

    # Table 5.3
    discount = 0.9
    begin_epsilon = 1.0
    end_epsilon = 0.01
    learning_rate = 0.01
    num_runs = 100
    num_updates_list = [100, 1000, 10000, 20000, 50000]
    env = DogMDPEnv()

    results_Q = {}
    for num_updates in num_updates_list:
        results_Q[num_updates] = []

    for num_updates in num_updates_list:
        for _ in range(num_runs):
            policy, Q = algos.q_learning(
                env,
                discount,
                begin_epsilon,
                end_epsilon,
                learning_rate,
                num_updates,
            )
            results_Q[num_updates].append(Q)

    for num_updates in num_updates_list:
        Q = np.array(results_Q[num_updates]).mean(axis=0)
        V = utils.compute_vstar_from_qstar(env, Q)
        print(f'State value function (number of updates={num_updates}):')
        utils.print_state_value(env, V)
