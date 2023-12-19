"""
Sample code of using Monte Carlo algorithms to solve the service dog MDP example.
"""

from envs.dog_mdp import DogMDPEnv
import numpy as np
import algos
import utils


if __name__ == '__main__':
    print('Running First-visit Monte Carlo on Service Dog MDP, this may take few minutes...')

    # Table 4.3
    discount = 0.9
    epsilon = 1.0
    num_runs = 100
    num_episodes_list = [10, 100, 1000, 10000]
    env = DogMDPEnv()

    results_Q = {}
    results_V = {}
    for num_episodes in num_episodes_list:
        results_Q[num_episodes] = []
        results_V[num_episodes] = None

    for num_episodes in num_episodes_list:
        for run in range(num_runs):
            policy, Q = algos.mc_policy_control(env, discount, epsilon, num_episodes, first_visit=True)
            results_Q[num_episodes].append(Q)

    for num_episodes in num_episodes_list:
        Q = np.array(results_Q[num_episodes]).mean(axis=0)
        V = utils.compute_vstar_from_qstar(env, Q)
        print(f'State value function (number of episodes={num_episodes}):')
        utils.print_state_value(env, V)
