"""
Sample code of using TD0 to do policy evaluation for the service dog MDP example.
"""

from envs.dog_mdp import DogMDPEnv
import numpy as np
import algos
import utils


if __name__ == '__main__':
    print('Running TD0 Policy Evaluation on Service Dog MDP, this may take few minutes...')

    discount = 0.9
    learning_rate = 0.01
    num_runs = 100
    env = DogMDPEnv()

    # Table 5.1
    learning_rate = 0.01
    num_updates_list = [100, 1000, 10000, 20000, 50000]

    random_policy = algos.create_random_policy(env)

    results_V = {}
    for num_updates in num_updates_list:
        results_V[num_updates] = []

    for num_updates in num_updates_list:
        for _ in range(num_runs):
            V = algos.td0_policy_evaluation(env, random_policy, discount, learning_rate, num_updates)
            results_V[num_updates].append(V)

    for num_updates in num_updates_list:
        print(f'State value function (number of updates={num_updates}):')
        V = np.array(results_V[num_updates]).mean(axis=0)
        utils.print_state_value(env, V)

    # Fig 4.3
    random_policy = algos.create_random_policy(env)
    results_Q = []
    for _ in range(num_runs):
        Q = algos.td0_policy_evaluation_qpi(env, random_policy, discount, learning_rate, 20000)
        results_Q.append(Q)

    results_Q = np.array(results_Q).mean(axis=0)
    print('State-action value function:')
    utils.print_state_action_value(env, results_Q)
