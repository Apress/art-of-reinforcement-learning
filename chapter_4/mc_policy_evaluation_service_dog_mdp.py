"""
Sample code of using Monte Carlo algorithms to do policy evaluation for the service dog MDP example.
"""
from envs.dog_mdp import DogMDPEnv
import numpy as np
import algos
import utils


if __name__ == '__main__':
    print('Running Monte Carlo Policy Evaluation on Service Dog MDP, this may take few minutes...')

    discount = 0.9
    num_episodes = 2000
    num_runs = 100
    env = DogMDPEnv()
    random_policy = algos.create_random_policy(env)

    # Table 4.1
    first_visit_V = []
    every_visit_V = []
    for i in range(num_runs):
        V = algos.mc_policy_evaluation(env, random_policy, discount, num_episodes, first_visit=True)
        first_visit_V.append(V)
        V = algos.mc_policy_evaluation(env, random_policy, discount, num_episodes, first_visit=False)
        every_visit_V.append(V)

    first_visit_V = np.array(first_visit_V).mean(axis=0)
    every_visit_V = np.array(every_visit_V).mean(axis=0)
    print('First-visit Monte Carlo policy evaluation:')
    utils.print_state_value(env, first_visit_V)
    print('Every-visit Monte Carlo policy evaluation:')
    utils.print_state_value(env, every_visit_V)
    print('\n')

    # Table 4.2
    first_visit_V = []
    incr_first_visit_V = []
    for i in range(num_runs):
        V = algos.mc_policy_evaluation(env, random_policy, discount, num_episodes, first_visit=True)
        first_visit_V.append(V)

        V = algos.incremental_mc_policy_evaluation(env, random_policy, discount, num_episodes, first_visit=True)
        incr_first_visit_V.append(V)

    first_visit_V = np.array(first_visit_V).mean(axis=0)
    incr_first_visit_V = np.array(incr_first_visit_V).mean(axis=0)
    print('First-visit Monte Carlo policy evaluation:')
    utils.print_state_value(env, first_visit_V)
    print('Incremental MC first-visit policy evaluation:')
    utils.print_state_value(env, incr_first_visit_V)
    print('\n')

    incr_first_visit_Q = []
    for i in range(num_runs):
        Q = algos.incremental_mc_policy_evaluation_qpi(env, random_policy, discount, num_episodes, first_visit=True)
        incr_first_visit_Q.append(Q)

    incr_first_visit_Q = np.array(incr_first_visit_Q).mean(axis=0)
    print('Incremental MC first-visit policy evaluation for state-action value function:')
    utils.print_state_action_value(env, incr_first_visit_Q)
    print('\n')

    print('Incremental MC every-visit policy evaluation:')
    V = algos.incremental_mc_policy_evaluation(env, random_policy, discount, num_episodes, first_visit=False)
    utils.print_state_value(env, V)
    print('\n')
