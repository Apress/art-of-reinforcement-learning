"""Extra sample code of using DP algorithms to solve the student MDP example in the David Silver RL course.

Course slides link at:
https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf
"""
from envs.student_mdp import StudentMDPEnv

import utils
import algos


if __name__ == '__main__':
    discount = 1.0
    env = StudentMDPEnv()

    # Find the optimal policy.
    # Option 1: use policy iteration algorithm.
    print('Start to run policy iteration algorithm:')
    optimal_pi, optimal_V, optimal_Q = algos.policy_iteration(env, discount)
    print('Optimal policy:')
    utils.print_policy(env, optimal_pi)
    print('Optimal state value function:')
    utils.print_state_value(env, optimal_V)
    print('\n')

    # Option 2: use value iteration algorithm.
    print('Start to run value iteration algorithm:')
    optimal_pi, optimal_V = algos.value_iteration(env, discount)
    print('Optimal policy:')
    utils.print_policy(env, optimal_pi)
    print('Optimal state value function:')
    utils.print_state_value(env, optimal_V)
    print('\n')

    for discount in [0, 0.3, 0.5, 0.7, 0.9, 1]:
        print(f'Start to run policy iteration algorithm (discount={discount}):')
        optimal_pi, optimal_V, optimal_Q = algos.policy_iteration(env, discount)
        print('Optimal state value function:')
        utils.print_state_value(env, optimal_V)
        print('\n')
