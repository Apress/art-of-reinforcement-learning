"""
Sample code of using DP algorithms to solve the service dog MDP example.
"""

from envs.dog_mdp import DogMDPEnv
import utils
import algos


if __name__ == '__main__':
    discount = 0.9
    env = DogMDPEnv()

    # Evaluate a random policy.
    random_policy = algos.create_random_policy(env)
    print('Random policy:')
    utils.print_policy(env, random_policy)
    random_pi_state_values = algos.policy_evaluation(env, random_policy, discount)
    print('Random policy state values:')
    utils.print_state_value(env, random_pi_state_values)
    print('\n')

    # Find the optimal policy.
    # Option 1: use policy iteration algorithm.
    print('Start to run policy iteration algorithm:')
    optimal_pi, optimal_V, optimal_Q = algos.policy_iteration(env, discount)
    print('Optimal policy:')
    utils.print_policy(env, optimal_pi)
    print('Optimal state value function:')
    utils.print_state_value(env, optimal_V)
    print('Optimal state-action value function:')
    utils.print_state_action_value(env, optimal_Q)
    print('\n')

    # Option 2: use value iteration algorithm.
    print('Start to run value iteration algorithm:')
    optimal_pi, optimal_V = algos.value_iteration(env, discount)
    print('Optimal policy:')
    utils.print_policy(env, optimal_pi)
    print('Optimal state value function:')
    utils.print_state_value(env, optimal_V)
    print('\n')

    # Optimal state value with different discount factor.
    for discount in [0, 0.3, 0.5, 0.7, 0.9, 1]:
        print(f'Start to run policy iteration algorithm (discount={discount}):')
        optimal_V = algos.policy_evaluation(env, optimal_pi, discount)
        print('Optimal state value function:')
        utils.print_state_value(env, optimal_V)
        print('\n')
