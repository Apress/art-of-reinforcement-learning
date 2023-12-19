"""Extra sample code of using SARSA algorithm to solve the student MDP example in the David Silver RL course.

Course slides link at:
https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf
"""

from envs.student_mdp import StudentMDPEnv
import algos
import utils


if __name__ == '__main__':
    print('Running SARSA on Student MDP, this may take few minutes...')

    discount = 0.9
    begin_epsilon = 1.0
    end_epsilon = 0.01
    learning_rate = 0.01
    num_updates = 50000
    env = StudentMDPEnv()

    print('SARSA:')
    policy, Q = algos.sarsa(env, discount, begin_epsilon, end_epsilon, learning_rate, num_updates)

    print('Found policy:')
    utils.print_policy(env, policy)
    print('\n')

    print('State-action value function:')
    utils.print_state_action_value(env, Q)
    print('\n')

    V = utils.compute_vstar_from_qstar(env, Q)
    print('State value function:')
    utils.print_state_value(env, V)
    print('\n')
