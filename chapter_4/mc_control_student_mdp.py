"""Extra sample code of using DP algorithms to solve the student MDP example in the David Silver RL course.

Course slides link at:
https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf
"""

from envs.student_mdp import StudentMDPEnv
import algos
import utils


if __name__ == '__main__':
    print('Running First-visit Monte Carlo on Student MDP, this may take few minutes...')

    discount = 0.9
    epsilon = 1.0
    num_episodes = 50000
    env = StudentMDPEnv()

    print('MC first-visit policy control:')
    policy, Q = algos.mc_policy_control(env, discount, epsilon, num_episodes, first_visit=True)

    print('Found policy:')
    utils.print_policy(env, policy)
    print('\n')

    print('State-action value function:')
    utils.print_state_action_value(env, Q)
    print('\n')

    V = utils.compute_vstar_from_qstar(Q)
    print('State value function:')
    utils.print_state_value(env, V)
    print('\n')
