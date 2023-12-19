"""Extra sample code of using DP algorithms to solve the student MRP example in the David Silver RL course.

Course slides link at:
https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf
"""
from envs.student_mrp import StudentMRPEnv

import utils
import algos

if __name__ == '__main__':
    env = StudentMRPEnv()

    for discount in [0, 0.3, 0.5, 0.7, 0.9, 1]:
        state_values = algos.compute_mrp_state_value(env, discount)
        print(f'State value function for student example MRP (discount={discount}):')
        utils.print_state_value(env, state_values)
        print('\n')
