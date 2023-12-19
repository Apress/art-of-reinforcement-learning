#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

for (( seed=1; seed<=5; seed++ ))
do
    python3 prioritized_dqn_atari.py --environment_name=Riverraid --num_iterations=50 --seed=$seed --results_csv_path=logs/prioritized_dqn/riverraid/$seed/results.csv
done
