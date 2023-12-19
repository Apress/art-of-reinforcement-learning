#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

for (( seed=1; seed<=5; seed++ ))
do
    python3 ppo_continuous.py --environment_name=Humanoid-v4 --num_iterations=80 --seed=$seed --results_csv_path=logs/ppo/humanoid/$seed/results.csv
done
