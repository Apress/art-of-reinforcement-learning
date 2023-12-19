#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

for (( seed=1; seed<=5; seed++ ))
do
    python3 ppo_continuous.py --environment_name=Ant-v4 --num_iterations=40 --seed=$seed --results_csv_path=logs/ppo/ant/$seed/results.csv
done
