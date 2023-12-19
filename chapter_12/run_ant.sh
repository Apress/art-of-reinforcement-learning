#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

for num_actors in 4 8 16 32;
do
    for (( seed=1; seed<=3; seed++ ))
    do
        python3 dist_ppo_continuous.py --environment_name=Ant-v4 --num_iterations=40 --num_epochs=4 --num_actors=$num_actors --seed=$seed --results_csv_path=logs/ppo/ant/$num_actors/$seed/results.csv
    done
done
