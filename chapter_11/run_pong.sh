#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

for (( seed=1; seed<=3; seed++ ))
do
    python3 ppo_atari.py --environment_name=Pong --num_iterations=30 --seed=$seed --results_csv_path=logs/ppo/pong/$seed/results.csv
done
