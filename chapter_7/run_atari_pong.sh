#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

## Runs DQN on Atari games
for (( seed=1; seed<=5; seed++ ))
do
    python3 dqn_atari.py --environment_name=Pong --num_iterations=20 --seed=$seed --results_csv_path=logs/dqn/pong/$seed/results.csv
done
