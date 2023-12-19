#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

for (( seed=1; seed<=5; seed++ ))
do
    python3 reinforce.py --environment_name=CartPole-v1 --seed=$seed --results_csv_path=logs/reinforce/cartpole/$seed/results.csv

    python3 reinforce_baseline.py --environment_name=CartPole-v1 --seed=$seed --results_csv_path=logs/reinforce_baseline/cartpole/$seed/results.csv

    python3 actor_critic.py --environment_name=CartPole-v1 --seed=$seed --results_csv_path=logs/actor_critic/cartpole/$seed/results.csv
done
