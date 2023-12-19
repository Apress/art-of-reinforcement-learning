#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

## Runs deep Q learning with different combination of target network and experience replay
for (( seed=1; seed<=5; seed++ ))
do
    python3 naive_deep_q.py --environment_name=CartPole-v1 --seed=$seed --results_csv_path=logs/naive_deep_q/cartpole/$seed/results.csv

    python3 deep_q_with_replay.py --environment_name=CartPole-v1 --seed=$seed --results_csv_path=logs/deep_q_replay/cartpole/$seed/results.csv

    python3 deep_q_with_targetnet.py --environment_name=CartPole-v1 --seed=$seed --results_csv_path=logs/deep_q_targetnet/cartpole/$seed/results.csv

    python3 dqn_classic.py --environment_name=CartPole-v1 --seed=$seed --results_csv_path=logs/dqn/cartpole/$seed/results.csv
done
