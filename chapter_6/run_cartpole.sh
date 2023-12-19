#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

## Runs linear VFA
for (( seed=1; seed<=5; seed++ ))
do
    python3 linear_mc_cartpole.py --seed=$seed --results_csv_path=logs/linear_mc/cartpole/$seed/results.csv
    python3 linear_q_cartpole.py --seed=$seed --results_csv_path=logs/linear_q/cartpole/$seed/results.csv
done
