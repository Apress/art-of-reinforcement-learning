#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

for (( seed=1; seed<=3; seed++ ))
do
    python3 rnd_ppo_atari.py --environment_name=MontezumaRevenge --num_iterations=80 --num_actors=32 --seed=$seed --results_csv_path=logs/rnd_ppo/montezumarevenge/$seed/results.csv
done
