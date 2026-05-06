#!/usr/bin/env bash
set -euo pipefail

# Run gradient reservoir sanity checks for all ODE systems.

for ode in lorenz tyson1999 markevich2004 tyson1991; do
  uv run src/tools/gradient_reservoir_check.py \
    --ode_name "$ode" \
    --n_initial 128 \
    --burn_in_steps 128 \
    --burn_in_dt 0.05 \
    --samples_per_traj 128 \
    --sample_dt 0.05 \
    --device cpu \
    --seed 0
done
