#!/usr/bin/env bash
set -euo pipefail

# MLP convergence calibration with tuned PPO hyperparameters.
# Trains MLP on all 13 active environments with generous budgets,
# then runs convergence analysis to determine per-env training steps.
#
# Expected runtime: ~24-48 hours (depending on CPU load).
#
# Usage:
#   nohup bash src/scripts/run_convergence_calibration.sh > results/convergence_calibration.log 2>&1 &

export MUJOCO_GL="${MUJOCO_GL:-egl}"

EXPERIMENT="convergence_mlp"

# Generous budgets for convergence detection.
# Classic control envs converge fast; MuJoCo needs much more.
declare -A BUDGETS=(
  ["CartPole-v1"]=500000
  ["Acrobot-v1"]=500000
  ["MountainCarContinuous-v0"]=500000
  ["Pendulum-v1"]=2000000
  ["Reacher-v4"]=2000000
  ["Pusher-v4"]=2000000
  ["Swimmer-v4"]=2000000
  ["Hopper-v4"]=3000000
  ["HalfCheetah-v4"]=3000000
  ["HumanoidStandup-v4"]=2000000
  ["PointMaze"]=2000000
  ["finger-spin"]=2000000
)

ENVS=(
  "CartPole-v1"
  "Acrobot-v1"
  "MountainCarContinuous-v0"
  "Pendulum-v1"
  "Reacher-v4"
  "Pusher-v4"
  "Swimmer-v4"
  "Hopper-v4"
  "HalfCheetah-v4"
  "HumanoidStandup-v4"
  "PointMaze"
  "finger-spin"
)

echo "================================================================"
echo "  MLP Convergence Calibration"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Environments: ${#ENVS[@]}"
echo "================================================================"

for env in "${ENVS[@]}"; do
  steps="${BUDGETS[$env]}"
  echo "[$(date '+%H:%M:%S')] START  ${env}  (${steps} steps, MLP)"
  t0=$(date +%s)

  uv run src/train_agent.py \
    -e "$env" \
    -r mlp \
    -t "$steps" \
    -n 1 \
    -s 0 \
    --experiment "$EXPERIMENT" || echo "  FAILED: $env"

  elapsed=$(( $(date +%s) - t0 ))
  echo "[$(date '+%H:%M:%S')] DONE   ${env}  (${elapsed}s)"
done

echo ""
echo "================================================================"
echo "  Running convergence analysis..."
echo "================================================================"

uv run src/scripts/analyze_convergence.py \
  --log_root "./results/experiments/${EXPERIMENT}/log" \
  --output "./results/experiments/${EXPERIMENT}/convergence.json"

echo ""
echo "  Calibration complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
