#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Convergence analysis: train LorenzSystem on each env for a generous budget,
# then analyze the reward curves to find actual convergence points.
#
# Uses tiered budgets based on environment complexity:
#   - Simple (classic control):  1M steps
#   - Medium (MuJoCo simple, others): 2M steps
#   - Hard (MuJoCo locomotion): 3M steps
#   - Very Hard (HumanoidStandup): 5M steps

RESERVOIR="LorenzSystem"
SEED=42
OUTPUT_DIR="./results/experiments/convergence_v2"
TIMING_LOG="${OUTPUT_DIR}/timing.log"

mkdir -p "$OUTPUT_DIR"

# Environment → budget mapping
declare -A BUDGETS=(
  # Simple: 1M
  ["CartPole-v1"]=1000000
  ["Acrobot-v1"]=1000000
  ["MountainCarContinuous-v0"]=1000000
  # Medium: 2M
  ["Pendulum-v1"]=2000000
  ["LunarLander-v3"]=2000000
  ["Reacher-v4"]=2000000
  ["Pusher-v4"]=2000000
  ["Swimmer-v4"]=2000000
  ["parking-v0"]=2000000
  ["FetchSlide-v2"]=2000000
  ["HandReach-v2"]=2000000
  ["PointMaze"]=2000000
  ["finger-spin"]=2000000
  # Hard: 3M
  ["BipedalWalker-v3"]=3000000
  ["Hopper-v4"]=3000000
  ["HalfCheetah-v4"]=3000000
  ["Ant-v4"]=3000000
  # Very Hard: 5M
  ["HumanoidStandup-v4"]=5000000
)

# Ordered env list
ENVS=(
  "CartPole-v1"
  "Acrobot-v1"
  "MountainCarContinuous-v0"
  "Pendulum-v1"
  "LunarLander-v3"
  "BipedalWalker-v3"
  "Reacher-v4"
  "Pusher-v4"
  "Swimmer-v4"
  "Hopper-v4"
  "HalfCheetah-v4"
  "Ant-v4"
  "HumanoidStandup-v4"
  "parking-v0"
  "FetchSlide-v2"
  "HandReach-v2"
  "PointMaze"
  "finger-spin"
)

fmt_elapsed() {
  local secs=$1
  printf "%02d:%02d:%02d" "$((secs/3600))" "$(((secs%3600)/60))" "$((secs%60))"
}

log() { echo "$1" | tee -a "$TIMING_LOG"; }

log ""
log "================================================================"
log "  CONVERGENCE ANALYSIS v2  $(date '+%Y-%m-%d %H:%M:%S')"
log "  Reservoir: ${RESERVOIR}  Seed: ${SEED}"
log "  Environments: ${#ENVS[@]}"
log "================================================================"

SCRIPT_START="$(date +%s)"

for env in "${ENVS[@]}"; do
  budget="${BUDGETS[$env]}"
  env_start="$(date +%s)"
  log "[$(date '+%H:%M:%S')]  START  ${env}  (${budget} steps)"

  if uv run src/train_agent.py \
       -e "$env" \
       -r "$RESERVOIR" \
       -t "$budget" \
       -n 1 \
       -s "$SEED" \
       --experiment convergence_v2; then
    env_end="$(date +%s)"
    log "[$(date '+%H:%M:%S')]  DONE   ${env}  $(fmt_elapsed "$((env_end - env_start))")"
  else
    env_end="$(date +%s)"
    log "[$(date '+%H:%M:%S')]  FAIL   ${env}  $(fmt_elapsed "$((env_end - env_start))")"
  fi
done

TOTAL_ELAPSED="$(( $(date +%s) - SCRIPT_START ))"
log "----------------------------------------------------------------"
log "  ANALYSIS FINISHED  $(date '+%Y-%m-%d %H:%M:%S')"
log "  Total wall time: $(fmt_elapsed "$TOTAL_ELAPSED")"
log "================================================================"

# Now analyze convergence
log ""
log "Analyzing convergence..."
uv run python3 src/scripts/analyze_convergence.py \
  --log_root "./results/experiments/convergence_v2/log" \
  --output "${OUTPUT_DIR}/convergence_results.json" | tee -a "$TIMING_LOG"
