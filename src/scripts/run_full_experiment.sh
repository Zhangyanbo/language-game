#!/usr/bin/env bash
set -euo pipefail

# Use EGL for headless MuJoCo rendering (no X11 needed)
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Full training matrix: 13 environments × 17 reservoirs.
# Excluded: FetchSlide-v2, HandReach-v2 (no learning); InvertedDoublePendulum-v4 (redundant);
# ball_in_cup-catch (unstable training); LunarLander-v3, BipedalWalker-v3, Ant-v4, parking-v0 (slow).
#
# Timesteps calibrated via MLP convergence analysis + tuned PPO params (2026-03-21).
# Total steps reduced 28% from old calibration (26.2M vs 36.5M per reservoir).
# Estimated wall time per reservoir (N=1): ~2.4h.
# Total 17 reservoirs: ~41h / 1.7 days (N=1), ~163h / 6.8 days (N=4).
#
# Usage:
#   bash src/scripts/run_full_experiment.sh              # N=4, seed=0
#   bash src/scripts/run_full_experiment.sh -n 1          # N=1, seed=0
#   bash src/scripts/run_full_experiment.sh -n 2 -s 2     # N=2, seed starts at 2
#
# Continuation example:
#   First run:     -n 2 -s 0   →  trains seed 0, 1
#   Continue with: -n 2 -s 2   →  trains seed 2, 3  (no overlap)
#
# Output (all under ./results/):
#   agents/<reservoir>_gradient/<env>/seed_<n>.zip   Trained PPO models
#   log/<reservoir>_gradient/<env>/seed_<n>/          Monitor CSVs (reward curves)
#   figures/reward_curve/<reservoir>/<env>/seed_<n>.png  Per-seed reward curves
#   videos/<reservoir>/<env>/seed_<SEED>.mp4          Video of first seed only
#   figures/rewards_gradient.{png,pdf}                Multi-env reward comparison
#   timing.log                                        Crash-safe per-env timing

N_EXPERIMENTS=4
BASE_SEED=0

usage() {
  cat <<EOF
Usage: $0 [-n N] [-s SEED] [-h]

Options:
  -n, --n_experiments  Number of repeats per (env, reservoir). Default: 4
  -s, --seed           Starting seed. Seeds used: SEED, SEED+1, ..., SEED+N-1.
                       Default: 0. Use this to continue previous runs without overlap.
  -h, --help           Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--n_experiments)
      N_EXPERIMENTS="${2:?Missing value for $1}"
      shift 2
      ;;
    -s|--seed)
      BASE_SEED="${2:?Missing value for $1}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Environment list (16 active)
# ---------------------------------------------------------------------------
ENVS=(
  # Classic Control / Box2D
  "CartPole-v1"
  "Acrobot-v1"
  "MountainCarContinuous-v0"
  "Pendulum-v1"
  # "LunarLander-v3"       # deferred: slow env (1006 steps/s), 50 min/run
  # "BipedalWalker-v3"     # deferred: slow env (2159 steps/s), 35 min/run
  # MuJoCo
  "Reacher-v4"
  "Pusher-v4"
  "Swimmer-v4"
  "Hopper-v4"
  "HalfCheetah-v4"
  # "Ant-v4"               # deferred: slow env (1951 steps/s), 39 min/run
  "HumanoidStandup-v4"
  # highway-env
  # "parking-v0"           # deferred: very slow env (266 steps/s), 31 min/run
  # gymnasium-robotics
  "PointMaze"
  # dm_control
  "finger-spin"
  # Atari (RAM)
  "BankHeist-ram"
  "KungFuMaster-ram"
  "CrazyClimber-ram"
  "Kangaroo-ram"
)

# Timesteps calibrated from MLP convergence analysis (2026-03-21) with tuned
# PPO hyperparameters. MLP baseline trained on 13 envs (500k-3M steps).
# Atari calibrated 2026-03-26 (MLP @ 3M steps).
#
# Policy: converge × 1.25 (CONVERGED), total × 1.5 (RISING). Min 200k.
declare -A TIMESTEPS=(
  # Classic Control / Box2D
  ["CartPole-v1"]=650000           # C @502k → 650k.  Reward: 161→488
  ["Acrobot-v1"]=200000            # C @107k → 200k.  Reward: -355→-80
  ["MountainCarContinuous-v0"]=600000  # C @491k → 600k.  Reward: -43→-2
  ["Pendulum-v1"]=1000000          # C @782k → 1M.    Reward: -503→-203
  ["LunarLander-v3"]=2500000       # (deferred, no MLP data — estimated)
  ["BipedalWalker-v3"]=3800000     # (deferred, no MLP data — estimated)
  # MuJoCo
  ["Reacher-v4"]=600000            # C @461k → 600k.  Reward: -53→-6
  ["Pusher-v4"]=900000             # C @702k → 900k.  Reward: -106→-31
  ["Swimmer-v4"]=2500000           # C @1.98M → 2.5M. Reward: 18→29
  ["Hopper-v4"]=3700000            # C @2.98M → 3.7M. Reward: 323→1231
  ["HalfCheetah-v4"]=6200000       # C @4.94M → 6.2M.  Reward: -5→1138
  ["Ant-v4"]=4500000               # (deferred, no MLP data — estimated)
  ["HumanoidStandup-v4"]=6000000   # C @5M → 6M.    Reward: 101k→155k (plateaus ~5M)
  # highway-env
  ["parking-v0"]=400000            # (deferred, no MLP data — estimated)
  # gymnasium-robotics
  ["PointMaze"]=550000             # C @421k → 550k.  Reward: 135→249
  # dm_control
  ["finger-spin"]=2000000          # C @1.62M → 2M.   Reward: 110→623
  # Atari (RAM) — calibrated from MLP @10-20M on HPC (2026-03-28)
  ["BankHeist-ram"]=10500000       # C @7.0M → 10.5M.  Reward: 481→1205
  ["KungFuMaster-ram"]=8000000     # C @5M → 8M.       Reward: 8466→18144
  ["CrazyClimber-ram"]=10000000    # C @7M → 10M.      Reward: 25118→65102
  ["Kangaroo-ram"]=8000000         # C @5M → 8M.       Reward: 371→1633
)

# ---------------------------------------------------------------------------
# Reservoir list (17 total)
# ---------------------------------------------------------------------------
RESERVOIRS=(
  # Baseline / controls
  "LorenzSystem"
  "identity"
  "mlp"
  # GRN models (14)
  "Tyson1999CircleLock"
  "Markevich2004MAPKDoublePhosphorylation"
  "Tyson1991CellCycle2Var"
  "Weimann2004CircadianOscillator"
  "Almeida2019CircadianClock"
  "Zatorsky2006P53Model4"
  "Gardner2000ToggleSwitch"
  "Liebal2012TranscriptionInhibition"
  "Gerard2010CellCycle"
  "Chickarmane2006StemCellSwitch"
  "Gardner1998CellCycleGoldbeter"
  "Leloup1999CircadianClock"
  "Chickarmane2008NanogGata6"
  "Kholodenko2000MAPKCascade"
)

# ---------------------------------------------------------------------------
# Timing log (append-only, crash-safe)
# ---------------------------------------------------------------------------
TIMING_LOG="./results/timing.log"
mkdir -p ./results

fmt_elapsed() {
  local secs=$1
  printf "%02d:%02d:%02d" "$((secs/3600))" "$(((secs%3600)/60))" "$((secs%60))"
}

# Append one line to timing log AND stdout. Flushed immediately via tee.
# Even if the script crashes later, all previously logged lines are on disk.
log() {
  echo "$1" | tee -a "$TIMING_LOG"
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
TOTAL_COMBOS=$(( ${#ENVS[@]} * ${#RESERVOIRS[@]} * N_EXPERIMENTS ))

log ""
log "================================================================"
log "  RUN STARTED  $(date '+%Y-%m-%d %H:%M:%S')"
log "  Environments: ${#ENVS[@]}  Reservoirs: ${#RESERVOIRS[@]}  N: ${N_EXPERIMENTS}"
log "  Seeds: ${BASE_SEED}..$(( BASE_SEED + N_EXPERIMENTS - 1 ))   Total combos: ${TOTAL_COMBOS}"
log "================================================================"

# ---------------------------------------------------------------------------
# Training loop
# Each env trains ALL reservoirs × N seeds in one train_agent.py call.
# Timing is per-env (the atomic unit that can fail independently).
# ---------------------------------------------------------------------------
SCRIPT_START="$(date +%s)"
COMPLETED=0
FAILED=0

for env in "${ENVS[@]}"; do
  steps="${TIMESTEPS[$env]}"
  env_start="$(date +%s)"
  log "[$(date '+%H:%M:%S')]  START  ${env}  (${steps} steps × ${#RESERVOIRS[@]} res × ${N_EXPERIMENTS} seeds)"

  if uv run src/train_agent.py \
       -e "$env" \
       -r "${RESERVOIRS[@]}" \
       -t "$steps" \
       -n "$N_EXPERIMENTS" \
       -s "$BASE_SEED" \
       --save_video; then
    env_end="$(date +%s)"
    env_elapsed="$(( env_end - env_start ))"
    log "[$(date '+%H:%M:%S')]  DONE   ${env}  $(fmt_elapsed "$env_elapsed")"
    COMPLETED=$(( COMPLETED + 1 ))
  else
    env_end="$(date +%s)"
    env_elapsed="$(( env_end - env_start ))"
    log "[$(date '+%H:%M:%S')]  FAIL   ${env}  $(fmt_elapsed "$env_elapsed")  (exit $?)"
    FAILED=$(( FAILED + 1 ))
  fi
done

# ---------------------------------------------------------------------------
# Plotting (best-effort: don't abort if plotting fails)
# ---------------------------------------------------------------------------
plot_start="$(date +%s)"
log "[$(date '+%H:%M:%S')]  START  plotting"
if bash src/scripts/plot_full_experiment.sh; then
  plot_elapsed="$(( $(date +%s) - plot_start ))"
  log "[$(date '+%H:%M:%S')]  DONE   plotting  $(fmt_elapsed "$plot_elapsed")"
else
  plot_elapsed="$(( $(date +%s) - plot_start ))"
  log "[$(date '+%H:%M:%S')]  FAIL   plotting  $(fmt_elapsed "$plot_elapsed")"
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
SCRIPT_END="$(date +%s)"
TOTAL_ELAPSED="$(( SCRIPT_END - SCRIPT_START ))"

log "----------------------------------------------------------------"
log "  RUN FINISHED  $(date '+%Y-%m-%d %H:%M:%S')"
log "  Envs completed: ${COMPLETED}/${#ENVS[@]}   Failed: ${FAILED}"
log "  Total wall time: $(fmt_elapsed "$TOTAL_ELAPSED")"
log "  Timing log: ${TIMING_LOG}"
log "================================================================"
