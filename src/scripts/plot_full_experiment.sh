#!/usr/bin/env bash
set -euo pipefail

N_SEEDS="${1:-}"
if [[ -n "${N_SEEDS}" && ! "${N_SEEDS}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [n_seeds]"
  exit 1
fi

LOG_SUFFIX="_gradient"
REWARD_PREFIX="rewards_gradient"
POLICY_PREFIX="policy_similarity_gradient"

ENVS=(
  # Classic Control
  "CartPole-v1"
  "Acrobot-v1"
  "MountainCarContinuous-v0"
  "Pendulum-v1"
  # MuJoCo
  "Reacher-v4"
  "Pusher-v4"
  "Swimmer-v4"
  "Hopper-v4"
  "HalfCheetah-v4"
  "HumanoidStandup-v4"
  # Gymnasium Robotics
  "PointMaze"
  # DeepMind Control Suite
  "finger-spin"
  # Atari (ALE) — RAM observations
  "BankHeist-ram"
  "KungFuMaster-ram"
  "CrazyClimber-ram"
  "Kangaroo-ram"
)

# Plot multi-env reward curves
uv run src/plot_rewards.py \
  --log_root ./results/log \
  --output_dir ./results/figures \
  --output_prefix "${REWARD_PREFIX}" \
  --envs "${ENVS[@]}" \
  --reservoirs \
    identity${LOG_SUFFIX} mlp${LOG_SUFFIX} \
    lorenzsystem${LOG_SUFFIX} \
    tyson1999circlelock${LOG_SUFFIX} \
    markevich2004mapkdoublephosphorylation${LOG_SUFFIX} \
    tyson1991cellcycle2var${LOG_SUFFIX} \
    weimann2004circadianoscillator${LOG_SUFFIX} \
    almeida2019circadianclock${LOG_SUFFIX} \
    zatorsky2006p53model4${LOG_SUFFIX} \
    gardner2000toggleswitch${LOG_SUFFIX} \
    liebal2012transcriptioninhibition${LOG_SUFFIX} \
    gerard2010cellcycle${LOG_SUFFIX} \
    chickarmane2006stemcellswitch${LOG_SUFFIX} \
    gardner1998cellcyclegoldbeter${LOG_SUFFIX} \
    leloup1999circadianclock${LOG_SUFFIX} \
    chickarmane2008nanoggata6${LOG_SUFFIX} \
    kholodenko2000mapkcascade${LOG_SUFFIX}

# Common args for policy similarity
SEED_ARGS=()
if [[ -n "${N_SEEDS}" ]]; then
  SEED_ARGS=(--n_seeds "${N_SEEDS}")
fi
RESERVOIR_ARGS=(
  identity MLP LorenzSystem
  Tyson1999CircleLock Weimann2004CircadianOscillator
  Almeida2019CircadianClock Leloup1999CircadianClock
  Tyson1991CellCycle2Var Gardner1998CellCycleGoldbeter Gerard2010CellCycle
  Chickarmane2006StemCellSwitch Chickarmane2008NanogGata6
  Zatorsky2006P53Model4
  Gardner2000ToggleSwitch Liebal2012TranscriptionInhibition
  Markevich2004MAPKDoublePhosphorylation Kholodenko2000MAPKCascade
)

# Representative envs matching the paper policy-similarity figure (Figure 3)
REPR_ENVS=(CartPole-v1 Pendulum-v1 BankHeist-ram HalfCheetah-v4 PointMaze finger-spin)

# All 16 active envs
ALL_ENVS=(
  CartPole-v1 Acrobot-v1 MountainCarContinuous-v0 Pendulum-v1
  Reacher-v4 Pusher-v4 Swimmer-v4
  Hopper-v4 HalfCheetah-v4 HumanoidStandup-v4 PointMaze
  finger-spin
  BankHeist-ram KungFuMaster-ram CrazyClimber-ram Kangaroo-ram
)

# Plot policy similarity — 6 representative envs (paper Figure 7)
uv run src/rational_agent.py \
  --log_root ./results/log \
  --agents_root ./results/agents \
  --output_dir ./results/figures \
  --output_prefix "${POLICY_PREFIX}" \
  "${SEED_ARGS[@]}" \
  --device cpu \
  --log_suffix "${LOG_SUFFIX}" \
  --agent_suffix "${LOG_SUFFIX}" \
  --envs "${REPR_ENVS[@]}" \
  --reservoirs "${RESERVOIR_ARGS[@]}"

# Plot policy similarity — all 16 envs (appendix Figure 8), 4x4 grid
uv run src/rational_agent.py \
  --log_root ./results/log \
  --agents_root ./results/agents \
  --output_dir ./results/figures \
  --output_prefix policy_similarity_full \
  "${SEED_ARGS[@]}" \
  --device cpu \
  --n_cols 4 \
  --log_suffix "${LOG_SUFFIX}" \
  --agent_suffix "${LOG_SUFFIX}" \
  --envs "${ALL_ENVS[@]}" \
  --reservoirs "${RESERVOIR_ARGS[@]}"

# Semantic primitives analysis — GRN properties vs task primitives heatmap
uv run python -m src.semantic.semantic_analysis \
  --log_root ./results/log \
  --output_dir ./results/figures
