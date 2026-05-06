"""PPO hyperparameter tuning for Reservoir RL.

Runs multiple PPO configs on representative environments with LorenzSystem reservoir,
saves monitor CSVs, and generates comparison plots.

Usage:
    uv run src/tools/tune_ppo.py --round 1
    uv run src/tools/tune_ppo.py --round 2
    uv run src/tools/tune_ppo.py --round 3
    uv run src/tools/tune_ppo.py --plot          # plot all rounds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from train_agent import train_agent, load_reservoir

# Representative environments — covers discrete/continuous, classic/mujoco.
# 32768 = 2 rollouts at n_envs=16, n_steps=1024. Kept small for fast iteration.
TUNE_ENVS = {
    "CartPole-v1": 32_768,          # Classic discrete
    "Pendulum-v1": 32_768,          # Classic continuous
    "Swimmer-v4": 32_768,           # MuJoCo continuous
}

# Task-type grouping for per-type analysis
ENV_GROUPS = {
    "classic": ["CartPole-v1", "Pendulum-v1"],
    "mujoco": ["Swimmer-v4"],
}

RESULTS_DIR = Path("./results/tuning")


def define_configs(round_num):
    """Define parameter configs for each tuning round.

    Reference (RL Zoo3 tuned params for standard MlpPolicy):
      CartPole-v1:  n_steps=32,   batch=256, epochs=20, gamma=0.98,   gae=0.8,  lr=1e-3
      Pendulum-v1:  n_steps=1024, batch=64,  epochs=10, gamma=0.9,    gae=0.95, lr=1e-3
      Swimmer-v4:   n_steps=1024, batch=256, epochs=?,  gamma=0.9999, gae=0.98, lr=6e-4
      HalfCheetah:  n_steps=512,  batch=64,  epochs=20, gamma=0.98,   gae=0.92, lr=2e-5
    """
    if round_num == 1:
        # Round 1: Broad exploration — isolate key parameter axes.
        # Current defaults may be suboptimal for MLP (tuned for tiny GRN reservoirs).
        base = dict(learning_rate=2e-3, batch_size=64, n_epochs=5,
                    gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
                    n_steps=1024, n_envs=16)
        return {
            # A: Current project defaults (control)
            "A_baseline": {**base},
            # B: SB3-like defaults — lower lr, standard gamma/gae, more epochs, no entropy
            "B_sb3_like": {**base, "learning_rate": 3e-4, "n_epochs": 10,
                           "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0},
            # C: Big batch — reduces gradient variance, speeds up wall-clock
            "C_big_batch": {**base, "batch_size": 256},
            # D: Best-guess combination — big batch + moderate lr + more epochs + standard gamma
            "D_combined": {**base, "batch_size": 256, "learning_rate": 1e-3,
                           "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95},
        }
    elif round_num == 2:
        # Round 2: Refine lr and test clip_range.
        # Round 1 findings: gamma=0.99, gae=0.95, epochs=10 all confirmed better.
        # B_sb3_like (batch=64) and D_combined (batch=256) nearly tied.
        # Use batch=256 as base for speed (2.7× faster, same AUC).
        # Remaining unknowns: optimal lr (3e-4 vs 1e-3) and clip_range (never tested).
        fast = dict(learning_rate=1e-3, batch_size=256, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.0,
                    n_steps=1024, n_envs=16)
        return {
            # E: D_combined from Round 1 (batch=256, lr=1e-3)
            "E_r1_fast": {**fast},
            # F: lr=5e-4 — sweet spot between B (3e-4) and D (1e-3)?
            "F_lr_5e4": {**fast, "learning_rate": 5e-4},
            # G: lr=3e-4 with batch=256 — B's lr with D's batch speed
            "G_lr_3e4": {**fast, "learning_rate": 3e-4},
            # H: clip_range=0.1 — RL Zoo3 recommends for continuous control
            "H_clip01": {**fast, "clip_range": 0.1},
        }
    elif round_num == 3:
        # Round 3: Fine-tune remaining levers.
        # Confirmed from R1+R2: batch=256, gamma=0.99, gae=0.95, clip=0.2.
        # lr: 3e-4 to 1e-3 all work, 5e-4 slightly best mean AUC.
        # Remaining: n_epochs (10 vs 15), ent_coef (0 vs small), n_steps (512 vs 1024).
        best = dict(learning_rate=5e-4, batch_size=256, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.0,
                    n_steps=1024, n_envs=16)
        return {
            # I: Best from R2 (reference)
            "I_best_r2": {**best},
            # J: More epochs — better sample reuse?
            "J_epochs15": {**best, "n_epochs": 15},
            # K: Small entropy for exploration
            "K_ent001": {**best, "ent_coef": 0.001},
            # L: Shorter rollouts — more frequent updates
            "L_nsteps512": {**best, "n_steps": 512},
        }
    return {}


def run_config(config_name, params, env_id, timesteps, seed=0):
    """Train one (config, env) combination and return the monitor path."""
    exp_dir = RESULTS_DIR / f"round_{params.get('_round', 1)}"
    log_root = str(exp_dir / "log")
    agent_root = str(exp_dir / "agents")
    figure_root = str(exp_dir / "figures")

    n_envs = params.get("n_envs", 16)
    n_steps = params.get("n_steps", 1024)

    # Load reservoir — LorenzSystem (dim=3, faster than MLP dim=256)
    reservoir, reservoir_dim = load_reservoir("LorenzSystem")

    # Import PPO directly to pass custom params
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    from core.policy import ReservoirCriticPolicy
    from core.envs import resolve_env_id, needs_flatten, make_env_fn, _ensure_registered
    from train_agent import set_seed_everywhere, clean_log_folder
    import torch

    resolved_id = resolve_env_id(env_id)
    _ensure_registered(resolved_id)
    flatten = needs_flatten(resolved_id)

    log_reservoir_id = f"{config_name}_lorenz"
    monitor_dir = f"{log_root}/{log_reservoir_id}/{env_id}/seed_{seed}/"
    clean_log_folder(monitor_dir)
    set_seed_everywhere(seed, use_cuda=False)

    vec_env = make_vec_env(
        make_env_fn(resolved_id, flatten=flatten),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
    )
    vec_env = VecNormalize(
        vec_env, norm_obs=False, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0,
    )

    model = PPO(
        ReservoirCriticPolicy,
        vec_env,
        n_steps=n_steps,
        device="cpu",
        verbose=0,
        batch_size=params["batch_size"],
        ent_coef=params["ent_coef"],
        learning_rate=params["learning_rate"],
        n_epochs=params["n_epochs"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params.get("clip_range", 0.2),
        vf_coef=params.get("vf_coef", 0.5),
        max_grad_norm=params.get("max_grad_norm", 0.5),
        policy_kwargs={
            "reservoir": reservoir,
            "reservoir_dim": reservoir_dim,
        },
    )

    model.learn(total_timesteps=timesteps, progress_bar=False)
    vec_env.close()
    return monitor_dir


def load_rewards(monitor_dir):
    """Load episode rewards from monitor CSVs and return (timesteps, rewards)."""
    import glob
    files = glob.glob(f"{monitor_dir}/*.monitor.csv")
    all_t = []
    all_r = []
    for f in files:
        df = pd.read_csv(f, comment="#")
        if "r" in df.columns and "l" in df.columns:
            # Compute cumulative timesteps
            cum_t = df["l"].cumsum().values
            all_t.extend(cum_t.tolist())
            all_r.extend(df["r"].tolist())
    if not all_t:
        return np.array([]), np.array([])
    # Sort by timestep
    order = np.argsort(all_t)
    return np.array(all_t)[order], np.array(all_r)[order]


def smooth(y, window=50):
    """Rolling mean smoothing."""
    if len(y) < window:
        return y
    return pd.Series(y).rolling(window, min_periods=1).mean().values


def plot_comparison(round_num):
    """Generate comparison plot for a given round."""
    exp_dir = RESULTS_DIR / f"round_{round_num}"
    log_dir = exp_dir / "log"
    if not log_dir.exists():
        print(f"No data for round {round_num}")
        return

    # Discover configs from directory structure
    configs = sorted([d.name.replace("_lorenz", "") for d in log_dir.iterdir() if d.is_dir()])
    envs = list(TUNE_ENVS.keys())

    n_envs = len(envs)
    n_cols = min(n_envs, 2)
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(configs), 10)))

    for j, env_id in enumerate(envs):
        ax = axes[j // n_cols, j % n_cols]
        for i, config in enumerate(configs):
            monitor_dir = f"{log_dir}/{config}_lorenz/{env_id}/seed_0/"
            t, r = load_rewards(monitor_dir)
            if len(t) > 0:
                r_smooth = smooth(r)
                ax.plot(t, r_smooth, label=config, color=colors[i], alpha=0.8)
        ax.set_title(env_id, fontsize=10)
        ax.set_xlabel("Timesteps")
        if j == 0:
            ax.set_ylabel("Episode Reward")
        ax.grid(True, alpha=0.3)

    # Single legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(configs), 6),
                   bbox_to_anchor=(0.5, -0.05), fontsize=9)

    fig.suptitle(f"PPO Tuning — Round {round_num}", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = exp_dir / f"comparison_round_{round_num}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def compute_auc(monitor_dir, max_timesteps):
    """Compute area under the reward curve (higher = faster convergence)."""
    t, r = load_rewards(monitor_dir)
    if len(t) == 0:
        return float("nan")
    r_smooth = smooth(r, window=30)
    # Normalize timesteps to [0, 1] for fair comparison across envs
    t_norm = t / max_timesteps
    # Trapezoidal integration
    return float(np.trapz(r_smooth, t_norm))


def summarize_round(round_num):
    """Print summary statistics for a round."""
    exp_dir = RESULTS_DIR / f"round_{round_num}"
    log_dir = exp_dir / "log"
    if not log_dir.exists():
        print(f"No data for round {round_num}")
        return {}

    configs = sorted([d.name.replace("_lorenz", "") for d in log_dir.iterdir() if d.is_dir()])
    envs = list(TUNE_ENVS.keys())

    results = {}
    print(f"\n{'='*80}")
    print(f"  Round {round_num} Summary — AUC (higher = better)")
    print(f"{'='*80}")
    header = f"{'Config':<25}" + "".join(f"{e:<22}" for e in envs) + f"{'Mean AUC':<12}"
    print(header)
    print("-" * len(header))

    for config in configs:
        aucs = []
        row = f"{config:<25}"
        for env_id in envs:
            max_t = TUNE_ENVS[env_id]
            monitor_dir = f"{log_dir}/{config}_lorenz/{env_id}/seed_0/"
            auc = compute_auc(monitor_dir, max_t)
            aucs.append(auc)
            row += f"{auc:<22.2f}"
        mean_auc = np.nanmean(aucs)
        row += f"{mean_auc:<12.2f}"
        print(row)
        results[config] = {"aucs": {e: a for e, a in zip(envs, aucs)}, "mean_auc": float(mean_auc)}

    print(f"{'='*80}")

    # Also get final rewards
    print(f"\n  Final Reward (last 20 episodes mean)")
    print(f"{'='*80}")
    header = f"{'Config':<25}" + "".join(f"{e:<22}" for e in envs)
    print(header)
    print("-" * len(header))
    for config in configs:
        row = f"{config:<25}"
        for env_id in envs:
            monitor_dir = f"{log_dir}/{config}_lorenz/{env_id}/seed_0/"
            t, r = load_rewards(monitor_dir)
            if len(r) >= 20:
                final_r = np.mean(r[-20:])
            elif len(r) > 0:
                final_r = np.mean(r)
            else:
                final_r = float("nan")
            row += f"{final_r:<22.1f}"
            results[config].setdefault("final_rewards", {})[env_id] = float(final_r)
        print(row)
    print(f"{'='*80}\n")

    # Per-task-type analysis
    print(f"\n  Per-Task-Type Mean AUC")
    print(f"{'='*80}")
    header = f"{'Config':<25}" + "".join(f"{g:<22}" for g in ENV_GROUPS)
    print(header)
    print("-" * len(header))
    for config in configs:
        row = f"{config:<25}"
        for group_name, group_envs in ENV_GROUPS.items():
            group_aucs = [results[config]["aucs"].get(e, float("nan")) for e in group_envs]
            row += f"{np.nanmean(group_aucs):<22.2f}"
        print(row)
    print(f"{'='*80}\n")

    # Save results JSON
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="Only plot existing results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.plot:
        for r in range(1, 4):
            plot_comparison(r)
            summarize_round(r)
    else:
        configs = define_configs(args.round)
        if not configs:
            print(f"No configs defined for round {args.round}. Update define_configs().")
            exit(1)

        import time
        total = len(configs) * len(TUNE_ENVS)
        done = 0
        t_start = time.time()
        for config_name, params in configs.items():
            params["_round"] = args.round
            for env_id, timesteps in TUNE_ENVS.items():
                done += 1
                print(f"[{done}/{total}] {config_name} × {env_id} ({timesteps} steps)...", flush=True)
                try:
                    t0 = time.time()
                    run_config(config_name, params, env_id, timesteps, seed=args.seed)
                    elapsed = time.time() - t0
                    print(f"  Done in {elapsed:.0f}s", flush=True)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
        print(f"\nTotal round time: {time.time() - t_start:.0f}s", flush=True)

        plot_comparison(args.round)
        summarize_round(args.round)
