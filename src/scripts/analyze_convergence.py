"""Analyze monitor CSVs to detect convergence and recommend timesteps."""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import io


def load_env_rewards(log_root: str, reservoir: str, env: str, seed_dir: str):
    """Load rewards and compute global cumulative timesteps from monitor CSVs."""
    seed_path = os.path.join(log_root, reservoir, env, seed_dir)
    if not os.path.isdir(seed_path):
        return None, None

    monitor_files = glob.glob(os.path.join(seed_path, "*.monitor.csv"))
    n_envs = len(monitor_files)
    if n_envs == 0:
        return None, None

    all_r = []
    all_steps = []
    for csv_file in sorted(
        monitor_files, key=lambda x: int(os.path.basename(x).split(".")[0])
    ):
        with open(csv_file, "r") as f:
            lines = f.readlines()
        lines = [l for l in lines if not l.lstrip().startswith("#") and l.strip()]
        if not lines:
            continue
        df = pd.read_csv(io.StringIO("".join(lines)))
        if df.empty:
            continue
        cum = df["l"].cumsum() * n_envs
        all_r.extend(df["r"].tolist())
        all_steps.extend(cum.tolist())

    if not all_r:
        return None, None

    steps = np.array(all_steps)
    rewards = np.array(all_r)

    idx = np.argsort(steps)
    return rewards[idx], steps[idx]


def detect_convergence(rewards, steps, n_bins=50, patience_frac=0.15):
    """
    Detect convergence point where reward improvement becomes negligible.

    Returns (converge_step, total_steps, bin_means).

    Algorithm:
    1. Bin rewards into n_bins
    2. Compute rolling improvement (slope) over a window of patience_frac * n_bins
    3. Convergence = first point where rolling improvement drops below 5% of
       the max improvement rate AND stays low for the rest of the training.
    """
    total_steps = float(max(steps))
    bin_edges = np.linspace(0, total_steps, n_bins + 1)
    bin_means = []
    bin_centers = []

    for i in range(n_bins):
        mask = (steps >= bin_edges[i]) & (steps < bin_edges[i + 1])
        if mask.any():
            bin_means.append(float(np.mean(rewards[mask])))
        else:
            bin_means.append(np.nan)
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    # Forward-fill NaN
    for i in range(1, len(bin_means)):
        if np.isnan(bin_means[i]):
            bin_means[i] = bin_means[i - 1]

    bin_means = np.array(bin_means)
    bin_centers = np.array(bin_centers)

    # Smooth with a small window
    window = max(3, n_bins // 10)
    smoothed = np.convolve(bin_means, np.ones(window) / window, mode="same")

    # Compute improvement per bin (absolute change)
    improvements = np.abs(np.diff(smoothed))

    # Total range of reward
    reward_range = max(np.max(smoothed) - np.min(smoothed), 1e-6)

    # Normalized improvement per bin
    norm_imp = improvements / reward_range

    # Convergence criterion: improvement per bin < threshold for patience bins
    threshold = 0.01  # 1% of total range per bin
    patience = max(3, int(n_bins * patience_frac))

    converge_bin = n_bins - 1  # default: never converged
    for i in range(len(norm_imp) - patience):
        if all(norm_imp[i : i + patience] < threshold):
            converge_bin = i
            break

    converge_step = int(bin_centers[converge_bin])

    # Check if still improving at the end
    last_quarter = bin_means[-n_bins // 4 :]
    mid_quarter = bin_means[n_bins // 2 : 3 * n_bins // 4]
    end_mean = np.mean(last_quarter)
    mid_mean = np.mean(mid_quarter)
    late_improvement = abs(end_mean - mid_mean) / max(reward_range, 1e-6)
    still_improving = late_improvement > 0.03  # >3% of range still improving

    return converge_step, total_steps, bin_means, still_improving


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = {}

    # Find all reservoir dirs
    for res_dir in sorted(os.listdir(args.log_root)):
        res_path = os.path.join(args.log_root, res_dir)
        if not os.path.isdir(res_path):
            continue

        for env_name in sorted(os.listdir(res_path)):
            env_path = os.path.join(res_path, env_name)
            if not os.path.isdir(env_path):
                continue

            # Use first seed found
            seed_dirs = sorted(os.listdir(env_path))
            if not seed_dirs:
                continue

            rewards, steps = load_env_rewards(
                args.log_root, res_dir, env_name, seed_dirs[0]
            )
            if rewards is None:
                continue

            converge_step, total_steps, bin_means, still_improving = (
                detect_convergence(rewards, steps)
            )

            # Recommended steps = convergence × 1.5, rounded to nearest 50k, floor 200k
            recommended = int(converge_step * 1.5)
            recommended = max(200000, round(recommended / 50000) * 50000)

            # If still improving at the end, recommend full budget × 1.5
            if still_improving:
                recommended = max(recommended, int(total_steps * 1.5))
                recommended = round(recommended / 50000) * 50000

            start_r = float(np.nanmean(bin_means[:5]))
            end_r = float(np.nanmean(bin_means[-5:]))

            results[env_name] = {
                "total_steps_trained": int(total_steps),
                "converge_step": int(converge_step),
                "still_improving": bool(still_improving),
                "recommended_steps": int(recommended),
                "start_reward": round(float(start_r), 2),
                "end_reward": round(float(end_r), 2),
            }

            status = "CONVERGED" if not still_improving else "NOT CONVERGED"
            print(
                f"  {env_name:35s}  {status:15s}  "
                f"converge@{converge_step:>10,}  "
                f"recommend={recommended:>10,}  "
                f"reward: {start_r:>10.1f} -> {end_r:>10.1f}"
            )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
