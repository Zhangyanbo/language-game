"""Analyze encoder/decoder node importance for GRN reservoirs across RL tasks.

For each GRN node i:
  Encoder importance = ||W_E[i, :]||_2  (row norm of Linear(obs_dim -> NUM_NODES) weight)
  Decoder importance = ||W_D[:, i]||_2  (col norm of Linear(NUM_NODES -> action_dim) weight)

Outputs (in results/figures/node_importance/):
  <grn>_weights.png  — 2-row heatmap (encoder + decoder) across all envs
  <grn>_scatter.png  — enc vs dec scatter per env (triangles = action-only nodes)
  summary_gini.png   — Gini coefficient of decoder per GRN x env
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

AGENTS_DIR = "results/agents"
OUTPUT_DIR = "results/figures/node_importance"

SKIP_RESERVOIRS = {"identity_gradient", "mlp_gradient"}

ENV_SHORT = {
    "CartPole-v1": "CartPole",
    "Acrobot-v1": "Acrobot",
    "MountainCarContinuous-v0": "MtnCar",
    "Pendulum-v1": "Pendulum",
    "HalfCheetah-v4": "Cheetah",
    "Hopper-v4": "Hopper",
    "Swimmer-v4": "Swimmer",
    "Reacher-v4": "Reacher",
    "Pusher-v4": "Pusher",
    "HumanoidStandup-v4": "Humanoid",
    "PointMaze": "Maze",
    "finger-spin": "FngSpin",
}

GRN_SHORT = {
    "almeida2019circadianclock": "Almeida19 Circ",
    "chickarmane2006stemcellswitch": "Chickar06 Stem",
    "chickarmane2008nanoggata6": "Chickar08 Nanog",
    "gardner1998cellcyclegoldbeter": "Gardner98 CC",
    "gardner2000toggleswitch": "Gardner00 Toggle",
    "gerard2010cellcycle": "Gerard10 CC",
    "kholodenko2000mapkcascade": "Kholod00 MAPK",
    "leloup1999circadianclock": "Leloup99 Circ",
    "liebal2012transcriptioninhibition": "Liebal12 Transc",
    "lorenzsystem": "Lorenz (chaotic)",
    "markevich2004mapkdoublephosphorylation": "Markev04 MAPK",
    "tyson1991cellcycle2var": "Tyson91 CC",
    "tyson1999circlelock": "Tyson99 CircLock",
    "weimann2004circadianoscillator": "Weimann04 Circ",
    "zatorsky2006p53model4": "Zatorsky06 p53",
}


def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n


def extract_weights(zip_path):
    """Return (enc_norms, dec_norms) of shape (NUM_NODES,) each."""
    from stable_baselines3 import PPO

    model = PPO.load(zip_path, device="cpu")
    policy = model.policy
    # Encoder: Linear(obs_dim, NUM_NODES) -> weight shape (NUM_NODES, obs_dim)
    W_E = policy.features_extractor.encoder.weight.detach().cpu().numpy()
    # Decoder: Linear(NUM_NODES, action_dim) -> weight shape (action_dim, NUM_NODES)
    W_D = policy.action_net.weight.detach().cpu().numpy()
    enc = np.linalg.norm(W_E, axis=1)  # row norms -> (NUM_NODES,)
    dec = np.linalg.norm(W_D, axis=0)  # col norms -> (NUM_NODES,)
    return enc, dec


def load_grn_data():
    """Return dict: grn_name -> env_name -> {'enc': array, 'dec': array} (mean over seeds)."""
    data = {}
    for res_dir in sorted(os.listdir(AGENTS_DIR)):
        if res_dir in SKIP_RESERVOIRS:
            continue
        res_path = os.path.join(AGENTS_DIR, res_dir)
        if not os.path.isdir(res_path):
            continue
        grn = res_dir.replace("_gradient", "")
        data[grn] = {}
        for env_name in sorted(os.listdir(res_path)):
            env_path = os.path.join(res_path, env_name)
            seeds = sorted(glob.glob(os.path.join(env_path, "seed_*.zip")))
            if not seeds:
                continue
            encs, decs = [], []
            for seed_path in seeds:
                try:
                    enc, dec = extract_weights(seed_path)
                    encs.append(enc)
                    decs.append(dec)
                except Exception as e:
                    print(f"  skip {seed_path}: {e}")
            if encs:
                data[grn][env_name] = {
                    "enc": np.mean(encs, axis=0),
                    "dec": np.mean(decs, axis=0),
                }
        print(f"  {grn}: loaded {len(data[grn])} envs")
    return data


def _col_normalize(mat):
    """Normalize each column to sum to 1 (relative importance within each env)."""
    col_sums = mat.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    return mat / col_sums


def plot_heatmap(grn, env_data, output_dir):
    """2-row heatmap: top=encoder, bottom=decoder. Columns are envs, rows are nodes."""
    envs = sorted(env_data.keys())
    if not envs:
        return
    num_nodes = len(next(iter(env_data.values()))["enc"])

    enc_mat = np.array([env_data[e]["enc"] for e in envs]).T  # (num_nodes, num_envs)
    dec_mat = np.array([env_data[e]["dec"] for e in envs]).T

    enc_n = _col_normalize(enc_mat)
    dec_n = _col_normalize(dec_mat)

    env_labels = [ENV_SHORT.get(e, e) for e in envs]
    node_labels = [f"n{i}" for i in range(num_nodes)]

    fig_w = max(5.5, len(envs) * 0.55)
    fig_h = fig_w / 1.618 * 1.9
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h))

    for ax, mat, subtitle in zip(
        axes,
        [enc_n, dec_n],
        ["Encoder weight norm (col-normalized)", "Decoder weight norm (col-normalized)"],
    ):
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=mat.max())
        ax.set_xticks(range(len(envs)))
        ax.set_xticklabels(env_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(num_nodes))
        ax.set_yticklabels(node_labels, fontsize=7)
        ax.set_title(subtitle, fontsize=8, fontweight="normal")
        for j in range(len(envs) - 1):
            ax.axvline(j + 0.5, color="white", linewidth=0.4)
        for i in range(num_nodes - 1):
            ax.axhline(i + 0.5, color="white", linewidth=0.4)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="rel. importance")

    short = GRN_SHORT.get(grn, grn)
    fig.suptitle(f"{short}  ({num_nodes} nodes)", fontsize=9, fontweight="normal")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{grn}_weights.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_scatter(grn, env_data, output_dir):
    """Scatter enc vs dec importance: one subplot per env. Triangles = action-only nodes."""
    envs = sorted(env_data.keys())
    if not envs:
        return
    num_nodes = len(next(iter(env_data.values()))["enc"])

    ncols = min(5, len(envs))
    nrows = (len(envs) + ncols - 1) // ncols
    fig_w = ncols * 2.0
    fig_h = nrows * 2.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    colors = plt.cm.tab20(np.linspace(0, 1, max(num_nodes, 2)))

    for idx, env in enumerate(envs):
        ax = axes[idx // ncols][idx % ncols]
        enc = env_data[env]["enc"]
        dec = env_data[env]["dec"]
        med_enc = np.median(enc)
        med_dec = np.median(dec)

        for i in range(num_nodes):
            action_only = dec[i] > med_dec and enc[i] < med_enc
            marker = "^" if action_only else "o"
            ax.scatter(enc[i], dec[i], color=colors[i], marker=marker, s=28, zorder=3)
            ax.annotate(str(i), (enc[i], dec[i]), fontsize=5, ha="left", va="bottom")

        ax.axvline(med_enc, color="gray", linewidth=0.5, linestyle="--")
        ax.axhline(med_dec, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(ENV_SHORT.get(env, env), fontsize=7, fontweight="normal")
        ax.set_xlabel("Enc norm", fontsize=5)
        ax.set_ylabel("Dec norm", fontsize=5)
        ax.tick_params(labelsize=5)

    for idx in range(len(envs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    short = GRN_SHORT.get(grn, grn)
    fig.suptitle(
        f"{short} — encoder vs decoder node importance\n(^ = action-only: high dec, low enc)",
        fontsize=8,
        fontweight="normal",
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{grn}_scatter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_summary_gini(data, output_dir):
    """Heatmap: rows=GRN models sorted by NUM_NODES, cols=envs, color=decoder Gini."""
    all_envs = sorted({e for grn_data in data.values() for e in grn_data})

    def num_nodes(grn):
        d = data[grn]
        return len(next(iter(d.values()))["dec"]) if d else 0

    grns = sorted(data.keys(), key=num_nodes)

    gini_mat = np.full((len(grns), len(all_envs)), np.nan)
    for i, grn in enumerate(grns):
        for j, env in enumerate(all_envs):
            if env in data[grn]:
                gini_mat[i, j] = gini(data[grn][env]["dec"])

    env_labels = [ENV_SHORT.get(e, e) for e in all_envs]
    grn_labels = [
        f"{GRN_SHORT.get(g, g)} ({num_nodes(g)}n)" for g in grns
    ]

    fig_w = max(8, len(all_envs) * 0.6)
    fig_h = max(4, len(grns) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("lightgray")
    masked = np.ma.array(gini_mat, mask=np.isnan(gini_mat))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(all_envs)))
    ax.set_xticklabels(env_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(grns)))
    ax.set_yticklabels(grn_labels, fontsize=7)

    for j in range(len(all_envs) - 1):
        ax.axvline(j + 0.5, color="white", linewidth=0.4)
    for i in range(len(grns) - 1):
        ax.axhline(i + 0.5, color="white", linewidth=0.4)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Gini (decoder importance)", pad=0.02)
    ax.set_title(
        "Decoder node concentration (Gini)\nhigher = weight concentrated on fewer nodes",
        fontsize=9,
        fontweight="normal",
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "summary_gini.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def print_top_nodes(data):
    print("\n=== Top-2 decoder nodes per GRN x env ===")
    for grn in sorted(data, key=lambda g: len(next(iter(data[g].values()))["dec"]) if data[g] else 0):
        for env in sorted(data[grn]):
            dec = data[grn][env]["dec"]
            top2 = np.argsort(dec)[::-1][:2]
            total = dec.sum()
            pct = dec[top2] / total * 100 if total > 0 else dec[top2] * 0
            short = GRN_SHORT.get(grn, grn)
            print(
                f"  {short:28s}  {ENV_SHORT.get(env, env):10s}  "
                f"top: n{top2[0]}({pct[0]:.0f}%), n{top2[1]}({pct[1]:.0f}%)"
                f"  gini={gini(dec):.2f}"
            )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading weights from all GRN checkpoints...")
    data = load_grn_data()

    for grn in sorted(data):
        if not data[grn]:
            continue
        print(f"\n[{grn}]")
        plot_heatmap(grn, data[grn], OUTPUT_DIR)
        plot_scatter(grn, data[grn], OUTPUT_DIR)

    print("\nPlotting summary Gini heatmap...")
    plot_summary_gini(data, OUTPUT_DIR)

    print_top_nodes(data)
    print("\nDone. Figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
