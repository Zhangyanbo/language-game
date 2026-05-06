import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from odes.ode_loader import load_ode_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ode_name", type=str, default="lorenz")
    parser.add_argument("--n_initial", type=int, default=64)
    parser.add_argument("--burn_in_steps", type=int, default=128)
    parser.add_argument("--burn_in_dt", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--sample_dt", type=float, default=0.05)
    parser.add_argument("--experiment", type=str, default="default")
    args = parser.parse_args()

    ode_model = load_ode_model(args.ode_name)
    ode_class = ode_model.__class__.__name__

    x = ode_model.random_initial_state(args.n_initial, device="cpu")
    x = ode_model.simulate(
        x, dt=args.burn_in_dt, T=args.burn_in_steps * args.burn_in_dt
    )[-1]

    states = []
    grads = []
    for _ in range(args.samples):
        x = ode_model.simulate(x, dt=args.sample_dt, T=args.sample_dt)[-1]
        g = ode_model(ode_model.denormalize(x))
        g = ode_model.normalize(g)
        states.append(x)
        grads.append(g)

    states = torch.stack(states, dim=0).reshape(-1, ode_model.NUM_NODES)
    grads = torch.stack(grads, dim=0).reshape(-1, ode_model.NUM_NODES)
    grads_norm = grads

    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states.cpu().numpy())
    grads_2d = pca.transform((states + grads).cpu().numpy()) - states_2d

    fig = plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    mag = np.linalg.norm(grads_norm.cpu().numpy(), axis=1)
    ax1.scatter(states_2d[:, 0], states_2d[:, 1], c=mag, s=12, cmap="viridis")

    # Subsample vectors to keep the field legible
    step = max(1, len(states_2d) // 200)
    idx = np.arange(0, len(states_2d), step)
    ax1.quiver(
        states_2d[idx, 0],
        states_2d[idx, 1],
        grads_2d[idx, 0],
        grads_2d[idx, 1],
        mag[idx],
        cmap="magma",
        alpha=0.7,
        scale=40,
        width=0.003,
    )
    ax1.set_title("Gradient Field (PCA plane)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    ax2 = plt.subplot(1, 2, 2)
    z = grads_norm.cpu().numpy().reshape(-1)
    ax2.hist(z, bins=30, density=True, color="gray", alpha=0.7)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1)
    ax2.set_title("Normalized Gradient Histogram")
    ax2.set_xlabel("z = (g - mu) / std")
    ax2.set_ylabel("Density")
    mean_z = float(z.mean())
    std_z = float(z.std())
    ax2.text(
        0.02,
        0.95,
        f"mean={mean_z:.3f}\\nstd={std_z:.3f}",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if args.experiment == "default":
        out_dir = "./results/figures"
    else:
        out_dir = f"./results/experiments/{args.experiment}/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = f"{out_dir}/gradient_reservoir_{ode_class.lower()}"
    plt.savefig(f"{out_prefix}.png", bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
