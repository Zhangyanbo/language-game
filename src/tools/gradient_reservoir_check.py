import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import random

import torch

from odes.ode_loader import load_ode_model


@torch.no_grad()
def gradient_reservoir_sanity_check(
    ode_model,
    n_initial: int = 128,
    burn_in_steps: int = 128,
    burn_in_dt: float = 0.05,
    samples_per_traj: int = 128,
    sample_dt: float = 0.05,
    device: str = "cpu",
):
    x = ode_model.random_initial_state(n_initial, device=device)
    x = ode_model.simulate(x, dt=burn_in_dt, T=burn_in_steps * burn_in_dt)[-1]
    grads = []

    for _ in range(samples_per_traj):
        x = ode_model.simulate(x, dt=sample_dt, T=sample_dt)[-1]
        x_denorm = ode_model.denormalize(x)
        g = ode_model(x_denorm)
        g = ode_model.normalize(g)
        grads.append(g)

    grads = torch.stack(grads, dim=0).reshape(-1, ode_model.NUM_NODES)
    mu = grads.mean(dim=0)
    std = grads.std(dim=0)
    return mu.cpu(), std.cpu()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ode_name", type=str, default="lorenz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_initial", type=int, default=128)
    parser.add_argument("--burn_in_steps", type=int, default=128)
    parser.add_argument("--burn_in_dt", type=float, default=0.05)
    parser.add_argument("--samples_per_traj", type=int, default=128)
    parser.add_argument("--sample_dt", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ode_model = load_ode_model(args.ode_name)
    mu, std = gradient_reservoir_sanity_check(
        ode_model,
        n_initial=args.n_initial,
        burn_in_steps=args.burn_in_steps,
        burn_in_dt=args.burn_in_dt,
        samples_per_traj=args.samples_per_traj,
        sample_dt=args.sample_dt,
        device=args.device,
    )
    print("Gradient reservoir sanity check:")
    print("  mean:", mu.tolist())
    print("  std:", std.tolist())


if __name__ == "__main__":
    main()
