import os
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from core.video import save_video
from core.policy import ReservoirCriticPolicy, IdentityReservoir, MLPTrainable
from core.envs import resolve_env_id, needs_flatten, make_env_fn, _ensure_registered, is_atari_ram
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from typing import Union, Callable
import random
import numpy as np


def set_seed_everywhere(seed: int, use_cuda: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(
        True, warn_only=True
    )  # warn_only=True to avoid crashing


def plot_reward(
    env_id: str, seed: int, reservoir: str, log_root: str, figure_root: str
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import glob
    import os

    # create folder if not exists
    os.makedirs(
        f"{figure_root}/reward_curve/{reservoir.lower()}/{env_id}/", exist_ok=True
    )

    # Get all monitor files in the folder
    monitor_files = glob.glob(
        f"{log_root}/{reservoir.lower()}/{env_id}/seed_{seed}/*.monitor.csv"
    )

    all_rewards = []
    max_episodes = 0

    # Load all monitor files
    for file in monitor_files:
        df = pd.read_csv(file, comment="#")
        episode_rewards = df["r"].tolist()
        all_rewards.append(episode_rewards)
        max_episodes = max(max_episodes, len(episode_rewards))

    # Pad shorter sequences with NaN and compute average
    averaged_rewards = []
    for episode_idx in range(max_episodes):
        episode_rewards = []
        for env_rewards in all_rewards:
            if episode_idx < len(env_rewards):
                episode_rewards.append(env_rewards[episode_idx])
        if episode_rewards:  # Only average if we have data
            averaged_rewards.append(sum(episode_rewards) / len(episode_rewards))

    os.makedirs(figure_root, exist_ok=True)
    plt.plot(averaged_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Average Reward Curve")
    os.makedirs(f"{figure_root}/reward_curve", exist_ok=True)
    plt.savefig(
        f"{figure_root}/reward_curve/{reservoir.lower()}/{env_id}/seed_{seed}.png"
    )
    plt.close()


def save_agent(env_id: str, seed: int, reservoir: str, model, agent_root: str):
    import os

    os.makedirs(f"{agent_root}/{reservoir.lower()}/{env_id}/", exist_ok=True)
    model.save(f"{agent_root}/{reservoir.lower()}/{env_id}/seed_{seed}.zip")


def clean_log_folder(folder: str):
    import os
    import glob

    # check if folder exists
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist")
        return

    for file in glob.glob(f"./{folder}/*.monitor.csv"):
        os.remove(file)
        print(f"Removed {file}")


class GradientReservoir:
    METHOD = "direct"

    def __init__(self, ode_model):
        self.ode_model = ode_model

    def to(self, device: str):
        return self

    def eval(self):
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return self

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        x_denorm = self.ode_model.denormalize(x)
        min_val = getattr(self.ode_model, "MIN_VALUE", None)
        max_val = getattr(self.ode_model, "MAX_VALUE", None)
        if min_val is not None or max_val is not None:
            # STE (Straight-Through Estimator) projection:
            # forward pass uses clamped value; backward pass gets full gradient.
            x_clamped = torch.clamp(x_denorm, min=min_val, max=max_val)
            x_denorm = x_denorm + (x_clamped - x_denorm).detach()
        g = self.ode_model(x_denorm)
        # Safety net: replace NaN/Inf with zero to prevent training crash
        g = torch.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.ode_model.normalize(g)


def load_reservoir(reservoir: str):
    if reservoir.lower() == "identity":
        return IdentityReservoir(), 256
    if reservoir.lower() == "mlp":
        return MLPTrainable(256), 256

    from odes.ode_loader import load_ode_model, _ODE_REGISTRY

    # Map ClassName (lowered) -> short registry key
    ode_lookup = {cls.__name__.lower(): key for key, cls in _ODE_REGISTRY.items()}
    ode_name = ode_lookup.get(reservoir.lower())
    if ode_name is None:
        raise ValueError(
            f"Unknown ODE name for gradient reservoir: {reservoir}. "
            f"Known: {list(ode_lookup.keys())}"
        )
    ode_model = load_ode_model(ode_name)
    model = GradientReservoir(ode_model)
    return model, ode_model.NUM_NODES


def train_agent(
    env_id: str,
    reservoir: Union[str, Callable],
    seed: int,
    n_envs: int = 16,
    n_steps: int = 512,
    total_timesteps: int = 10_0000,
    reservoir_dim: int = None,
    device: str = "cpu",
    verbose: int = 1,
    log_root: str = "./results/log",
    agent_root: str = "./results/agents",
    figure_root: str = "./results/figures",
    render_mode: str = None,
):
    if isinstance(reservoir, str):
        log_reservoir_id = f"{reservoir.lower()}_gradient"
    else:
        log_reservoir_id = f"{reservoir.__class__.__name__.lower()}_gradient"

    # Resolve aliases (e.g. Pusher-v4 → Pusher-v5, finger-spin → dm_control/finger-spin-v0)
    resolved_id = resolve_env_id(env_id)
    _ensure_registered(resolved_id)
    extra_kwargs = {"obs_type": "ram"} if is_atari_ram(env_id) else {}
    flatten = needs_flatten(resolved_id, extra_kwargs or None)

    clean_log_folder(f"{log_root}/{log_reservoir_id}/{env_id}/seed_{seed}/")
    set_seed_everywhere(seed, use_cuda=(device != "cpu"))

    if isinstance(reservoir, str):
        reservoir, reservoir_dim = load_reservoir(reservoir)

    vec_env = make_vec_env(
        make_env_fn(resolved_id, flatten=flatten, render_mode=render_mode, extra_kwargs=extra_kwargs or None),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=f"{log_root}/{log_reservoir_id}/{env_id}/seed_{seed}/",
    )

    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,  # keep the original observation space
        norm_reward=True,  # only normalize the reward visible to the algorithm
        clip_obs=10.0,
        clip_reward=10.0,
    )

    model = PPO(
        ReservoirCriticPolicy,
        vec_env,
        n_steps=n_steps,
        device=device,
        verbose=verbose,
        batch_size=256,
        ent_coef=0.0,
        learning_rate=5e-4,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs={
            "reservoir": reservoir,
            "reservoir_dim": reservoir_dim,
        },
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.policy.set_training_mode(False)
    return model, vec_env


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", "-e", type=str, nargs="+", default=["CartPole-v1"])
    parser.add_argument(
        "--reservoir",
        "-r",
        type=str,
        nargs="+",
        default=[
            "LorenzSystem",
            "Tyson1999CircleLock",
            "Markevich2004MAPKDoublePhosphorylation",
            "Tyson1991CellCycle2Var",
        ],
    )
    parser.add_argument("--total_timesteps", "-t", type=int, default=10_0000)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--n_experiments", "-n", type=int, default=1)
    parser.add_argument("--n_envs", "-v", type=int, default=16)
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument(
        "--save_video", action="store_true", help="Save video of the trained agent"
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    args = parser.parse_args()
    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"

    if len(args.env_id) > 1 or len(args.reservoir) > 1:
        verbose = 0
    else:
        verbose = 1

    if args.experiment == "default":
        log_root = "./results/log"
        agent_root = "./results/agents"
        figure_root = "./results/figures"
    else:
        base_root = f"./results/experiments/{args.experiment}"
        log_root = f"{base_root}/log"
        agent_root = f"{base_root}/agents"
        figure_root = f"{base_root}/figures"

    bar = tqdm(total=len(args.env_id) * len(args.reservoir) * args.n_experiments)
    for env_id in args.env_id:
        for reservoir in args.reservoir:
            for i in range(args.n_experiments):
                current_seed = args.seed + i
                # Only enable render for the first seed when --save_video
                want_video = args.save_video and i == 0
                bar.set_description(
                    f"env: {env_id}, reservoir: {reservoir}, exp: {i+1}/{args.n_experiments}"
                )

                try:
                    model, vec_env = train_agent(
                        env_id,
                        reservoir,
                        current_seed,
                        total_timesteps=args.total_timesteps,
                        verbose=verbose,
                        device=device,
                        n_envs=args.n_envs,
                        log_root=log_root,
                        agent_root=agent_root,
                        figure_root=figure_root,
                        render_mode="rgb_array" if want_video else None,
                    )
                except Exception as e:
                    print(f"ERROR: training failed for {env_id} × {reservoir} seed={current_seed}: {e}")
                    bar.update(1)
                    continue

                log_reservoir_id = f"{reservoir.lower()}_gradient"
                plot_reward(
                    env_id,
                    current_seed,
                    log_reservoir_id,
                    log_root=log_root,
                    figure_root=figure_root,
                )
                if want_video:
                    try:
                        save_video(
                            vec_env, env_id, current_seed, reservoir, model, max_env=4
                        )
                    except Exception as e:
                        print(f"WARNING: video recording failed ({e}), skipping.")
                save_agent(
                    env_id, current_seed, log_reservoir_id, model, agent_root=agent_root
                )
                bar.update(1)
