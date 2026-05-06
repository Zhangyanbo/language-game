import sys
import core.policy
sys.modules.setdefault("policy", core.policy)

from stable_baselines3 import PPO
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym
from plot_rewards import Records, average_rewards
import numpy as np
from gymnasium.spaces import Box, Discrete
import glob
import os


def policy_vector(policy, s):
    obs, _ = policy.obs_to_tensor(s)
    with torch.no_grad():
        features = policy.extract_features(obs)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        vec = policy.action_net(latent_pi)
    return vec


@torch.no_grad()
def get_action_vector(policy, x):
    features = policy.extract_features(x)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent_pi)


def load_model(
    ode_name: str,
    env_name: str,
    seed: int = 0,
    agents_root: str = "./results/agents",
    agent_suffix: str = "",
    device: str = "cpu",
):
    model = PPO.load(
        f"{agents_root}/{ode_name.lower()}{agent_suffix}/{env_name}/seed_{seed}.zip"
    )

    policy = model.policy
    policy.set_training_mode(False)
    policy.to(device)

    def V(s):
        obs, vec = policy.obs_to_tensor(s)
        v = policy.predict_values(obs).cpu()
        return v

    def pi(s, *args, **kwargs):
        a = policy_vector(policy, s)
        return a

    return pi, V, model


def list_seed_ids(
    agents_root: str,
    reservoir_name: str,
    env_name: str,
    agent_suffix: str,
) -> list[int]:
    agent_dir = os.path.join(
        agents_root, f"{reservoir_name.lower()}{agent_suffix}", env_name
    )
    pattern = os.path.join(agent_dir, "seed_*.zip")
    seed_ids = []
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        if not base.startswith("seed_") or not base.endswith(".zip"):
            continue
        seed_str = base[len("seed_") : -len(".zip")]
        try:
            seed_ids.append(int(seed_str))
        except ValueError:
            continue
    return sorted(set(seed_ids))


@torch.no_grad()
def get_latent(policy, x):
    features = policy.extract_features(x)
    latent_pi, latent_vf = policy.mlp_extractor(features)
    # our action network is identity
    return latent_pi


def action_similarity_matrix(
    policies, env, n: int = 64, reference_policies: list = None, device: str = "cpu"
):
    x0 = [env.reset()[0] for _ in range(n)]
    x0 = np.stack(x0)
    x = torch.from_numpy(x0).float().reshape(n, -1).to(device)

    def _append_unit_dim_if_1d(action_tensor: torch.Tensor) -> torch.Tensor:
        if action_tensor.shape[-1] == 1:
            ones = torch.ones_like(action_tensor) * 0.1
            action_tensor = torch.cat([action_tensor, ones], dim=-1)
        return action_tensor

    actions = []
    for i in range(len(policies)):
        a = get_action_vector(policies[i].policy, x)  # [n, d]
        if isinstance(env.action_space, Discrete):
            a = a - a.mean(dim=-1, keepdim=True)
        a = _append_unit_dim_if_1d(a)
        v = a.reshape(-1).detach().cpu().numpy()
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
        actions.append(v)

    actions_reference = []
    if reference_policies is not None:
        for i in range(len(reference_policies)):
            a = get_action_vector(reference_policies[i].policy, x)
            if isinstance(env.action_space, Discrete):
                a = a - a.mean(dim=-1, keepdim=True)
            a = _append_unit_dim_if_1d(a)
            v = a.reshape(-1).detach().cpu().numpy()
            v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
            actions_reference.append(v)
    else:
        actions_reference = actions

    return np.stack(actions) @ np.stack(actions_reference).T, actions, actions_reference


def sorted_policy(
    policies: list[torch.nn.Module],
    reservoir_name: str,
    task_name: str,
    reward_records: Records,
    n_bins: int = 10,
    seed_ids: list[int] | None = None,
):
    n_policies = len(policies)
    if seed_ids is None:
        record = reward_records[reservoir_name, task_name]
    else:
        record = {
            seed: reward_records[reservoir_name, task_name, seed] for seed in seed_ids
        }
    rewards = np.stack(average_rewards(record, n=n_bins))[:n_policies, -1]
    sorted_idx = np.argsort(rewards)
    sorted_policies = [policies[i] for i in sorted_idx]
    return sorted_policies


def policy_similarity_matrix(policies: dict, env, device: str = "cpu"):
    dim = sum([len(policy_list) for policy_list in policies.values()])
    sim_matrix = []

    for i, (name, policy) in enumerate(policies.items()):
        sim_list = []
        for j, (name_ref, policy_ref) in enumerate(policies.items()):
            m, _, _ = action_similarity_matrix(
                policy, reference_policies=policy_ref, env=env, device=device
            )
            sim_list.append(m)
        sim_matrix.append(sim_list)

    # merge block matrix
    sim_matrix = np.block(sim_matrix)
    return sim_matrix


if __name__ == "__main__":
    import argparse
    from core.envs import resolve_env_id, _ensure_registered, is_atari_ram
    from gymnasium.wrappers import FlattenObservation

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=str, default="./results/log")
    parser.add_argument("--agents_root", type=str, default="./results/agents")
    parser.add_argument("--output_dir", type=str, default="./results/figures")
    parser.add_argument("--output_prefix", type=str, default="policy_similarity_gradient")
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help="Seeds per reservoir/env; default uses all available seeds",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--n_cols",
        type=int,
        default=3,
        help="Number of subplot columns in the grid layout",
    )
    parser.add_argument(
        "--log_suffix",
        type=str,
        default="_gradient",
        help="Log directory suffix, default '_gradient'",
    )
    parser.add_argument(
        "--agent_suffix",
        type=str,
        default="_gradient",
        help="Agents directory suffix, default '_gradient'",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=[
            "CartPole-v1",
            "Pendulum-v1",
            "BankHeist-ram",
            "HalfCheetah-v4",
            "PointMaze",
            "finger-spin",
        ],
    )
    parser.add_argument(
        "--reservoirs",
        nargs="+",
        default=[
            "identity",
            "MLP",
            "LorenzSystem",
            "Tyson1999CircleLock",
            "Weimann2004CircadianOscillator",
            "Almeida2019CircadianClock",
            "Leloup1999CircadianClock",
            "Tyson1991CellCycle2Var",
            "Gardner1998CellCycleGoldbeter",
            "Gerard2010CellCycle",
            "Chickarmane2006StemCellSwitch",
            "Chickarmane2008NanogGata6",
            "Zatorsky2006P53Model4",
            "Gardner2000ToggleSwitch",
            "Liebal2012TranscriptionInhibition",
            "Markevich2004MAPKDoublePhosphorylation",
            "Kholodenko2000MAPKCascade",
        ],
    )
    args = parser.parse_args()

    reward_records = Records(args.log_root)

    reservoir_names = args.reservoirs
    display_names = {
        "identity": "Linear",
        "MLP": "MLP",
        "LorenzSystem": "Lorenz",
        "Tyson1999CircleLock": "Tyson99",
        "Weimann2004CircadianOscillator": "Weimann04",
        "Almeida2019CircadianClock": "Almeida19",
        "Leloup1999CircadianClock": "Leloup99",
        "Tyson1991CellCycle2Var": "Tyson91",
        "Gardner1998CellCycleGoldbeter": "Gardner98",
        "Gerard2010CellCycle": "Gerard10",
        "Chickarmane2006StemCellSwitch": "StemCell",
        "Chickarmane2008NanogGata6": "NanogGata6",
        "Zatorsky2006P53Model4": "p53",
        "Gardner2000ToggleSwitch": "Toggle",
        "Liebal2012TranscriptionInhibition": "Liebal12",
        "Markevich2004MAPKDoublePhosphorylation": "MAPK",
        "Kholodenko2000MAPKCascade": "MAPKcasc",
    }

    n_envs = len(args.envs)
    n_cols = args.n_cols
    n_rows = int(np.ceil(n_envs / n_cols)) if n_envs > 0 else 1

    TEXTWIDTH = 6.5
    subplot_w = TEXTWIDTH / n_cols
    row_h = subplot_w * 0.9
    fig_h = n_rows * row_h + 0.3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(TEXTWIDTH, fig_h), squeeze=False)
    axes = axes.flatten()

    def resolve_seed_ids(reservoir_name: str, task_name: str) -> list[int]:
        available = list_seed_ids(
            agents_root=args.agents_root,
            reservoir_name=reservoir_name,
            env_name=task_name,
            agent_suffix=args.agent_suffix,
        )
        if not available:
            return []
        if args.n_seeds is None or args.n_seeds <= 0:
            return available
        if args.n_seeds > len(available):
            print(
                f"[warning] requested {args.n_seeds} seeds, but only {len(available)} "
                f"available for {reservoir_name}/{task_name}; using all available seeds."
            )
            return available
        return available[: args.n_seeds]

    def make_env(task_name: str):
        gym_id = resolve_env_id(task_name)
        _ensure_registered(gym_id)
        kwargs = {}
        if is_atari_ram(task_name):
            kwargs["obs_type"] = "ram"
        env = gym.make(gym_id, **kwargs)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObservation(env)
        return env

    for i, task in enumerate(args.envs):
        print(f"[{i + 1}/{n_envs}] {task}")
        env = make_env(task)
        policy_list = []
        available_names = []
        for name in reservoir_names:
            seed_ids = resolve_seed_ids(name, task)
            if not seed_ids:
                print(f"  skip {name}: no agents found")
                continue
            models = [
                load_model(
                    name,
                    task,
                    seed,
                    agents_root=args.agents_root,
                    agent_suffix=args.agent_suffix,
                    device=args.device,
                )[-1]
                for seed in seed_ids
            ]
            policy_list.append(
                sorted_policy(
                    models,
                    f"{name.lower()}{args.log_suffix}",
                    task,
                    reward_records,
                    seed_ids=seed_ids,
                )
            )
            available_names.append(name)

        policies = {name: policy_list[j] for j, name in enumerate(available_names)}
        sim_matrix = policy_similarity_matrix(policies, env, device=args.device)

        ax = axes[i]
        ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu")

        idx = 0
        x_ticks = []
        y_ticks = []
        for name, policy in policies.items():
            idx += len(policy)
            center = idx - len(policy) / 2
            ax.axvline(idx - 0.5, color="white", linewidth=0.5)
            ax.axhline(idx - 0.5, color="white", linewidth=0.5)
            x_ticks.append(center)
            y_ticks.append(center)
        for spine in ax.spines.values():
            spine.set_color("white")

        labels = [display_names.get(n, n) for n in policies]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        if i + n_cols >= n_envs:
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
        else:
            ax.set_xticklabels([])
        if i % n_cols == 0:
            ax.set_yticklabels(labels, fontsize=5)
        else:
            ax.set_yticklabels([])
        ax.set_title(task, fontsize=8)
        env.close()

    for i in range(n_envs, len(axes)):
        axes[i].set_visible(False)

    fig.subplots_adjust(left=0.11, right=0.87, top=0.96, bottom=0.12,
                        hspace=0.20, wspace=0.12)
    cbar_ax = fig.add_axes([0.89, 0.12, 0.015, 0.83])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-1, vmax=1), cmap="RdBu"),
        cax=cbar_ax,
    )
    cbar.set_label("Cosine Similarity", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    os.makedirs(args.output_dir, exist_ok=True)
    fig.savefig(f"{args.output_dir}/{args.output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.output_dir}/{args.output_prefix}.png", bbox_inches="tight", dpi=300)
    print(f"Saved to {args.output_dir}/{args.output_prefix}.{{pdf,png}}")
    plt.close()
