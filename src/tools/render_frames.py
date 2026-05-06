"""Render snapshot images of trained agents in all RL environments.

Usage:
    uv run src/tools/render_frames.py                    # all envs, MLP reservoir, 3 frames each
    uv run src/tools/render_frames.py --n_frames 5       # 5 frames per env
    uv run src/tools/render_frames.py --envs CartPole-v1 Pendulum-v1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from core.envs import resolve_env_id, needs_flatten, make_env_fn, _ensure_registered

ALL_ENVS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "Reacher-v4",
    "Pusher-v4",
    "Swimmer-v4",
    "Hopper-v4",
    "HalfCheetah-v4",
    "HumanoidStandup-v4",
    "PointMaze",
    "finger-spin",
]


def render_env_frames(
    env_id: str,
    reservoir: str = "mlp",
    seed: int = 0,
    n_frames: int = 3,
    agents_root: str = "./results/agents",
    output_root: str = "./results/figures/env_snapshots",
):
    """Load a trained agent and save n_frames evenly-spaced snapshots."""
    reservoir_id = f"{reservoir.lower()}_gradient"
    agent_path = f"{agents_root}/{reservoir_id}/{env_id}/seed_{seed}.zip"
    if not os.path.exists(agent_path):
        print(f"SKIP: {agent_path} not found")
        return

    model = PPO.load(agent_path, device="cpu")
    model.policy.set_training_mode(False)

    resolved_id = resolve_env_id(env_id)
    _ensure_registered(resolved_id)
    flatten = needs_flatten(resolved_id)

    env = gym.make(resolved_id, render_mode="rgb_array")
    if flatten:
        from gymnasium.wrappers import FlattenObservation

        env = FlattenObservation(env)

    obs, _ = env.reset(seed=seed)
    frame = env.render()

    # Run a full episode to know total length
    frames_buffer = [(0, frame)]
    done = False
    step = 0
    max_steps = 2000
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        frame = env.render()
        if frame is not None:
            frames_buffer.append((step, frame))

    env.close()

    total_steps = len(frames_buffer)
    if total_steps == 0:
        print(f"SKIP: {env_id} - no frames captured")
        return

    # Pick n_frames evenly spaced indices (include first and last)
    if n_frames >= total_steps:
        indices = list(range(total_steps))
    else:
        indices = np.linspace(0, total_steps - 1, n_frames, dtype=int).tolist()

    out_dir = f"{output_root}/{env_id}"
    os.makedirs(out_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        step_num, img = frames_buffer[idx]
        img_pil = Image.fromarray(img)
        path = f"{out_dir}/frame_{i:02d}_step{step_num:04d}.png"
        img_pil.save(path)

    print(f"  {env_id}: saved {len(indices)} frames to {out_dir}/ (episode length: {step})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render snapshot images of trained agents")
    parser.add_argument("--envs", nargs="+", default=ALL_ENVS, help="Environments to render")
    parser.add_argument("--reservoir", default="mlp", help="Reservoir type (default: mlp)")
    parser.add_argument("--seed", type=int, default=0, help="Agent seed to use")
    parser.add_argument("--n_frames", type=int, default=3, help="Number of frames per env")
    parser.add_argument("--agents_root", default="./results/agents")
    parser.add_argument("--output_root", default="./results/figures/env_snapshots")
    args = parser.parse_args()

    print(f"Rendering {args.n_frames} frames per env using {args.reservoir} reservoir (seed={args.seed})")
    for env_id in args.envs:
        try:
            render_env_frames(
                env_id=env_id,
                reservoir=args.reservoir,
                seed=args.seed,
                n_frames=args.n_frames,
                agents_root=args.agents_root,
                output_root=args.output_root,
            )
        except Exception as e:
            print(f"  ERROR: {env_id} - {e}")
