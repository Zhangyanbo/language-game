import os
import numpy as np
import imageio.v2 as imageio
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize


def save_video(
    env_or_venv,
    env_id: str,
    seed: int,
    reservoir: str,
    model,
    video_length: int = 2000,
    max_env: int = 0,
    fps: int = 30,
):
    """
    Record a video of the policy running on env_or_venv.
    - env_or_venv: Can be a Gymnasium single environment / VecEnv / VecNormalize
    - max_env: Only record the max_env-th parallel environment (0-based)
    - video_length: Maximum recording steps; early termination if the sub-environment finishes first
    - Automatically switch VecNormalize to evaluation mode (training=False, norm_reward=False)
    - Output to ./results/videos/{reservoir}/{env_id}/seed_{seed}.mp4
    """
    os.makedirs(f"./results/videos/{reservoir.lower()}/{env_id}/", exist_ok=True)

    # 1) Normalize to VecEnv
    if isinstance(env_or_venv, VecEnv):
        venv = env_or_venv
    else:
        venv = DummyVecEnv([lambda: env_or_venv])

    # 2) If VecNormalize, switch to evaluation mode
    if isinstance(venv, VecNormalize):
        venv.training = False
        venv.norm_reward = False

    # 3) Select the sub-environment index to record
    idx = int(max_env) if max_env is not None else 0
    if idx < 0:
        idx = 0
    if idx >= venv.num_envs:
        idx = venv.num_envs - 1  # clamp

    # 4) Reset and start rollout
    obs = venv.reset()
    if isinstance(
        obs, tuple
    ):  # Compatible with some old wrappers that return (obs, info)
        obs = obs[0]

    dones = np.zeros(venv.num_envs, dtype=bool)
    frames = []

    # Helper function: safely get a frame from a specific env
    def grab_frame():
        frame = None
        # VecEnv (Subproc/Dummy) usually implements get_images() -> List[np.ndarray]
        if hasattr(venv, "get_images"):
            images = venv.get_images()
            if isinstance(images, (list, tuple)) and len(images) > 0:
                # clamp idx
                j = min(idx, len(images) - 1)
                frame = images[j]
        # Fallback: call render(); some VecEnv returns tiled large image or None
        if frame is None:
            try:
                out = (
                    venv.render()
                )  # Some implementations need render(mode="rgb_array")
                # If it's a tiled large image, accept it; we'll record it
                frame = out
            except Exception:
                frame = None
        return frame

    # First grab a frame to check if rendering is possible
    first = grab_frame()
    if first is None:
        raise RuntimeError(
            "Current environment failed to return image frames. "
            "Please ensure: 1) The environment (or its sub-environment) used for recording has rendering enabled when created, e.g., make_vec_env(..., env_kwargs={'render_mode':'rgb_array'}); "
            "2) Or create a separate evaluation environment with n_envs=1 for recording."
        )
    frames.append(first)

    # 5) Loop execution and frame capture, only check the done status of the idx-th env
    t = 0
    while (not dones[idx]) and t < video_length - 1:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        frame = grab_frame()
        if frame is not None:
            frames.append(frame)
        t += 1

    # 6) Write video
    out_path = f"./results/videos/{reservoir.lower()}/{env_id}/seed_{seed}.mp4"
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Video saved to: {out_path}  (env index = {idx}, frames = {len(frames)})")
