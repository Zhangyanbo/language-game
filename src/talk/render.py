"""Render RL environment frames from LLM-designed internal states.

Pipeline: set internal state → env produces observation → render frame.

Internal state formats by environment family:
  - Classic Control: env.unwrapped.state (low-dim vector)
  - MuJoCo: qpos + qvel (joint positions and velocities)
  - Atari RAM: 128 bytes of console RAM (applied on top of a valid system state)
  - dm_control: physics.set_state() (joint angles + velocities)
  - gymnasium-robotics PointMaze: point position + velocity
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
from PIL import Image

from core.envs import (
    _ensure_registered,
    is_atari_ram,
    needs_flatten,
    resolve_env_id,
)

# ─────────────────────────────────────────────────────────────────────────────
# Environment creation
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIC_CONTROL = {"CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0", "Pendulum-v1"}
_MUJOCO = {"Reacher-v4", "Pusher-v4", "Swimmer-v4", "Hopper-v4", "HalfCheetah-v4", "HumanoidStandup-v4"}

_ATARI_WARMUP_STEPS = 20  # Steps to run after reset to reach a valid game state

# Documented RAM addresses per Atari game (from doc/games/*.md).
# Only these addresses are overwritten when setting the LLM-designed state;
# the rest keep the warmup values (maze layout, floor patterns, etc.).
_ATARI_KEY_ADDRS: dict[str, list[int]] = {
    "BankHeist-ram": [
        8, 9, 10, 11,       # entity Y (player + slots 0-2)
        12,                  # dynamite Y
        24, 25, 26,          # entity type (slots 0-2)
        28, 29, 30, 31,      # X positions (player + slots 0-2)
        32,                  # dynamite X
        41,                  # player direction
        78,                  # death timer
        85, 86,              # lives, gas
        88, 89, 90,          # score digits
    ],
    "KungFuMaster-ram": [
        24, 25, 26,          # score digits
        27, 28,              # timer
        29,                  # lives
        35,                  # enemy Y offset
        46,                  # player Y
        50,                  # enemy type
        63,                  # extra henchmen bitmask
        72,                  # enemy X
        73,                  # projectile X
        74,                  # player X
        75, 76,              # health bars
        90,                  # player facing
        92,                  # enemy facing
        93,                  # player animation
        96,                  # projectile Y offset
    ],
    "CrazyClimber-ram": [
        2, 3, 4, 5,          # score digits
        9,                   # helicopter Y
        14, 15, 16,          # enemy column/row/offset
        19,                  # left hand pull
        21,                  # right hand pull
        24,                  # player X
        34,                  # bird/helicopter X
        42,                  # lives
        58,                  # vertical scroll offset
        81, 82, 83, 85,      # projectile active/Y/type/X
        84,                  # enemy type
    ] + list(range(46, 58))  # window layout (12 rows)
      + list(range(95, 103)) # window closing progress
      + list(range(108, 116)), # window closing pattern
    "Kangaroo-ram": [
        0, 1, 2, 3,          # monkey state (slots 3,2,1,0)
        8, 9, 10, 11,        # monkey Y (slots 3,2,1,0)
        12, 13, 14, 15,      # monkey X (slots 3,2,1,0)
        16, 17, 18,          # player Y, X, state
        25, 26, 27,          # thrown coconut Y (slots 0-2)
        28, 29, 30,          # thrown coconut X (slots 0-2)
        33, 34,              # falling coconut Y, X
        36,                  # level/floor
        39, 40,              # score
        41,                  # bell state
        42, 43, 44,          # fruit type/state
        45,                  # lives
        54,                  # crash state
        59,                  # timer
        83,                  # child X
    ],
}


def _make_render_env(env_name: str):
    """Create a single env with render_mode='rgb_array', properly wrapped."""
    resolved_id = resolve_env_id(env_name)
    _ensure_registered(resolved_id)
    extra_kwargs = {"obs_type": "ram"} if is_atari_ram(env_name) else {}
    kwargs = dict(extra_kwargs)
    kwargs["render_mode"] = "rgb_array"

    env = gym.make(resolved_id, **kwargs)
    flatten = needs_flatten(resolved_id, extra_kwargs or None)
    if flatten:
        from gymnasium.wrappers import FlattenObservation
        env = FlattenObservation(env)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# State setting + observation extraction per environment family
# ─────────────────────────────────────────────────────────────────────────────


def _set_state_and_get_obs(env, env_name: str, state: np.ndarray) -> np.ndarray:
    """Set the environment's internal state and return the resulting observation."""
    if env_name in _CLASSIC_CONTROL:
        return _set_classic(env, env_name, state)
    elif env_name in _MUJOCO:
        return _set_mujoco(env, env_name, state)
    elif is_atari_ram(env_name):
        return _set_atari_ram(env, env_name, state)
    elif env_name == "finger-spin":
        return _set_dm_control(env, state)
    elif env_name in ("PointMaze", "PointMaze_UMaze-v3"):
        return _set_pointmaze(env, state)
    else:
        return state


def _set_classic(env, env_name: str, state: np.ndarray) -> np.ndarray:
    """Classic Control: state IS the internal state directly."""
    uw = env.unwrapped
    internal_len = len(uw.state)
    uw.state = np.array(state[:internal_len], dtype=np.float64)

    if env_name == "Pendulum-v1":
        return uw._get_obs()
    elif env_name == "Acrobot-v1":
        s = uw.state
        return np.array([
            np.cos(s[0]), np.sin(s[0]),
            np.cos(s[1]), np.sin(s[1]),
            s[2], s[3],
        ], dtype=np.float64)
    else:
        return np.array(uw.state, dtype=np.float64)


def _set_mujoco(env, env_name: str, state: np.ndarray) -> np.ndarray:
    """MuJoCo: internal state = qpos + qvel."""
    uw = env.unwrapped
    nq = uw.data.qpos.shape[0]
    nv = uw.data.qvel.shape[0]

    qpos = np.zeros(nq, dtype=np.float64)
    qvel = np.zeros(nv, dtype=np.float64)

    n = len(state)
    qpos[:min(nq, n)] = state[:min(nq, n)]
    if n > nq:
        remaining = min(nv, n - nq)
        qvel[:remaining] = state[nq : nq + remaining]

    uw.set_state(qpos, qvel)
    return uw._get_obs()


def _set_atari_ram(env, env_name: str, state: np.ndarray) -> np.ndarray:
    """Atari RAM: selectively modify documented RAM addresses on a valid state.

    Atari 2600 rendering depends on the full emulator state (CPU, TIA, RIOT,
    etc.), not just the 128 bytes of RAM. Overwriting all RAM destroys visual
    structures (maze layouts, floor patterns, etc.) stored in undocumented
    addresses.

    Strategy:
    1. Warm up the env to reach a valid game state.
    2. Only overwrite the documented key addresses (player position, enemy
       positions, score, lives, etc.) with the LLM-designed values.
    3. Step NOOP to let the emulator render a frame reflecting the changes.
    4. Build a full 128-byte observation: designed values at key addresses,
       original warmup values elsewhere.
    """
    ale = env.unwrapped.ale

    # Warm up to valid game state
    for _ in range(_ATARI_WARMUP_STEPS):
        env.step(0)

    original_ram = ale.getRAM().copy()
    designed_ram = np.clip(state, 0, 255).astype(np.uint8)
    key_addrs = _ATARI_KEY_ADDRS.get(env_name, list(range(128)))

    # Selectively overwrite only documented key addresses
    for addr in key_addrs:
        if addr < len(designed_ram):
            ale.setRAM(addr, int(designed_ram[addr]))

    # Step NOOP to render with modified RAM
    env.step(0)

    # Build observation: designed values at key addresses, original elsewhere
    obs = original_ram.astype(np.float64)
    for addr in key_addrs:
        if addr < len(designed_ram):
            obs[addr] = float(designed_ram[addr])

    return obs


def _set_dm_control(env, state: np.ndarray) -> np.ndarray:
    """dm_control (finger-spin): physics state = joint angles + velocities."""
    uw = env.unwrapped
    dm_env = uw._env
    physics = dm_env.physics
    physics_state = physics.get_state()
    n = min(len(state), len(physics_state))
    new_state = physics_state.copy()
    new_state[:n] = state[:n]
    with physics.reset_context():
        physics.set_state(new_state)
    obs_dict = dm_env.task.get_observation(physics)
    obs = np.concatenate([v.flatten() for v in obs_dict.values()])
    return obs


def _set_pointmaze(env, state: np.ndarray) -> np.ndarray:
    """PointMaze: internal state = [x, y, vx, vy]."""
    uw = env.unwrapped
    pe = uw.point_env
    pos = state[:2] if len(state) >= 2 else np.zeros(2)
    vel = state[2:4] if len(state) >= 4 else np.zeros(2)
    pe.data.qpos[:] = pos
    pe.data.qvel[:] = vel
    point_obs, _ = pe._get_obs()
    obs = uw._get_obs(point_obs)
    if isinstance(obs, dict):
        return np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Action conversion
# ─────────────────────────────────────────────────────────────────────────────


def _convert_action(env, action: list[float]):
    """Convert raw action list to the format expected by the environment."""
    act_space = env.action_space
    if hasattr(act_space, "n"):
        return int(np.argmax(action[: act_space.n]))
    else:
        act = np.array(action, dtype=np.float32)
        return np.clip(act, act_space.low, act_space.high)


# ─────────────────────────────────────────────────────────────────────────────
# Main rendering function
# ─────────────────────────────────────────────────────────────────────────────


def render_state(
    env_name: str,
    state: list[float],
    action: list[float] | None = None,
    output_dir: str = ".",
) -> tuple[str, str | None, np.ndarray]:
    """Render frames before and after action execution.

    Returns (before_path, after_path_or_None, observation).
    """
    env = _make_render_env(env_name)
    env.reset()

    state_arr = np.array(state, dtype=np.float64)
    os.makedirs(output_dir, exist_ok=True)

    obs = _set_state_and_get_obs(env, env_name, state_arr)

    # Render before frame
    before_frame = env.render()
    before_path = os.path.join(output_dir, "before.png")
    Image.fromarray(before_frame).save(before_path)

    # Step with action and render after frame
    after_path = None
    if action is not None:
        act = _convert_action(env, action)
        try:
            env.step(act)
            after_frame = env.render()
            after_path = os.path.join(output_dir, "after.png")
            Image.fromarray(after_frame).save(after_path)
        except Exception as e:
            print(f"  [Render] step failed ({e}), skipping after frame")

    env.close()
    return before_path, after_path, obs
