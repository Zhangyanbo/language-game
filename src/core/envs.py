"""Environment registry and utilities for creating RL environments.

Handles:
- Mapping user-friendly env names to actual gymnasium IDs
- Auto-importing packages that register external envs (highway-env, gymnasium-robotics, shimmy)
- Wrapping Dict observation spaces with FlattenObservation for compatibility with the policy
"""

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# Maps user-facing aliases → actual gymnasium env IDs.
# Environments not listed here are passed through as-is.
ENV_ALIASES = {
    # MuJoCo (gymnasium built-in)
    "Reacher-v4": "Reacher-v4",
    "Pusher-v4": "Pusher-v5",  # v4 requires mujoco<3
    "Swimmer-v4": "Swimmer-v4",
    "Hopper-v4": "Hopper-v4",
    "HalfCheetah-v4": "HalfCheetah-v4",
    "Ant-v4": "Ant-v4",
    "HumanoidStandup-v4": "HumanoidStandup-v4",
    # highway-env
    "parking-v0": "parking-v0",
    # gymnasium-robotics
    "FetchSlide-v2": "FetchSlide-v4",  # v2 deprecated
    "HandReach-v2": "HandReach-v3",  # v2 deprecated
    "PointMaze": "PointMaze_UMaze-v3",
    "PointMaze_UMaze-v3": "PointMaze_UMaze-v3",
    # dm_control via shimmy
    "finger-spin": "dm_control/finger-spin-v0",
    # Atari (ALE) — RAM observations (128-dim vector, no CNN needed)
    "BankHeist-ram": "ALE/BankHeist-v5",
    "KungFuMaster-ram": "ALE/KungFuMaster-v5",
    "CrazyClimber-ram": "ALE/CrazyClimber-v5",
    "Kangaroo-ram": "ALE/Kangaroo-v5",
}

# Packages to import for registering external environments
_IMPORT_TRIGGERS = {
    "parking": "highway_env",
    "Fetch": "gymnasium_robotics",
    "Hand": "gymnasium_robotics",
    "PointMaze": "gymnasium_robotics",
    "dm_control": "shimmy",
    "ALE": "ale_py",
}

# Aliases that need obs_type="ram" passed to gym.make()
_ATARI_RAM_ALIASES = {
    "BankHeist-ram", "KungFuMaster-ram", "CrazyClimber-ram", "Kangaroo-ram",
}


def _ensure_registered(env_id: str):
    """Import the package that registers the environment if needed."""
    for prefix, package in _IMPORT_TRIGGERS.items():
        if prefix in env_id:
            __import__(package)
            return


def resolve_env_id(env_id: str) -> str:
    """Resolve a user-facing env name to an actual gymnasium env ID."""
    return ENV_ALIASES.get(env_id, env_id)


def is_atari_ram(env_id: str) -> bool:
    """Return True if the user-facing env_id is an Atari RAM alias."""
    return env_id in _ATARI_RAM_ALIASES


def _has_dict_obs(env_id: str, extra_kwargs: dict | None = None) -> bool:
    """Check if environment has a Dict observation space."""
    _ensure_registered(env_id)
    env = gym.make(env_id, **(extra_kwargs or {}))
    is_dict = isinstance(env.observation_space, gym.spaces.Dict)
    env.close()
    return is_dict


# Cache which env IDs need flattening (computed once per env_id)
_FLATTEN_CACHE: dict[str, bool] = {}


def needs_flatten(env_id: str, extra_kwargs: dict | None = None) -> bool:
    """Return True if the env has Dict obs and needs FlattenObservation wrapper."""
    key = (env_id, frozenset((extra_kwargs or {}).items()))
    if key not in _FLATTEN_CACHE:
        _FLATTEN_CACHE[key] = _has_dict_obs(env_id, extra_kwargs)
    return _FLATTEN_CACHE[key]


def make_env_fn(
    env_id: str,
    flatten: bool = False,
    render_mode: str | None = None,
    extra_kwargs: dict | None = None,
):
    """Return a factory function for creating (optionally wrapped) environments.

    Used with stable_baselines3.common.env_util.make_vec_env's env_id parameter
    when we need to apply wrappers or set render_mode.
    """
    _ensure_registered(env_id)
    extra = extra_kwargs or {}

    if not flatten and render_mode is None and not extra:
        return env_id  # Just return the string; make_vec_env handles it

    # Return a callable that creates and wraps the env
    def _make():
        kwargs = dict(extra)
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        env = gym.make(env_id, **kwargs)
        if flatten:
            env = FlattenObservation(env)
        return env

    return _make
