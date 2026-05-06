"""Inference policy architectures and model loading for Talk to GRN.

Two checkpoint formats exist:

  OLD format (*_gradient, no aux suffix):
    - LayerNorm in feature extractors (encoder_ln + log_epsilon)
    - Extra encoder_extractor sub-module inside mlp_extractor
    - share_features_extractor=False (separate pi/vf extractors)

  NEW format (*_gradient_aux*):
    - BatchNorm in feature extractors (encoder_bn)
    - No encoder_extractor in mlp_extractor
    - share_features_extractor=False (separate pi/vf extractors)

Architecture is auto-detected from the checkpoint's state-dict keys.
"""

from __future__ import annotations

import io
import os
import zipfile

import torch
import torch.nn as nn

# Project root: two levels up from this file (src/talk/ -> src/ -> project root)
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_AGENTS_ROOT = os.path.join(_PROJECT_ROOT, "results", "agents")


# ─────────────────────────────────────────────────────────────────────────────
# Shared components
# ─────────────────────────────────────────────────────────────────────────────


class _MLPCritic(nn.Module):
    """Matches policy.MLPCritic architecture."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── Old format (LayerNorm) ───────────────────────────────────────────────────


class _OldFeatureExtractor(nn.Module):
    """Matches old LinearEncoderExtractor: Linear -> LayerNorm + log_epsilon."""

    def __init__(self, obs_dim: int, out_dim: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Linear(obs_dim, out_dim)
        self.encoder_ln = nn.LayerNorm(out_dim)
        self.log_epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_ln(self.encoder(self.flatten(x)))


class _OldEncoderExtractor(nn.Module):
    """Matches mlp_extractor.encoder_extractor in old checkpoints."""

    def __init__(self, obs_dim: int, reservoir_dim: int):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, reservoir_dim)
        self.encoder_ln = nn.LayerNorm(reservoir_dim)
        self.log_epsilon = nn.Parameter(torch.zeros(1))


class _OldMlpExtractor(nn.Module):
    """Matches old ReservoirMLPExtractor: encoder_extractor + pre_decoder_bn."""

    def __init__(self, obs_dim: int, reservoir_dim: int, reservoir: nn.Module, critic_latent_dim: int):
        super().__init__()
        self.reservoir = reservoir
        self.dt_logit = nn.Parameter(torch.zeros(1))
        self.encoder_extractor = _OldEncoderExtractor(obs_dim, reservoir_dim)
        self.pre_decoder_bn = nn.BatchNorm1d(reservoir_dim)
        self.critic_mlp = _MLPCritic(reservoir_dim, critic_latent_dim)

    def get_dt(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.dt_logit).expand(x.shape[0]).reshape(-1, 1)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        dt = self.get_dt(features)
        return self.pre_decoder_bn(self.reservoir(features, dt))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_mlp(features)


class _OldStylePolicy(nn.Module):
    """Inference policy matching old checkpoint format."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        reservoir_dim: int,
        reservoir: nn.Module,
        critic_latent_dim: int = 64,
        continuous: bool = True,
    ):
        super().__init__()
        self.features_extractor = _OldFeatureExtractor(obs_dim, reservoir_dim)
        self.pi_features_extractor = _OldFeatureExtractor(obs_dim, reservoir_dim)
        self.vf_features_extractor = _OldFeatureExtractor(obs_dim, reservoir_dim)
        self.mlp_extractor = _OldMlpExtractor(obs_dim, reservoir_dim, reservoir, critic_latent_dim)
        self.action_net = nn.Linear(reservoir_dim, action_dim)
        self.value_net = nn.Linear(critic_latent_dim, 1)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.pi_features_extractor(obs.float())
        latent = self.mlp_extractor.forward_actor(features)
        return self.action_net(latent)

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.vf_features_extractor(obs.float())
        latent = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent)


# ── New format (BatchNorm) ───────────────────────────────────────────────────


class _NewFeatureExtractor(nn.Module):
    """Matches current LinearEncoderExtractor: Linear -> BatchNorm1d."""

    def __init__(self, obs_dim: int, out_dim: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Linear(obs_dim, out_dim)
        self.encoder_bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_bn(self.encoder(self.flatten(x)))


class _NewMlpExtractor(nn.Module):
    """Matches current ReservoirMLPExtractor: pre_decoder_bn, no encoder_extractor."""

    def __init__(self, reservoir_dim: int, reservoir: nn.Module, critic_latent_dim: int):
        super().__init__()
        self.reservoir = reservoir
        self.dt_logit = nn.Parameter(torch.zeros(1))
        self.pre_decoder_bn = nn.BatchNorm1d(reservoir_dim)
        self.critic_mlp = _MLPCritic(reservoir_dim, critic_latent_dim)

    def get_dt(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.dt_logit).expand(x.shape[0]).reshape(-1, 1)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        dt = self.get_dt(features)
        return self.pre_decoder_bn(self.reservoir(features, dt))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_mlp(features)


class _NewStylePolicy(nn.Module):
    """Inference policy matching new checkpoint format (BatchNorm, no encoder_extractor)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        reservoir_dim: int,
        reservoir: nn.Module,
        critic_latent_dim: int = 64,
        continuous: bool = True,
    ):
        super().__init__()
        self.features_extractor = _NewFeatureExtractor(obs_dim, reservoir_dim)
        self.pi_features_extractor = _NewFeatureExtractor(obs_dim, reservoir_dim)
        self.vf_features_extractor = _NewFeatureExtractor(obs_dim, reservoir_dim)
        self.mlp_extractor = _NewMlpExtractor(reservoir_dim, reservoir, critic_latent_dim)
        self.action_net = nn.Linear(reservoir_dim, action_dim)
        self.value_net = nn.Linear(critic_latent_dim, 1)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.pi_features_extractor(obs.float())
        latent = self.mlp_extractor.forward_actor(features)
        return self.action_net(latent)

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.vf_features_extractor(obs.float())
        latent = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent)


# ─────────────────────────────────────────────────────────────────────────────
# Reservoir helpers
# ─────────────────────────────────────────────────────────────────────────────


class _GradientReservoir(nn.Module):
    """ODE-based gradient reservoir (no trainable parameters)."""

    def __init__(self, ode_model):
        super().__init__()
        self.ode_model = ode_model

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_denorm = self.ode_model.denormalize(x)
        min_val = getattr(self.ode_model, "MIN_VALUE", None)
        max_val = getattr(self.ode_model, "MAX_VALUE", None)
        if min_val is not None or max_val is not None:
            x_clamped = torch.clamp(x_denorm, min=min_val, max=max_val)
            x_denorm = x_denorm + (x_clamped - x_denorm).detach()
        g = self.ode_model(x_denorm)
        g = torch.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.ode_model.normalize(g)


def _build_reservoir(system: str, reservoir_dim: int):
    """Construct the reservoir object from a system folder name."""
    base = system.split("_aux")[0]
    ode_part = base[: base.index("_gradient")] if "_gradient" in base else base

    if ode_part == "identity":
        from core.policy import IdentityReservoir
        return IdentityReservoir()

    if ode_part == "mlp":
        from core.policy import MLPTrainable
        return MLPTrainable(reservoir_dim)

    ode_lookup = {
        "lorenzsystem": "lorenz",
        "tyson1999circlelock": "tyson1999",
        "markevich2004mapkdoublephosphorylation": "markevich2004",
        "tyson1991cellcycle2var": "tyson1991",
        "weimann2004circadianoscillator": "weimann2004",
        "almeida2019circadianclock": "almeida2019",
        "zatorsky2006p53model4": "zatorsky2006",
        "gardner2000toggleswitch": "gardner2000",
        "liebal2012transcriptioninhibition": "liebal2012",
        "gerard2010cellcycle": "gerard2010",
        "chickarmane2006stemcellswitch": "chickarmane2006",
        "gardner1998cellcyclegoldbeter": "gardner1998",
        "leloup1999circadianclock": "leloup1999",
        "chickarmane2008nanoggata6": "chickarmane2008",
        "kholodenko2000mapkcascade": "kholodenko2000",
    }
    ode_name = ode_lookup.get(ode_part)
    if ode_name is None:
        raise ValueError(
            f"Cannot determine reservoir for system '{system}' "
            f"(parsed ODE part: '{ode_part}'). "
            f"Known ODE names: {list(ode_lookup.keys())}"
        )
    from odes.ode_loader import load_ode_model
    return _GradientReservoir(load_ode_model(ode_name))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model(
    env_name: str,
    system: str,
    seed: int = 0,
    agents_root: str = _DEFAULT_AGENTS_ROOT,
    device: str = "cpu",
) -> nn.Module:
    """Load a trained policy for inference.

    Auto-detects checkpoint format (old LayerNorm vs new BatchNorm).
    """
    path = os.path.join(agents_root, system, env_name, f"seed_{seed}.zip")
    with zipfile.ZipFile(path) as z:
        sd = torch.load(
            io.BytesIO(z.read("policy.pth")),
            map_location=device,
            weights_only=False,
        )

    reservoir_dim = sd["pi_features_extractor.encoder.weight"].shape[0]
    obs_dim = sd["pi_features_extractor.encoder.weight"].shape[1]
    action_dim = sd["action_net.weight"].shape[0]
    critic_latent_dim = sd["value_net.weight"].shape[1]
    continuous = "log_std" in sd

    reservoir = _build_reservoir(system, reservoir_dim)

    is_old_format = "pi_features_extractor.encoder_ln.weight" in sd
    if is_old_format:
        policy = _OldStylePolicy(
            obs_dim, action_dim, reservoir_dim, reservoir, critic_latent_dim, continuous
        )
    else:
        policy = _NewStylePolicy(
            obs_dim, action_dim, reservoir_dim, reservoir, critic_latent_dim, continuous
        )

    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [load_model] missing keys (will use init): {missing}")
    if unexpected:
        print(f"  [load_model] unexpected keys (ignored): {unexpected}")

    policy.eval()
    policy.to(device)
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ─────────────────────────────────────────────────────────────────────────────


def list_systems(agents_root: str = _DEFAULT_AGENTS_ROOT) -> list[str]:
    if not os.path.isdir(agents_root):
        return []
    return sorted(
        d for d in os.listdir(agents_root)
        if os.path.isdir(os.path.join(agents_root, d))
    )


def list_envs(system: str, agents_root: str = _DEFAULT_AGENTS_ROOT) -> list[str]:
    system_dir = os.path.join(agents_root, system)
    if not os.path.isdir(system_dir):
        return []
    return sorted(
        d for d in os.listdir(system_dir)
        if os.path.isdir(os.path.join(system_dir, d))
    )
