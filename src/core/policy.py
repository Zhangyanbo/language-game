import torch
import torch.nn as nn
import torch
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Tuple
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.policies import ActorCriticPolicy

from copy import deepcopy


# Pre-defined Reservoir functions


class IdentityReservoir(nn.Module):
    METHOD = "direct"

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return x


class MLPTrainable(nn.Module):
    METHOD = "direct"

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x, *args, **kwargs):
        return self.mlp(x)


class LinearEncoderExtractor(BaseFeaturesExtractor):
    """
    Shared linear encoder:
        z = W_E * flatten(obs) + b_E
    """

    def __init__(self, observation_space: spaces.Space, output_dim: int):
        super().__init__(observation_space, features_dim=output_dim)
        in_dim = get_flattened_obs_dim(observation_space)
        self.flatten = nn.Flatten()
        self.encoder = nn.Linear(in_dim, output_dim)
        self.encoder_bn = nn.BatchNorm1d(output_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.flatten(observations))
        return self.encoder_bn(encoded)


class MLPCritic(nn.Module):
    def __init__(self, reservoir_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(reservoir_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class ReservoirMLPExtractor(nn.Module):
    def __init__(
        self,
        features_dim: int,
        reservoir: nn.Module,
        critic_latent_dim: int = 64,
    ):
        super().__init__()

        self.latent_dim_pi = features_dim  # actor latent dim
        self.latent_dim_vf = critic_latent_dim  # critic latent dim

        # Models
        self.reservoir = reservoir
        self.dt_logit = nn.Parameter(torch.tensor([0], dtype=torch.float32))
        self.pre_decoder_bn = nn.BatchNorm1d(features_dim)
        self.critic_mlp = MLPCritic(features_dim, critic_latent_dim)

    def get_dt(self, input):
        batch_size = input.shape[0]
        device = input.device
        return torch.sigmoid(self.dt_logit).expand(batch_size).to(device).reshape(-1, 1)

    def forward_actor(self, features: torch.Tensor, dt=None) -> torch.Tensor:
        # features = z (output of shared linear encoder)
        if dt is None:
            dt = self.get_dt(features)
        h_T = self.reservoir(features, dt)

        actor_latent = h_T
        # Apply BatchNorm right before decoder (action_net), not inside reservoir.
        return self.pre_decoder_bn(actor_latent)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_mlp(features)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class ReservoirCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        reservoir: Callable,  # reservoir function
        reservoir_dim: int,  # reservoir dimension (R: (reservoir_dim, 1) -> reservoir_dim)
        critic_latent_dim: int = 64,
        reservoir_mode: str | None = None,  # backward compatibility for old checkpoints
        *args,
        **kwargs,
    ):
        reservoir.eval()
        reservoir_state = deepcopy(reservoir.state_dict())  # backup reservoir state

        kwargs["ortho_init"] = True
        kwargs["share_features_extractor"] = True

        kwargs.setdefault("features_extractor_class", LinearEncoderExtractor)
        kwargs.setdefault("features_extractor_kwargs", {"output_dim": reservoir_dim})

        self.reservoir_dim = reservoir_dim
        self.reservoir_factory = (
            lambda: reservoir
        )  # factory function to create reservoir, avoid error when adding module before super().__init__()

        self._critic_latent_dim = int(critic_latent_dim)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        # Restore reservoir state
        self.mlp_extractor.reservoir.load_state_dict(reservoir_state)
        self.mlp_extractor.reservoir.eval()

    def _build_mlp_extractor(self) -> None:
        reservoir = self.reservoir_factory()
        # (features_dim is determined by LinearEncoderExtractor(output_dim=reservoir_dim))

        self.mlp_extractor = ReservoirMLPExtractor(
            features_dim=self.features_dim,
            reservoir=reservoir,
            critic_latent_dim=self._critic_latent_dim,
        )

    def set_training_mode(self, mode: bool) -> None:
        # Ensure reservoir is always in eval mode
        super().set_training_mode(mode)
        self.mlp_extractor.reservoir.eval()
