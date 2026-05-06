import torch
from .utils import rk4
from .ode_base import ODEBase


class LorenzSystem(ODEBase):
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    time_scale = 1.0

    NUM_NODES = 3

    mu = torch.tensor([-0.07, -0.02, 24.76])
    std = torch.tensor([8.18, 9.16, 7.66])

    MIN_VALUE = -1e8
    MAX_VALUE = 1e8

    def __call__(self, x):
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        dx1 = self.sigma * (x2 - x1)
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta * x3
        return torch.stack((dx1, dx2, dx3), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        """Generate random initial state in original space."""
        x = torch.randn(batch_size, self.NUM_NODES, device=device)
        return self.denormalize(x)


__all__ = ["LorenzSystem"]
