import torch
from ..ode_base import ODEBase


class Tyson1991CellCycle2Var(ODEBase):
    DESCRIPTION = "Tyson1991 - Cell Cycle 2 variables"
    NODE_NAMES = {
        1: "EmptySet",
        2: "u",
        3: "z",
        4: "v",
    }
    PARAMETER_NAMES = {
        1: "kappa",
        2: "k6",
        3: "k4",
        4: "k4prime",
        5: "alpha",
        6: "cell",
        7: "EmptySet",
    }
    URL = "https://odebase.org/detail/1328"
    NUM_NODES = 4
    NUM_PARAMS = 7
    time_scale = 36
    mu = [0.5, 0.015, 0.2, 0.1]
    PHASE_PORTRAIT_DIMS = (1, 3, "u (active MPF)", "v (total cyclin)")
    CATEGORY = "Cell Cycle"
    std = [0.5, 0.02, 0.07, 0.07]

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)
        self.k[1] = 3 / 200
        self.k[2] = 1
        self.k[3] = 180
        self.k[4] = 9 / 500
        self.k[5] = 1 / 10000
        self.k[6] = 1
        self.k[7] = 1

    def project_to_manifold(self, x):
        x4 = x[..., 1:2] + x[..., 2:3]  # x_3 = x_4 - x_2  <=>  x_4 = x_2 + x_3
        return torch.cat((x[..., :3], x4), dim=-1)

    def ode(self, x):
        x1, x2, x4 = x[..., 0], x[..., 1], x[..., 3]
        dx1 = torch.zeros_like(x1)
        dx2 = self.k[3] * (x4 - x2) * (self.k[4] / self.k[3] + x2**2) - self.k[2] * x2
        dx4 = self.k[1] - self.k[2] * x2
        dx3 = dx4 - dx2
        return torch.stack((dx1, dx2, dx3, dx4), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)

        # reference: https://odebase.org/detail/1328
        x[..., 0] = 1.0  # empty set
        x[..., 1] *= 0.2  # u
        x[..., 2] *= 0.2  # z
        x[..., 3] = x[..., 1] + x[..., 2]  # since, z = v - u, v = z + u
        return x
