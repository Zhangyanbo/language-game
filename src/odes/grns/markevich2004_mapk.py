import torch
from ..ode_base import ODEBase


class Markevich2004MAPKDoublePhosphorylation(ODEBase):
    DESCRIPTION = (
        "Markevich2004 - MAPK double phosphorylation, ordered Michaelis-Menton"
    )
    NODE_NAMES = {
        1: "M",
        2: "Mp",
        3: "Mpp",
        4: "MAPKK",
        5: "MKP3",
    }
    PARAMETER_NAMES = {
        0: "k1cat",
        1: "Km1",
        2: "k2cat",
        3: "Km2",
        4: "k3cat",
        5: "Km3",
        6: "k4cat",
        7: "Km4",
        8: "Km5",
        9: "uVol",
        10: "__cm_k11",
    }
    URL = "https://odebase.org/detail/1343"
    NUM_NODES = 5
    NUM_PARAMS = 11
    time_scale = 420
    mu = [250, 250, 250, 50, 100]
    std = [250, 250, 250, 50, 100]
    PHASE_PORTRAIT_DIMS = (0, 2, "M (MAPK)", "Mpp (MAPK-PP)")
    CATEGORY = "Signal Transduction"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)
        self.k[1] = 1 / 100
        self.k[2] = 50
        self.k[3] = 15
        self.k[4] = 500
        self.k[5] = 21 / 250
        self.k[6] = 22
        self.k[7] = 3 / 50
        self.k[8] = 18
        self.k[9] = 78
        self.k[10] = 1
        self.k[11] = 500

    def project_to_manifold(self, x):
        c_prime = x[..., :3].sum(dim=-1, keepdim=True)
        scale = self.k[11] / torch.clamp(c_prime, min=1e-12)
        x123 = x[..., :3] * scale
        return torch.cat((x123, x[..., 3:]), dim=-1)

    def ode(self, x):
        x1, x2, x3, x4, x5 = x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4]
        dx1 = (-1) * self.k[10] * self.k[1] * x4 * x1 / self.k[2] / (
            1 + x1 / self.k[2] + x2 / self.k[4]
        ) + 1 * self.k[10] * self.k[7] * x5 * x2 / self.k[8] / (
            1 + x3 / self.k[6] + x2 / self.k[8] + x1 / self.k[9]
        ) / self.k[
            10
        ]
        dx2 = (
            1
            * self.k[10]
            * self.k[1]
            * x4
            * x1
            / self.k[2]
            / (1 + x1 / self.k[2] + x2 / self.k[4])
            - 1
            * self.k[10]
            * self.k[3]
            * x4
            * x2
            / self.k[4]
            / (1 + x1 / self.k[2] + x2 / self.k[4])
            + 1
            * self.k[10]
            * self.k[5]
            * x5
            * x3
            / self.k[6]
            / (1 + x3 / self.k[6] + x2 / self.k[8] + x1 / self.k[9])
            - 1
            * self.k[10]
            * self.k[7]
            * x5
            * x2
            / self.k[8]
            / (1 + x3 / self.k[6] + x2 / self.k[8] + x1 / self.k[9])
            / self.k[10]
        )
        dx3 = (
            1
            * self.k[10]
            * self.k[3]
            * x4
            * x2
            / self.k[4]
            / (1 + x1 / self.k[2] + x2 / self.k[4])
            - 1
            * self.k[10]
            * self.k[5]
            * x5
            * x3
            / self.k[6]
            / (1 + x3 / self.k[6] + x2 / self.k[8] + x1 / self.k[9])
            / self.k[10]
        )
        dx4 = torch.zeros_like(x4) / self.k[10]
        dx5 = torch.zeros_like(x5) / self.k[10]
        return torch.stack((dx1, dx2, dx3, dx4, dx5), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)

        # reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC2172246/
        z = x[..., :3].sum(dim=-1, keepdim=True)
        x[..., :3] = x[..., :3] / z * self.k[11]  # M + Mp + Mpp = k_{11}
        x[..., 3] *= 100  # [MAPKK]
        x[..., 4] *= 200  # [MKP3]
        return x
