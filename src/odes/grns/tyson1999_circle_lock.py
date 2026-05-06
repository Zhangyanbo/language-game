import torch
from ..ode_base import ODEBase


class Tyson1999CircleLock(ODEBase):
    DESCRIPTION = "Tyson1999 - Circle lock"
    NODE_NAMES = {
        1: "EmptySet",
        2: "M",
        3: "P",
    }
    PARAMETER_NAMES = {
        1: "N_A",
        2: "default",
        3: "CYTOPLASM",
        4: "__lp_r2_Vm",
        5: "__lp_r2_Pcrit",
        6: "__lp_r2_Keq",
        7: "__lp_r3_V",
        8: "__lp_r4_D",
        9: "__lp_r5_D",
        10: "__lp_r6_k1",
        11: "__lp_r6_k2",
        12: "__lp_r6_J",
        13: "__lp_r6_Keq",
        14: "EmptySet",
    }
    URL = "https://odebase.org/detail/1351"
    NUM_NODES = 3
    time_scale = 24
    NUM_PARAMS = 14
    mu = [0, 1.3, 1.9]
    std = [1, 0.7, 1.1]
    PHASE_PORTRAIT_DIMS = (1, 2, "M (mRNA)", "P (Protein)")
    CATEGORY = "Circadian Clock"

    def __init__(self):
        self.k = [0] * 15
        self.k[1] = 602213670000000000000000
        self.k[2] = 1
        self.k[3] = 1
        self.k[4] = 1
        self.k[5] = 1 / 10
        self.k[6] = 200
        self.k[7] = 1 / 2
        self.k[8] = 1 / 10
        self.k[9] = 1 / 10
        self.k[10] = 10
        self.k[11] = 3 / 100
        self.k[12] = 1 / 20
        self.k[13] = 200
        self.k[14] = 0

    def _safe_sqrt(self, val):
        """sqrt guarded against negative arguments."""
        return torch.sqrt(torch.clamp(val, min=0.0))

    def ode(self, x):
        # x.shape = (..., 3)
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        eps = 1e-6
        dx1 = torch.zeros_like(x1)

        # Guard sqrt args: 1 + 8*k*x3 can be negative if x3 < 0
        sqrt_arg1 = 1 + 8 * self.k[6] * x3
        sqrt_arg2 = 1 + 8 * self.k[13] * x3

        dx2 = (
            self.k[3]
            * self.k[4]
            / (
                1
                + (
                    x3
                    * (1 - 2 / (1 + self._safe_sqrt(sqrt_arg1) + eps))
                    / (2 * self.k[5])
                )
                ** 2
            )
            + (-1) * self.k[8] * x2 * self.k[3]
        ) / self.k[3]
        dx3 = (
            self.k[7] * x2 * self.k[3]
            + (-1) * self.k[9] * x3 * self.k[3]
            + (-1)
            * self.k[3]
            * (
                self.k[10] * x3 * 2 / (1 + self._safe_sqrt(sqrt_arg2) + eps)
                + self.k[11] * x3
            )
            / (self.k[12] + x3 + eps)
        ) / self.k[3]
        return torch.stack((dx1, dx2, dx3), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)

        # reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC1300518/
        x *= 3.5
        return x
