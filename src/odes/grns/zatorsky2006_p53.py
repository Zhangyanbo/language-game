import torch
from ..ode_base import ODEBase


class Zatorsky2006P53Model4(ODEBase):
    DESCRIPTION = "Zatorsky2006 - p53 Model4"
    NODE_NAMES = {
        1: "x (p53)",
        2: "y (Mdm2)",
        3: "y0 (Mdm2 precursor)",
    }
    PARAMETER_NAMES = {
        1: "beta_x",
        2: "psi",
        3: "alpha_x",
        4: "beta_y",
        5: "alpha_y",
        6: "alpha_0",
        7: "k",
        8: "alpha_k",
        9: "compartment",
    }
    URL = "https://odebase.org/detail/1422"
    NUM_NODES = 3
    NUM_PARAMS = 9
    time_scale = 7
    mu = [0.40, 0.55, 0.55]
    std = [0.30, 0.17, 0.26]
    PHASE_PORTRAIT_DIMS = (0, 1, "p53", "Mdm2")
    CATEGORY = "p53 / DNA Damage"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)
        self.k[1] = 0.9       # beta_x
        self.k[2] = 1.0       # psi
        self.k[3] = 0.0       # alpha_x
        self.k[4] = 1.1       # beta_y
        self.k[5] = 0.8       # alpha_y
        self.k[6] = 0.8       # alpha_0
        self.k[7] = 0.0001    # k
        self.k[8] = 1.7       # alpha_k
        self.k[9] = 1.0       # compartment

    def ode(self, x):
        x1 = x[..., 0]  # p53
        x2 = x[..., 1]  # Mdm2
        x3 = x[..., 2]  # Mdm2 precursor

        # dx1/dt = k1*k2 - k3*x1 - k8*x2*x1/(x1 + k7)
        # k7=0.0001 is very small; add eps to prevent division by near-zero
        dx1 = (self.k[1] * self.k[2]
               - self.k[3] * x1
               - self.k[8] * x2 * x1 / (x1 + self.k[7] + 1e-6))

        # dx2/dt = k6*x3 - k5*x2
        dx2 = self.k[6] * x3 - self.k[5] * x2

        # dx3/dt = k4*x1*k2 - k6*x3
        dx3 = self.k[4] * x1 * self.k[2] - self.k[6] * x3

        return torch.stack((dx1, dx2, dx3), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)
        # All species are concentrations > 0
        # Scale to reasonable ranges around steady state
        x[..., 0] *= 2.0   # p53
        x[..., 1] *= 2.0   # Mdm2
        x[..., 2] *= 2.0   # Mdm2 precursor
        return x
