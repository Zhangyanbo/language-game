import torch
from ..ode_base import ODEBase


class Liebal2012TranscriptionInhibition(ODEBase):
    """Liebal2012 - B.subtilis transcription inhibition model.

    A rational (non-polynomial) ODE model of transcription inhibition in
    Bacillus subtilis, with 4 species and 8 parameters. IPTG (x1) is constant;
    sigB (x2) drives production of lacZ (x3) and x (x4), both inhibited by x4.

    Reference: https://odebase.org/detail/1616
    BioModels: BIOMD0000000461
    """

    DESCRIPTION = "Liebal2012 - B.subtilis transcription inhibition model"
    NODE_NAMES = {
        1: "IPTG",
        2: "sigb",
        3: "lacz",
        4: "x",
    }
    PARAMETER_NAMES = {
        1: "kbd",
        2: "kbs",
        3: "kxd",
        4: "kxs",
        5: "kzd",
        6: "kzs",
        7: "compartment",
        8: "IPTG",
    }
    URL = "https://odebase.org/detail/1616"
    NUM_NODES = 4
    NUM_PARAMS = 8

    # Calibrated from 20 trials, RK4, dt=0.01, T=200
    time_scale = 100
    mu = [100.0, 226660.0, 15.75, 137.85]
    std = [1.0, 46480.0, 4.1, 19.25]
    PHASE_PORTRAIT_DIMS = (1, 2, "sigB", "lacZ")
    CATEGORY = "Synthetic Gene Circuit"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)
        self.k[1] = 11 / 250       # kbd = 0.044
        self.k[2] = 100             # kbs = 100
        self.k[3] = 9               # kxd = 9
        self.k[4] = 19 / 25         # kxs = 0.76
        self.k[5] = 41 / 1000       # kzd = 0.041
        self.k[6] = 1 / 2500        # kzs = 0.0004
        self.k[7] = 1               # compartment = 1
        self.k[8] = 100             # IPTG = 100

    def ode(self, x):
        x1 = x[..., 0]  # IPTG (constant)
        x2 = x[..., 1]  # sigb
        x3 = x[..., 2]  # lacz
        x4 = x[..., 3]  # x

        inv_1_plus_x4 = 1.0 / (1.0 + x4)

        # dx1/dt = 0  (IPTG is constant)
        dx1 = torch.zeros_like(x1)

        # Reaction fluxes (from odebase LaTeX):
        #   R1 = k8*k2 - k1*x2             (sigb synthesis - degradation)
        #   R2 = -k5*x3 + k6*x2/(1+x4)    (lacz production inhibited by x4)
        #   R3 = -k3*x4 + k4*x2/(1+x4)    (x production inhibited by x4)
        #
        # dx2/dt = (R1 - R2 - R3) / k7
        # dx3/dt = R2 / k7
        # dx4/dt = R3 / k7

        r1 = self.k[8] * self.k[2] - self.k[1] * x2
        r2 = -self.k[5] * x3 + self.k[6] * x2 * inv_1_plus_x4
        r3 = -self.k[3] * x4 + self.k[4] * x2 * inv_1_plus_x4

        dx2 = (r1 - r2 - r3) / self.k[7]
        dx3 = r2 / self.k[7]
        dx4 = r3 / self.k[7]

        return torch.stack((dx1, dx2, dx3, dx4), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)

        # IPTG is constant at k8
        x[..., 0] = self.k[8]
        # sigb: positive, around steady-state range
        x[..., 1] = x[..., 1] * 10000 + 1000
        # lacz: small positive
        x[..., 2] = x[..., 2] * 2.0 + 0.01
        # x: small positive
        x[..., 3] = x[..., 3] * 0.2 + 0.001
        return x
