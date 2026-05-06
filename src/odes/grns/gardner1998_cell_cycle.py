import torch
from ..ode_base import ODEBase


class Gardner1998CellCycleGoldbeter(ODEBase):
    DESCRIPTION = "Gardner1998 - Cell Cycle Goldbeter"
    NODE_NAMES = {
        1: "C",
        2: "X",
        3: "M",
        4: "Y",
        5: "Z",
    }
    PARAMETER_NAMES = {
        1: "V1",
        2: "K6",
        3: "V1p",
        4: "V3",
        5: "V3p",
        6: "Cell",
        7: "vi",
        8: "k1",
        9: "K5",
        10: "kd",
        11: "K1",
        12: "V2",
        13: "K2",
        14: "K3",
        15: "K4",
        16: "V4",
        17: "a1",
        18: "a2",
        19: "alpha",
        20: "d1",
        21: "kd_Z",
        22: "alpha_Z",
        23: "vs",
        24: "d1_Y",
    }
    URL = "https://odebase.org/detail/1329"
    NUM_NODES = 5
    NUM_PARAMS = 24
    time_scale = 16
    mu = [0.12, 0.24, 0.29, 3.96, 0.41]
    std = [0.06, 0.09, 0.15, 0.03, 0.03]
    PHASE_PORTRAIT_DIMS = (0, 2, "C (cyclin)", "M (cdc2 kinase)")
    CATEGORY = "Cell Cycle"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)
        # k1=V1 and k4=V3 are assignment rules, not free parameters
        self.k[1] = 0            # V1 (assigned: C * V1p / (C + K6))
        self.k[2] = 3 / 10      # K6 = 0.3
        self.k[3] = 3 / 4       # V1p = 0.75
        self.k[4] = 0            # V3 (assigned: M * V3p)
        self.k[5] = 3 / 10      # V3p = 0.3
        self.k[6] = 1            # Cell = 1
        self.k[7] = 1 / 10      # vi = 0.1
        self.k[8] = 1 / 2       # k1 = 0.5
        self.k[9] = 1 / 50      # K5 = 0.02
        self.k[10] = 1 / 50     # kd = 0.02
        self.k[11] = 1 / 10     # K1 = 0.1
        self.k[12] = 1 / 4      # V2 = 0.25
        self.k[13] = 1 / 10     # K2 = 0.1
        self.k[14] = 1 / 5      # K3 = 0.2
        self.k[15] = 1 / 10     # K4 = 0.1
        self.k[16] = 1 / 10     # V4 = 0.1
        self.k[17] = 1 / 20     # a1 = 0.05
        self.k[18] = 1 / 20     # a2 = 0.05
        self.k[19] = 1 / 10     # alpha = 0.1
        self.k[20] = 1 / 20     # d1 = 0.05
        self.k[21] = 1 / 50     # kd (for Z degradation) = 0.02
        self.k[22] = 1 / 10     # alpha (for Z degradation) = 0.1
        self.k[23] = 1 / 5      # vs = 0.2
        self.k[24] = 1 / 20     # d1 (for Y degradation) = 0.05

    def project_to_manifold(self, x):
        return x

    def ode(self, x):
        x1, x2, x3, x4, x5 = (
            x[..., 0],  # C (cyclin)
            x[..., 1],  # X (protease)
            x[..., 2],  # M (cdc2 kinase)
            x[..., 3],  # Y (cyclin inhibitor)
            x[..., 4],  # Z (inhibitor-cyclin complex)
        )

        # Assignment rules (not free params, computed from state)
        # V1 = C * V1p / (C + K6)
        V1 = x1 * self.k[3] / (x1 + self.k[2])
        # V3 = M * V3p
        V3 = x3 * self.k[5]

        # dC/dt: creation - cdc2k-triggered degradation - default degradation
        #        - binding with Y + dissociation of Z + desinhibition of Z
        dx1 = (
            self.k[7]                                    # vi (creation of cyclin)
            - self.k[8] * x2 * x1 / (self.k[9] + x1)   # k1*X*C/(K5+C) (cdc2k-triggered degradation)
            - self.k[10] * x1                            # kd*C (default degradation)
            - self.k[17] * x1 * x4                       # a1*C*Y (binding)
            + self.k[18] * x5                            # a2*Z (dissociation)
            + self.k[19] * self.k[20] * x5               # alpha*d1*Z (desinhibition)
        ) / self.k[6]

        # dX/dt: activation - deactivation of cyclin protease
        # Goldbeter-Koshland terms: V3*(1-X)/(K3+1-X) - V4*X/(K4+X)
        # x2 can exceed [0,1] after denormalization; add eps to prevent div-by-zero
        eps = 1e-6
        dx2 = (
            V3 * (1 - x2) / (self.k[14] + 1 - x2 + eps)      # activation
            - self.k[16] * x2 / (self.k[15] + x2 + eps)       # deactivation
        ) / self.k[6]

        # dM/dt: activation - deactivation of cdc2 kinase
        # V1*(1-M)/(K1+1-M) - V2*M/(K2+M)
        dx3 = (
            V1 * (1 - x3) / (self.k[11] + 1 - x3 + eps)      # activation
            - self.k[12] * x3 / (self.k[13] + x3 + eps)       # deactivation
        ) / self.k[6]

        # dY/dt: -binding + dissociation + degraded-Z-releases-Y + creation - degradation
        dx4 = (
            - self.k[17] * x1 * x4                       # -a1*C*Y (binding)
            + self.k[18] * x5                            # a2*Z (dissociation)
            + self.k[22] * self.k[21] * x5               # alpha*kd*Z (degradation of Z releases Y)
            + self.k[23]                                  # vs (creation of inhibitor)
            - self.k[24] * x4                            # d1*Y (degradation of inhibitor)
        ) / self.k[6]

        # dZ/dt: binding - dissociation - desinhibition - degradation-release
        dx5 = (
            self.k[17] * x1 * x4                         # a1*C*Y (binding)
            - self.k[18] * x5                            # a2*Z (dissociation)
            - self.k[19] * self.k[20] * x5               # alpha*d1*Z (desinhibition)
            - self.k[22] * self.k[21] * x5               # alpha*kd*Z (degradation of Z)
        ) / self.k[6]

        return torch.stack((dx1, dx2, dx3, dx4, dx5), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)
        # Reference initial values from BIOMD0000000008:
        # C=0, X=0, M=0, Y=1, Z=1
        # Use random perturbations around biologically reasonable ranges
        x[..., 0] *= 0.5     # C (cyclin): [0, 0.5]
        x[..., 1] *= 1.0     # X (protease): [0, 1] (fraction active)
        x[..., 2] *= 1.0     # M (cdc2k): [0, 1] (fraction active)
        x[..., 3] *= 2.0     # Y (inhibitor): [0, 2]
        x[..., 4] *= 2.0     # Z (complex): [0, 2]
        return x
