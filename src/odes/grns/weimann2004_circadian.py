import torch
from ..ode_base import ODEBase


class Weimann2004CircadianOscillator(ODEBase):
    DESCRIPTION = "Weimann2004 - Circadian Oscillator"
    NODE_NAMES = {
        1: "Per2_Cry_mRNA",
        2: "Per2_Cry_cytoplasm",
        3: "Per2_Cry_nucleus",
        4: "Bmal1_mRNA",
        5: "Bmal1_cytoplasm",
        6: "Bmal1_nucleus",
        7: "Bmal1_nucleus_active",
    }
    PARAMETER_NAMES = {
        1: "trans_per2_cry",
        2: "v1b",
        3: "c",
        4: "k1b",
        5: "k1i",
        6: "hill_coeff",
        7: "trans_Bmal1",
        8: "v4b",
        9: "r",
        10: "k4b",
        11: "y5_y6_y7",
        12: "k1d",
        13: "k2b",
        14: "q",
        15: "k2d",
        16: "k2t",
        17: "k3t",
        18: "k3d",
        19: "k4d",
        20: "k5b",
        21: "k5d",
        22: "k5t",
        23: "k6t",
        24: "k6d",
        25: "k6a",
        26: "k7a",
        27: "k7d",
        28: "Nucleus",
        29: "Cytoplasm",
    }
    URL = "https://odebase.org/detail/1432"
    NUM_NODES = 7
    NUM_PARAMS = 29
    time_scale = 24  # ~24h circadian oscillation period
    mu = [0.74, 0.95, 1.64, 1.16, 0.70, 1.18, 1.15]
    std = [0.51, 1.54, 1.92, 0.81, 0.42, 0.64, 0.55]
    PHASE_PORTRAIT_DIMS = (0, 5, "Per2/Cry mRNA", "Bmal1 nucleus")
    CATEGORY = "Circadian Clock"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)  # 1-indexed
        self.k[1] = 10011752955904 / 234756138908881  # ~0.04265
        self.k[2] = 9.0
        self.k[3] = 0.01
        self.k[4] = 1.0
        self.k[5] = 0.56
        self.k[6] = 8.0
        self.k[7] = 598950 / 1426087  # ~0.42
        self.k[8] = 3.6
        self.k[9] = 3.0
        self.k[10] = 2.16
        self.k[11] = 3.05
        self.k[12] = 0.12
        self.k[13] = 0.3
        self.k[14] = 2.0
        self.k[15] = 0.05
        self.k[16] = 0.24
        self.k[17] = 0.02
        self.k[18] = 0.12
        self.k[19] = 0.75
        self.k[20] = 0.24
        self.k[21] = 0.06
        self.k[22] = 0.45
        self.k[23] = 0.06
        self.k[24] = 0.12
        self.k[25] = 0.09
        self.k[26] = 0.003
        self.k[27] = 0.09
        self.k[28] = 1.0  # Nucleus volume
        self.k[29] = 1.0  # Cytoplasm volume

    def project_to_manifold(self, x):
        return x

    def ode(self, x):
        # x.shape = (..., 7)
        x1 = x[..., 0]  # Per2/Cry mRNA
        x2 = x[..., 1]  # Per2/Cry protein (cytoplasm)
        x3 = x[..., 2]  # Per2/Cry protein (nucleus)
        x4 = x[..., 3]  # Bmal1 mRNA
        x5 = x[..., 4]  # Bmal1 protein (cytoplasm)
        x6 = x[..., 5]  # Bmal1 protein (nucleus)
        x7 = x[..., 6]  # Bmal1 protein (nucleus, active)

        # Since k28 = k29 = 1 (compartment volumes), the ODEs simplify:
        # dx1/dt = v1b * (x7 + c) / (k1b * (1 + (x3/k1i)^hill) + x7 + c) - k1d * x1
        dx1 = (
            self.k[2] * (x7 + self.k[3])
            / (self.k[4] * (1 + (x3 / self.k[5]) ** self.k[6]) + x7 + self.k[3])
            - self.k[12] * x1
        )

        # dx2/dt = k2b * x1^q - k2d * x2 - k2t * x2 + k3t * x3
        dx2 = (
            self.k[13] * x1 ** self.k[14]
            - self.k[15] * x2
            - self.k[16] * x2
            + self.k[17] * x3
        )

        # dx3/dt = k2t * x2 - k3t * x3 - k3d * x3
        dx3 = self.k[16] * x2 - self.k[17] * x3 - self.k[18] * x3

        # dx4/dt = v4b * x3^r / (k4b^r + x3^r) - k4d * x4
        dx4 = (
            self.k[8] * x3 ** self.k[9]
            / (self.k[10] ** self.k[9] + x3 ** self.k[9])
            - self.k[19] * x4
        )

        # dx5/dt = k5b * x4 - k5d * x5 - k5t * x5 + k6t * x6
        dx5 = (
            self.k[20] * x4
            - self.k[21] * x5
            - self.k[22] * x5
            + self.k[23] * x6
        )

        # dx6/dt = k5t * x5 - k6t * x6 - k6d * x6 - k6a * x6 + k7a * x7
        dx6 = (
            self.k[22] * x5
            - self.k[23] * x6
            - self.k[24] * x6
            - self.k[25] * x6
            + self.k[26] * x7
        )

        # dx7/dt = k6a * x6 - k7a * x7 - k7d * x7
        dx7 = self.k[25] * x6 - self.k[26] * x7 - self.k[27] * x7

        return torch.stack((dx1, dx2, dx3, dx4, dx5, dx6, dx7), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)
        # Based on initial values from BioModels:
        # y1=0.2, y2=0.0, y3=1.1, y4=0.8, y5=1.0, y6=1.0, y7=1.05
        # Scale random values to plausible biological ranges
        x[..., 0] *= 2.0    # Per2/Cry mRNA: 0-2
        x[..., 1] *= 2.0    # Per2/Cry cytoplasm: 0-2
        x[..., 2] *= 3.0    # Per2/Cry nucleus: 0-3
        x[..., 3] *= 2.0    # Bmal1 mRNA: 0-2
        x[..., 4] *= 3.0    # Bmal1 cytoplasm: 0-3
        x[..., 5] *= 3.0    # Bmal1 nucleus: 0-3
        x[..., 6] *= 3.0    # Bmal1 nucleus active: 0-3
        x = torch.clamp(x, min=1e-4)
        return x
