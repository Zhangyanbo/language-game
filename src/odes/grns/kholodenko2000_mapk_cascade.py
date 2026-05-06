import torch
from ..ode_base import ODEBase


class Kholodenko2000MAPKCascade(ODEBase):
    DESCRIPTION = "Kholodenko2000 - 3-tier MAPK cascade with MAPK-PP negative feedback"
    NODE_NAMES = {
        1: "MKKK",
        2: "MKKK_P",
        3: "MKK",
        4: "MKK_P",
        5: "MKK_PP",
        6: "MAPK",
        7: "MAPK_P",
        8: "MAPK_PP",
    }
    PARAMETER_NAMES = {
        1: "uVol",
        2: "V1 (MKKK activation Vmax)",
        3: "Ki (MAPK_PP feedback Ki)",
        4: "n (Hill coefficient)",
        5: "K1 (MKKK activation Km)",
        6: "V2 (MKKK dephospho Vmax)",
        7: "KK2 (MKKK dephospho Km)",
        8: "k3 (MKK 1st phospho rate)",
        9: "KK3 (MKK 1st phospho Km)",
        10: "k4 (MKK 2nd phospho rate)",
        11: "KK4 (MKK 2nd phospho Km)",
        12: "V5 (MKK_PP dephospho Vmax)",
        13: "KK5 (MKK_PP dephospho Km)",
        14: "V6 (MKK_P dephospho Vmax)",
        15: "KK6 (MKK_P dephospho Km)",
        16: "k7 (MAPK 1st phospho rate)",
        17: "KK7 (MAPK 1st phospho Km)",
        18: "k8 (MAPK 2nd phospho rate)",
        19: "KK8 (MAPK 2nd phospho Km)",
        20: "V9 (MAPK_PP dephospho Vmax)",
        21: "KK9 (MAPK_PP dephospho Km)",
        22: "V10 (MAPK_P dephospho Vmax)",
        23: "KK10 (MAPK_P dephospho Km)",
        24: "MKKK_total",
        25: "MKK_total",
        26: "MAPK_total",
    }
    URL = "https://odebase.org/detail/1331"
    NUM_NODES = 8
    NUM_PARAMS = 26
    time_scale = 1300
    # Calibrated from 20-trajectory simulation (latter half)
    mu = [78.43, 21.57, 212.43, 45.79, 41.77, 79.29, 38.67, 182.04]
    std = [17.54, 17.54, 77.98, 33.64, 52.41, 76.26, 26.07, 97.91]
    PHASE_PORTRAIT_DIMS = (1, 7, "MKKK_P", "MAPK_PP")
    CATEGORY = "Signal Transduction"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)
        self.k[1] = 1       # uVol (cancels in all ODEs)
        self.k[2] = 2.5     # V1
        self.k[3] = 9       # Ki
        self.k[4] = 1       # n (Hill exponent; linear feedback)
        self.k[5] = 10      # K1
        self.k[6] = 0.25    # V2
        self.k[7] = 8       # KK2
        self.k[8] = 0.025   # k3
        self.k[9] = 15      # KK3
        self.k[10] = 0.025  # k4
        self.k[11] = 15     # KK4
        self.k[12] = 0.75   # V5
        self.k[13] = 15     # KK5
        self.k[14] = 0.75   # V6
        self.k[15] = 15     # KK6
        self.k[16] = 0.025  # k7
        self.k[17] = 15     # KK7
        self.k[18] = 0.025  # k8
        self.k[19] = 15     # KK8
        self.k[20] = 0.5    # V9
        self.k[21] = 15     # KK9
        self.k[22] = 0.5    # V10
        self.k[23] = 15     # KK10
        self.k[24] = 100    # MKKK_total (conservation: MKKK + MKKK_P = 100)
        self.k[25] = 300    # MKK_total  (conservation: MKK + MKK_P + MKK_PP = 300)
        self.k[26] = 300    # MAPK_total (conservation: MAPK + MAPK_P + MAPK_PP = 300)

    def project_to_manifold(self, x):
        """Enforce conservation laws by rescaling each tier."""
        # MKKK tier: x1 + x2 = MKKK_total
        s1 = x[..., :2].sum(dim=-1, keepdim=True)
        x12 = x[..., :2] * (self.k[24] / torch.clamp(s1, min=1e-12))

        # MKK tier: x3 + x4 + x5 = MKK_total
        s2 = x[..., 2:5].sum(dim=-1, keepdim=True)
        x345 = x[..., 2:5] * (self.k[25] / torch.clamp(s2, min=1e-12))

        # MAPK tier: x6 + x7 + x8 = MAPK_total
        s3 = x[..., 5:8].sum(dim=-1, keepdim=True)
        x678 = x[..., 5:8] * (self.k[26] / torch.clamp(s3, min=1e-12))

        return torch.cat((x12, x345, x678), dim=-1)

    def ode(self, x):
        x1 = x[..., 0]  # MKKK (inactive)
        x2 = x[..., 1]  # MKKK_P (active)
        x3 = x[..., 2]  # MKK
        x4 = x[..., 3]  # MKK_P
        x5 = x[..., 4]  # MKK_PP (active)
        x6 = x[..., 5]  # MAPK
        x7 = x[..., 6]  # MAPK_P
        x8 = x[..., 7]  # MAPK_PP (active, feeds back to inhibit MKKK activation)

        # MKKK tier: activation inhibited by MAPK_PP feedback (Hill n=1 → linear)
        feedback = 1.0 + x8 / self.k[3]  # 1 + MAPK_PP / Ki
        act_mkkk = self.k[2] * x1 / (feedback * (self.k[5] + x1))
        deact_mkkk = self.k[6] * x2 / (self.k[7] + x2)
        dx1 = -act_mkkk + deact_mkkk
        dx2 = act_mkkk - deact_mkkk

        # MKK tier: MKKK_P drives double phosphorylation; phosphatases dephosphorylate
        mkk_p1 = self.k[8] * x2 * x3 / (self.k[9] + x3)       # MKK -> MKK_P
        mkk_p2 = self.k[10] * x2 * x4 / (self.k[11] + x4)     # MKK_P -> MKK_PP
        mkk_d1 = self.k[12] * x5 / (self.k[13] + x5)           # MKK_PP -> MKK_P
        mkk_d2 = self.k[14] * x4 / (self.k[15] + x4)           # MKK_P -> MKK
        dx3 = -mkk_p1 + mkk_d2
        dx4 = mkk_p1 - mkk_p2 + mkk_d1 - mkk_d2
        dx5 = mkk_p2 - mkk_d1

        # MAPK tier: MKK_PP drives double phosphorylation; phosphatases dephosphorylate
        mapk_p1 = self.k[16] * x5 * x6 / (self.k[17] + x6)    # MAPK -> MAPK_P
        mapk_p2 = self.k[18] * x5 * x7 / (self.k[19] + x7)    # MAPK_P -> MAPK_PP
        mapk_d1 = self.k[20] * x8 / (self.k[21] + x8)          # MAPK_PP -> MAPK_P
        mapk_d2 = self.k[22] * x7 / (self.k[23] + x7)          # MAPK_P -> MAPK
        dx6 = -mapk_p1 + mapk_d2
        dx7 = mapk_p1 - mapk_p2 + mapk_d1 - mapk_d2
        dx8 = mapk_p2 - mapk_d1

        return torch.stack((dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)

        # MKKK tier: distribute randomly such that x1 + x2 = MKKK_total = 100
        r1 = x[..., :2]
        x[..., :2] = r1 / r1.sum(dim=-1, keepdim=True) * self.k[24]

        # MKK tier: distribute randomly such that x3 + x4 + x5 = MKK_total = 300
        r2 = x[..., 2:5]
        x[..., 2:5] = r2 / r2.sum(dim=-1, keepdim=True) * self.k[25]

        # MAPK tier: distribute randomly such that x6 + x7 + x8 = MAPK_total = 300
        r3 = x[..., 5:8]
        x[..., 5:8] = r3 / r3.sum(dim=-1, keepdim=True) * self.k[26]

        return x
