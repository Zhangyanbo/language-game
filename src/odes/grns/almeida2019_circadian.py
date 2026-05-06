import torch
from ..ode_base import ODEBase


class Almeida2019CircadianClock(ODEBase):
    DESCRIPTION = "Almeida2019 - Transcription-based circadian mechanism"
    NODE_NAMES = {
        1: "BMAL1",
        2: "ROR",
        3: "REV",
        4: "DBP",
        5: "E4BP4",
        6: "CRY",
        7: "PER",
        8: "PERCRY",
    }
    PARAMETER_NAMES = {
        1: "V_R",
        2: "k_R",
        3: "k_R_r",
        4: "V_E",
        5: "k_E",
        6: "k_E_r",
        7: "V_D",
        8: "k_D",
        9: "k_D_r",
        10: "gamma_ror",
        11: "gamma_rev",
        12: "gamma_p",
        13: "gamma_c",
        14: "gamma_db",
        15: "gamma_E4",
        16: "gamma_pc",
        17: "gamma_cp",
        18: "gamma_bp",
        19: "E_box",
        20: "R_box",
        21: "D_box",
        22: "ModelValue_6",
        23: "ModelValue_3",
        24: "ModelValue_0",
        25: "ModelValue_8",
        26: "ModelValue_4",
        27: "ModelValue_5",
        28: "ModelValue_1",
        29: "ModelValue_2",
        30: "compartment",
    }
    URL = "https://odebase.org/detail/1850"
    NUM_NODES = 8
    NUM_PARAMS = 30
    time_scale = 20
    mu = [1.94, 3.08, 107.82, 1.40, 51.80, 3.36, 21.48, 6.70]
    std = [2.96, 1.83, 22.46, 0.74, 20.40, 2.82, 10.53, 4.40]
    PHASE_PORTRAIT_DIMS = (5, 6, "CRY", "PER")
    CATEGORY = "Circadian Clock"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)  # 1-indexed
        self.k[1] = 222 / 5          # V_R = 44.4
        self.k[2] = 177 / 50         # k_R = 3.54
        self.k[3] = 801 / 10         # k_R_r = 80.1
        self.k[4] = 303 / 10         # V_E = 30.3
        self.k[5] = 214              # k_E = 214
        self.k[6] = 31 / 25          # k_E_r = 1.24
        self.k[7] = 202              # V_D = 202
        self.k[8] = 133 / 25         # k_D = 5.32
        self.k[9] = 947 / 10         # k_D_r = 94.7
        self.k[10] = 51 / 20         # gamma_ror = 2.55
        self.k[11] = 241 / 1000      # gamma_rev = 0.241
        self.k[12] = 211 / 250       # gamma_p = 0.844
        self.k[13] = 117 / 50        # gamma_c = 2.34
        self.k[14] = 39 / 250        # gamma_db = 0.156
        self.k[15] = 59 / 200        # gamma_E4 = 0.295
        self.k[16] = 191 / 1000      # gamma_pc = 0.191
        self.k[17] = 141 / 1000      # gamma_cp = 0.141
        self.k[18] = 129 / 50        # gamma_bp = 2.58
        self.k[19] = 505 / 3604      # E_box
        self.k[20] = 1424354220 / 420313427  # R_box
        self.k[21] = 2391175 / 138013  # D_box
        self.k[22] = 202              # ModelValue_6 (= V_D)
        self.k[23] = 303 / 10         # ModelValue_3 (= V_E)
        self.k[24] = 222 / 5          # ModelValue_0 (= V_R)
        self.k[25] = 947 / 10         # ModelValue_8 (= k_D_r)
        self.k[26] = 214              # ModelValue_4 (= k_E)
        self.k[27] = 31 / 25          # ModelValue_5 (= k_E_r)
        self.k[28] = 177 / 50         # ModelValue_1 (= k_R)
        self.k[29] = 801 / 10         # ModelValue_2 (= k_R_r)
        self.k[30] = 1                # compartment

    def project_to_manifold(self, x):
        # No constraints for this model
        return x

    def ode(self, x):
        # x.shape = (..., 8)
        x1 = x[..., 0]  # BMAL1
        x2 = x[..., 1]  # ROR
        x3 = x[..., 2]  # REV
        x4 = x[..., 3]  # DBP
        x5 = x[..., 4]  # E4BP4
        x6 = x[..., 5]  # CRY
        x7 = x[..., 6]  # PER
        x8 = x[..., 7]  # PERCRY

        # Shared sub-expressions (k30=1, so it cancels out)
        # R-box activation: k24 * x2 / (x2 + k28) * k29^2 / (k29^2 + x3^2)
        rbox = self.k[24] * x2 / (x2 + self.k[28]) * self.k[29] ** 2 / (self.k[29] ** 2 + x3 ** 2)

        # E-box activation: k23 * x1 / (x1 + k26 + k27 * x1 * x6)
        ebox = self.k[23] * x1 / (x1 + self.k[26] + self.k[27] * x1 * x6)

        # D-box activation: k22 * x4 / (x4 + k8) * k25 / (k25 + x5)
        dbox = self.k[22] * x4 / (x4 + self.k[8]) * self.k[25] / (self.k[25] + x5)

        # dx1/dt = rbox - k18 * x1 * x8
        dx1 = rbox - self.k[18] * x1 * x8

        # dx2/dt = ebox + rbox - k10 * x2
        dx2 = ebox + rbox - self.k[10] * x2

        # dx3/dt = 2 * ebox + dbox - k11 * x3
        dx3 = 2 * ebox + dbox - self.k[11] * x3

        # dx4/dt = ebox - k14 * x4
        dx4 = ebox - self.k[14] * x4

        # dx5/dt = 2 * rbox - k15 * x5
        dx5 = 2 * rbox - self.k[15] * x5

        # dx6/dt = 2 * rbox + ebox - k16 * x7 * x6 + k17 * x8 - k13 * x6
        dx6 = 2 * rbox + ebox - self.k[16] * x7 * x6 + self.k[17] * x8 - self.k[13] * x6

        # dx7/dt = -k16 * x7 * x6 + k17 * x8 + ebox + dbox - k12 * x7
        dx7 = -self.k[16] * x7 * x6 + self.k[17] * x8 + ebox + dbox - self.k[12] * x7

        # dx8/dt = k16 * x7 * x6 - k17 * x8 - k18 * x1 * x8
        dx8 = self.k[16] * x7 * x6 - self.k[17] * x8 - self.k[18] * x1 * x8

        return torch.stack((dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)
        # Scale to reasonable biological ranges based on observed steady-state oscillations
        x[..., 0] = x[..., 0] * 9.0 + 0.1    # BMAL1: ~0.1-9.6
        x[..., 1] = x[..., 1] * 5.0 + 1.0    # ROR: ~1.0-6.4
        x[..., 2] = x[..., 2] * 65.0 + 71.0  # REV: ~71-137
        x[..., 3] = x[..., 3] * 2.0 + 0.5    # DBP: ~0.5-2.8
        x[..., 4] = x[..., 4] * 59.0 + 26.0  # E4BP4: ~26-86
        x[..., 5] = x[..., 5] * 8.5 + 0.7    # CRY: ~0.7-9.3
        x[..., 6] = x[..., 6] * 30.0 + 5.5   # PER: ~5.5-35.7
        x[..., 7] = x[..., 7] * 12.0 + 0.6   # PERCRY: ~0.6-12.6
        return x
