import torch
from ..ode_base import ODEBase


class Gerard2010CellCycle(ODEBase):
    DESCRIPTION = "Gerard2010 - Progression of mammalian cell cycle"
    NODE_NAMES = {
        1: "cyclin_D_Cdk4_6",
        2: "transcription_factor_E2F_active",
        3: "cyclin_E_Cdk2",
        4: "cyclin_A_Cdk2",
        5: "cyclin_B_Cdk1",
        6: "Cdc20_active",
        7: "E2F_total",
        8: "Cdc20_total",
    }
    PARAMETER_NAMES = {
        1: "GF",
        2: "Kda",
        3: "Kdb",
        4: "Kdd",
        5: "Kde",
        6: "Kgf",
        7: "K1cdc20",
        8: "K2cdc20",
        9: "K1e2f",
        10: "K2e2f",
        11: "Vda",
        12: "Vdb",
        13: "Vdd",
        14: "Vde",
        15: "vsa",
        16: "vsb",
        17: "vsd",
        18: "vse",
        19: "V1cdc20",
        20: "V2cdc20",
        21: "V1e2f",
        22: "V2e2f",
        23: "nuclear",
    }
    URL = "https://odebase.org/detail/1928"
    NUM_NODES = 8
    NUM_PARAMS = 23
    time_scale = 24
    mu = [0.19, 1.22, 0.76, 1.36, 1.20, 1.37, 3.0, 5.0]
    std = [0.01, 1.25, 0.86, 1.14, 1.13, 0.66, 0.01, 0.01]
    PHASE_PORTRAIT_DIMS = (3, 4, "cyclin A/Cdk2", "cyclin B/Cdk1")
    CATEGORY = "Cell Cycle"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)  # 1-indexed
        self.k[1] = 1              # GF
        self.k[2] = 1 / 10         # Kda = 0.1
        self.k[3] = 1 / 200        # Kdb = 0.005
        self.k[4] = 1 / 10         # Kdd = 0.1
        self.k[5] = 1 / 10         # Kde = 0.1
        self.k[6] = 1 / 10         # Kgf = 0.1
        self.k[7] = 1              # K1cdc20 = 1.0
        self.k[8] = 1              # K2cdc20 = 1.0
        self.k[9] = 1 / 100        # K1e2f = 0.01
        self.k[10] = 1 / 100       # K2e2f = 0.01
        self.k[11] = 49 / 200      # Vda = 0.245
        self.k[12] = 7 / 25        # Vdb = 0.28
        self.k[13] = 49 / 200      # Vdd = 0.245
        self.k[14] = 7 / 20        # Vde = 0.35
        self.k[15] = 7 / 40        # vsa = 0.175
        self.k[16] = 21 / 100      # vsb = 0.21
        self.k[17] = 7 / 40        # vsd = 0.175
        self.k[18] = 21 / 100      # vse = 0.21
        self.k[19] = 21 / 100      # V1cdc20 = 0.21
        self.k[20] = 7 / 20        # V2cdc20 = 0.35
        self.k[21] = 161 / 200     # V1e2f = 0.805
        self.k[22] = 7 / 10        # V2e2f = 0.7
        self.k[23] = 1             # nuclear compartment volume

    def project_to_manifold(self, x):
        return x

    def _safe_div(self, num, denom, eps=1e-6):
        """Safe division: clamp denominator away from zero."""
        return num / (denom + eps * torch.sign(denom + eps))

    def ode(self, x):
        # x.shape = (..., 8)
        x1 = x[..., 0]  # cyclin D / Cdk4-6
        x2 = x[..., 1]  # E2F active
        x3 = x[..., 2]  # cyclin E / Cdk2
        x4 = x[..., 3]  # cyclin A / Cdk2
        x5 = x[..., 4]  # cyclin B / Cdk1
        x6 = x[..., 5]  # Cdc20 active
        x7 = x[..., 6]  # E2F total (constant)
        x8 = x[..., 7]  # Cdc20 total (constant)

        eps = 1e-6

        # dx1/dt = vsd * GF / (Kgf + GF) - Vdd * x1 / (Kdd + x1)
        dx1 = (
            self.k[17] * self.k[1] / (self.k[6] + self.k[1])
            - self.k[13] * x1 / (self.k[4] + x1 + eps)
        )

        # dx2/dt: difference terms (x7-x2) can make denominator near zero
        diff_e2f = x7 - x2
        dx2 = (
            self.k[21] * diff_e2f / (self.k[9] + diff_e2f + eps) * (x1 + x3)
            - self.k[22] * x2 / (self.k[10] + x2 + eps) * x4
        )

        # dx3/dt = vse * x2 - Vde * x4 * x3 / (Kde + x3)
        dx3 = (
            self.k[18] * x2
            - self.k[14] * x4 * x3 / (self.k[5] + x3 + eps)
        )

        # dx4/dt = vsa * x2 - Vda * x6 * x4 / (Kda + x4)
        dx4 = (
            self.k[15] * x2
            - self.k[11] * x6 * x4 / (self.k[2] + x4 + eps)
        )

        # dx5/dt = vsb * x4 - Vdb * x6 * x5 / (Kdb + x5)
        dx5 = (
            self.k[16] * x4
            - self.k[12] * x6 * x5 / (self.k[3] + x5 + eps)
        )

        # dx6/dt: difference term (x8-x6) can make denominator near zero
        diff_cdc20 = x8 - x6
        dx6 = (
            self.k[19] * x5 * diff_cdc20 / (self.k[7] + diff_cdc20 + eps)
            - self.k[20] * x6 / (self.k[8] + x6 + eps)
        )

        # dx7/dt = 0 (E2F total is conserved)
        dx7 = torch.zeros_like(x7)

        # dx8/dt = 0 (Cdc20 total is conserved)
        dx8 = torch.zeros_like(x8)

        return torch.stack((dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)
        # Scale to biologically plausible ranges based on BioModels initial conditions:
        # x1=0.7205, x2=2.4855, x3=2.0, x4=2.6, x5=1.0, x6=0.5, x7=3.0, x8=5.0
        x[..., 0] *= 1.5       # cyclin D / Cdk4-6: 0-1.5
        x[..., 1] *= 3.0       # E2F active: 0-3.0 (must be < E2F_total)
        x[..., 2] *= 3.0       # cyclin E / Cdk2: 0-3.0
        x[..., 3] *= 4.0       # cyclin A / Cdk2: 0-4.0
        x[..., 4] *= 2.0       # cyclin B / Cdk1: 0-2.0
        x[..., 5] *= 1.0       # Cdc20 active: 0-1.0 (must be < Cdc20_total)
        x[..., 6] = 3.0        # E2F total (constant)
        x[..., 7] = 5.0        # Cdc20 total (constant)
        x = torch.clamp(x, min=1e-4)
        return x
