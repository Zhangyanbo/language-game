import torch
from ..ode_base import ODEBase


class Chickarmane2006StemCellSwitch(ODEBase):
    """Chickarmane2006 - Stem cell switch reversible (BIOMD0000000203).

    A model of the core pluripotency network involving OCT4, SOX2, NANOG
    and their regulatory interactions. Species x1-x7 are constant (gene loci,
    degradation sink, p53, activator A) and x8-x12 are dynamic proteins.

    The model uses rational (non-polynomial) kinetics with activator/repressor
    terms resembling Hill-like regulation.
    """

    DESCRIPTION = "Chickarmane2006 - Stem cell switch reversible"
    NODE_NAMES = {
        1: "OCT4_Gene",
        2: "NANOG_Gene",
        3: "SOX2_Gene",
        4: "targetGene",
        5: "degradation",
        6: "p53",
        7: "A",
        8: "OCT4",
        9: "SOX2",
        10: "NANOG",
        11: "OCT4_SOX2",
        12: "Protein",
    }
    PARAMETER_NAMES = {
        1: "eta1",
        2: "a1",
        3: "a2",
        4: "a3",
        5: "f",
        6: "b1",
        7: "b2",
        8: "b3",
        9: "gamma1",
        10: "eta5",
        11: "e1",
        12: "e2",
        13: "f2",
        14: "f1",
        15: "f3",
        16: "gamma2",
        17: "k1c",
        18: "k2c",
        19: "k3c",
        20: "eta3",
        21: "c1",
        22: "c2",
        23: "c3",
        24: "d1",
        25: "d2",
        26: "d3",
        27: "gamma3",
        28: "g1",
        29: "eta7",
        30: "h1",
        31: "h2",
        32: "gamma4",
        33: "compartment",
        34: "OCT4_Gene",
        35: "NANOG_Gene",
        36: "SOX2_Gene",
        37: "targetGene",
        38: "degradation",
        39: "p53",
        40: "A",
    }
    URL = "https://odebase.org/detail/1450"
    NUM_NODES = 12
    NUM_PARAMS = 40
    time_scale = 50  # transient dynamics settle within ~50 time units
    # x1-x7: constant species (genes, p53, A); x8-x12: dynamic proteins
    # mu: latter-half means; std: all-time std (20 trials, RK4 dt=0.01 T=200)
    # Constant species use their fixed value as mu, std=1 (they never change)
    mu = [0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 10.0,
          7.26, 7.26, 0.003, 0.53, 4.26]
    std = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 0.5, 0.2, 1.1]
    PHASE_PORTRAIT_DIMS = (9, 11, "NANOG", "Protein")
    CATEGORY = "Cell Fate Decision"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)  # 1-indexed
        self.k[1] = 0.0001    # eta1
        self.k[2] = 1.0       # a1
        self.k[3] = 0.01      # a2
        self.k[4] = 0.2       # a3
        self.k[5] = 1000.0    # f
        self.k[6] = 0.0011    # b1
        self.k[7] = 0.001     # b2
        self.k[8] = 0.0007    # b3
        self.k[9] = 1.0       # gamma1
        self.k[10] = 0.0001   # eta5
        self.k[11] = 0.005    # e1
        self.k[12] = 0.1      # e2
        self.k[13] = 0.000995  # f2
        self.k[14] = 0.001    # f1
        self.k[15] = 0.01     # f3
        self.k[16] = 1.0      # gamma2
        self.k[17] = 0.05     # k1c
        self.k[18] = 0.001    # k2c
        self.k[19] = 5.0      # k3c
        self.k[20] = 0.0001   # eta3
        self.k[21] = 1.0      # c1
        self.k[22] = 0.01     # c2
        self.k[23] = 0.2      # c3
        self.k[24] = 0.0011   # d1
        self.k[25] = 0.001    # d2
        self.k[26] = 0.0007   # d3
        self.k[27] = 1.0      # gamma3
        self.k[28] = 0.1      # g1
        self.k[29] = 0.0001   # eta7
        self.k[30] = 0.0019   # h1
        self.k[31] = 0.05     # h2
        self.k[32] = 0.01     # gamma4
        self.k[33] = 1.0      # compartment
        self.k[34] = 0.0      # OCT4_Gene (constant species x1)
        self.k[35] = 0.0      # NANOG_Gene (constant species x2)
        self.k[36] = 0.0      # SOX2_Gene (constant species x3)
        self.k[37] = 0.01     # targetGene (constant species x4)
        self.k[38] = 0.0      # degradation (constant species x5)
        self.k[39] = 0.0      # p53 (constant species x6)
        self.k[40] = 10.0     # A (constant species x7)

    def project_to_manifold(self, x):
        return x

    def ode(self, x):
        # x shape: (..., 12)
        # x1-x7 are constant species; x8-x12 are dynamic
        x8 = x[..., 7]    # OCT4
        x9 = x[..., 8]    # SOX2
        x10 = x[..., 9]   # NANOG
        x11 = x[..., 10]  # OCT4_SOX2
        x12 = x[..., 11]  # Protein

        # Constant species values (from parameters k34-k40)
        # x6 = k39 (p53), x7 = k40 (A)
        k39 = self.k[39]  # p53
        k40 = self.k[40]  # A

        comp = self.k[33]  # compartment volume

        # dx1/dt through dx7/dt = 0 (constant species)
        zeros = torch.zeros_like(x8)

        # dx8/dt (OCT4):
        # (eta1 + a1*A + a2*OCT4_SOX2 + a3*OCT4_SOX2*NANOG) /
        #   (1 + eta1/f + b1*A + b2*OCT4_SOX2 + b3*OCT4_SOX2*NANOG)
        # - gamma1*OCT4 - (k1c*OCT4*SOX2 - k2c*OCT4_SOX2)
        numer8 = (self.k[1] + self.k[2] * k40
                  + self.k[3] * x11 + self.k[4] * x11 * x10)
        denom8 = (1.0 + self.k[1] / self.k[5] + self.k[6] * k40
                  + self.k[7] * x11 + self.k[8] * x11 * x10)
        binding = self.k[17] * x8 * x9 - self.k[18] * x11
        dx8 = (numer8 / denom8 - self.k[9] * x8 - binding) / comp

        # dx9/dt (SOX2):
        # -(k1c*OCT4*SOX2 - k2c*OCT4_SOX2)
        # + (eta3 + c1*A + c2*OCT4_SOX2 + c3*OCT4_SOX2*NANOG) /
        #   (1 + eta3/f + d1*A + d2*OCT4_SOX2 + d3*OCT4_SOX2*NANOG)
        # - gamma3*SOX2
        numer9 = (self.k[20] + self.k[21] * k40
                  + self.k[22] * x11 + self.k[23] * x11 * x10)
        denom9 = (1.0 + self.k[20] / self.k[5] + self.k[24] * k40
                  + self.k[25] * x11 + self.k[26] * x11 * x10)
        dx9 = (-binding + numer9 / denom9 - self.k[27] * x9) / comp

        # dx10/dt (NANOG):
        # (eta5 + e1*OCT4_SOX2 + e2*OCT4_SOX2*NANOG) /
        #   (1 + eta5/f + f2*OCT4_SOX2 + f1*OCT4_SOX2*NANOG + f3*p53)
        # - gamma2*NANOG
        numer10 = (self.k[10] + self.k[11] * x11
                   + self.k[12] * x11 * x10)
        denom10 = (1.0 + self.k[10] / self.k[5] + self.k[13] * x11
                   + self.k[14] * x11 * x10 + self.k[15] * k39)
        dx10 = (numer10 / denom10 - self.k[16] * x10) / comp

        # dx11/dt (OCT4_SOX2):
        # (k1c*OCT4*SOX2 - k2c*OCT4_SOX2) - k3c*OCT4_SOX2
        dx11 = (binding - self.k[19] * x11) / comp

        # dx12/dt (Protein / target gene product):
        # (g1*OCT4_SOX2 + eta7) /
        #   (1 + eta7/f + h1*OCT4_SOX2 + h2*OCT4_SOX2*NANOG)
        # - gamma4*Protein
        numer12 = self.k[28] * x11 + self.k[29]
        denom12 = (1.0 + self.k[29] / self.k[5] + self.k[30] * x11
                   + self.k[31] * x11 * x10)
        dx12 = (numer12 / denom12 - self.k[32] * x12) / comp

        return torch.stack(
            (zeros, zeros, zeros, zeros, zeros, zeros, zeros,
             dx8, dx9, dx10, dx11, dx12),
            dim=-1,
        )

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.zeros(batch_size, self.NUM_NODES, device=device)
        # Constant species (x1-x7): set to their parameter values
        x[..., 0] = self.k[34]  # OCT4_Gene = 0
        x[..., 1] = self.k[35]  # NANOG_Gene = 0
        x[..., 2] = self.k[36]  # SOX2_Gene = 0
        x[..., 3] = self.k[37]  # targetGene = 0.01
        x[..., 4] = self.k[38]  # degradation = 0
        x[..., 5] = self.k[39]  # p53 = 0
        x[..., 6] = self.k[40]  # A = 10

        # Dynamic species (x8-x12): random positive values
        # Based on the network structure, concentrations are typically O(1)
        rand = torch.rand(batch_size, 5, device=device)
        x[..., 7] = rand[..., 0] * 2.0    # OCT4: 0-2
        x[..., 8] = rand[..., 1] * 2.0    # SOX2: 0-2
        x[..., 9] = rand[..., 2] * 2.0    # NANOG: 0-2
        x[..., 10] = rand[..., 3] * 1.0   # OCT4_SOX2: 0-1
        x[..., 11] = rand[..., 4] * 2.0   # Protein: 0-2
        x = torch.clamp(x, min=0.0)
        return x
