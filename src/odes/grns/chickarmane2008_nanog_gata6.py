import torch
from ..ode_base import ODEBase


class Chickarmane2008NanogGata6(ODEBase):
    """Chickarmane2008 - Stem cell lineage - NANOG GATA-6 switch

    This model describes the gene regulatory network controlling the NANOG/GATA-6
    bistable switch in stem cell differentiation. The network includes OCT4, SOX2,
    NANOG, GATA6, CDX2, and GCNF as dynamic species.

    The original SBML model (BIOMD0000000210) has 18 species, but 12 are boundary
    species (constant, dx/dt=0): gene promoters, degradation sinks, p53, A (=10),
    OCT4_SOX2 (=0.1), and Protein. Only 6 species are dynamically modeled here.

    Source: https://odebase.org/detail/1456
    Paper: Chickarmane & Peterson (2008) PNAS
    """

    DESCRIPTION = "Chickarmane2008 - Stem cell lineage - NANOG GATA-6 switch"
    NODE_NAMES = {
        1: "OCT4",
        2: "SOX2",
        3: "NANOG",
        4: "GATA6",
        5: "CDX2",
        6: "GCNF",
    }
    PARAMETER_NAMES = {
        1: "a0",
        2: "a1",
        3: "a2",
        4: "a3",
        5: "b0",
        6: "b1",
        7: "b2",
        8: "b3",
        9: "b4",
        10: "b5",
        11: "gamma1",
        12: "c0",
        13: "c1",
        14: "c2",
        15: "d0",
        16: "d1",
        17: "d2",
        18: "d3",
        19: "gamma2",
        20: "e0",
        21: "e1",
        22: "e2",
        23: "f0",
        24: "f1",
        25: "f2",
        26: "f3",
        27: "gamma3",
        28: "g0",
        29: "g1",
        30: "h0",
        31: "h1",
        32: "gamma4",
        33: "i0",
        34: "i1",
        35: "i2",
        36: "j0",
        37: "j1",
        38: "gamma5",
        39: "p0",
        40: "p1",
        41: "p2",
        42: "q0",
        43: "q1",
        44: "q2",
        45: "gammag",
        46: "gamman",
        47: "cell",
        48: "OCT4_Gene",
        49: "NANOG_Gene",
        50: "SOX2_Gene",
        51: "GATA6_Gene",
        52: "CDX2_Gene",
        53: "GCNF_Gene",
        54: "targetGene",
        55: "degradation",
        56: "p53",
        57: "A",
    }
    URL = "https://odebase.org/detail/1456"
    NUM_NODES = 6
    NUM_PARAMS = 57

    # Calibrated from 20-trial RK4 simulation (dt=0.01, T=200)
    # Using full-trajectory stats to capture transient dynamics
    time_scale = 560.0
    mu = [0.05, 0.04, 0.62, 2.44, 8.30, 4.98]
    std = [0.08, 0.09, 0.27, 1.47, 1.15, 0.75]
    PHASE_PORTRAIT_DIMS = (2, 4, "NANOG", "CDX2")
    CATEGORY = "Cell Fate Decision"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)  # 1-indexed
        # From odebase parameter values
        self.k[1] = 1 / 1000       # a0 = 0.001
        self.k[2] = 1 / 50         # a1 = 0.02
        self.k[3] = 1 / 80         # a2 = 0.0125
        self.k[4] = 1 / 40         # a3 = 0.025
        self.k[5] = 1              # b0 = 1
        self.k[6] = 1 / 50         # b1 = 0.02
        self.k[7] = 1 / 80         # b2 = 0.0125
        self.k[8] = 3 / 100        # b3 = 0.03
        self.k[9] = 10             # b4 = 10
        self.k[10] = 10            # b5 = 10
        self.k[11] = 1 / 10        # gamma1 = 0.1
        self.k[12] = 1 / 1000      # c0 = 0.001
        self.k[13] = 1 / 20        # c1 = 0.05
        self.k[14] = 1 / 80        # c2 = 0.0125
        self.k[15] = 1 / 1000      # d0 = 0.001
        self.k[16] = 1 / 20        # d1 = 0.05
        self.k[17] = 1 / 80        # d2 = 0.0125
        self.k[18] = 1 / 20        # d3 = 0.05
        self.k[19] = 1 / 10        # gamma2 = 0.1
        self.k[20] = 1 / 1000      # e0 = 0.001
        self.k[21] = 1 / 10        # e1 = 0.1
        self.k[22] = 1 / 10        # e2 = 0.1
        self.k[23] = 1 / 1000      # f0 = 0.001
        self.k[24] = 1 / 10        # f1 = 0.1
        self.k[25] = 1 / 10        # f2 = 0.1
        self.k[26] = 10            # f3 = 10
        self.k[27] = 1 / 10        # gamma3 = 0.1
        self.k[28] = 1 / 1000      # g0 = 0.001
        self.k[29] = 2             # g1 = 2
        self.k[30] = 2             # h0 = 2
        self.k[31] = 5             # h1 = 5
        self.k[32] = 1 / 10        # gamma4 = 0.1
        self.k[33] = 1 / 1000      # i0 = 0.001
        self.k[34] = 1 / 10        # i1 = 0.1
        self.k[35] = 1 / 10        # i2 = 0.1
        self.k[36] = 1 / 10        # j0 = 0.1
        self.k[37] = 1 / 10        # j1 = 0.1
        self.k[38] = 1 / 10        # gamma5 = 0.1
        self.k[39] = 1 / 10        # p0 = 0.1
        self.k[40] = 1             # p1 = 1
        self.k[41] = 1 / 4000      # p2 = 0.00025
        self.k[42] = 1             # q0 = 1
        self.k[43] = 1 / 4000      # q1 = 0.00025
        self.k[44] = 15            # q2 = 15
        self.k[45] = 1 / 100       # gammag = 0.01
        self.k[46] = 1 / 100       # gamman = 0.01
        self.k[47] = 1             # cell (compartment volume)
        # Boundary species initial values (constants)
        self.k[48] = 0             # OCT4_Gene
        self.k[49] = 0             # NANOG_Gene
        self.k[50] = 0             # SOX2_Gene
        self.k[51] = 0             # GATA6_Gene
        self.k[52] = 0             # CDX2_Gene
        self.k[53] = 0             # GCNF_Gene
        self.k[54] = 1 / 100       # targetGene = 0.01
        self.k[55] = 0             # degradation
        self.k[56] = 0             # p53
        self.k[57] = 10            # A = 10

        # OCT4_SOX2 is a boundary species (constant) with initial value 0.1
        # It is used in NANOG and GATA6 synthesis equations
        self.oct4_sox2 = 0.1

    def project_to_manifold(self, x):
        # No constraints for this model
        return x

    def ode(self, x):
        # x.shape = (..., 6)
        # Dynamic species:
        #   x[...,0] = OCT4  (x11 in odebase)
        #   x[...,1] = SOX2  (x12 in odebase)
        #   x[...,2] = NANOG (x13 in odebase)
        #   x[...,3] = GATA6 (x14 in odebase)
        #   x[...,4] = CDX2  (x15 in odebase)
        #   x[...,5] = GCNF  (x16 in odebase)
        oct4 = x[..., 0]
        sox2 = x[..., 1]
        nanog = x[..., 2]
        gata6 = x[..., 3]
        cdx2 = x[..., 4]
        gcnf = x[..., 5]

        A = self.k[57]             # constant boundary species A = 10
        OS = self.oct4_sox2         # constant boundary species OCT4_SOX2 = 0.1
        cell = self.k[47]          # compartment volume = 1

        # R1: OCT4 synthesis (uses OCT4*SOX2 product directly, and A as activator)
        # Rate = (a0 + a1*A + a2*OCT4*SOX2 + a3*OCT4*SOX2*NANOG)
        #      / (1 + b0*A + b1*OCT4 + b2*OCT4*SOX2 + b3*OCT4*SOX2*NANOG
        #         + b4*CDX2*OCT4 + b5*GCNF)
        oct4_sox2_prod = oct4 * sox2
        r1_num = (self.k[1] + self.k[2] * A + self.k[3] * oct4_sox2_prod
                  + self.k[4] * oct4_sox2_prod * nanog)
        r1_den = (1 + self.k[5] * A + self.k[6] * oct4
                  + self.k[7] * oct4_sox2_prod + self.k[8] * oct4_sox2_prod * nanog
                  + self.k[9] * cdx2 * oct4 + self.k[10] * gcnf)
        r1 = r1_num / r1_den

        # R2: OCT4 degradation = gamma1 * OCT4
        # dx_OCT4/dt = (R1 - R2) / cell
        d_oct4 = (r1 - self.k[11] * oct4) / cell

        # R3: SOX2 synthesis
        # Rate = (c0 + c1*OCT4*SOX2 + c2*OCT4*SOX2*NANOG)
        #      / (1 + d0*OCT4 + d1*OCT4*SOX2 + d2*OCT4*SOX2*NANOG)
        r3_num = (self.k[12] + self.k[13] * oct4_sox2_prod
                  + self.k[14] * oct4_sox2_prod * nanog)
        r3_den = (1 + self.k[15] * oct4 + self.k[16] * oct4_sox2_prod
                  + self.k[17] * oct4_sox2_prod * nanog)
        r3 = r3_num / r3_den

        # R4: SOX2 degradation = gamma2 * SOX2
        # dx_SOX2/dt = (R3 - R4) / cell
        d_sox2 = (r3 - self.k[19] * sox2) / cell

        # R5: NANOG synthesis (uses OCT4_SOX2 boundary species = 0.1)
        # Rate = (a1*OCT4_SOX2 + a2*OCT4_SOX2*NANOG)
        #      / (1 + b1*OCT4_SOX2 + b2*OCT4_SOX2*NANOG + b3*OCT4_SOX2*GATA6)
        r5_num = self.k[2] * OS + self.k[3] * OS * nanog
        r5_den = 1 + self.k[6] * OS + self.k[7] * OS * nanog + self.k[8] * OS * gata6
        r5 = r5_num / r5_den

        # R6: NANOG degradation = gamman * NANOG
        # dx_NANOG/dt = (R5 - R6) / cell
        d_nanog = (r5 - self.k[46] * nanog) / cell

        # R11: GATA6 synthesis (uses OCT4_SOX2 boundary species = 0.1)
        # Rate = (c1*OCT4_SOX2 + c2*GATA6)
        #      / (1 + d1*OCT4_SOX2 + d2*GATA6 + d3*NANOG)
        r11_num = self.k[13] * OS + self.k[14] * gata6
        r11_den = 1 + self.k[16] * OS + self.k[17] * gata6 + self.k[18] * nanog
        r11 = r11_num / r11_den

        # R12: GATA6 degradation = gammag * GATA6
        # dx_GATA6/dt = (R11 - R12) / cell
        d_gata6 = (r11 - self.k[45] * gata6) / cell

        # R7: CDX2 synthesis
        # Rate = (g0 + g1*CDX2) / (1 + h0*CDX2 + h1*CDX2*OCT4)
        r7_num = self.k[28] + self.k[29] * cdx2
        r7_den = 1 + self.k[30] * cdx2 + self.k[31] * cdx2 * oct4
        r7 = r7_num / r7_den

        # R8: CDX2 degradation = gamma4 * CDX2
        # dx_CDX2/dt = (R7 - R8) / cell
        d_cdx2 = (r7 - self.k[32] * cdx2) / cell

        # R9: GCNF synthesis
        # Rate = (i0 + i1*CDX2 + i2*GATA6) / (1 + j0*CDX2 + j1*GATA6)
        r9_num = self.k[33] + self.k[34] * cdx2 + self.k[35] * gata6
        r9_den = 1 + self.k[36] * cdx2 + self.k[37] * gata6
        r9 = r9_num / r9_den

        # R10: GCNF degradation = gamma5 * GCNF
        # dx_GCNF/dt = (R9 - R10) / cell
        d_gcnf = (r9 - self.k[38] * gcnf) / cell

        return torch.stack((d_oct4, d_sox2, d_nanog, d_gata6, d_cdx2, d_gcnf), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)
        # Ranges based on observed simulation dynamics
        x[..., 0] = x[..., 0] * 0.9 + 0.003   # OCT4:  ~0.003-0.9
        x[..., 1] = x[..., 1] * 0.9 + 0.01    # SOX2:  ~0.01-0.94
        x[..., 2] = x[..., 2] * 1.5 + 0.26    # NANOG: ~0.26-1.76
        x[..., 3] = x[..., 3] * 3.9 + 0.03    # GATA6: ~0.03-3.9
        x[..., 4] = x[..., 4] * 8.8 + 0.004   # CDX2:  ~0.004-8.85
        x[..., 5] = x[..., 5] * 5.5 + 0.06    # GCNF:  ~0.06-5.6
        return x
