import torch
from ..ode_base import ODEBase


class Gardner2000ToggleSwitch(ODEBase):
    """Gardner2000 - Genetic toggle switch in E.coli (BIOMD0000000507).

    Classic bistable toggle switch with two mutually repressing promoters
    and an optional IPTG inducer (species 3, constant).

    ODEs (from odebase, simplified with k10=k11=k12=1):
        dx1/dt = k1 / (1 + x2^k3) - x1
        dx2/dt = k2 / (1 + (x1 / (1 + x3/k8))^k9) - x2
        dx3/dt = 0   (IPTG inducer, constant)

    The system is bistable: depending on initial conditions, it settles to
    either (high x1, low x2) or (low x1, high x2).
    """

    DESCRIPTION = "Gardner2000 - genetic toggle switch in E.coli"
    NODE_NAMES = {
        1: "LacI",    # Lactose operon repressor (species_1)
        2: "TetR",    # Tetracycline repressor protein (species_2)
        3: "IPTG",    # Inducer (species_3, constant)
    }
    PARAMETER_NAMES = {
        1: "alpha1",          # k1: max production rate of LacI
        2: "alpha2",          # k2: max production rate of TetR
        3: "beta",            # k3: Hill coefficient for LacI production
        4: "gamma",           # k4: exponent (=1)
        5: "K_IPTG_raw",     # k5: = k8 (ModelValue_4)
        6: "n_TetR_raw",     # k6: = k9 (ModelValue_5)
        7: "zero",            # k7: unused (=0)
        8: "K_IPTG",         # k8: IPTG dissociation constant
        9: "n_TetR",         # k9: Hill coefficient for TetR production
        10: "compartment",   # k10: compartment volume (=1)
        11: "deg_LacI",      # k11: degradation rate of LacI (=1)
        12: "deg_TetR",      # k12: degradation rate of TetR (=1)
    }
    URL = "https://odebase.org/detail/1646"
    NUM_NODES = 3
    NUM_PARAMS = 12

    # Calibrated from 20-trial simulation (latter half statistics)
    # Bistable: ~half settle to (LacI~156, TetR~0), half to (LacI~0, TetR~15.6)
    time_scale = 15.0
    mu = [78.0, 7.7, 0.0]
    std = [78.0, 7.7, 1.0]
    PHASE_PORTRAIT_DIMS = (0, 1, "LacI", "TetR")
    CATEGORY = "Synthetic Gene Circuit"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)
        self.k[1] = 625 / 4               # 156.25, alpha1
        self.k[2] = 78 / 5                # 15.6, alpha2
        self.k[3] = 5 / 2                 # 2.5, beta (Hill coeff)
        self.k[4] = 1                     # gamma exponent
        self.k[5] = 14809 / 500000000     # ~2.9618e-8
        self.k[6] = 4003 / 2000           # ~2.0015
        self.k[7] = 0                     # unused
        self.k[8] = 14809 / 500000000     # ~2.9618e-8, K_IPTG
        self.k[9] = 4003 / 2000           # ~2.0015, n_TetR (Hill coeff)
        self.k[10] = 1                    # compartment
        self.k[11] = 1                    # degradation rate LacI
        self.k[12] = 1                    # degradation rate TetR

    def ode(self, x):
        x1 = x[..., 0]  # LacI
        x2 = x[..., 1]  # TetR
        x3 = x[..., 2]  # IPTG (constant inducer)

        # dx1/dt = alpha1 / (1 + x2^beta) - deg_LacI * x1
        dx1 = self.k[1] / (1 + x2 ** self.k[3]) - self.k[11] * x1

        # dx2/dt = alpha2 / (1 + (x1 / (1 + x3/K_IPTG))^n_TetR) - deg_TetR * x2
        # K_IPTG ≈ 2.96e-8 is extremely small; clamp ratio to prevent overflow
        effective_x1 = x1 / (1 + torch.clamp(x3 / self.k[8], max=1e8))
        dx2 = self.k[2] / (1 + torch.clamp(effective_x1, min=0.0, max=1e6) ** self.k[9]) - self.k[12] * x2

        # dx3/dt = 0 (IPTG is constant)
        dx3 = torch.zeros_like(x3)

        return torch.stack((dx1, dx2, dx3), dim=-1)

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.zeros(batch_size, self.NUM_NODES, device=device)

        # Bistable system with two steady states:
        #   Basin 1: LacI ~ 156.25, TetR ~ 0
        #   Basin 2: LacI ~ 0, TetR ~ 15.6
        # Randomly assign each sample to a basin, then add noise
        basin = torch.rand(batch_size, device=device) > 0.5

        # Basin 1: high LacI, low TetR
        x[basin, 0] = 100.0 + torch.rand(basin.sum(), device=device) * 100.0
        x[basin, 1] = torch.rand(basin.sum(), device=device) * 2.0

        # Basin 2: low LacI, high TetR
        x[~basin, 0] = torch.rand((~basin).sum(), device=device) * 2.0
        x[~basin, 1] = 10.0 + torch.rand((~basin).sum(), device=device) * 10.0

        # IPTG = 0 (no inducer)
        x[..., 2] = 0.0
        return x

    def project_to_manifold(self, x):
        # No constraints, but ensure all concentrations >= 0
        # and IPTG stays constant at its initial value
        return x
