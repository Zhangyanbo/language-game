import torch
from ..ode_base import ODEBase


class Leloup1999CircadianClock(ODEBase):
    """Leloup1999 - Drosophila PER/TIM Circadian Clock (BIOMD0000000021).

    10-species model of the Drosophila circadian oscillator with:
    - Mp, Mt: per and tim mRNAs
    - P0, P1, P2: PER protein (un/mono/bi-phosphorylated)
    - T0, T1, T2: TIM protein (un/mono/bi-phosphorylated)
    - CC: PER-TIM complex (cytoplasm)
    - Cn: PER-TIM complex (nucleus, acts as transcriptional repressor)

    Uses Hill functions for transcriptional repression and Michaelis-Menten
    kinetics for phosphorylation/dephosphorylation cascades.

    Reference: Leloup & Goldbeter (1999) J Biol Rhythms 14(6):433-48.
    """

    DESCRIPTION = "Leloup1999 - Drosophila PER/TIM Circadian Clock"
    NODE_NAMES = {
        1: "P0",
        2: "T0",
        3: "P1",
        4: "T1",
        5: "P2",
        6: "T2",
        7: "CC",
        8: "Cn",
        9: "Mp",
        10: "Mt",
    }
    PARAMETER_NAMES = {
        1: "Pt", 2: "Tt", 3: "V_mT", 4: "V_dT",
        5: "Cell", 6: "compartment_0000002",
        7: "K_1P", 8: "V_1P", 9: "K_1T", 10: "V_1T",
        11: "K_2P", 12: "V_2P", 13: "K_2T", 14: "V_2T",
        15: "K_3P", 16: "V_3P", 17: "K_3T", 18: "V_3T",
        19: "K_4P", 20: "V_4P", 21: "K_4T", 22: "V_4T",
        23: "kd_P0", 24: "kd_T0", 25: "kd_P1", 26: "kd_T1",
        27: "kd_P2", 28: "V_dP", 29: "K_dP",
        30: "kd_T2", 31: "K_dT",
        32: "k3", 33: "k4", 34: "k1", 35: "k2",
        36: "k_dC", 37: "k_dN",
        38: "v_sP", 39: "K_IP", 40: "n_P",
        41: "V_sT", 42: "K_IT", 43: "n_T",
        44: "k_sP", 45: "k_sT",
        46: "kd_Mp", 47: "V_mP", 48: "K_mP",
        49: "kd_Mt", 50: "K_mT",
    }
    URL = "https://odebase.org/detail/1338"
    NUM_NODES = 10
    NUM_PARAMS = 50

    # Calibrated from 20-trial simulation (latter half, after transients)
    time_scale = 24.0
    mu = [0.35, 0.35, 0.34, 0.34, 0.33, 0.33, 0.41, 1.16, 1.03, 1.03]
    std = [0.32, 0.32, 0.32, 0.32, 0.33, 0.33, 0.31, 0.51, 0.89, 0.89]
    PHASE_PORTRAIT_DIMS = (7, 8, "Cn (nuclear complex)", "Mp (per mRNA)")
    CATEGORY = "Circadian Clock"

    def __init__(self):
        self.k = [0] * (self.NUM_PARAMS + 1)  # 1-indexed
        self.k[1] = 0            # Pt (total PER, not used in ODEs)
        self.k[2] = 0            # Tt (total TIM, not used in ODEs)
        self.k[3] = 0.7          # V_mT
        self.k[4] = 2.0          # V_dT
        self.k[5] = 1.0          # Cell
        self.k[6] = 1.0          # compartment_0000002
        self.k[7] = 2.0          # K_1P
        self.k[8] = 8.0          # V_1P
        self.k[9] = 2.0          # K_1T
        self.k[10] = 8.0         # V_1T
        self.k[11] = 2.0         # K_2P
        self.k[12] = 1.0         # V_2P
        self.k[13] = 2.0         # K_2T
        self.k[14] = 1.0         # V_2T
        self.k[15] = 2.0         # K_3P
        self.k[16] = 8.0         # V_3P
        self.k[17] = 2.0         # K_3T
        self.k[18] = 8.0         # V_3T
        self.k[19] = 2.0         # K_4P
        self.k[20] = 1.0         # V_4P
        self.k[21] = 2.0         # K_4T
        self.k[22] = 1.0         # V_4T
        self.k[23] = 0.01        # kd_P0
        self.k[24] = 0.01        # kd_T0
        self.k[25] = 0.01        # kd_P1
        self.k[26] = 0.01        # kd_T1
        self.k[27] = 0.01        # kd_P2
        self.k[28] = 2.0         # V_dP
        self.k[29] = 0.2         # K_dP
        self.k[30] = 0.01        # kd_T2
        self.k[31] = 0.2         # K_dT
        self.k[32] = 1.2         # k3 (PER-TIM complex formation)
        self.k[33] = 0.6         # k4 (PER-TIM complex dissociation)
        self.k[34] = 0.6         # k1 (nuclear import)
        self.k[35] = 0.2         # k2 (nuclear export)
        self.k[36] = 0.01        # k_dC (cytoplasmic complex degradation)
        self.k[37] = 0.01        # k_dN (nuclear complex degradation)
        self.k[38] = 1.0         # v_sP (max transcription rate per)
        self.k[39] = 1.0         # K_IP (Hill constant for per repression)
        self.k[40] = 4.0         # n_P (Hill coefficient for per)
        self.k[41] = 1.0         # V_sT (max transcription rate tim)
        self.k[42] = 1.0         # K_IT (Hill constant for tim repression)
        self.k[43] = 4.0         # n_T (Hill coefficient for tim)
        self.k[44] = 0.9         # k_sP (per mRNA translation rate)
        self.k[45] = 0.9         # k_sT (tim mRNA translation rate)
        self.k[46] = 0.01        # kd_Mp (per mRNA basal degradation)
        self.k[47] = 0.7         # V_mP (max per mRNA degradation rate)
        self.k[48] = 0.2         # K_mP (Michaelis constant for per mRNA deg)
        self.k[49] = 0.01        # kd_Mt (tim mRNA basal degradation)
        self.k[50] = 0.2         # K_mT (Michaelis constant for tim mRNA deg)

    def ode(self, x):
        # x.shape = (..., 10)
        # Species: P0, T0, P1, T1, P2, T2, CC, Cn, Mp, Mt
        P0 = x[..., 0]
        T0 = x[..., 1]
        P1 = x[..., 2]
        T1 = x[..., 3]
        P2 = x[..., 4]
        T2 = x[..., 5]
        CC = x[..., 6]
        Cn = x[..., 7]
        Mp = x[..., 8]
        Mt = x[..., 9]

        k = self.k

        # Note: k[5]=Cell=1, k[6]=compartment=1, so they cancel out everywhere.
        # The ODEs simplify considerably.

        # --- Michaelis-Menten terms for phosphorylation/dephosphorylation ---
        # PER: P0 -> P1 -> P2
        V1P_P0 = k[8] * P0 / (k[7] + P0)    # P0 phosphorylation
        V2P_P1 = k[12] * P1 / (k[11] + P1)   # P1 dephosphorylation
        V3P_P1 = k[16] * P1 / (k[15] + P1)   # P1 phosphorylation
        V4P_P2 = k[20] * P2 / (k[19] + P2)   # P2 dephosphorylation

        # TIM: T0 -> T1 -> T2
        V1T_T0 = k[10] * T0 / (k[9] + T0)    # T0 phosphorylation
        V2T_T1 = k[14] * T1 / (k[13] + T1)   # T1 dephosphorylation
        V3T_T1 = k[18] * T1 / (k[17] + T1)   # T1 phosphorylation
        V4T_T2 = k[22] * T2 / (k[21] + T2)   # T2 dephosphorylation

        # --- Complex formation/dissociation ---
        complex_form = k[32] * P2 * T2        # k3 * P2 * T2
        complex_dissoc = k[33] * CC            # k4 * CC

        # --- Nuclear transport ---
        nuc_import = k[34] * CC                # k1 * CC (cytoplasm -> nucleus)
        nuc_export = k[35] * Cn                # k2 * Cn (nucleus -> cytoplasm)
        # Note: k[6]=1 so k[6]*k[35]*Cn = k[35]*Cn

        # --- Hill function transcriptional repression by nuclear complex Cn ---
        # v_sP * K_IP^n / (K_IP^n + Cn^n)
        KIP_n = k[39] ** k[40]
        Cn_nP = Cn ** k[40]
        hill_per = k[38] * KIP_n / (KIP_n + Cn_nP)

        KIT_n = k[42] ** k[43]
        Cn_nT = Cn ** k[43]
        hill_tim = k[41] * KIT_n / (KIT_n + Cn_nT)

        # --- mRNA degradation (basal + Michaelis-Menten) ---
        deg_Mp = k[46] * Mp + k[47] * Mp / (k[48] + Mp)
        deg_Mt = k[49] * Mt + k[3] * Mt / (k[50] + Mt)  # k[3] = V_mT

        # --- P2 specific degradation ---
        deg_P2_specific = k[27] * P2 + k[28] * P2 / (k[29] + P2)

        # --- T2 specific degradation ---
        deg_T2_specific = k[30] * T2 + k[4] * T2 / (k[31] + T2)  # k[4] = V_dT

        # ===== ODEs =====

        # dP0/dt = -V_1P*P0/(K_1P+P0) + V_2P*P1/(K_2P+P1) - kd_P0*P0 + k_sP*Mp
        dP0 = -V1P_P0 + V2P_P1 - k[23] * P0 + k[44] * Mp

        # dT0/dt = -V_1T*T0/(K_1T+T0) + V_2T*T1/(K_2T+T1) - kd_T0*T0 + k_sT*Mt
        dT0 = -V1T_T0 + V2T_T1 - k[24] * T0 + k[45] * Mt

        # dP1/dt = V_1P*P0/(K_1P+P0) - V_2P*P1/(K_2P+P1) - V_3P*P1/(K_3P+P1) + V_4P*P2/(K_4P+P2) - kd_P1*P1
        dP1 = V1P_P0 - V2P_P1 - V3P_P1 + V4P_P2 - k[25] * P1

        # dT1/dt = V_1T*T0/(K_1T+T0) - V_2T*T1/(K_2T+T1) - V_3T*T1/(K_3T+T1) + V_4T*T2/(K_4T+T2) - kd_T1*T1
        dT1 = V1T_T0 - V2T_T1 - V3T_T1 + V4T_T2 - k[26] * T1

        # dP2/dt = V_3P*P1/(K_3P+P1) - V_4P*P2/(K_4P+P2) - (kd_P2*P2 + V_dP*P2/(K_dP+P2)) - k3*P2*T2 + k4*CC
        dP2 = V3P_P1 - V4P_P2 - deg_P2_specific - complex_form + complex_dissoc

        # dT2/dt = V_3T*T1/(K_3T+T1) - V_4T*T2/(K_4T+T2) - (kd_T2*T2 + V_dT*T2/(K_dT+T2)) - k3*P2*T2 + k4*CC
        dT2 = V3T_T1 - V4T_T2 - deg_T2_specific - complex_form + complex_dissoc

        # dCC/dt = k3*P2*T2 - k4*CC - k1*CC + k2*Cn - k_dC*CC
        #   Note: k2*Cn has factor k[6]=1
        dCC = complex_form - complex_dissoc - nuc_import + nuc_export - k[36] * CC

        # dCn/dt = (k1*CC - k2*Cn) / k[6] - k_dN*Cn
        #   k[6]=1, so: k1*CC - k2*Cn - k_dN*Cn
        # From odebase: dCn/dt = (k5*k34*CC - k6*k35*Cn - k6*k37*Cn) / k6
        #   = k34*CC - k35*Cn - k37*Cn  (since k5=k6=1)
        dCn = k[34] * CC - k[35] * Cn - k[37] * Cn

        # dMp/dt = v_sP * K_IP^n / (K_IP^n + Cn^n) - (kd_Mp*Mp + V_mP*Mp/(K_mP+Mp))
        dMp = hill_per - deg_Mp

        # dMt/dt = V_sT * K_IT^n / (K_IT^n + Cn^n) - (kd_Mt*Mt + V_mT*Mt/(K_mT+Mt))
        dMt = hill_tim - deg_Mt

        return torch.stack(
            (dP0, dT0, dP1, dT1, dP2, dT2, dCC, dCn, dMp, dMt), dim=-1
        )

    def project_to_manifold(self, x):
        # No constraints for this model
        return x

    def _random_initial_state(self, batch_size=1, device="cpu"):
        x = torch.rand(batch_size, self.NUM_NODES, device=device)
        # Scale to observed oscillation ranges (from calibration)
        x[..., 0] = x[..., 0] * 0.95 + 0.01  # P0:  ~0.01-0.95
        x[..., 1] = x[..., 1] * 0.95 + 0.01  # T0:  ~0.01-0.95
        x[..., 2] = x[..., 2] * 0.92 + 0.01  # P1:  ~0.01-0.93
        x[..., 3] = x[..., 3] * 0.92 + 0.01  # T1:  ~0.01-0.93
        x[..., 4] = x[..., 4] * 0.93 + 0.02  # P2:  ~0.02-0.95
        x[..., 5] = x[..., 5] * 0.93 + 0.02  # T2:  ~0.02-0.95
        x[..., 6] = x[..., 6] * 0.96 + 0.13  # CC:  ~0.13-1.08
        x[..., 7] = x[..., 7] * 1.51 + 0.53  # Cn:  ~0.53-2.04
        x[..., 8] = x[..., 8] * 2.53 + 0.03  # Mp:  ~0.03-2.56
        x[..., 9] = x[..., 9] * 2.53 + 0.03  # Mt:  ~0.03-2.56
        return x
