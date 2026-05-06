"""Semantic Primitives Analysis: GRN biological properties × RL task primitives.

Ranking-based statistical analysis of relationships between GRN dynamical/biological
properties and RL task semantic primitives, using GRN-level exact permutation tests
with Benjamini-Hochberg FDR correction.

Each GRN's mean rank across environments within a primitive yields one independent
observation per GRN, avoiding the pseudo-replication of pooling correlated (GRN, env)
rows.  Property labels are permuted at the GRN level; p-values come from the exact
permutation distribution (all C(n, k) assignments are enumerated).
"""

import argparse
import json
import os
from collections import OrderedDict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from src.plot_rewards import Records, average_rewards

# ---------------------------------------------------------------------------
# Semantic Primitives: primitive name -> list of env names
# ---------------------------------------------------------------------------
SEMANTIC_PRIMITIVES = OrderedDict([
    ("Stabilization", [
        "CartPole-v1", "Pendulum-v1", "HumanoidStandup-v4",
    ]),
    ("Energy Pumping", [
        "Acrobot-v1", "MountainCarContinuous-v0", "Pendulum-v1",
    ]),
    ("Target Reaching", [
        "Reacher-v4", "Acrobot-v1", "PointMaze", "MountainCarContinuous-v0",
        "Pusher-v4", "HumanoidStandup-v4",
    ]),
    ("Periodic Motion", [
        "Swimmer-v4", "Hopper-v4", "HalfCheetah-v4", "finger-spin",
    ]),
    ("Survival", [
        "CartPole-v1", "Hopper-v4",
    ]),
    ("High-DOF Coord.", [
        "Pusher-v4", "HumanoidStandup-v4", "HalfCheetah-v4",
    ]),
    ("Manipulation", [
        "Pusher-v4", "finger-spin",
    ]),
    ("Contact-rich", [
        "Pusher-v4", "finger-spin", "Hopper-v4",
    ]),
])

# ---------------------------------------------------------------------------
# GRN Biological Properties: property name -> list of reservoir base names
# (base names without _gradient suffix)
# ---------------------------------------------------------------------------
GRN_PROPERTIES = OrderedDict([
    # --- A. Dynamical Behavior ---
    ("Oscillatory", [
        "tyson1999circlelock", "weimann2004circadianoscillator",
        "almeida2019circadianclock", "leloup1999circadianclock",
        "tyson1991cellcycle2var", "gardner1998cellcyclegoldbeter",
        "gerard2010cellcycle", "zatorsky2006p53model4",
        "kholodenko2000mapkcascade",
    ]),
    ("Bistable", [
        "gardner2000toggleswitch", "chickarmane2006stemcellswitch",
        "chickarmane2008nanoggata6",
    ]),
    ("Neg. Feedback", [
        "tyson1999circlelock", "weimann2004circadianoscillator",
        "almeida2019circadianclock", "leloup1999circadianclock",
        "zatorsky2006p53model4", "kholodenko2000mapkcascade",
    ]),
    ("Ultrasensitivity", [
        "markevich2004mapkdoublephosphorylation", "gardner1998cellcyclegoldbeter",
        "gerard2010cellcycle", "gardner2000toggleswitch",
    ]),
    ("Non-oscillatory", [
        "chickarmane2006stemcellswitch", "chickarmane2008nanoggata6",
        "gardner2000toggleswitch", "liebal2012transcriptioninhibition",
        "markevich2004mapkdoublephosphorylation",
    ]),
    # --- B. Biological Process ---
    ("Circadian", [
        "tyson1999circlelock", "weimann2004circadianoscillator",
        "almeida2019circadianclock", "leloup1999circadianclock",
    ]),
    ("Cell Cycle", [
        "tyson1991cellcycle2var", "gardner1998cellcyclegoldbeter",
        "gerard2010cellcycle",
    ]),
    ("Cell Fate", [
        "chickarmane2006stemcellswitch", "chickarmane2008nanoggata6",
    ]),
    ("Signal Transd.", [
        "markevich2004mapkdoublephosphorylation", "kholodenko2000mapkcascade",
        "almeida2019circadianclock",
    ]),
    ("Transcriptional", [
        "almeida2019circadianclock", "liebal2012transcriptioninhibition",
        "chickarmane2006stemcellswitch", "chickarmane2008nanoggata6",
    ]),
    # --- C. Structural/Mechanistic ---
    ("Phosphorylation", [
        "markevich2004mapkdoublephosphorylation", "kholodenko2000mapkcascade",
        "leloup1999circadianclock", "gerard2010cellcycle",
        "gardner1998cellcyclegoldbeter",
    ]),
    ("Complex Form.", [
        "almeida2019circadianclock", "leloup1999circadianclock",
        "chickarmane2006stemcellswitch",
    ]),
    ("Conservation", [
        "tyson1991cellcycle2var", "markevich2004mapkdoublephosphorylation",
        "kholodenko2000mapkcascade",
    ]),
])

# Category separators for heatmap row grouping (index after which to draw line)
PROPERTY_CATEGORY_BREAKS = [4, 9]  # after "Non-oscillatory", after "Transcriptional"

# All 14 biological GRN reservoir base names (exclude identity, mlp)
ALL_BIO_RESERVOIRS = sorted(set(
    name for grns in GRN_PROPERTIES.values() for name in grns
))

# Controls excluded from statistical comparison
CONTROLS = ["identity", "mlp"]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def extract_final_rewards(
    records: Records,
    reservoirs: list[str],
    envs: list[str],
    n_bins: int = 10,
    tail_frac: float = 0.2,
) -> dict[tuple[str, str], float]:
    """Extract mean reward over the last tail_frac of training for each (reservoir, env).

    Returns dict mapping (reservoir_base_name, env) -> float.
    Averages across seeds.
    """
    n_tail = max(1, int(n_bins * tail_frac))
    rewards = {}
    for res in reservoirs:
        for env in envs:
            try:
                record = records[res, env]
            except KeyError:
                continue
            bins = average_rewards(record, n=n_bins)
            if len(bins) == 0:
                continue
            # bins is list of arrays (one per seed), each of length n_bins
            seed_finals = []
            for seed_bin in bins:
                seed_finals.append(float(np.mean(seed_bin[-n_tail:])))
            base = res.replace("_gradient", "")
            rewards[(base, env)] = float(np.mean(seed_finals))
    return rewards


def rank_within_env(
    rewards: dict[tuple[str, str], float],
    envs: list[str],
    reservoirs_base: list[str],
) -> dict[tuple[str, str], float]:
    """Rank reservoirs within each environment (1=best, higher=worse).

    Only ranks reservoirs present in reservoirs_base.
    """
    ranks = {}
    for env in envs:
        scores = []
        res_names = []
        for res in reservoirs_base:
            if (res, env) in rewards:
                scores.append(rewards[(res, env)])
                res_names.append(res)
        if not scores:
            continue
        # Higher reward = better = lower rank
        arr = np.array(scores)
        # sp_stats.rankdata gives rank 1=smallest; we want 1=largest
        raw_ranks = sp_stats.rankdata(-arr, method="average")
        for res, rank in zip(res_names, raw_ranks):
            ranks[(res, env)] = float(rank)
    return ranks


def compute_property_primitive_stats(
    ranks: dict[tuple[str, str], float],
    properties: dict[str, list[str]],
    primitives: dict[str, list[str]],
    bio_reservoirs: list[str],
) -> list[dict]:
    """GRN-level exact permutation test for each (property, primitive) pair.

    For each GRN, computes its mean rank across all environments in the
    primitive (one independent observation per GRN).  Then tests whether GRNs
    possessing the property have different mean ranks from those without it,
    using an exact two-sided permutation test over all C(n, k) label
    assignments.  Effect size is rank-biserial r on the GRN-level means.
    """
    results = []
    for prop_name, prop_grns in properties.items():
        prop_set = set(prop_grns)
        for prim_name, prim_envs in primitives.items():
            # -- Step 1: one observation per GRN (mean rank across envs) --
            grn_mean_ranks: dict[str, float] = {}
            for res in bio_reservoirs:
                env_ranks = [ranks[(res, env)]
                             for env in prim_envs if (res, env) in ranks]
                if env_ranks:
                    grn_mean_ranks[res] = float(np.mean(env_ranks))

            grns_with = [g for g in grn_mean_ranks if g in prop_set]
            grns_without = [g for g in grn_mean_ranks if g not in prop_set]
            n_with = len(grns_with)
            n_without = len(grns_without)

            row = {
                "property": prop_name,
                "primitive": prim_name,
                "n_with": n_with,
                "n_without": n_without,
                "mean_rank_with": (float(np.mean([grn_mean_ranks[g] for g in grns_with]))
                                   if grns_with else np.nan),
                "mean_rank_without": (float(np.mean([grn_mean_ranks[g] for g in grns_without]))
                                      if grns_without else np.nan),
            }

            if n_with >= 2 and n_without >= 2:
                vals_with = np.array([grn_mean_ranks[g] for g in grns_with])
                vals_without = np.array([grn_mean_ranks[g] for g in grns_without])

                # Observed test statistic: difference in group means
                observed_diff = float(np.mean(vals_with) - np.mean(vals_without))

                # -- Step 2: exact permutation test --
                all_grns = list(grn_mean_ranks.keys())
                all_vals = np.array([grn_mean_ranks[g] for g in all_grns])
                n_total = len(all_grns)

                count_extreme = 0
                count_total = 0
                for combo in combinations(range(n_total), n_with):
                    perm_mean_with = all_vals[list(combo)].mean()
                    perm_mean_without = np.delete(all_vals, list(combo)).mean()
                    perm_diff = perm_mean_with - perm_mean_without
                    if abs(perm_diff) >= abs(observed_diff) - 1e-12:
                        count_extreme += 1
                    count_total += 1

                p_value = count_extreme / count_total

                # -- Step 3: rank-biserial r on GRN-level means --
                U, _ = sp_stats.mannwhitneyu(
                    vals_with, vals_without, alternative="two-sided"
                )
                r_effect = 1.0 - 2.0 * U / (n_with * n_without)

                row["U"] = float(U)
                row["p_value"] = float(p_value)
                row["effect_r"] = float(r_effect)
                row["n_permutations"] = count_total
            else:
                row["U"] = np.nan
                row["p_value"] = np.nan
                row["effect_r"] = np.nan
                row["n_permutations"] = 0

            results.append(row)
    return results


def compute_grn_mean_ranks(
    ranks: dict[tuple[str, str], float],
    primitives: dict[str, list[str]],
    bio_reservoirs: list[str],
) -> dict[str, dict[str, float]]:
    """For each primitive, compute each GRN's mean rank across its environments.

    Returns {primitive_name: {grn_name: mean_rank}}.
    """
    result = {}
    for prim_name, prim_envs in primitives.items():
        grn_means = {}
        for res in bio_reservoirs:
            env_ranks = [ranks[(res, env)]
                         for env in prim_envs if (res, env) in ranks]
            if env_ranks:
                grn_means[res] = float(np.mean(env_ranks))
        result[prim_name] = grn_means
    return result


def omnibus_permutation_test(
    grn_mean_ranks: dict[str, dict[str, float]],
    properties: dict[str, list[str]],
    bio_reservoirs: list[str],
    n_perm: int = 100_000,
    rng_seed: int = 42,
) -> dict:
    """Global permutation test: are biological property labels informative overall?

    Test statistic: sum of squared rank-biserial r across all (property, primitive)
    cells.  Null distribution: permute the GRN-to-property-vector mapping at the
    GRN level (each GRN's full property vector stays intact, but which GRN gets
    which vector is shuffled).

    Returns dict with observed statistic, p-value, n_permutations.
    """
    rng = np.random.RandomState(rng_seed)

    prim_names = list(grn_mean_ranks.keys())
    prop_names = list(properties.keys())

    # Build property matrix: (n_grn, n_props) binary
    grn_list = [g for g in bio_reservoirs if g in grn_mean_ranks[prim_names[0]]]
    n_grn = len(grn_list)
    n_props = len(prop_names)
    prop_matrix = np.zeros((n_grn, n_props), dtype=bool)
    for j, (prop_name, prop_grns) in enumerate(properties.items()):
        prop_set = set(prop_grns)
        for i, g in enumerate(grn_list):
            prop_matrix[i, j] = g in prop_set

    # Build mean-rank matrix: (n_grn, n_prims)
    n_prims = len(prim_names)
    rank_matrix = np.full((n_grn, n_prims), np.nan)
    for k, prim_name in enumerate(prim_names):
        for i, g in enumerate(grn_list):
            if g in grn_mean_ranks[prim_name]:
                rank_matrix[i, k] = grn_mean_ranks[prim_name][g]

    def compute_stat(prop_mat):
        """Sum of squared mean-rank differences across all (property, primitive) cells."""
        total = 0.0
        for j in range(n_props):
            mask = prop_mat[:, j]
            n_with = int(mask.sum())
            n_without = n_grn - n_with
            if n_with < 2 or n_without < 2:
                continue
            for k in range(n_prims):
                vals = rank_matrix[:, k]
                vals_w = vals[mask]
                vals_wo = vals[~mask]
                diff = np.nanmean(vals_w) - np.nanmean(vals_wo)
                total += diff * diff
        return total

    observed = compute_stat(prop_matrix)

    # Permutation: shuffle rows of prop_matrix (= reassign property vectors)
    count_ge = 0
    for _ in range(n_perm):
        perm_idx = rng.permutation(n_grn)
        perm_stat = compute_stat(prop_matrix[perm_idx])
        if perm_stat >= observed - 1e-12:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perm + 1)  # +1 for observed itself

    return {
        "observed_stat": float(observed),
        "p_value": float(p_value),
        "n_permutations": n_perm,
        "n_grns": n_grn,
        "n_properties": n_props,
        "n_primitives": n_prims,
    }


# ---------------------------------------------------------------------------
# A priori hypotheses: (property, primitive, predicted_direction)
# direction: "positive" means property helps (lower rank = better),
#            "negative" means property hurts.
# These are biologically motivated predictions, stated before seeing the data:
#   H1: Oscillatory GRNs should excel at periodic motion (oscillation → rhythm)
#   H2: Circadian GRNs should excel at periodic motion (evolved for rhythm)
#   H3: Negative feedback GRNs should excel at periodic motion (oscillation mechanism)
#   H4: Ultrasensitive GRNs should struggle at periodic motion (switch ≠ smooth)
#   H5: Complex-forming GRNs should excel at manipulation (multi-component coupling)
# ---------------------------------------------------------------------------
A_PRIORI_HYPOTHESES = [
    ("Oscillatory",      "Periodic Motion", "positive"),
    ("Circadian",        "Periodic Motion", "positive"),
    ("Neg. Feedback",    "Periodic Motion", "positive"),
    ("Ultrasensitivity", "Periodic Motion", "negative"),
    ("Complex Form.",    "Manipulation",    "positive"),
]


def test_a_priori_hypotheses(
    ranks: dict[tuple[str, str], float],
    properties: dict[str, list[str]],
    primitives: dict[str, list[str]],
    bio_reservoirs: list[str],
    hypotheses: list[tuple[str, str, str]] = A_PRIORI_HYPOTHESES,
) -> list[dict]:
    """One-sided exact permutation tests for a priori hypotheses.

    Since the direction is predicted a priori, we use one-sided tests
    (more powerful than two-sided).  FDR correction is applied only over
    these hypotheses (not all 104 cells).
    """
    results = []
    for prop_name, prim_name, direction in hypotheses:
        prop_set = set(properties[prop_name])
        prim_envs = primitives[prim_name]

        # One observation per GRN
        grn_mean_ranks: dict[str, float] = {}
        for res in bio_reservoirs:
            env_ranks = [ranks[(res, env)]
                         for env in prim_envs if (res, env) in ranks]
            if env_ranks:
                grn_mean_ranks[res] = float(np.mean(env_ranks))

        grns_with = [g for g in grn_mean_ranks if g in prop_set]
        grns_without = [g for g in grn_mean_ranks if g not in prop_set]
        n_with, n_without = len(grns_with), len(grns_without)

        vals_with = np.array([grn_mean_ranks[g] for g in grns_with])
        vals_without = np.array([grn_mean_ranks[g] for g in grns_without])

        # Observed difference: mean_with - mean_without
        # "positive" hypothesis: property helps → with group has LOWER ranks
        #   → observed_diff should be negative (lower = better)
        observed_diff = float(np.mean(vals_with) - np.mean(vals_without))

        # Exact permutation: one-sided p-value
        all_grns = list(grn_mean_ranks.keys())
        all_vals = np.array([grn_mean_ranks[g] for g in all_grns])
        n_total = len(all_grns)

        count_extreme = 0
        count_total = 0
        for combo in combinations(range(n_total), n_with):
            perm_diff = all_vals[list(combo)].mean() - np.delete(all_vals, list(combo)).mean()
            if direction == "positive":
                # Property helps → with group should have lower ranks
                if perm_diff <= observed_diff + 1e-12:
                    count_extreme += 1
            else:
                # Property hurts → with group should have higher ranks
                if perm_diff >= observed_diff - 1e-12:
                    count_extreme += 1
            count_total += 1

        p_value = count_extreme / count_total

        # Effect size: rank-biserial r
        U, _ = sp_stats.mannwhitneyu(vals_with, vals_without, alternative="two-sided")
        r_effect = 1.0 - 2.0 * U / (n_with * n_without)

        results.append({
            "property": prop_name,
            "primitive": prim_name,
            "direction": direction,
            "n_with": n_with,
            "n_without": n_without,
            "effect_r": float(r_effect),
            "observed_diff": observed_diff,
            "p_value_one_sided": float(p_value),
            "n_permutations": count_total,
        })

    return results


def fdr_correction(results: list[dict], alpha: float = 0.05) -> list[dict]:
    """Apply Benjamini-Hochberg FDR correction."""
    p_values = np.array([r["p_value"] for r in results])
    valid = ~np.isnan(p_values)

    q_values = np.full_like(p_values, np.nan)
    if valid.sum() > 0:
        valid_p = p_values[valid]
        n_valid = len(valid_p)
        sorted_idx = np.argsort(valid_p)
        sorted_p = valid_p[sorted_idx]
        # BH procedure
        bh_vals = sorted_p * n_valid / (np.arange(1, n_valid + 1))
        # Enforce monotonicity from right
        bh_corrected = np.minimum.accumulate(bh_vals[::-1])[::-1]
        bh_corrected = np.clip(bh_corrected, 0, 1)
        # Map back
        valid_q = np.empty(n_valid)
        valid_q[sorted_idx] = bh_corrected
        q_values[valid] = valid_q

    for i, r in enumerate(results):
        r["q_value"] = float(q_values[i]) if not np.isnan(q_values[i]) else np.nan
        if not np.isnan(r["q_value"]):
            if r["q_value"] < 0.001:
                r["sig"] = "***"
            elif r["q_value"] < 0.01:
                r["sig"] = "**"
            elif r["q_value"] < 0.05:
                r["sig"] = "*"
            else:
                r["sig"] = ""
        else:
            r["sig"] = ""
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_heatmap(
    results: list[dict],
    properties: dict[str, list[str]],
    primitives: dict[str, list[str]],
    output_path: str,
    category_breaks: list[int] | None = None,
):
    """Transposed layout: primitives (rows) × properties (columns)."""
    TEXTWIDTH = 6.5
    prop_names = list(properties.keys())
    prim_names = list(primitives.keys())
    n_props = len(prop_names)
    n_prims = len(prim_names)

    # Build effect matrix in (property, primitive) then transpose to (primitive, property)
    effect_raw = np.full((n_props, n_prims), np.nan)
    sig_raw = [[""] * n_prims for _ in range(n_props)]
    for r in results:
        i = prop_names.index(r["property"])
        j = prim_names.index(r["primitive"])
        effect_raw[i, j] = r["effect_r"]
        sig_raw[i][j] = r["sig"]

    # Transpose: rows=primitives, cols=properties
    effect_matrix = effect_raw.T  # shape (n_prims, n_props)
    sig_matrix = [[sig_raw[i][j] for i in range(n_props)] for j in range(n_prims)]

    n_rows = n_prims   # 7
    n_cols = n_props    # 13

    # Figure size — landscape
    cell_w = 0.44
    cell_h = 0.44
    left_margin = 1.15
    right_margin = 0.55
    top_margin = 1.15
    bottom_margin = 0.35
    fig_w = left_margin + n_cols * cell_w + right_margin
    fig_h = top_margin + n_rows * cell_h + bottom_margin

    scale = TEXTWIDTH / fig_w
    fig_w = TEXTWIDTH
    fig_h *= scale
    cell_w *= scale
    cell_h *= scale

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked = np.ma.masked_invalid(effect_matrix)
    vmax = max(0.3, np.nanmax(np.abs(effect_matrix[~np.isnan(effect_matrix)])))
    vmin = -vmax

    im = ax.imshow(
        masked, cmap="RdBu", vmin=vmin, vmax=vmax, aspect="auto",
        interpolation="nearest",
    )

    # X-axis: property labels (top)
    prop_labels = [f"{name}\n(n={len(properties[name])})" for name in prop_names]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(prop_labels, fontsize=5.5, rotation=45, ha="left",
                       rotation_mode="anchor")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Y-axis: primitive labels
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(prim_names, fontsize=7)

    # Cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = effect_matrix[i, j]
            sig = sig_matrix[i][j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=5, color="#bbbbbb")
            elif sig:
                txt_color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(j, i - 0.12, f"{val:+.2f}", ha="center", va="center",
                        fontsize=5, color=txt_color)
                ax.text(j, i + 0.16, sig, ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color=txt_color)
            else:
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=4.5, color="#aaaaaa")

    # Grid lines (draw first, separators will overlay)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)

    # Category separator lines (vertical, between property groups)
    # Draw on top of grid; use zorder and clip_on to ensure exact alignment
    if category_breaks:
        for brk in category_breaks:
            ax.axvline(x=brk + 0.5, color="white", linewidth=3.5,
                       zorder=5, clip_on=True)

    # Category labels between heatmap and colorbar
    if category_breaks:
        cat_labels = ["Dynamical", "Biological", "Structural"]
        cat_ranges = [
            (0, category_breaks[0]),
            (category_breaks[0] + 1, category_breaks[1]),
            (category_breaks[1] + 1, n_cols - 1),
        ]
        trans = ax.get_xaxis_transform()
        for label, (start, end) in zip(cat_labels, cat_ranges):
            mid = (start + end) / 2
            ax.text(mid, -0.06, label, ha="center", va="top",
                    fontsize=5.5, color="#555555", transform=trans)

    # Colorbar — horizontal below the heatmap (extra pad for category labels)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        fraction=0.06, pad=0.12, aspect=30)
    cbar.set_label("Rank-biserial r  (blue = property helps, red = hurts)",
                   fontsize=5.5)
    cbar.ax.tick_params(labelsize=5.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path} and .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Semantic Primitives Analysis")
    parser.add_argument("--log_root", default="./results/log")
    parser.add_argument("--output_dir", default="./results/figures")
    parser.add_argument("--output_prefix", default="semantic_primitives")
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--tail_frac", type=float, default=0.2)
    parser.add_argument(
        "--reservoirs", nargs="+",
        default=[
            "identity_gradient", "mlp_gradient", "lorenzsystem_gradient",
            "tyson1999circlelock_gradient",
            "weimann2004circadianoscillator_gradient",
            "almeida2019circadianclock_gradient",
            "leloup1999circadianclock_gradient",
            "tyson1991cellcycle2var_gradient",
            "gardner1998cellcyclegoldbeter_gradient",
            "gerard2010cellcycle_gradient",
            "chickarmane2006stemcellswitch_gradient",
            "chickarmane2008nanoggata6_gradient",
            "zatorsky2006p53model4_gradient",
            "gardner2000toggleswitch_gradient",
            "liebal2012transcriptioninhibition_gradient",
            "markevich2004mapkdoublephosphorylation_gradient",
            "kholodenko2000mapkcascade_gradient",
        ],
    )
    parser.add_argument(
        "--envs", nargs="+",
        default=[
            "CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0",
            "Pendulum-v1", "Reacher-v4",
            "Pusher-v4", "Swimmer-v4", "Hopper-v4", "HalfCheetah-v4",
            "HumanoidStandup-v4", "PointMaze", "finger-spin",
        ],
    )
    args = parser.parse_args()

    print("Loading records...")
    records = Records(args.log_root)
    print(f"  {records}")

    # All reservoir base names that participate in ranking (including Lorenz, excluding controls)
    bio_bases = [r.replace("_gradient", "") for r in args.reservoirs
                 if r.replace("_gradient", "") not in CONTROLS]

    print("Extracting final rewards...")
    rewards = extract_final_rewards(
        records, args.reservoirs, args.envs,
        n_bins=args.n_bins, tail_frac=args.tail_frac,
    )
    print(f"  {len(rewards)} (reservoir, env) pairs")

    print("Ranking within environments...")
    ranks = rank_within_env(rewards, args.envs, bio_bases)
    print(f"  {len(ranks)} rank entries")

    # Remove Lorenz from bio_reservoirs for the property comparison
    # (Lorenz participates in ranking but has no biological property)
    bio_for_test = [r for r in bio_bases if r != "lorenzsystem"]

    print("Computing GRN-level mean ranks...")
    grn_mean_ranks = compute_grn_mean_ranks(ranks, SEMANTIC_PRIMITIVES, bio_for_test)

    print("Running omnibus permutation test...")
    omnibus = omnibus_permutation_test(
        grn_mean_ranks, GRN_PROPERTIES, bio_for_test,
    )
    print(f"  Omnibus: stat = {omnibus['observed_stat']:.3f}, "
          f"p = {omnibus['p_value']:.4f} "
          f"({omnibus['n_permutations']} permutations)")

    print("Computing per-cell statistics...")
    results = compute_property_primitive_stats(
        ranks, GRN_PROPERTIES, SEMANTIC_PRIMITIVES, bio_for_test,
    )
    results = fdr_correction(results)

    # Print summary
    sig_count = sum(1 for r in results if r["sig"])
    print(f"  {len(results)} tests, {sig_count} significant after FDR correction")
    for r in results:
        if r["sig"]:
            direction = "helps" if r["effect_r"] > 0 else "hurts"
            print(f"    {r['property']:20s} × {r['primitive']:18s}: "
                  f"r={r['effect_r']:+.3f} {r['sig']} ({direction})")

    # Print top uncorrected associations
    valid = [r for r in results if r["p_value"] == r["p_value"]]
    valid.sort(key=lambda r: r["p_value"])
    print(f"\n  Top 10 associations by uncorrected p-value:")
    for r in valid[:10]:
        direction = "+" if r["effect_r"] > 0 else ""
        print(f"    {r['property']:20s} × {r['primitive']:18s}: "
              f"r={r['effect_r']:+.3f}  p={r['p_value']:.4f}")

    # A priori hypothesis tests (one-sided, FDR over 5 tests only)
    print("\nA priori hypothesis tests (one-sided):")
    a_priori = test_a_priori_hypotheses(
        ranks, GRN_PROPERTIES, SEMANTIC_PRIMITIVES, bio_for_test,
    )
    a_priori_p = np.array([r["p_value_one_sided"] for r in a_priori])
    n_ap = len(a_priori_p)
    sorted_idx = np.argsort(a_priori_p)
    bh_vals = a_priori_p[sorted_idx] * n_ap / (np.arange(1, n_ap + 1))
    bh_corrected = np.minimum.accumulate(bh_vals[::-1])[::-1]
    bh_corrected = np.clip(bh_corrected, 0, 1)
    q_ap = np.empty(n_ap)
    q_ap[sorted_idx] = bh_corrected
    for i, r in enumerate(a_priori):
        r["q_value"] = float(q_ap[i])
        r["sig"] = "***" if q_ap[i] < 0.001 else "**" if q_ap[i] < 0.01 else "*" if q_ap[i] < 0.05 else ""
    ap_sig = sum(1 for r in a_priori if r["sig"])
    print(f"  {n_ap} tests, {ap_sig} significant after FDR correction")
    for r in a_priori:
        tag = f" {r['sig']}" if r["sig"] else ""
        print(f"  H: {r['property']:18s} → {r['primitive']:18s} ({r['direction']:>8s}): "
              f"r={r['effect_r']:+.3f}  p={r['p_value_one_sided']:.4f}  "
              f"q={r['q_value']:.4f}{tag}")

    # Save stats
    output_json = os.path.join(args.output_dir, f"{args.output_prefix}_stats.json")
    os.makedirs(args.output_dir, exist_ok=True)
    all_stats = {"omnibus": omnibus, "per_cell": results, "a_priori": a_priori}
    with open(output_json, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nSaved stats: {output_json}")

    # Plot
    output_pdf = os.path.join(args.output_dir, f"{args.output_prefix}.pdf")
    plot_heatmap(
        results, GRN_PROPERTIES, SEMANTIC_PRIMITIVES,
        output_pdf, PROPERTY_CATEGORY_BREAKS,
    )


if __name__ == "__main__":
    main()
