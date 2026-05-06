"""Per-cell 2x2 contingency-table test for (property, environment) pairs.

For each environment ``e``, binarise the 14 biological GRNs into "good" and
"bad" by a median split of their seed-averaged final reward (top 7 vs.
bottom 7).  For every biological property ``b``, the 14 GRNs are also
partitioned into "with b" and "without b".  The two binarisations combine
into a 2x2 contingency table

                    good   bad
    with b           a      b
    without b        c      d

on which a two-sided Fisher exact test is run.  Repeating this for every
(property, environment) pair yields a 13 x 16 matrix of signed associations,
with Benjamini-Hochberg FDR correction across all 13 * 16 = 208 tests.

Unlike the universal sign test (Appendix E), this analysis localises the
effect of every biological property to individual environments: it asks
*which* environments each property helps or hurts on, not just whether the
aggregate direction is positive or negative.

Inputs
------
- ``results/log/`` (training monitor CSVs).

Outputs
-------
- ``results/figures/contingency.{pdf,png}``: heatmap of signed -log10(q)
  with stars marking cells that survive FDR correction.
- ``results/figures/contingency_stats.json``: per-cell contingency tables,
  odds ratios, raw and FDR-corrected p-values.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from src.plot_rewards import Records
from src.semantic.semantic_analysis import (
    CONTROLS,
    GRN_PROPERTIES,
    PROPERTY_CATEGORY_BREAKS,
    extract_final_rewards,
)
from src.semantic.sign_test_analysis import DEFAULT_ENVS, DEFAULT_RESERVOIRS, ENV_SHORT


def binarise_by_median(
    rewards: dict[tuple[str, str], float],
    envs: list[str],
    bio_reservoirs: list[str],
) -> dict[tuple[str, str], int]:
    """Top-half vs bottom-half split of biological GRNs within each env.

    Returns a dict mapping (grn, env) -> 1 (good) or 0 (bad).  Cells with
    missing reward data are simply absent from the dict.  With 14 GRNs the
    split is 7 good / 7 bad; ties at the median (rare with float rewards)
    are resolved by assigning ranks in argsort order, which keeps the split
    exactly balanced.
    """
    labels: dict[tuple[str, str], int] = {}
    for env in envs:
        pairs = [(g, rewards[(g, env)]) for g in bio_reservoirs
                 if (g, env) in rewards]
        if not pairs:
            continue
        # Sort descending by reward, top half -> good.
        pairs_sorted = sorted(pairs, key=lambda x: -x[1])
        n = len(pairs_sorted)
        n_good = n // 2 + (n % 2)  # upper half on odd sizes (not used here)
        for rank, (g, _) in enumerate(pairs_sorted):
            labels[(g, env)] = 1 if rank < n_good else 0
    return labels


def compute_contingency(
    good_labels: dict[tuple[str, str], int],
    properties: dict[str, list[str]],
    envs: list[str],
    bio_reservoirs: list[str],
) -> list[dict]:
    """Per (property, env) Fisher exact test on the 2x2 good/bad x with/without table."""
    results: list[dict] = []
    for prop_name, prop_grns in properties.items():
        prop_set = set(prop_grns)
        per_env: list[dict] = []
        for env in envs:
            grns_here = [g for g in bio_reservoirs if (g, env) in good_labels]
            if not grns_here:
                continue
            a = sum(1 for g in grns_here if g in prop_set
                    and good_labels[(g, env)] == 1)
            b = sum(1 for g in grns_here if g in prop_set
                    and good_labels[(g, env)] == 0)
            c = sum(1 for g in grns_here if g not in prop_set
                    and good_labels[(g, env)] == 1)
            d = sum(1 for g in grns_here if g not in prop_set
                    and good_labels[(g, env)] == 0)

            table = [[a, b], [c, d]]
            # Two-sided Fisher exact; scipy returns odds ratio + p-value.
            odds, p = sp_stats.fisher_exact(table, alternative="two-sided")
            # Haldane-Anscombe corrected log-odds for heatmap colouring
            # (stable when any cell is zero).
            lor = float(np.log(
                ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
            ))
            per_env.append({
                "env": env,
                "a": int(a), "b": int(b), "c": int(c), "d": int(d),
                "n_with": int(a + b), "n_without": int(c + d),
                "n_good": int(a + c), "n_bad": int(b + d),
                "odds_ratio": float(odds) if np.isfinite(odds) else None,
                "log_odds_ratio_corrected": lor,
                "p_value": float(p),
            })
        results.append({
            "property": prop_name,
            "n_property_grns": len(prop_grns),
            "per_env": per_env,
        })
    return results


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; NaN in, NaN out."""
    q = np.full_like(pvals, np.nan, dtype=float)
    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return q
    vp = pvals[valid]
    n = len(vp)
    order = np.argsort(vp)
    ranked = vp[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n)
    out[order] = adj
    q[valid] = out
    return q


def sig_mark(q: float) -> str:
    if np.isnan(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def plot_contingency_heatmap(
    results: list[dict],
    envs: list[str],
    output_path: str,
    category_breaks: list[int] | None = None,
    alpha_mark: float = 0.05,
):
    """Heatmap of property x env signed -log10(p_raw); stars mark FDR-significant cells.

    Colour uses signed -log10(p_raw) so that the visual strength reflects
    raw evidence direction.  Stars annotate cells whose FDR-corrected q
    crosses the standard thresholds.
    """
    TEXTWIDTH = 6.5

    prop_names = [r["property"] for r in results]
    n_props = len(prop_names)
    n_envs = len(envs)

    signed_nlp = np.full((n_props, n_envs), np.nan)
    q_mat = np.full((n_props, n_envs), np.nan)
    p_mat = np.full((n_props, n_envs), np.nan)
    lor_mat = np.full((n_props, n_envs), np.nan)

    for i, r in enumerate(results):
        for er in r["per_env"]:
            if er["env"] not in envs:
                continue
            j = envs.index(er["env"])
            p_mat[i, j] = er["p_value"]
            lor_mat[i, j] = er["log_odds_ratio_corrected"]

    # FDR correct across all cells in one pass.
    q_flat = fdr_bh(p_mat.flatten())
    q_mat = q_flat.reshape(p_mat.shape)

    # Signed -log10(p_raw): sign taken from corrected log-odds ratio.
    sign = np.sign(lor_mat)
    sign[sign == 0] = 1.0
    signed_nlp = -np.log10(np.clip(p_mat, 1e-6, 1.0)) * sign

    fig_h = 4.2
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, fig_h))

    vmax = max(0.5, float(np.nanmax(np.abs(signed_nlp)))
               if np.any(~np.isnan(signed_nlp)) else 1.0)
    vmin = -vmax

    im = ax.imshow(
        np.ma.masked_invalid(signed_nlp),
        cmap="RdBu_r",
        vmin=vmin, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )

    # Annotations: stars for FDR-significant cells, a small dot for cells
    # that are nominally interesting (raw p < 0.10) but not FDR-significant.
    for i in range(n_props):
        for j in range(n_envs):
            q = q_mat[i, j]
            p = p_mat[i, j]
            if np.isnan(p):
                ax.text(j, i, "-", ha="center", va="center",
                        fontsize=5, color="#bbbbbb")
                continue
            if not np.isnan(q) and q < 0.05:
                ax.text(j, i, sig_mark(q), ha="center", va="center",
                        fontsize=6.5, color="black", fontweight="bold")
            elif p < 0.10:
                ax.text(j, i, "\u00b7", ha="center", va="center",
                        fontsize=9, color="black")

    ax.set_xticks(range(n_envs))
    ax.set_xticklabels([ENV_SHORT.get(e, e) for e in envs],
                       fontsize=5.5, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(n_props))
    ax.set_yticklabels(prop_names, fontsize=6.5)

    ax.set_xticks(np.arange(-0.5, n_envs, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_props, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)
    ax.tick_params(axis="both", which="major", length=2)

    if category_breaks:
        for brk in category_breaks:
            ax.axhline(y=brk + 0.5, color="white", linewidth=2.5,
                       zorder=5, clip_on=True)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                        fraction=0.035, pad=0.02)
    cbar.set_label(
        r"signed $-\log_{10}\,p_\mathrm{raw}$ "
        r"(red: property over-represented in good half)",
        fontsize=5.5,
    )
    cbar.ax.tick_params(labelsize=5.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path} and .png")

    return q_mat, p_mat, lor_mat


def main():
    parser = argparse.ArgumentParser(
        description="Per-cell 2x2 contingency test for (property, env) pairs",
    )
    parser.add_argument("--log_root", default="./results/log")
    parser.add_argument("--output_dir", default="./results/figures")
    parser.add_argument("--output_prefix", default="contingency")
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--tail_frac", type=float, default=0.2)
    parser.add_argument("--reservoirs", nargs="+", default=DEFAULT_RESERVOIRS)
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    args = parser.parse_args()

    print("Loading records...")
    records = Records(args.log_root)
    print(f"  {records}")

    print("Extracting seed-averaged final rewards...")
    rewards = extract_final_rewards(
        records, args.reservoirs, args.envs,
        n_bins=args.n_bins, tail_frac=args.tail_frac,
    )
    print(f"  {len(rewards)} (reservoir, env) pairs")

    bio_reservoirs = sorted({g for grns in GRN_PROPERTIES.values() for g in grns})
    bio_reservoirs = [r for r in bio_reservoirs if r not in CONTROLS]
    print(f"  {len(bio_reservoirs)} biological GRNs")

    missing = [(g, e) for g in bio_reservoirs for e in args.envs
               if (g, e) not in rewards]
    if missing:
        print(f"  warning: {len(missing)} (GRN, env) cells missing")

    print("Binarising performance within each environment (median split)...")
    good = binarise_by_median(rewards, args.envs, bio_reservoirs)

    print("Running per-cell Fisher exact tests...")
    results = compute_contingency(good, GRN_PROPERTIES, args.envs, bio_reservoirs)

    # Flatten p-values for FDR across all cells.
    pvals_flat = []
    index_map: list[tuple[int, int]] = []
    for i, r in enumerate(results):
        per_env_map = {er["env"]: er for er in r["per_env"]}
        for j, env in enumerate(args.envs):
            er = per_env_map.get(env)
            if er is None:
                continue
            pvals_flat.append(er["p_value"])
            index_map.append((i, j))
    pvals_arr = np.array(pvals_flat)
    qvals_arr = fdr_bh(pvals_arr)

    # Write q back into per-cell dicts.
    q_lookup = {ij: q for ij, q in zip(index_map, qvals_arr)}
    for i, r in enumerate(results):
        for er in r["per_env"]:
            j = args.envs.index(er["env"])
            er["q_value"] = float(q_lookup.get((i, j), np.nan))
            er["sig"] = sig_mark(er["q_value"])

    # Console summary: list all cells with raw p < 0.1, sorted by p.
    print()
    print("Per-cell results (raw p < 0.10):")
    print(f"  {'property':18s} {'env':14s} "
          f"{'a/b/c/d':>12s} {'lnOR':>7s} {'p':>8s} {'q(BH)':>8s}")
    flat_rows = []
    for r in results:
        for er in r["per_env"]:
            if er["p_value"] < 0.10:
                flat_rows.append((r["property"], er))
    flat_rows.sort(key=lambda x: x[1]["p_value"])
    for prop, er in flat_rows:
        tab = f"{er['a']}/{er['b']}/{er['c']}/{er['d']}"
        print(f"  {prop:18s} {ENV_SHORT.get(er['env'], er['env']):14s} "
              f"{tab:>12s} {er['log_odds_ratio_corrected']:+7.2f} "
              f"{er['p_value']:8.4f} {er['q_value']:8.4f} {er['sig']}")

    n_sig_05 = sum(1 for _, er in flat_rows if er["q_value"] < 0.05)
    n_sig_10 = sum(1 for _, er in flat_rows if er["q_value"] < 0.10)
    print(f"\n  FDR q<0.05: {n_sig_05} cells; q<0.10: {n_sig_10} cells; "
          f"total tested: {len(pvals_arr)}")

    out_json = os.path.join(args.output_dir, f"{args.output_prefix}_stats.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "envs": args.envs,
            "bio_reservoirs": bio_reservoirs,
            "results": results,
        }, f, indent=2, default=str)
    print(f"Saved stats: {out_json}")

    out_pdf = os.path.join(args.output_dir, f"{args.output_prefix}.pdf")
    plot_contingency_heatmap(results, args.envs, out_pdf,
                             PROPERTY_CATEGORY_BREAKS)


if __name__ == "__main__":
    main()
