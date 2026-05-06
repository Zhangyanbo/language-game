"""Sign test for universal effects of GRN properties on RL reward.

For each property ``b`` and each environment ``e``, partition the 14
biological GRNs into two groups (with ``b`` vs. without) and compare their
median seed-averaged reward in ``e``. This yields one signed observation per
environment. Across all environments, a binomial sign test asks whether the
proportion of positive signs deviates from 0.5, i.e., whether ``b`` universally
helps or hurts, independent of any semantic-primitive taxonomy.

The analysis is deliberately assumption-light: no per-primitive grouping, no
rank-biserial effect size, no task-semantic labelling.  It is the simplest
non-parametric test that "generally helps/hurts" admits.

Inputs
------
- ``results/log/`` (training monitor CSVs).

Outputs
-------
- ``results/figures/sign_test.{pdf,png}``: two-panel figure.
    * Left: (GRN property x env) matrix of signed normalised effect sizes.
    * Right: per-property bar of positive-sign proportion with 95% Wilson
      confidence interval, annotated with k/n and raw / FDR-corrected p.
- ``results/figures/sign_test_stats.json``: machine-readable stats.
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


# Full 16-env matrix (12 non-Atari + 4 Atari RAM).
DEFAULT_ENVS = [
    "CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0",
    "Pendulum-v1", "Reacher-v4",
    "Pusher-v4", "Swimmer-v4", "Hopper-v4", "HalfCheetah-v4",
    "HumanoidStandup-v4", "PointMaze", "finger-spin",
    "BankHeist-ram", "KungFuMaster-ram", "CrazyClimber-ram", "Kangaroo-ram",
]

DEFAULT_RESERVOIRS = [
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
]

# Short labels for the sign matrix x-axis.
ENV_SHORT = {
    "CartPole-v1": "CartPole",
    "Acrobot-v1": "Acrobot",
    "MountainCarContinuous-v0": "MountainCar",
    "Pendulum-v1": "Pendulum",
    "Reacher-v4": "Reacher",
    "Pusher-v4": "Pusher",
    "Swimmer-v4": "Swimmer",
    "Hopper-v4": "Hopper",
    "HalfCheetah-v4": "HalfCheetah",
    "HumanoidStandup-v4": "HumanoidSU",
    "PointMaze": "PointMaze",
    "finger-spin": "finger-spin",
    "BankHeist-ram": "BankHeist",
    "KungFuMaster-ram": "KungFu",
    "CrazyClimber-ram": "CrazyClimber",
    "Kangaroo-ram": "Kangaroo",
}


def standardize_within_env(
    rewards: dict[tuple[str, str], float],
    envs: list[str],
    reservoirs_base: list[str],
) -> dict[tuple[str, str], float]:
    """Z-score rewards within each env across the provided reservoirs."""
    z = {}
    for env in envs:
        pairs = [(r, rewards[(r, env)]) for r in reservoirs_base
                 if (r, env) in rewards]
        if not pairs:
            continue
        arr = np.array([v for _, v in pairs], dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std())
        if sd == 0.0:
            sd = 1.0
        for (name, _), val in zip(pairs, arr):
            z[(name, env)] = (val - mu) / sd
    return z


def compute_sign_test(
    rewards: dict[tuple[str, str], float],
    z_scores: dict[tuple[str, str], float],
    properties: dict[str, list[str]],
    envs: list[str],
    bio_reservoirs: list[str],
) -> list[dict]:
    """Per-property binomial sign test over the per-env median comparison."""
    results = []
    for prop_name, prop_grns in properties.items():
        prop_set = set(prop_grns)

        env_records: list[dict] = []
        for env in envs:
            with_raw = [rewards[(g, env)] for g in bio_reservoirs
                        if g in prop_set and (g, env) in rewards]
            without_raw = [rewards[(g, env)] for g in bio_reservoirs
                           if g not in prop_set and (g, env) in rewards]
            if not with_raw or not without_raw:
                continue

            med_with = float(np.median(with_raw))
            med_without = float(np.median(without_raw))

            if med_with > med_without:
                sign = 1
            elif med_with < med_without:
                sign = -1
            else:
                sign = 0

            with_z = [z_scores[(g, env)] for g in bio_reservoirs
                      if g in prop_set and (g, env) in z_scores]
            without_z = [z_scores[(g, env)] for g in bio_reservoirs
                         if g not in prop_set and (g, env) in z_scores]
            effect_z = float(np.median(with_z) - np.median(without_z))

            env_records.append({
                "env": env,
                "sign": sign,
                "median_with": med_with,
                "median_without": med_without,
                "effect_z": effect_z,
                "n_with": len(with_raw),
                "n_without": len(without_raw),
            })

        signs = np.array([r["sign"] for r in env_records], dtype=int)
        n_pos = int((signs > 0).sum())
        n_neg = int((signs < 0).sum())
        n_tie = int((signs == 0).sum())
        n_eff = n_pos + n_neg  # ties excluded from binomial test

        if n_eff >= 1:
            bt = sp_stats.binomtest(n_pos, n_eff, p=0.5, alternative="two-sided")
            p_two = float(bt.pvalue)
            ci = bt.proportion_ci(confidence_level=0.95, method="wilson")
            ci_lo, ci_hi = float(ci.low), float(ci.high)
            prop_pos = n_pos / n_eff
        else:
            p_two = np.nan
            ci_lo = ci_hi = prop_pos = np.nan

        results.append({
            "property": prop_name,
            "n_property_grns": len(prop_grns),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_tie": n_tie,
            "n_effective": n_eff,
            "n_envs_tested": len(env_records),
            "prop_positive": prop_pos,
            "wilson_ci_lo": ci_lo,
            "wilson_ci_hi": ci_hi,
            "p_value": p_two,
            "per_env": env_records,
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sign_test(
    results: list[dict],
    envs: list[str],
    output_path: str,
    category_breaks: list[int] | None = None,
):
    """Two-panel figure: sign matrix (left) + per-property summary bars (right)."""
    TEXTWIDTH = 6.5

    prop_names = [r["property"] for r in results]
    n_props = len(prop_names)
    n_envs = len(envs)

    # --- Build effect matrix (property x env) in z-score units ------------
    eff = np.full((n_props, n_envs), np.nan)
    sgn = np.zeros((n_props, n_envs), dtype=int)
    for i, r in enumerate(results):
        for er in r["per_env"]:
            if er["env"] in envs:
                j = envs.index(er["env"])
                eff[i, j] = er["effect_z"]
                sgn[i, j] = er["sign"]

    # --- Layout -------------------------------------------------------------
    fig_h = 4.2
    fig, (ax_mat, ax_bar) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH, fig_h),
        gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.42},
    )

    # --- Sign matrix --------------------------------------------------------
    # Colour alone encodes the direction and magnitude of the (with - without)
    # median reward difference, so no +/- overlay is needed.
    vmax = max(0.5, np.nanmax(np.abs(eff))) if np.any(~np.isnan(eff)) else 1.0
    vmin = -vmax

    im = ax_mat.imshow(
        np.ma.masked_invalid(eff), cmap="RdBu", vmin=vmin, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )

    for i in range(n_props):
        for j in range(n_envs):
            if np.isnan(eff[i, j]):
                ax_mat.text(j, i, "-", ha="center", va="center",
                            fontsize=5, color="#bbbbbb")

    ax_mat.set_xticks(range(n_envs))
    ax_mat.set_xticklabels([ENV_SHORT.get(e, e) for e in envs],
                           fontsize=5.5, rotation=45, ha="right",
                           rotation_mode="anchor")
    ax_mat.set_yticks(range(n_props))
    ax_mat.set_yticklabels(prop_names, fontsize=6.5)

    ax_mat.set_xticks(np.arange(-0.5, n_envs, 1), minor=True)
    ax_mat.set_yticks(np.arange(-0.5, n_props, 1), minor=True)
    ax_mat.grid(which="minor", color="white", linewidth=0.5)
    ax_mat.tick_params(which="minor", bottom=False, left=False, top=False)
    ax_mat.tick_params(axis="both", which="major", length=2)

    if category_breaks:
        for brk in category_breaks:
            ax_mat.axhline(y=brk + 0.5, color="white", linewidth=2.5,
                           zorder=5, clip_on=True)

    # Colourbar below the matrix
    cbar = fig.colorbar(im, ax=ax_mat, orientation="horizontal",
                        fraction=0.05, pad=0.22, aspect=30)
    cbar.set_label(
        "median reward z-score: with property - without property (within env)",
        fontsize=5.5,
    )
    cbar.ax.tick_params(labelsize=5.5)

    # --- Per-property summary bars ---------------------------------------
    props_pos = np.array([r["prop_positive"] for r in results])
    ci_lo = np.array([r["wilson_ci_lo"] for r in results])
    ci_hi = np.array([r["wilson_ci_hi"] for r in results])
    qvals = np.array([r["q_value"] for r in results])
    n_eff = np.array([r["n_effective"] for r in results])
    n_pos = np.array([r["n_pos"] for r in results])

    ys = np.arange(n_props)
    err = np.vstack([props_pos - ci_lo, ci_hi - props_pos])
    colors = ["#3b6fb6" if p >= 0.5 else "#c03a2b" for p in props_pos]

    ax_bar.barh(ys, props_pos, height=0.72, color=colors, alpha=0.85,
                edgecolor="black", linewidth=0.4)
    ax_bar.errorbar(props_pos, ys, xerr=err, fmt="none",
                    ecolor="black", elinewidth=0.5, capsize=1.5)

    ax_bar.axvline(0.5, color="black", linewidth=0.6, linestyle="--")
    ax_bar.set_xlim(0.0, 1.30)  # extra room for k/n and significance stars
    ax_bar.set_ylim(n_props - 0.5, -0.5)  # match matrix orientation
    ax_bar.set_yticks(ys)
    ax_bar.set_yticklabels(prop_names, fontsize=6.5)
    ax_bar.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.tick_params(axis="both", which="major", labelsize=5.5, length=2)
    ax_bar.set_xlabel("proportion of envs with positive sign",
                      fontsize=6)
    ax_mat.set_ylabel("GRN Property",
                      fontsize=6)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_linewidth(0.4)
    ax_bar.spines["bottom"].set_linewidth(0.4)

    # Annotation: k/n and significance star, placed just past the CI
    for i in range(n_props):
        label = f"{n_pos[i]}/{n_eff[i]}"
        star = sig_mark(qvals[i])
        if star:
            label += f"  {star}"
        x_anchor = min(max(ci_hi[i], props_pos[i]) + 0.02, 1.02)
        ax_bar.text(x_anchor, i, label, ha="left", va="center",
                    fontsize=5.5)

    if category_breaks:
        for brk in category_breaks:
            ax_bar.axhline(y=brk + 0.5, color="#888888", linewidth=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path} and .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sign test for GRN property effects"
    )
    parser.add_argument("--log_root", default="./results/log")
    parser.add_argument("--output_dir", default="./results/figures")
    parser.add_argument("--output_prefix", default="sign_test")
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

    # Bio reservoirs only (drop controls and the Lorenz baseline: Lorenz has no
    # GRN property, so it does not belong to either
    # side of the sign test partition).
    bio_reservoirs = sorted({g for grns in GRN_PROPERTIES.values() for g in grns})
    bio_reservoirs = [r for r in bio_reservoirs if r not in CONTROLS]
    print(f"  {len(bio_reservoirs)} biological GRNs")

    # Warn about missing (GRN, env) cells, so partial data is visible.
    missing = [(g, e) for g in bio_reservoirs for e in args.envs
               if (g, e) not in rewards]
    if missing:
        print(f"  warning: {len(missing)} (GRN, env) cells missing")

    print("Standardising rewards within environments...")
    z_scores = standardize_within_env(rewards, args.envs, bio_reservoirs)

    print("Running per-property sign tests...")
    results = compute_sign_test(
        rewards, z_scores, GRN_PROPERTIES, args.envs, bio_reservoirs,
    )

    pvals = np.array([r["p_value"] for r in results])
    qvals = fdr_bh(pvals)
    for r, q in zip(results, qvals):
        r["q_value"] = float(q) if not np.isnan(q) else np.nan
        r["sig"] = sig_mark(r["q_value"])

    # Print summary table.
    print()
    print(f"{'property':22s} {'k/n':>7s}  {'prop+':>6s}  "
          f"{'95% CI':>16s}  {'p':>8s}  {'q(BH)':>8s}")
    print("-" * 78)
    for r in results:
        ci = f"[{r['wilson_ci_lo']:.2f},{r['wilson_ci_hi']:.2f}]"
        print(f"{r['property']:22s} "
              f"{r['n_pos']:2d}/{r['n_effective']:2d}  "
              f"{r['prop_positive']:6.3f}  "
              f"{ci:>16s}  "
              f"{r['p_value']:8.4f}  {r['q_value']:8.4f}  {r['sig']}")

    # Save stats.
    out_json = os.path.join(args.output_dir, f"{args.output_prefix}_stats.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "envs": args.envs,
            "bio_reservoirs": bio_reservoirs,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSaved stats: {out_json}")

    # Plot.
    out_pdf = os.path.join(args.output_dir, f"{args.output_prefix}.pdf")
    plot_sign_test(results, args.envs, out_pdf, PROPERTY_CATEGORY_BREAKS)


if __name__ == "__main__":
    main()
