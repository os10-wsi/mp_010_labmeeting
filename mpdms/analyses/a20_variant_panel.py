"""A20 - replicate-level dot plot for a handful of named variants against synonymous WT.

For a short list of specific alleles (by default the published ones in
configs/_literature.yaml), plot every replicate's normalised fitness as a dot,
one column per variant, with the synonymous distribution as the reference
column. Scores are already normalised so synonymous median = 0 and nonsense
median = -1, per replicate, so "relative to synonymous wild type" is the scale.

TWO COMPARISONS, VERY DIFFERENT POWER - this is the whole point of the module:

  variant vs synonymous   well powered. The reference is hundreds of synonymous
                          variants, so their spread is a real estimate of what
                          "no effect" looks like, including variant-to-variant
                          noise and not just replicate noise. This is the test
                          to quote.

  variant vs variant      3 replicates against 3. Only a difference of roughly
                          3 pooled SDs is detectable at 80% power, so a null
                          result here means "underpowered", NOT "the same".
                          The module prints the minimum detectable difference
                          so that can be stated rather than implied.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..io import replicate_cols
from ..stats import bh_fdr
from .a19_literature import load_literature
from .base import dirs, fmt_p, result, skipped

SYN_COLOR, MUT_COLOR = "#1B9E77", "#2F6DB5"
HILITE = "#B8860B"
SYN_LABEL = "synonymous\n(WT)"
MDE_D = 3.10  # Cohen's d detectable at 80% power, alpha 0.05 two-sided, n = 3 vs 3
SYN_DISPLAY_N = 20   # synonymous variants drawn in the dot plot (fixed seed)
SYN_DISPLAY_SEED = 20
MISS_COLOR = "#2F6DB5"


def parse_substitution(s: str) -> tuple[str, int, str] | None:
    """'Q149A' -> ('Q', 149, 'A'); None if it is not a simple point substitution."""
    m = re.fullmatch(r"\s*([A-Z])(\d+)([A-Z])\s*", str(s or "").upper())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def wanted_variants(cfg) -> list[str]:
    """Explicit config list if present, else every published substitution for this gene."""
    named = cfg.get_path("report.variant_panel", None)
    if named:
        return [str(x) for x in named]
    lit = load_literature(str(cfg.get_path("protein.gene", cfg.id))) or {}
    return [e["substitution"] for e in (lit.get("measured") or []) if e.get("substitution")]


def collect(df: pd.DataFrame, cfg, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Long replicate-level tables for the named variants and for synonymous, plus misses."""
    reps = replicate_cols(df)
    if not reps:
        return pd.DataFrame(), pd.DataFrame(), ["no per-replicate columns in the dataset"]
    rows, missing = [], []
    for lab in labels:
        p = parse_substitution(lab)
        if p is None:
            missing.append(f"{lab}: not a point substitution")
            continue
        wt, pos, mut = p
        hit = df[(df.pos == pos) & (df.mut == mut) & (df.vclass == "missense")]
        if not len(hit):
            missing.append(f"{lab}: not measured in this dataset")
            continue
        r = hit.iloc[0]
        if str(r.wt).upper() != wt:
            missing.append(f"{lab}: dataset has {r.wt}{pos}, not {wt}{pos} - numbering mismatch, skipped")
            continue
        if not bool(r.pass_filter):
            missing.append(f"{lab}: present but fails the read/replicate filter")
            continue
        for k, c in enumerate(reps, 1):
            v = r[c]
            if pd.notna(v):
                rows.append({"variant": lab, "replicate": f"rep{k}", "value": float(v)})
    syn = df[(df.vclass == "synonymous") & df.pass_filter]
    srows = [{"variant": SYN_LABEL, "replicate": f"rep{k}", "value": float(v), "syn_index": i}
             for i, (_, r) in enumerate(syn.iterrows()) for k, c in enumerate(reps, 1)
             if pd.notna(r[c]) for v in [r[c]]]
    return pd.DataFrame(rows), pd.DataFrame(srows), missing


def sample_for_display(syn: pd.DataFrame, n: int = SYN_DISPLAY_N) -> pd.DataFrame:
    """A fixed random subset of synonymous VARIANTS (all their replicates) to draw.

    Display only. The vs-synonymous test keeps the full set: cutting the null to 20
    would raise the empirical p floor to 1/21 = 0.048, so nothing could reach
    significance. The figure says which number is which.
    """
    if not len(syn) or "syn_index" not in syn:
        return syn
    ids = syn.syn_index.unique()
    if len(ids) <= n:
        return syn
    keep = np.random.default_rng(SYN_DISPLAY_SEED).choice(ids, n, replace=False)
    return syn[syn.syn_index.isin(keep)]


def stats_table(mut: pd.DataFrame, syn: pd.DataFrame, syn_means: np.ndarray) -> pd.DataFrame:
    """Each variant against synonymous, then every mutant pair. BH across all of them."""
    rows = []
    syn_vals = syn.value.to_numpy()
    syn_sd = float(np.std(syn_means, ddof=1)) if len(syn_means) > 1 else np.nan
    centre = float(np.median(syn_means)) if len(syn_means) else 0.0
    for v, g in mut.groupby("variant", sort=False):
        x = g.value.to_numpy()
        m = float(np.mean(x))
        # empirical: how extreme is this mean among synonymous variants' own means?
        emp = (1 + int(np.sum(np.abs(syn_means - centre) >= abs(m - centre)))) / (len(syn_means) + 1) \
            if len(syn_means) else np.nan
        t = ss.ttest_ind(x, syn_vals, equal_var=False) if len(x) > 1 and len(syn_vals) > 1 else None
        # The empirical p is primary. A Welch test of 3 replicates against every synonymous
        # MEASUREMENT pretends replicates of one variant sample the same variance as different
        # variants do, which they do not, so it reads far too significant. Comparing this
        # variant's mean to the spread of synonymous variant MEANS asks the right question:
        # is this further from WT than synonymous variants get on their own?
        rows.append({"comparison": f"{v} vs synonymous", "kind": "vs_synonymous", "a": v, "b": SYN_LABEL,
                     "n_a": len(x), "n_b": len(syn_means), "mean_a": m, "mean_b": centre,
                     "difference": m - centre,
                     "z_vs_syn_variants": (m - centre) / syn_sd if syn_sd and np.isfinite(syn_sd) else np.nan,
                     "p": emp, "p_welch_replicates": float(t.pvalue) if t else np.nan,
                     "p_floor": 1 / (len(syn_means) + 1) if len(syn_means) else np.nan})
    for a, b in itertools.combinations(sorted(mut.variant.unique()), 2):
        xa = mut.loc[mut.variant == a, "value"].to_numpy()
        xb = mut.loc[mut.variant == b, "value"].to_numpy()
        if len(xa) < 2 or len(xb) < 2:
            continue
        t = ss.ttest_ind(xa, xb, equal_var=False)
        sd = float(np.sqrt((np.var(xa, ddof=1) + np.var(xb, ddof=1)) / 2))
        rows.append({"comparison": f"{a} vs {b}", "kind": "mutant_pair", "a": a, "b": b,
                     "n_a": len(xa), "n_b": len(xb), "mean_a": float(np.mean(xa)), "mean_b": float(np.mean(xb)),
                     "difference": float(np.mean(xa) - np.mean(xb)), "z_vs_syn_variants": np.nan,
                     "p": float(t.pvalue),
                     "min_detectable_difference": MDE_D * sd})
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = bh_fdr(t.p.fillna(1.0))
    return t


def figure(cfg, mut: pd.DataFrame, syn: pd.DataFrame, syn_means: np.ndarray,
           st: pd.DataFrame, figdir):
    syn_shown = sample_for_display(syn)
    n_shown = int(syn_shown.syn_index.nunique()) if "syn_index" in syn_shown else 0
    order = [SYN_LABEL] + list(dict.fromkeys(mut.variant))
    fig, ax = plt.subplots(figsize=(1.35 * len(order) + 2.6, 4.6), layout="constrained")
    rng = np.random.default_rng(20)
    syn_vals = syn.value.to_numpy()
    # the shaded band is the null the vs-synonymous test uses, so it must be the spread of
    # synonymous variant means - not of individual measurements, which is wider
    sd = float(np.std(syn_means, ddof=1)) if len(syn_means) > 1 else 0.0
    ax.axhspan(-sd, sd, color=SYN_COLOR, alpha=0.10, lw=0, zorder=0)
    ax.axhline(0, color=SYN_COLOR, lw=0.8, zorder=1)
    ax.axhline(-1, color="#D62728", lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.text(len(order) - 0.45, -1, "nonsense median", fontsize=6, color="#D62728", va="bottom", ha="right")

    for i, v in enumerate(order):
        vals = (syn_shown if v == SYN_LABEL else mut).query("variant == @v").value.to_numpy()
        col = SYN_COLOR if v == SYN_LABEL else MUT_COLOR
        # synonymous is a distribution of many variants; the mutants are 3 replicates each
        s, a = (26, 0.55) if v == SYN_LABEL else (42, 0.95)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(vals)), vals, s=s, color=col, alpha=a,
                   lw=0.5 if v != SYN_LABEL else 0, edgecolor="white", zorder=3)
        if len(vals):
            m = float(np.mean(syn.value)) if v == SYN_LABEL else float(np.mean(vals))
            ax.plot([i - 0.26, i + 0.26], [m, m], color=P.INK, lw=1.6, zorder=4)
            if len(vals) > 1:
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                ax.plot([i, i], [m - se, m + se], color=P.INK, lw=1.0, zorder=4)
            ax.annotate(f"{m:.2f}", (i, m), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=7, color=P.INK, zorder=5)

    vs = st[st.kind == "vs_synonymous"].set_index("a") if len(st) else pd.DataFrame()
    lo = min([syn_vals.min() if len(syn_vals) else 0, mut.value.min() if len(mut) else 0, -1.05])
    for i, v in enumerate(order):
        if v == SYN_LABEL or v not in vs.index:
            continue
        r = vs.loc[v]
        at_floor = np.isfinite(r.p_floor) and r.p <= r.p_floor + 1e-12
        ax.text(i, lo - 0.10, f"vs syn\n{fmt_p(r.p)}{' (floor)' if at_floor else ''}\nq = {r.q:.3f}",
                ha="center", va="top", fontsize=6.5, color=P.INK)
    pair = st[st.kind == "mutant_pair"] if len(st) else pd.DataFrame()
    if len(pair):
        mde = float(pair.min_detectable_difference.max())
        bits = [f"{r.comparison}: Δ = {r.difference:+.2f}, q = {r.q:.3f}" for r in pair.itertuples()]
        ax.text(0.5, -0.185, "Between variants (Welch, 3 vs 3): " + " | ".join(bits),
                transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=P.INK)
        ax.text(0.5, -0.255, f"at n = 3 per side only a difference of ≈ {mde:.2f} is detectable at 80% "
                             "power, so a null here is underpowered, not equivalent",
                transform=ax.transAxes, ha="center", va="top", fontsize=6, color=P.MUTED, style="italic")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(lo - 0.40, max(0.45, (mut.value.max() if len(mut) else 0) + 0.3))
    ax.set_ylabel("Normalised fitness (synonymous = 0, nonsense = −1)")
    ax.set_title(f"(a) Replicate fitness per variant; green band = ±1 SD of {len(syn_means)} "
                 f"synonymous variant means\n{n_shown} synonymous variants drawn at random "
                 f"(display only - the test below uses all {len(syn_means)})",
                 loc="left", fontsize=8.5)
    P.title(fig, cfg, "A20 named variants vs synonymous wild type")
    return P.save(fig, figdir, "a20_variant_panel", cfg, "A20")


def percentiles(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Where each named variant sits in this protein's missense fitness distribution."""
    miss = df[(df.vclass == "missense") & df.pass_filter].dropna(subset=["score_z"])
    vals = miss.score_z.to_numpy()
    rows = []
    for lab in labels:
        p = parse_substitution(lab)
        if p is None:
            continue
        _, pos, mut = p
        hit = miss[(miss.pos == pos) & (miss.mut == mut)]
        if not len(hit):
            continue
        x = float(hit.score_z.iloc[0])
        rows.append({"variant": lab, "score_z": x,
                     "percentile": float(100 * np.mean(vals < x)),
                     "n_missense": len(vals),
                     "rank_from_bottom": int(np.sum(vals < x)) + 1})
    return pd.DataFrame(rows)


def percentile_figure(cfg, df: pd.DataFrame, pc: pd.DataFrame, figdir):
    """Missense fitness distribution with the named variants placed in it."""
    miss = df[(df.vclass == "missense") & df.pass_filter].dropna(subset=["score_z"])
    syn = df[(df.vclass == "synonymous") & df.pass_filter].score_z.dropna()
    non = df[(df.vclass == "nonsense") & df.pass_filter].score_z.dropna()
    fig, ax = plt.subplots(figsize=(7.8, 4.4), layout="constrained")
    ax.hist(miss.score_z, bins=60, color=MISS_COLOR, alpha=0.55, lw=0, label=f"missense (n = {len(miss)})")
    if len(syn):
        ax.axvline(float(syn.median()), color=SYN_COLOR, lw=1.2, label="synonymous median")
    if len(non):
        ax.axvline(float(non.median()), color="#D62728", lw=1.2, ls=(0, (3, 3)), label="nonsense median")

    top = ax.get_ylim()[1]
    for k, r in enumerate(pc.sort_values("score_z").itertuples()):
        # stagger vertically, and lean the text left/right alternately: two variants a
        # couple of percentiles apart sit almost on top of each other otherwise
        y = top * (0.95 - 0.15 * (k % 3))
        side = -1 if k % 2 else 1
        ax.annotate("", xy=(r.score_z, 0), xytext=(r.score_z, y),
                    arrowprops=dict(arrowstyle="-", color=HILITE, lw=1.3))
        ax.scatter([r.score_z], [y], s=46, color=HILITE, zorder=5)
        ax.annotate(f"{r.variant}  {r.score_z:.2f}\n{r.percentile:.0f}th percentile",
                    (r.score_z, y), xytext=(8 * side, 8), textcoords="offset points",
                    ha="left" if side > 0 else "right", fontsize=7.5, color=HILITE,
                    fontweight="bold", zorder=6)
    ax.set_ylim(0, top * 1.22)
    ax.set_xlabel("Normalised fitness (synonymous = 0, nonsense = −1)")
    ax.set_ylabel("Missense variants")
    ax.set_title("(b) Where the named variants sit in the missense distribution; "
                 "percentile = % of missense variants less fit", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5)
    P.title(cfg=cfg, fig=fig, text="A20b named variants in the missense distribution")
    return P.save(fig, figdir, "a20b_missense_percentile", cfg, "A20")


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    labels = wanted_variants(cfg)
    if not labels:
        return skipped("a20", cfg, f"no variants named for {cfg.get_path('protein.gene', cfg.id)} "
                                   "(set report.variant_panel, or add measured entries to _literature.yaml)")
    mut, syn, missing = collect(df, cfg, labels)
    if not len(mut):
        return skipped("a20", cfg, "none of the named variants are usable: " + "; ".join(missing))

    syn_means = df[(df.vclass == "synonymous") & df.pass_filter].score_z.dropna().to_numpy()
    st = stats_table(mut, syn, syn_means)
    tp = tabdir / "a20_variant_stats.csv"
    st.to_csv(tp, index=False)
    dp = tabdir / "a20_variant_replicates.csv"
    pd.concat([mut, syn.assign(variant=SYN_LABEL)]).to_csv(dp, index=False)
    pc = percentiles(df, labels)
    figs = figure(cfg, mut, syn, syn_means, st, figdir)
    if len(pc):
        figs = figs + percentile_figure(cfg, df, pc, figdir)
        pc.to_csv(tabdir / "a20b_percentiles.csv", index=False)

    vs = st[st.kind == "vs_synonymous"]
    pair = st[st.kind == "mutant_pair"]
    sig = vs[vs.q < 0.05]
    caveats = [f"not plotted - {m}" for m in missing]
    n_rep = int(mut.groupby("variant").size().min())
    if len(pair):
        mde = float(pair.min_detectable_difference.max())
        caveats.append(f"mutant-vs-mutant tests have {n_rep} replicates per side: only a difference of about "
                       f"{mde:.2f} fitness units is detectable at 80% power, so a null there means "
                       "underpowered, not equivalent")
    floor = float(vs.p_floor.max()) if len(vs) and vs.p_floor.notna().any() else np.nan
    caveats.append(f"vs-synonymous p is empirical against {len(syn_means)} synonymous variant means, so it "
                   f"cannot go below {floor:.2g}; p_welch_replicates in the table is the parametric "
                   "alternative and reads optimistically because replicates of one variant are correlated")
    caveats.append("a variant indistinguishable from synonymous here is NOT evidence it is functionally "
                   "silent - this assay reads abundance, and a mechanistic residue is expected to look normal")

    head = (f"{mut.variant.nunique()} variants x {n_rep} replicates vs {len(syn)} synonymous measurements; "
            + (f"differ from synonymous at q<0.05: {', '.join(sig.a)}" if len(sig) else
               "none differ from synonymous at q<0.05")
            + (f"; largest |Δ| = {vs.difference.abs().max():.2f}" if len(vs) else ""))
    return result("a20", cfg, head,
                  {"n_variants": int(mut.variant.nunique()), "n_replicates": n_rep,
                   "n_synonymous_measurements": int(len(syn)),
                   "vs_synonymous": [{"variant": r.a, "mean": r.mean_a, "difference": r.difference,
                                      "z_vs_syn_variants": r.z_vs_syn_variants,
                                      "p": r.p, "p_welch_replicates": r.p_welch_replicates, "q": r.q}
                                     for r in vs.itertuples()],
                   "mutant_pairs": [{"comparison": r.comparison, "difference": r.difference,
                                     "p": r.p, "q": r.q} for r in pair.itertuples()],
                   "not_plotted": missing,
                   "missense_percentiles": [{"variant": r.variant, "score_z": r.score_z,
                                             "percentile": r.percentile,
                                             "rank_from_bottom": r.rank_from_bottom}
                                            for r in pc.itertuples()]},
                  caveats, figs, [tp, dp] + ([tabdir / "a20b_percentiles.csv"] if len(pc) else []))
