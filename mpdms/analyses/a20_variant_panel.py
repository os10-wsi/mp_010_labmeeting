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

TABLES

  a20_variant_estimates.csv   one row per named variant: normalised fitness, sd, se,
                              the 95% interval, the synonymous median it is measured
                              against, and the test result. This is the one to analyse.
                              `error_kind` says which error it is - the se of the
                              replicate mean, or the dataset's own sigma rescaled -
                              because the two are not interchangeable.
  a20_variant_stats.csv       one row per comparison, including every mutant pair.
  a20_variant_replicates.csv  the individual replicate values behind the dots.
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

# the synonymous reference is blue - fill, band, median line and zero line all from
# the same two values - and the mutants are black points on top of it, so the figure
# reads in one colour family and the two things never blur into each other
SYN_COLOR, SYN_FILL = "#2166AC", "#A6C8E8"
MUT_COLOR = "#111111"
HILITE = "#2166AC"
SYN_LABEL = "synonymous\n(WT)"
MDE_D = 3.10  # Cohen's d detectable at 80% power, alpha 0.05 two-sided, n = 3 vs 3
SYN_DISPLAY_N = 20   # synonymous variants drawn in the dot plot (fixed seed)
SYN_DISPLAY_SEED = 20
# the vs-synonymous test uses the SAME subset the figure draws, so the picture and the
# p-value are about one set of variants. The cost is an empirical p floor of
# 1/(n+1) = 0.048, which is reported on the figure and in every row of the table.
SYN_TEST_N = SYN_DISPLAY_N
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


def wanted_sites(cfg) -> tuple[list[int], dict[int, str], dict[int, str]]:
    """Positions to examine, their expected wild-type residue, and a substitution to mark.

    Entries may name a bare residue (Phe160) or a substitution (D358N); both contribute
    the position, which is what the per-site analysis needs. A position is only marked
    with a specific allele when exactly one is named for it - with three (K492A/Q/E)
    there is no single allele to point at, so the panel just shows the position.
    """
    named = cfg.get_path("report.variant_panel", None)
    entries = ([{"substitution": str(x)} for x in named] if named else
               ((load_literature(str(cfg.get_path("protein.gene", cfg.id))) or {}).get("measured") or []))
    order, wt, muts = [], {}, {}
    for e in entries:
        pos, w, mut = e.get("pos"), e.get("wt"), None
        if e.get("substitution"):
            parsed = parse_substitution(e["substitution"])
            if parsed:
                w, pos, mut = parsed[0], parsed[1], parsed[2]
        if pos is None:
            continue
        pos = int(pos)
        if pos not in order:
            order.append(pos)
        if w:
            wt[pos] = w
        if mut:
            muts.setdefault(pos, []).append(mut)
    marked = {p_: m[0] for p_, m in muts.items() if len(set(m)) == 1}
    return order, wt, marked


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


def sample_means(means: np.ndarray, n: int = SYN_TEST_N,
                 seed: int = SYN_DISPLAY_SEED) -> np.ndarray:
    """The subset of synonymous variant means the vs-synonymous test is run against.

    Keeping the reference to the same twenty the figure draws means the picture and the
    p-value are about one set of variants rather than two. It costs power, and the cost
    is not small: an empirical p cannot go below 1/(n+1), so with twenty the floor is
    0.048 and a variant can at best be called marginally significant. The floor is
    reported on the figure, in `p_floor` on every row of the table, and in the caveats.
    """
    means = np.asarray(means, float)
    means = means[np.isfinite(means)]
    if n is None or len(means) <= n:
        return means
    return means[np.random.default_rng(seed).choice(len(means), n, replace=False)]


def sample_for_display(syn: pd.DataFrame, n: int = SYN_DISPLAY_N) -> pd.DataFrame:
    """A fixed random subset of synonymous VARIANTS (all their replicates) to draw.

    The vs-synonymous test is run against the same number (SYN_TEST_N), so the dots
    and the p-value describe one set of variants. These are drawn from the replicate
    table and the test's are drawn from the per-variant means, with the same seed and
    the same count but not necessarily the same variants - the figure says how many of
    each.
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
        sd_a = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
        rows.append({"comparison": f"{v} vs synonymous", "kind": "vs_synonymous", "a": v, "b": SYN_LABEL,
                     "n_a": len(x), "n_b": len(syn_means), "mean_a": m, "mean_b": centre,
                     "sd_a": sd_a, "se_a": sd_a / np.sqrt(len(x)) if len(x) else np.nan,
                     "sd_b": syn_sd,
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
                     "sd_a": float(np.std(xa, ddof=1)), "sd_b": float(np.std(xb, ddof=1)),
                     "se_a": float(np.std(xa, ddof=1) / np.sqrt(len(xa))),
                     "se_b": float(np.std(xb, ddof=1) / np.sqrt(len(xb))),
                     "difference": float(np.mean(xa) - np.mean(xb)), "z_vs_syn_variants": np.nan,
                     "p": float(t.pvalue),
                     "min_detectable_difference": MDE_D * sd})
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = bh_fdr(t.p.fillna(1.0))
    return t


def estimates(mut: pd.DataFrame, syn_means: np.ndarray, st: pd.DataFrame) -> pd.DataFrame:
    """One row per named variant: the value, its error, and its interval.

    The stats table is organised by comparison, which is the right shape for reading and
    the wrong shape for analysis - a variant appears in several rows and its error does
    not appear at all. This is the other shape: one row per variant, the normalised
    fitness and the uncertainty on it, ready to be read straight into something else.

    The error is the standard error of the replicate mean. With three replicates that is
    a two-degree-of-freedom estimate, so the interval uses t rather than 1.96, and the
    column saying which is which travels with the numbers rather than living in a caption.
    """
    centre = float(np.median(syn_means)) if len(syn_means) else 0.0
    vs = st[st.kind == "vs_synonymous"].set_index("a") if len(st) else pd.DataFrame()
    rows = []
    for v, g in mut.groupby("variant", sort=False):
        x = g.value.to_numpy(float)
        x = x[np.isfinite(x)]
        n = len(x)
        m = float(np.mean(x)) if n else np.nan
        sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        half = float(ss.t.ppf(0.975, n - 1) * se) if n > 1 and np.isfinite(se) else np.nan
        pr = parse_substitution(v)
        r = vs.loc[v] if len(vs) and v in vs.index else None
        rows.append({"variant": v,
                     "wt": pr[0] if pr else "", "pos": pr[1] if pr else np.nan,
                     "mut": pr[2] if pr else "",
                     "n_replicates": n, "score_z": m, "sd": sd, "se": se,
                     "error_kind": "se of the replicate mean",
                     "ci95_lo": m - half, "ci95_hi": m + half,
                     "synonymous_median": centre, "difference": m - centre,
                     "z_vs_syn_variants": float(r.z_vs_syn_variants) if r is not None else np.nan,
                     "p": float(r.p) if r is not None else np.nan,
                     "q": float(r.q) if r is not None and "q" in r else np.nan})
    return pd.DataFrame(rows)


def estimates_single(named: pd.DataFrame, st: pd.DataFrame,
                     syn_all: np.ndarray) -> pd.DataFrame:
    """The same table where the dataset gives one estimate and its sigma, not replicates.

    The error is then the assay's own sigma carried through the normalisation, not
    replicate scatter, and the column says so - the two are not interchangeable and a
    table that called both "se" without comment would invite them to be averaged.
    """
    centre = float(np.median(syn_all)) if len(syn_all) else 0.0
    keyed = st.set_index("variant") if len(st) else pd.DataFrame()
    rows = []
    for r in named.itertuples():
        se = float(r.se) if np.isfinite(r.se) else np.nan
        half = 1.96 * se if np.isfinite(se) else np.nan
        q = keyed.loc[r.variant] if len(keyed) and r.variant in keyed.index else None
        rows.append({"variant": r.variant, "wt": r.wt, "pos": r.pos, "mut": r.mut,
                     "n_replicates": 1, "score_z": float(r.score_z), "sd": np.nan,
                     "se": se, "error_kind": "sigma from the dataset, rescaled",
                     "ci95_lo": r.score_z - half, "ci95_hi": r.score_z + half,
                     "synonymous_median": centre,
                     "difference": float(r.score_z) - centre,
                     "z_vs_syn_variants": float(q.z_vs_synonymous) if q is not None else np.nan,
                     "p": float(q.p) if q is not None else np.nan,
                     "q": float(q.q) if q is not None and "q" in q else np.nan})
    return pd.DataFrame(rows)


def figure(cfg, mut: pd.DataFrame, syn: pd.DataFrame, syn_means: np.ndarray,
           st: pd.DataFrame, figdir):
    syn_shown = sample_for_display(syn)
    n_shown = int(syn_shown.syn_index.nunique()) if "syn_index" in syn_shown else 0
    order = [SYN_LABEL] + list(dict.fromkeys(mut.variant))
    # the canvas is kept near-square on purpose: MPDMS_SQUARE sets the axes box to a
    # square at save time, and in a wide figure that leaves a small square plot adrift
    # in the middle with every annotation overlapping it
    w = 1.35 * len(order) + 2.6
    fig, ax = plt.subplots(figsize=(w, max(4.6, 0.88 * w)), layout="constrained")
    rng = np.random.default_rng(20)
    syn_vals = syn.value.to_numpy()
    # the shaded band is the null the vs-synonymous test uses, so it must be the spread of
    # synonymous variant means - not of individual measurements, which is wider
    sd = float(np.std(syn_means, ddof=1)) if len(syn_means) > 1 else 0.0
    ax.axhspan(-sd, sd, color=SYN_COLOR, alpha=0.10, lw=0, zorder=0)
    ax.axhline(0, color=SYN_COLOR, lw=0.8, zorder=1)
    ax.axhline(-1, color="#D62728", lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.text(len(order) - 0.45, -1, "nonsense median", fontsize=6, color="#D62728", va="bottom", ha="right")

    # synonymous is a distribution over many variants, so it gets a violin; each mutant
    # is three replicates, where a violin would be a lie - those stay as the raw points
    # violin over every synonymous variant for the shape; the 20 sampled ones drawn on
    # top so individual measurements stay visible without burying the mutant columns
    P.violin_box(ax, {SYN_LABEL: syn.value.to_numpy()}, colors={SYN_LABEL: SYN_FILL},
                 points=0, width=0.8, show_medians=False)
    sv = syn_shown.value.to_numpy()
    ax.scatter(rng.uniform(-0.12, 0.12, len(sv)), sv, s=5, color=SYN_COLOR, alpha=0.75, lw=0, zorder=4)
    for i, v in enumerate(order):
        if v == SYN_LABEL:
            continue
        vals = mut.query("variant == @v").value.to_numpy()
        ax.scatter(i + rng.uniform(-0.09, 0.09, len(vals)), vals, s=30, color=MUT_COLOR,
                   lw=0.5, edgecolor="white", zorder=4)
        if len(vals):
            m = float(np.mean(vals))
            ax.plot([i - 0.22, i + 0.22], [m, m], color=P.INK, lw=1.4, zorder=5)
            if len(vals) > 1:
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                ax.plot([i, i], [m - se, m + se], color=P.INK, lw=0.9, zorder=5)
            ax.annotate(f"{m:.2f}", (i, m), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=6.5, color=P.INK, zorder=6)

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
        import textwrap
        # wrapped to the figure's own width: one long line was running off the page as
        # soon as the panel was squared
        body = textwrap.fill("Between variants (Welch, 3 vs 3): " + " | ".join(bits),
                             width=int(12 * fig.get_figwidth()))
        ax.text(0.5, -0.185, body,
                transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=P.INK)
        ax.text(0.5, -0.185 - 0.035 * (1 + body.count(chr(10))),
                f"at n = 3 per side only a difference of ≈ {mde:.2f} is detectable at 80% "
                             "power, so a null here is underpowered, not equivalent",
                transform=ax.transAxes, ha="center", va="top", fontsize=6, color=P.MUTED, style="italic")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(lo - 0.40, max(0.45, (mut.value.max() if len(mut) else 0) + 0.3))
    ax.set_ylabel("Normalised fitness (synonymous = 0, nonsense = −1)")
    floor = 1 / (len(syn_means) + 1) if len(syn_means) else float("nan")
    ax.set_title(f"(a) Replicate fitness per variant; blue band = ±1 SD of "
                 f"{len(syn_means)} synonymous variant means, which is also the "
                 f"reference the test uses (empirical p cannot go below {floor:.3f}); "
                 f"violin = all synonymous variants, dots = {n_shown} drawn at random",
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
    fig, ax = plt.subplots(figsize=(6.6, 3.6), layout="constrained")
    P.density_by_class(ax, {"missense": miss.score_z.to_numpy(),
                            "synonymous": syn.to_numpy(), "nonsense": non.to_numpy()})
    for v, c in ((syn, SYN_COLOR), (non, "#D62728")):
        if len(v):
            ax.axvline(float(v.median()), color=c, lw=0.7, ls=(0, (3, 3)), zorder=1)

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
    ax.set_title("(b) Named variants in the missense fitness distribution; "
                 "percentile = % of missense variants less fit", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5)
    P.title(cfg=cfg, fig=fig, text="A20b named variants in the missense distribution")
    return P.save(fig, figdir, "a20b_missense_percentile", cfg, "A20")



# ------------------------------------------------- single-estimate datasets
def collect_single(df: pd.DataFrame, labels: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """The named variants when the table carries one estimate and its error, not replicates.

    DiMSum writes a fitness and a sigma per variant rather than per-replicate columns, so
    there are no three values to draw. The error bar is that sigma carried through the same
    linear rescaling as the score, which is what `se` holds.
    """
    rows, missing = [], []
    for lab in labels:
        pr = parse_substitution(lab)
        if pr is None:
            missing.append(f"{lab}: not a point substitution")
            continue
        wt, pos, mut = pr
        hit = df[(df.pos == pos) & (df.mut == mut) & (df.vclass == "missense")]
        if not len(hit):
            missing.append(f"{lab}: not measured in this dataset")
            continue
        r = hit.iloc[0]
        if str(r.wt).upper() != wt:
            missing.append(f"{lab}: dataset has {r.wt}{pos}, not {wt}{pos} - numbering mismatch")
            continue
        if not bool(r.pass_filter):
            missing.append(f"{lab}: present but fails the read/replicate filter")
            continue
        rows.append({"variant": lab, "pos": pos, "wt": wt, "mut": mut,
                     "score_z": float(r.score_z),
                     "se": float(r.se) if "se" in hit.columns and np.isfinite(r.se) else np.nan})
    return pd.DataFrame(rows), missing


def site_distributions(df: pd.DataFrame, positions: list[int]) -> dict:
    """Every substitution measured at each named position: the distribution behind the point."""
    d = df[df.pass_filter & (df.vclass == "missense")]
    return {p: d.loc[d.pos == p, "score_z"].dropna().to_numpy() for p in positions}


def stats_single_wt(named: pd.DataFrame) -> pd.DataFrame:
    """Each variant against the wild type at 0, using its own standard error.

    Without synonymous variants there is no empirical picture of what "no effect" looks
    like, so the only test available is whether the variant's own interval excludes zero.
    That is a weaker claim: it uses measurement error alone and knows nothing about how
    much a variant that truly does nothing still moves between wells.
    """
    rows = []
    for r in named.itertuples():
        z = r.score_z / r.se if np.isfinite(r.se) and r.se > 0 else np.nan
        rows.append({"variant": r.variant, "score_z": r.score_z, "se": r.se,
                     "reference": "wild type (0)", "z_vs_wild_type": z,
                     "p": float(2 * ss.norm.sf(abs(z))) if np.isfinite(z) else np.nan})
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = bh_fdr(t.p)
    return t


def figure_single(cfg, named: pd.DataFrame, syn, sites: dict, figdir):
    """(a) each named variant against the synonymous spread; (b) its whole position."""
    order = named.variant.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(4.2 + 0.55 * len(order), 3.6),
                             layout="constrained", gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    has_syn = syn is not None and len(syn) >= 5
    if has_syn:
        # the synonymous variants are the reference: hundreds of measurements of "no
        # effect". Its median is 0 by construction and the label would land on the
        # two-line tick label, so it is left off here.
        P.violin_box(ax, {SYN_LABEL: syn}, colors={SYN_LABEL: SYN_COLOR}, width=0.6,
                     show_medians=False)
    x = np.arange(1 if has_syn else 0, len(order) + (1 if has_syn else 0))
    ax.errorbar(x, named.score_z, yerr=1.96 * named.se.fillna(0), fmt="o", ms=5,
                color=MUT_COLOR, ecolor=P.INK, elinewidth=0.9, capsize=2.5, zorder=4)
    if has_syn:
        lo, hi = np.percentile(syn, [2.5, 97.5])
        ax.axhspan(lo, hi, color=SYN_COLOR, alpha=0.12, lw=0, zorder=0)
    ax.axhline(0, color=SYN_COLOR, lw=1.1,
               label="wild type" if not has_syn else None)
    ax.axhline(-1, color="#C0392B", lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(np.arange(0, len(order) + (1 if has_syn else 0)))
    ax.set_xticklabels(([SYN_LABEL] if has_syn else []) + order,
                       rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Normalised fitness (wild type = 0, nonsense = −1)"
                  if not has_syn else
                  "Normalised fitness (synonymous = 0, nonsense = −1)")
    ax.set_title("(a) Named variants vs "
                 + ("the synonymous spread" if has_syn else "wild type")
                 + ", ±95% CI from sigma", loc="left", fontsize=8.5)
    if not has_syn:
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")

    ax = axes[1]
    groups = {SYN_LABEL: syn} if has_syn else {}
    for lab, pos in zip(order, named.pos):
        v = sites.get(pos, np.array([]))
        if len(v) >= 3:
            groups[f"{named.loc[named.variant == lab, 'wt'].iloc[0]}{pos}"] = v
    cols = {k: (SYN_COLOR if k == SYN_LABEL else "#B8912F") for k in groups}
    P.violin_box(ax, groups, colors=cols, width=0.7)
    for k, (lab, pos) in enumerate(zip(order, named.pos), start=1):
        key = f"{named.loc[named.variant == lab, 'wt'].iloc[0]}{pos}"
        if key in groups:
            j = list(groups).index(key)
            ax.plot([j], [named.loc[named.variant == lab, "score_z"].iloc[0]], "o", ms=5,
                    color=HILITE, zorder=6, markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0, color=SYN_COLOR, lw=0.8)
    ax.axhline(-1, color="#C0392B", lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups), rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Normalised fitness")
    ax.set_title("(b) Every substitution at each named position; gold = the alanine",
                 loc="left", fontsize=8.5)
    P.title(cfg=cfg, fig=fig, text="A20 named variants (single-estimate dataset)")
    return P.save(fig, figdir, "a20_variant_panel", cfg, "A20")


def stats_single(named: pd.DataFrame, syn: np.ndarray) -> pd.DataFrame:
    """Each variant against the synonymous spread - the comparison that is powered."""
    rows = []
    mu, sd = float(np.mean(syn)), float(np.std(syn, ddof=1))
    for r in named.itertuples():
        z = (r.score_z - mu) / sd if sd > 0 else np.nan
        rows.append({"variant": r.variant, "score_z": r.score_z, "se": r.se,
                     "syn_mean": mu, "syn_sd": sd, "z_vs_synonymous": z,
                     "percentile_in_synonymous": float((syn < r.score_z).mean() * 100),
                     "p": float(2 * ss.norm.sf(abs(z))) if np.isfinite(z) else np.nan})
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = bh_fdr(t.p)
    return t


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    labels = wanted_variants(cfg)
    if not labels:
        return skipped("a20", cfg, f"no variants named for {cfg.get_path('protein.gene', cfg.id)} "
                                   "(set report.variant_panel, or add measured entries to _literature.yaml)")
    syn_all = df[(df.vclass == "synonymous") & df.pass_filter].score_z.dropna().to_numpy()
    if not replicate_cols(df):
        # DiMSum output carries one fitness and one sigma per variant; draw that, rather
        # than skipping an analysis the dataset can perfectly well support
        named, missing = collect_single(df, labels)
        if not len(named):
            return skipped("a20", cfg, "none of the named variants are usable: "
                           + "; ".join(missing))
        # No synonymous variants is a normal library design, not a reason to skip. The
        # reference is then the wild type itself, which normalisation has put at 0, and
        # the spread of "no effect" has to come from somewhere else - here the variants
        # whose intervals cover 0.
        ref_syn = syn_all if len(syn_all) >= 5 else None
        sites = site_distributions(df, named.pos.tolist())
        st = stats_single(named, syn_all) if ref_syn is not None else stats_single_wt(named)
        st.to_csv(tabdir / "a20_variant_stats.csv", index=False)
        named.to_csv(tabdir / "a20_variant_values.csv", index=False)
        est = estimates_single(named, st, syn_all)
        est.to_csv(tabdir / "a20_variant_estimates.csv", index=False)
        figs = figure_single(cfg, named, ref_syn, sites, figdir)
        pc = percentiles(df, labels)
        if len(pc):
            figs = figs + percentile_figure(cfg, df, pc, figdir)
            pc.to_csv(tabdir / "a20b_percentiles.csv", index=False)
        sig = st[st.q < 0.05]
        ref_name = "synonymous" if ref_syn is not None else "wild type"
        head = (f"{len(named)} named variants, one estimate each (no replicate columns); "
                f"{len(sig)} differ from {ref_name} at q < 0.05"
                + (": " + ", ".join(sig.variant) if len(sig) else ""))
        return result("a20", cfg, head,
                      {"n_named": int(len(named)), "n_significant": int(len(sig)),
                       "has_replicates": False,
                       "variants": st.set_index("variant").score_z.to_dict()},
                      [f"not plotted - {m}" for m in missing]
                      + ["one estimate per variant, so the error bar is the table's sigma, "
                         "not replicate scatter"]
                      + ([] if ref_syn is not None else
                         ["no synonymous variants in this library: the reference is the wild "
                          "type at 0 and each variant is tested against its own sigma, which "
                          "is a weaker comparison than a synonymous spread would give"]),
                      figs, [tabdir / "a20_variant_stats.csv",
                             tabdir / "a20_variant_estimates.csv"])

    mut, syn, missing = collect(df, cfg, labels)
    if not len(mut):
        return skipped("a20", cfg, "none of the named variants are usable: " + "; ".join(missing))

    all_syn_means = df[(df.vclass == "synonymous")
                       & df.pass_filter].score_z.dropna().to_numpy()
    syn_means = sample_means(all_syn_means, SYN_TEST_N)
    st = stats_table(mut, syn, syn_means)
    tp = tabdir / "a20_variant_stats.csv"
    st.to_csv(tp, index=False)
    dp = tabdir / "a20_variant_replicates.csv"
    pd.concat([mut, syn.assign(variant=SYN_LABEL)]).to_csv(dp, index=False)
    est = estimates(mut, syn_means, st)
    ep = tabdir / "a20_variant_estimates.csv"
    est.to_csv(ep, index=False)
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
    caveats.append(f"the vs-synonymous reference was cut to {len(syn_means)} of "
                   f"{len(all_syn_means)} synonymous variant means (SYN_TEST_N), which "
                   "is a deliberate loss of power, not a property of the data")
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
                  caveats, figs, [tp, ep, dp] + ([tabdir / "a20b_percentiles.csv"] if len(pc) else []))
