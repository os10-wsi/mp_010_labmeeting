"""`python -m mpdms dual data/raw/SLC22A1/slc22a1_s1.xlsx --id SLC22A1`

Function against abundance, in a protein measured both ways.

A variant can lose function for two different reasons: there is less protein, or the
protein that is there does not work. Measuring abundance and function in the same
library separates them, and the separation is the point. Regress function on abundance
and most variants fall on one curve - less protein, less function. The residuals are
what is interesting: a position whose substitutions lose far more function than their
abundance accounts for is a position doing a job, not holding the fold together. That
is the signature of a substrate site, a gating residue or a catalytic one.

    d01  the assays against each other, with the anchors, and the no-selection control
    d02  LOESS of function on abundance, and the sites whose residuals are real

This is A15's logic with a measured abundance axis in place of an ESM-1v prediction,
and it is the stronger version of the same argument: A15 asks whether a model expected
more constraint than the assay found, this asks whether the cell made the protein and
it still did not work.

Both assays are put on the pipeline's usual two-point scale,

    score_z = (x - median(synonymous)) / (median(synonymous) - median(dead))

so 0 is wild type and -1 is a dead variant in BOTH assays. The sign is handled by that
formula rather than by hand: in a survival selection where a broken transporter lets the
cell live, the dead anchor sits ABOVE the wild type, the denominator is negative, and
the rescaled score comes out the right way up with nothing to remember.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .analyses.a15_esm_functional import loess_fit
from .config import REPO_ROOT
from .stats import bh_fdr

OUT = REPO_ROOT / "outputs" / "_dual"
MISSENSE = re.compile(r"^p\.\(([A-Z])(\d+)([A-Z])\)$")
DELETION = re.compile(r"^p\.\(([A-Z])(\d+)del\)$")
MIN_VARIANTS = 5          # at a position, before its residuals get a test
Q_SITE = 0.05             # BH-FDR, for calling a site
Z_SITE = -1.5             # and how far below the fit its median must sit
LOESS_FRAC = 0.3


# --------------------------------------------------------------------- loading
def parse_hgvs(s: pd.Series) -> pd.DataFrame:
    """Split `p.(A107C)` into wild type, position and mutant, and label each row.

    Deletions are kept, not dropped: in a protein with no nonsense variants they are
    the only loss-of-function anchor the library contains, and without an anchor the
    two assays cannot be put on a common scale.
    """
    s = s.astype(str)
    m = s.str.extract(MISSENSE)
    d = s.str.extract(DELETION)
    out = pd.DataFrame({"wt": m[0], "pos": pd.to_numeric(m[1], errors="coerce"),
                        "mut": m[2]})
    out["kind"] = np.where(out.wt.isna(), "other",
                           np.where(out.wt == out.mut, "synonymous", "missense"))
    isdel = d[0].notna()
    out.loc[isdel, ["wt", "pos", "mut", "kind"]] = np.column_stack(
        [d[0][isdel], pd.to_numeric(d[1][isdel], errors="coerce"),
         np.full(int(isdel.sum()), "del"), np.full(int(isdel.sum()), "deletion")])
    out["pos"] = pd.to_numeric(out.pos, errors="coerce")
    return out


def anchors(d: pd.DataFrame, col: str) -> dict:
    """The wild-type and dead medians for one assay column."""
    syn = float(d.loc[d.kind == "synonymous", col].median())
    dead = float(d.loc[d.kind == "deletion", col].median())
    return {"synonymous": syn, "dead": dead, "scale": syn - dead}


def rescale(d: pd.DataFrame, col: str, a: dict) -> pd.Series:
    """The pipeline's two-point scale: 0 at wild type, -1 at a dead variant."""
    if not np.isfinite(a["scale"]) or a["scale"] == 0:
        return pd.Series(np.nan, index=d.index)
    return (d[col] - a["synonymous"]) / a["scale"]


def read_table(path: Path, abundance: str, function: str,
               control: str | None = None, sheet=0) -> tuple[pd.DataFrame, dict]:
    """A dual-assay supplementary table, as one tidy frame on the common scale."""
    raw = (pd.read_excel(path, sheet_name=sheet) if path.suffix.lower() in
           (".xlsx", ".xls", ".xlsm") else pd.read_csv(path, sep=None, engine="python"))
    hg = next((c for c in raw.columns if c.lower() in ("hgvs", "variant", "hgvs_pro")),
              None)
    if hg is None:
        raise SystemExit(f"{path}: no hgvs/variant column; found {list(raw.columns)}")
    need = [c for c in (abundance, function, control) if c]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise SystemExit(f"{path}: no column(s) {missing}; found {list(raw.columns)}")
    d = pd.concat([raw, parse_hgvs(raw[hg])], axis=1)
    info = {}
    for lab, col in (("abundance", abundance), ("function", function),
                     ("control", control)):
        if not col:
            continue
        a = anchors(d, col)
        if lab == "control" and "function" in info:
            # the control is the same selection with nothing selecting, so its own
            # dead-to-wild-type range is noise. Dividing by that range would magnify
            # the noise until the null control looked like a result; it is centred on
            # its own wild type and scaled by the FUNCTION assay's range instead, so
            # "no separation" is drawn as no separation, in comparable units
            a = {**a, "scale": info["function"]["scale"],
                 "own_dead": a["dead"], "own_scale": a["synonymous"] - a["dead"]}
        d[lab] = rescale(d, col, a)
        info[lab] = {"column": col, **a}
        se = f"{col.rsplit('_', 1)[0]}_SE"
        if se in raw.columns and np.isfinite(a["scale"]) and a["scale"]:
            d[f"{lab}_se"] = raw[se].abs() / abs(a["scale"])
    return d, info


# ------------------------------------------------- d01 the assays against each other
def d01_assays(d: pd.DataFrame, info: dict, name: str, outdir: Path) -> dict:
    """Each assay against each other, with the anchors drawn and the control beside it.

    The control panel is not decoration. A selection assay with no selection applied
    should separate nothing, and seeing that its dead variants sit on top of its
    synonymous ones is what licenses reading the other panel as biology rather than as
    a batch effect.
    """
    has_ctl = "control" in d.columns and d.control.notna().any()
    cols = 2 if has_ctl else 1
    fig, axes = plt.subplots(1, cols, figsize=(3.6 * cols + 0.6, 3.4), squeeze=False,
                             layout="constrained")
    rows = []
    panels = [("function", "function (0 = wild type, −1 = dead)")]
    if has_ctl:
        panels.append(("control", "no-selection control, same scale"))
    for j, (ycol, ylab) in enumerate(panels):
        ax = axes[0][j]
        mis = d[(d.kind == "missense")].dropna(subset=["abundance", ycol])
        ax.scatter(mis.abundance, mis[ycol], s=5, color="#4D4D4D", lw=0, alpha=0.3,
                   rasterized=True)
        for kind, col, lab in (("synonymous", "#2166AC", "synonymous"),
                               ("deletion", "#B2182B", "deletion")):
            s = d[(d.kind == kind)].dropna(subset=["abundance", ycol])
            if len(s):
                ax.scatter(s.abundance, s[ycol], s=9, color=col, lw=0.2,
                           edgecolor=P.INK, alpha=0.8, label=f"{lab} (n = {len(s):,})")
        r = (ss.spearmanr(mis.abundance, mis[ycol]) if len(mis) > 10
             else (np.nan, np.nan))
        rows.append({"protein": name, "y": ycol, "n": int(len(mis)),
                     "spearman": float(r[0]), "p": float(r[1])})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axhline(-1, color="#B2182B", lw=0.6, ls=(0, (1, 2)))
        ax.axvline(0, color=P.MUTED, lw=0.6)
        ax.axvline(-1, color="#B2182B", lw=0.6, ls=(0, (1, 2)))
        ax.set_xlabel("abundance (0 = wild type, −1 = dead)", fontsize=7.5)
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.set_title(f"ρ = {r[0]:+.2f}" if np.isfinite(r[0]) else "", loc="left",
                     fontsize=8)
        if j == 0:
            ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.suptitle(f"{name}: both assays on one scale, anchored on synonymous variants "
                 f"and in-frame deletions.", fontsize=7.5, x=0.01, ha="left")
    P.save(fig, outdir, "d01_assays", None, "D01")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "d01_assay_correlations.csv", index=False)
    return {"status": "ok", "correlations": t.to_dict("records"),
            "anchors": info,
            "headline": "; ".join(f"{r.y} vs abundance ρ {r.spearman:+.2f} (n {r.n:,})"
                                  for r in t.itertuples())}


# ----------------------------------------- d02 the residual, and the sites that carry it
def residuals(d: pd.DataFrame, frac: float = LOESS_FRAC) -> tuple[pd.DataFrame, float]:
    """Function minus what this variant's abundance predicts, in MAD-scaled units."""
    m = d[(d.kind == "missense")].dropna(subset=["abundance", "function"]).copy()
    fit = loess_fit(m.abundance.to_numpy(float), m.function.to_numpy(float), frac=frac)
    m["expected"] = fit(m.abundance.to_numpy(float))
    m["residual"] = m.function - m.expected
    scale = float(1.4826 * ss.median_abs_deviation(m.residual))
    m["z"] = m.residual / (scale or 1.0)
    return m, scale


def site_table(m: pd.DataFrame, q_site: float = Q_SITE,
               z_site: float = Z_SITE) -> pd.DataFrame:
    """One row per position: is its function systematically below the fit?

    A one-sided signed-rank test on the position's own variant residuals, so a site is
    called on the agreement of its substitutions rather than on one extreme variant,
    and BH across positions because every position is tested.
    """
    rows = []
    for pos, g in m.groupby("pos"):
        if len(g) < MIN_VARIANTS:
            continue
        p = (float(ss.wilcoxon(g.residual, alternative="less").pvalue)
             if (g.residual != 0).any() else 1.0)
        rows.append({"pos": int(pos), "wt": g.wt.iloc[0], "n_variants": int(len(g)),
                     "median_abundance": float(g.abundance.median()),
                     "median_function": float(g.function.median()),
                     "median_residual": float(g.residual.median()),
                     "median_z": float(g.z.median()), "p": p})
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    t["q"] = bh_fdr(t.p)
    t["functional"] = (t.q < q_site) & (t.median_z <= z_site)
    return t.sort_values("median_z").reset_index(drop=True)


def d02_functional_sites(d: pd.DataFrame, name: str, outdir: Path,
                         frac: float = LOESS_FRAC, q_site: float = Q_SITE,
                         z_site: float = Z_SITE, label_n: int = 18) -> dict:
    """The LOESS fit, the residuals, and the positions whose residuals are real."""
    m, scale = residuals(d, frac)
    sites = site_table(m, q_site, z_site)
    func = sites[sites.functional] if len(sites) else sites
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), layout="constrained",
                             gridspec_kw={"height_ratios": [1.25, 1.0]})

    ax = axes[0]
    ax.scatter(m.abundance, m.function, s=5, color="#BFBFBF", lw=0, alpha=0.5,
               rasterized=True)
    hot = m[m.pos.isin(set(func.pos))] if len(func) else m.iloc[:0]
    ax.scatter(hot.abundance, hot.function, s=9, color="#B2182B", lw=0.2,
               edgecolor=P.INK, alpha=0.85,
               label=f"at a called site (n = {len(hot):,} variants)")
    xs = np.linspace(float(m.abundance.quantile(0.005)),
                     float(m.abundance.quantile(0.995)), 240)
    ax.plot(xs, loess_fit(m.abundance.to_numpy(float), m.function.to_numpy(float),
                          frac)(xs), color=P.INK, lw=1.8, label="LOESS fit")
    ax.axhline(0, color=P.MUTED, lw=0.6)
    ax.axvline(0, color=P.MUTED, lw=0.6)
    ax.set_xlabel("abundance (0 = wild type, −1 = dead)", fontsize=8)
    ax.set_ylabel("function (0 = wild type, −1 = dead)", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title("(a) every missense variant, and what its abundance predicts",
                 loc="left", fontsize=8)

    ax = axes[1]
    if len(sites):
        ax.scatter(sites.pos, sites.median_z, s=10, color="#BFBFBF", lw=0)
        if len(func):
            ax.scatter(func.pos, func.median_z, s=18, color="#B2182B", lw=0.3,
                       edgecolor=P.INK, zorder=5)
            for r in func.nsmallest(label_n, "median_z").itertuples():
                ax.annotate(f"{r.wt}{r.pos}", (r.pos, r.median_z), fontsize=5.5,
                            xytext=(0, -7), textcoords="offset points", ha="center",
                            color="#B2182B")
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axhline(z_site, color="#B2182B", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel("residue", fontsize=8)
    ax.set_ylabel("site median residual z", fontsize=8)
    ax.set_title(f"(b) {len(func)} of {len(sites)} sites lose more function than their "
                 f"abundance accounts for (q < {q_site}, z ≤ {z_site})",
                 loc="left", fontsize=8)
    fig.suptitle(f"{name}: a negative residual is function lost while the protein is "
                 f"still made - the signature of a site that does a job rather than "
                 f"one that holds the fold together. Residual scale (1.4826 × MAD) = "
                 f"{scale:.3f}.", fontsize=7.5, x=0.01, ha="left")
    P.save(fig, outdir, "d02_functional_sites", None, "D02")
    plt.close(fig)

    m.to_csv(outdir / "d02_variant_residuals.csv", index=False)
    if len(sites):
        sites.to_csv(outdir / "d02_sites.csv", index=False)
    return {"status": "ok", "n_sites": int(len(sites)), "n_functional": int(len(func)),
            "residual_scale": scale,
            "functional_sites": [f"{r.wt}{r.pos}" for r in func.itertuples()],
            "headline": f"{len(func)} of {len(sites)} positions called functional "
                        f"(q < {q_site}, median z ≤ {z_site})"}


ANALYSES = {"d01": "the two assays against each other, with the control",
            "d02": "LOESS of function on abundance, and the sites off it"}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "divide by zero|Degrees of freedom|"
                                              "All-NaN|tight_layout).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms dual",
        description="Function against abundance in a protein measured both ways: the "
                    "residual is the functional site.")
    ap.add_argument("table", help="supplementary table (.xlsx or delimited text)")
    ap.add_argument("--id", default="protein", help="name for the figures")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--abundance", required=True, help="the abundance score column")
    ap.add_argument("--function", required=True, help="the function score column")
    ap.add_argument("--control", default=None,
                    help="a no-selection control column, plotted beside the function "
                         "panel as the negative control")
    ap.add_argument("--frac", type=float, default=LOESS_FRAC)
    ap.add_argument("--q", type=float, default=Q_SITE)
    ap.add_argument("--z", type=float, default=Z_SITE,
                    help="how far below the fit a site's median must sit (negative)")
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)
    keys = [k.strip() for k in (a.only.split(",") if a.only else ANALYSES)]
    bad = [k for k in keys if k not in ANALYSES]
    if bad:
        raise SystemExit(f"unknown analysis {bad}; available: {', '.join(ANALYSES)}")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sheet = int(a.sheet) if str(a.sheet).isdigit() else a.sheet
    d, info = read_table(Path(a.table), a.abundance, a.function, a.control, sheet)
    counts = d.kind.value_counts()
    print(f"  {a.id:<12} " + "  ".join(f"{k} {n:,}" for k, n in counts.items()),
          flush=True)
    for lab, v in info.items():
        extra = (f"  (own range {abs(v['own_scale']):.3f}, drawn on the function "
                 f"assay's scale)" if "own_scale" in v else "")
        print(f"    {lab:<10} {v['column']:<16} wild type {v['synonymous']:+.3f}  "
              f"dead {v['dead']:+.3f}  range {abs(v['scale']):.3f}{extra}"
              + ("   (dead ABOVE wild type: a survival selection, rescaled)"
                 if v["scale"] < 0 and "own_scale" not in v else ""), flush=True)
    d.to_csv(outdir / "variants.csv", index=False)
    calls = {"d01": lambda: d01_assays(d, info, a.id, outdir),
             "d02": lambda: d02_functional_sites(d, a.id, outdir, a.frac, a.q, a.z)}
    out = {}
    print()
    for k in keys:
        try:
            r = calls[k]()
        except Exception as e:
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        out[k] = r
        print(f"  {k}  {r.get('status'):<8} {r.get('headline') or r.get('reason', '')}",
              flush=True)
    (outdir / "dual.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
