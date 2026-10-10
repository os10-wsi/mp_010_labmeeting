"""`python -m mpdms panel configs/*.yaml --only a01,a05,a05b,a06,a15,a22a,a22b`

One figure per analysis, with one panel per protein, so eight datasets can be read side
by side instead of eight PDFs apart. Each panel is drawn by the analysis's own code
wherever that code already computes the quantity, so a panel here and the single-protein
figure cannot drift apart.

Axis limits are shared across panels of a figure: comparing eight panels that each
auto-scaled to their own data is the fastest way to read a difference that is not there.
A protein that cannot supply an analysis (no ESM table, no topology) gets a panel saying
so rather than being dropped, so the grid keeps the same proteins in the same places.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P
from .config import REPO_ROOT, load_config
from .io import load_dataset

OUT = REPO_ROOT / "outputs" / "_panels"


class NotAvailable(Exception):
    """This protein cannot supply this panel; the reason is shown in its place."""


# --------------------------------------------------------------------- panels
def panel_a01(ax, df, cfg, shared):
    """Score distributions by variant class."""
    from .analyses.a01_dynamic_range import syn_bounds
    d = df[df.pass_filter]
    # no per-class n in the labels: one shared legend cannot carry eight proteins' counts
    groups = {P.CLASS_LABELS[c]: d.loc[d.vclass == c, "score_z"].dropna().to_numpy()
              for c in P.CLASS_ORDER}
    groups = {k: v for k, v in groups.items() if len(v) > 2}
    if not groups:
        raise NotAvailable("no scored variants")
    colors = {P.CLASS_LABELS[c]: P.CLASS_COLORS[c] for c in P.CLASS_ORDER}
    P.density_by_class(ax, groups, colors=colors, counts=False)
    lo, hi, sd = syn_bounds(df)
    if np.isfinite(lo):
        ax.axvspan(lo, hi, color=P.CLASS_COLORS["synonymous"], alpha=0.12, lw=0, zorder=0)
    ax.set_yticks([])
    ax.set_xlabel("Normalised fitness")
    ax.set_ylabel("Density")
    mis = d.loc[d.vclass == "missense", "score_z"].dropna()
    note = f"{float((mis < lo).mean()):.0%} deleterious" if np.isfinite(lo) and len(mis) else ""
    return {"note": note, "legend": ax, "share": "x"}


def panel_a05(ax, df, cfg, shared):
    """Standardised physicochemical coefficients, TM vs the rest."""
    from .analyses.a05_substitution_physchem import TERMS, fit
    from .analyses.base import missense
    from .annot import add_deltas
    d = add_deltas(missense(df))
    d["is_tm"] = d.seg_type == "TM"
    rows = []
    for name, sub in (("TM", d[d.is_tm]), ("non-TM", d[~d.is_tm])):
        m = fit(sub, "biological")
        if m is None:
            continue
        ci = m.conf_int()
        for t in TERMS:
            rows.append({"subset": name, "term": t, "coef": m.params[t],
                         "lo": ci.loc[t, 0], "hi": ci.loc[t, 1]})
    c = pd.DataFrame(rows)
    if not len(c):
        raise NotAvailable("too few variants to fit")
    y = np.arange(len(TERMS))
    for j, (name, col) in enumerate((("TM", "#1B4B80"), ("non-TM", "#9DB8DB"))):
        cc = c[c.subset == name].set_index("term").reindex(TERMS)
        if cc.coef.isna().all():
            continue
        ax.errorbar(cc.coef, y + (j - 0.5) * 0.26,
                    xerr=[cc.coef - cc.lo, cc.hi - cc.coef], fmt="o", ms=3.2,
                    color=col, lw=1, capsize=0, label=name)
    ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels(["Δ hydrophobicity", "Δ charge", "Δ volume", "Δ helix prop."], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Standardised coefficient (±95% CI)")
    return {"legend": ax, "share": "x", "ytick_shared": True}


def panel_a05b(ax, df, cfg, shared):
    """TM minus non-TM substitution matrix."""
    from .analyses.a05_substitution_physchem import MIN_CELL
    from .analyses.base import missense
    from .annot import HYDROPHOBICITY_ORDER
    order = list(HYDROPHOBICITY_ORDER)
    d = missense(df)
    d = d[d.seg_type.notna()]
    tm, rest = d[d.seg_type == "TM"], d[d.seg_type != "TM"]
    if not len(tm) or not len(rest):
        raise NotAvailable("no TM/non-TM split (topology missing)")
    mats = {}
    for lab, sub in (("TM", tm), ("rest", rest)):
        m = sub.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="mean")
        n = sub.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="size")
        mats[lab] = m.where(n >= MIN_CELL).reindex(index=order, columns=order)
    diff = (mats["TM"] - mats["rest"]).to_numpy(float)
    if not np.isfinite(diff).any():
        raise NotAvailable("no (wt, mut) cell occurs in both subsets")
    im = ax.imshow(np.ma.masked_invalid(diff), cmap=P.diverging_cmap(),
                   vmin=-shared["lim"], vmax=shared["lim"], interpolation="nearest")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=4, family="monospace")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=4, family="monospace")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Mutant", fontsize=6)
    ax.set_ylabel("Wild type", fontsize=6)
    return {"image": im, "note": f"{int(np.isfinite(diff).sum())} shared cells"}


def _a05b_limit(loaded):
    """One colour scale for every protein, else the panels cannot be compared."""
    from .analyses.a05_substitution_physchem import MIN_CELL
    from .analyses.base import missense
    from .annot import HYDROPHOBICITY_ORDER
    order = list(HYDROPHOBICITY_ORDER)
    best = []
    for cfg, df in loaded:
        d = missense(df)
        d = d[d.seg_type.notna()]
        tm, rest = d[d.seg_type == "TM"], d[d.seg_type != "TM"]
        if not len(tm) or not len(rest):
            continue
        ms = []
        for sub in (tm, rest):
            m = sub.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="mean")
            n = sub.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="size")
            ms.append(m.where(n >= MIN_CELL).reindex(index=order, columns=order))
        v = (ms[0] - ms[1]).to_numpy(float)
        if np.isfinite(v).any():
            best.append(np.nanpercentile(np.abs(v), 98))
    return {"lim": float(max(best)) if best else 1.0}


def panel_a06(ax, df, cfg, shared):
    """Charge introduced across the membrane, cytosolic end to lumenal end."""
    from .analyses.a06_charge_topology import ACID, BASE, FLANK, annotate
    from .analyses.base import missense
    from .annot import segments_df
    segs = segments_df(cfg)
    if segs.empty or not (segs.type == "TM").any():
        raise NotAvailable("no TM segments")
    d = missense(df)
    d = d[~d.wt.isin(list(ACID | BASE)) & d.mut.isin(list(ACID | BASE))].copy()
    if not len(d):
        raise NotAvailable("no charge-introducing variants")
    d["charge"] = np.where(d.mut.isin(list(ACID)), "acidic", "basic")
    a = annotate(d, segs)
    if a.empty:
        raise NotAvailable("no charge variants inside or beside a TM helix")
    # annotate() already orients x so 0 is the cytosolic end in both in_out and out_in
    # helices, so the orientations pool instead of needing a facet each
    bins = np.concatenate([np.linspace(-0.5, 0, 3), np.linspace(0, 1, 7)[1:],
                           np.linspace(1, 1.5, 3)[1:]])
    ax.axvspan(0, 1, color=P.TM_GREY, lw=0, zorder=0)
    for ch, col, lab in (("acidic", "#C0392B", "D/E introduced"),
                         ("basic", "#1B4B80", "K/R introduced")):
        s = a[a.charge == ch]
        if len(s) < 6:
            continue
        m = s.groupby(pd.cut(s.x, bins, include_lowest=True),
                      observed=False).score_z.agg(["mean", "sem"])
        ax.errorbar([iv.mid for iv in m.index], m["mean"], yerr=m["sem"],
                    fmt="-o", ms=2.6, lw=1, color=col, capsize=0, label=lab)
    ax.axhline(0, color=P.MUTED, lw=0.5)
    ax.set_xticks([-0.25, 0.5, 1.25])
    ax.set_xticklabels(["cyto\nflank", "centre", "lumen\nflank"], fontsize=5.5)
    ax.set_ylabel("Normalised fitness")
    return {"legend": ax, "share": "y", "note": f"{a.tm.nunique()} helices"}


def panel_a15(ax, df, cfg, shared):
    """Abundance against ESM-1v, with the functional sites picked out."""
    from .analyses.a15_esm_functional import (MIN_VARIANTS, Q_SITE, Z_SITE, FUNC_COLOR,
                                              loess_fit, read_esm)
    from .analyses.base import missense
    from .stats import bh_fdr
    from scipy import stats as ss
    p = cfg.resolve(cfg.get_path("evolution.esm_scores"))
    if p is None or not Path(p).exists():
        raise NotAvailable("no ESM-1v scores")
    esm = read_esm(p)
    if esm is None:
        raise NotAvailable("ESM table lacks pos/mut/score")
    d = missense(df)[["pos", "wt", "mut", "score_z"]].merge(esm, on=["pos", "mut"],
                                                            how="inner", suffixes=("", "_esm"))
    if "wt_esm" in d:
        d = d[d.wt == d.wt_esm]
    d = d.dropna(subset=["score_z", "esm1v"])
    if len(d) < 100:
        raise NotAvailable(f"only {len(d)} variants with both scores")
    x, y = d.score_z.to_numpy(), d.esm1v.to_numpy()
    f = loess_fit(x, y)
    d["residual"] = y - f(x)
    d["z"] = d.residual / (1.4826 * ss.median_abs_deviation(d.residual))
    rows = []
    for pos, g in d.groupby("pos"):
        if len(g) < MIN_VARIANTS:
            continue
        pv = ss.wilcoxon(g.residual, alternative="less").pvalue if (g.residual != 0).any() else 1.0
        rows.append({"pos": pos, "median_z": g.z.median(), "p": pv})
    sites = pd.DataFrame(rows)
    func = set()
    if len(sites):
        sites["q"] = bh_fdr(sites.p)
        func = set(sites.loc[(sites.q < Q_SITE) & (sites.median_z <= Z_SITE), "pos"])
    isf = d.pos.isin(func)
    grid = np.linspace(np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5), 120)
    P.scatter_fit(ax, d.loc[~isf, "score_z"], d.loc[~isf, "esm1v"], fit=(grid, f(grid)),
                  stat=f"ρ = {ss.spearmanr(x, y)[0]:.2f}\nn = {len(d):,}", s=1.4, alpha=0.28)
    if isf.any():
        ax.scatter(d.loc[isf, "score_z"], d.loc[isf, "esm1v"], s=4, color=FUNC_COLOR, lw=0,
                   alpha=0.85, zorder=4, label="functional sites")
    ax.set_xlabel("Abundance (normalised fitness)")
    ax.set_ylabel("ESM-1v score")
    return {"legend": ax, "share": "both", "note": f"{len(func)} functional sites"}


def panel_a22a(ax, df, cfg, shared):
    """Missense fitness by topology class."""
    from .analyses.a22_topology_violins import CLASS_COLORS, CLASS_ORDER, classify
    if not (cfg.get_path("topology.segments") or []):
        raise NotAvailable("no topology")
    d = classify(df, cfg)
    order = [c for c in CLASS_ORDER if (d.topo_class == c).sum() > 5]
    if not order:
        raise NotAvailable("too few variants per class")
    P.violin_box(ax, {c: d.loc[d.topo_class == c, "score_z"].to_numpy() for c in order},
                 colors=CLASS_COLORS)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([c[:5] for c in order], fontsize=6)
    ax.axhline(0, color="#1B9E77", lw=0.7)
    ax.axhline(-1, color="#D62728", lw=0.6, ls=(0, (3, 3)))
    ax.set_ylabel("Normalised fitness")
    return {"share": "y"}


def panel_a22b(ax, df, cfg, shared):
    """Missense fitness per transmembrane helix, N to C."""
    from .analyses.a22_topology_violins import _helix_key, classify
    if not (cfg.get_path("topology.segments") or []):
        raise NotAvailable("no topology")
    tm = classify(df, cfg)
    tm = tm[tm.seg_type == "TM"]
    order = [h for h in sorted(set(tm.segment), key=_helix_key)
             if (tm.segment == h).sum() > 5]
    if not order:
        raise NotAvailable("no TM helices with enough variants")
    orient = tm.groupby("segment", observed=True).seg_side.first().to_dict()
    cols = {h: ("#B8912F" if orient.get(h) == "in_out" else "#8C6D1F") for h in order}
    P.violin_box(ax, {h: tm.loc[tm.segment == h, "score_z"].to_numpy() for h in order},
                 colors=cols, width=0.8)
    med = float(tm.score_z.median())
    ax.axhline(med, color=P.INK, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(0, color="#1B9E77", lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(h).replace("TM", "") for h in order], fontsize=5)
    ax.set_xlabel("TM helix, N to C", fontsize=6)
    ax.set_ylabel("Normalised fitness")
    return {"share": "y", "note": f"{len(order)} helices"}


# key -> (draw, default columns, panel size, figure title, pre-pass over all proteins)
PANELS = {
    "a01": (panel_a01, 4, (2.5, 2.0), "Score distributions by variant class", None),
    "a05": (panel_a05, 4, (2.7, 2.1), "Physicochemical determinants, TM vs rest", None),
    "a05b": (panel_a05b, 4, (2.6, 2.6), "Substitution matrix: TM minus rest of protein", _a05b_limit),
    "a06": (panel_a06, 4, (2.5, 2.1), "Charge introduced across the membrane", None),
    "a15": (panel_a15, 4, (2.6, 2.2), "Abundance vs ESM-1v and functional sites", None),
    "a22a": (panel_a22a, 4, (2.3, 2.2), "Missense fitness by topology class", None),
    "a22b": (panel_a22b, 2, (5.2, 2.2), "Missense fitness per transmembrane helix", None),
}


# ---------------------------------------------------------------------- driver
def harmonise(axes, how: str) -> None:
    """One pair of limits for every panel; eight self-scaled panels cannot be compared."""
    live = [ax for ax in axes if ax.has_data()]
    if len(live) < 2:
        return
    if how in ("x", "both"):
        lo = min(ax.get_xlim()[0] for ax in live)
        hi = max(ax.get_xlim()[1] for ax in live)
        for ax in live:
            ax.set_xlim(lo, hi)
    if how in ("y", "both"):
        lo = min(ax.get_ylim()[0] for ax in live)
        hi = max(ax.get_ylim()[1] for ax in live)
        for ax in live:
            ax.set_ylim(lo, hi)


def build(key: str, loaded: list, ncols: int | None = None) -> tuple:
    draw, defcols, (pw, ph), title, pre = PANELS[key]
    ncols = ncols or defcols
    n = len(loaded)
    nrows = math.ceil(n / ncols)
    fig, axgrid = plt.subplots(nrows, ncols, figsize=(pw * ncols, ph * nrows),
                               squeeze=False, layout="constrained")
    axes = axgrid.ravel()
    shared = pre(loaded) if pre else {}
    legend_ax, image, shares, drawn, missing = None, None, set(), [], []
    ytick_shared = False
    for ax, (cfg, df) in zip(axes, loaded):
        name = cfg.display_name or cfg.id
        try:
            info = draw(ax, df, cfg, shared) or {}
        except NotAvailable as e:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(0.5, 0.5, f"{name}\n{e}", transform=ax.transAxes, ha="center",
                    va="center", fontsize=7, color=P.MUTED)
            missing.append((name, str(e)))
            continue
        drawn.append(ax)
        cand = info.get("legend")
        if cand is not None and (legend_ax is None or
                                 len(cand.get_legend_handles_labels()[0]) >
                                 len(legend_ax.get_legend_handles_labels()[0])):
            legend_ax = cand
        image = image or info.get("image")
        if info.get("share"):
            shares.add(info["share"])
        ytick_shared = ytick_shared or bool(info.get("ytick_shared"))
        head = name + (f"  ·  {info['note']}" if info.get("note") else "")
        ax.set_title(head, loc="left", fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    for how in (("both",) if "both" in shares else tuple(shares)):
        harmonise(drawn, how)
    # one legend and one colour bar for the figure, not eight of each
    if legend_ax is not None:
        h, lb = legend_ax.get_legend_handles_labels()
        if h:
            fig.legend(h, lb, loc="outside lower center", ncol=len(lb), frameon=False, fontsize=7)
    for ax in drawn:
        if ax.get_legend():
            ax.get_legend().remove()
    if image is not None:
        fig.colorbar(image, ax=list(axes[:n]), fraction=0.02, pad=0.01,
                     label="TM − rest (mean normalised fitness)")
    # inner panels keep their ticks but lose repeated axis labels
    for i, ax in enumerate(axes[:n]):
        if i % ncols:
            ax.set_ylabel("")
            if ytick_shared:          # the same categories in all eight panels
                ax.set_yticklabels([])
        if i < n - ncols:
            ax.set_xlabel("")
    if not P.paper():
        fig.suptitle(f"{key.upper()} — {title} ({len(drawn)} of {n} proteins)",
                     fontsize=10, x=0.01, ha="left")
    return fig, drawn, missing


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|singular|boundary|No artists|invalid value).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms panel",
        description="One figure per analysis with one panel per protein.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--only", help=f"comma-separated keys, default all: {','.join(PANELS)}")
    ap.add_argument("--ncols", type=int, help="override the per-analysis column count")
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)

    P.use_style(a.style)
    keys = [k.strip() for k in (a.only.split(",") if a.only else PANELS)]
    bad = [k for k in keys if k not in PANELS]
    if bad:
        raise SystemExit(f"unknown panel(s) {bad}; available: {', '.join(PANELS)}")

    loaded = []
    for c in a.configs:
        cfg = load_config(Path(c))
        loaded.append((cfg, load_dataset(cfg)))
        print(f"  loaded {cfg.id:<16} {len(loaded[-1][1]):>8,} variants", flush=True)
    if not loaded:
        raise SystemExit("no configs given")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for key in keys:
        fig, drawn, missing = build(key, loaded, a.ncols)
        paths = P.save(fig, outdir, f"{key}_by_protein", None, key.upper())
        plt.close(fig)
        written += paths
        note = "; ".join(f"{n}: {w}" for n, w in missing)
        print(f"  {key:<5} {len(drawn)}/{len(loaded)} panels -> "
              f"{paths[0].relative_to(REPO_ROOT)}" + (f"   [{note}]" if note else ""), flush=True)
    print(f"\n{len(written)} files in {outdir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
