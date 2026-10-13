"""A15 - DMS abundance vs ESM-1v: sites more constrained in evolution than abundance explains.

x = DMS normalised fitness (abundance), y = ESM-1v masked-marginal score. A LOESS of y on x
gives the ESM-1v score expected from the abundance effect. For each variant
    residual = ESM-1v - LOESS(abundance),  z = residual / (1.4826 * MAD of all residuals).
Site-level test: one-sided Wilcoxon signed-rank of the site's variant residuals < 0, BH-FDR.
Functional site: q < 0.05 and median z <= -1 (ESM-1v predicts it markedly worse than its
abundance effect explains). Sites whose abundance is itself tolerant (median >= -0.5) are
flagged separately: those are the cleanest "functional, not folding" candidates.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss
from statsmodels.nonparametric.smoothers_lowess import lowess

from .. import plotting as P
from ..stats import bh_fdr
from .base import dirs, fmt_p, missense, result, skipped

def _z_site() -> float:
    """How far below the fit a site's median residual must sit to be called functional,
    in MAD-scaled units. Raising the magnitude calls fewer, more confident sites.

    Set MPDMS_Z_SITE=1.5 to change it for a run. It is read once, at import, so every
    module that uses it sees the same number and no figure can draw a line its own table
    did not use. The sign is taken care of here: the threshold is a lower bound on a
    residual, so 1.5 and -1.5 both mean -1.5.
    """
    raw = os.environ.get("MPDMS_Z_SITE", "1.0")
    try:
        return -abs(float(raw))
    except ValueError:
        print(f"  note: MPDMS_Z_SITE={raw!r} is not a number; using 1.0")
        return -1.0


Z_SITE = _z_site()
Q_SITE = 0.05
TOLERANT = -0.5
MIN_VARIANTS = 5
FUNC_COLOR = "#D62728"


HIGHLIGHT_COLOR = "#111111"


def highlight_positions(cfg) -> list[int]:
    """Residues to call out by name on every A15 panel.

    These are the positions someone already cares about - a mutant they have made, a
    site from the literature - and the point of marking them is to see where they fall
    among the sites the analysis picked out on its own, including when the analysis did
    not pick them. They are drawn in black so they read as an annotation rather than as
    a result: red is reserved for what the statistics called functional.

    From `plotting.highlight_positions` in the config, or MPDMS_HIGHLIGHT=335,336,370
    to override for one run.
    """
    raw = os.environ.get("MPDMS_HIGHLIGHT")
    vals = raw.split(",") if raw is not None else (
        cfg.get_path("plotting.highlight_positions", []) or [])
    out = []
    for v in vals:
        try:
            out.append(int(str(v).strip()))
        except ValueError:
            print(f"  note: highlight position {v!r} is not a number; ignored")
    return sorted(set(out))


def read_esm(path) -> pd.DataFrame | None:
    """Per-variant ESM-1v scores, from this project's own csv or an ensemble table.

    Two layouts are accepted. The one `mpdms esm` writes has pos, mut and esm1v. The
    other is the ensemble layout - one row per variant with the substitution packed into
    a `mutation` column (M1A) and the models averaged into `mean` - which is what comes
    out of a batch ESM-1v run over many proteins. `mean` is only read as a score when a
    `mutation` column is there to identify the layout, since `mean` on its own is far
    too generic a name to claim.
    """
    e = pd.read_csv(path)
    e.columns = [c.strip().lower() for c in e.columns]
    mcol = next((c for c in ("mutation", "variant", "protein_variant") if c in e.columns),
                None)
    if mcol is not None and not {"pos", "mut"} <= set(e.columns):
        v = e[mcol].astype(str).str.extract(r"^([A-Za-z])(\d+)([A-Za-z])$")
        keep = v[0].notna()
        e = e[keep].copy()
        e["wt"], e["pos"], e["mut"] = (v[0][keep].str.upper(), v[1][keep].astype(int),
                                       v[2][keep].str.upper())
    names = ("esm1v", "esm1v_score", "llr", "esm_llr", "score", "esm_score")
    col = next((c for c in names if c in e.columns), None)
    if col is None and mcol is not None and "mean" in e.columns:
        col = "mean"
    if col is None or not {"pos", "mut"} <= set(e.columns):
        return None
    out = e[["pos", "mut"] + (["wt"] if "wt" in e.columns else []) + [col]].rename(
        columns={col: "esm1v"})
    out["pos"] = out.pos.astype(int)
    out.attrs["source_column"] = col
    return out.reset_index(drop=True)


def loess_fit(x, y, frac=0.3):
    """LOESS of y on x, returned as a callable.

    `delta` makes lowess interpolate between points closer than 1% of the x range
    instead of fitting each one. Without it this is O(n^2) and the 61 fits below
    (one real + 60 bootstrap) take minutes on a full-length protein; with it they
    take about a second. Checked at n = 5,000 and 10,000: the fitted values move
    by < 0.002 and the functional-site calls are identical.
    """
    x = np.asarray(x, dtype=float)
    span = float(np.nanmax(x) - np.nanmin(x)) if len(x) else 0.0
    f = lowess(y, x, frac=frac, return_sorted=True, delta=0.01 * span)
    xs, idx = np.unique(f[:, 0], return_index=True)
    return lambda q: np.interp(q, xs, f[idx, 1])


def functional_list(func: pd.DataFrame, width: int = 140) -> list[str]:
    """Every functional residue, in sequence order, wrapped to `width` characters.

    Panels (b) and (c) can only label the handful with the smallest q before the text
    collides, so the figure also carries the full set. Sequence order, not q order:
    this list is meant to be read off against a sequence or pasted into a selection.
    """
    import textwrap
    if not len(func):
        return ["none"]
    f = func.sort_values("pos")
    items = [f"{r.wt}{r.pos}" + ("*" if getattr(r, "abundance_tolerant", False) else "") for r in f.itertuples()]
    return textwrap.wrap("  ".join(items), width=width, break_long_words=False) or ["none"]


def _ranges(nums, sep="+"):
    """[1,2,3,7,9,10] -> '1-3+7+9-10' (PyMOL) or '1-3,7,9-10' with sep=','"""
    nums = sorted(set(int(n) for n in nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return sep.join(out)


Z_COLOR_RANGE = 4.0  # structure colouring: -4 red ... 0 white ... +4 blue
# Ray-tracing two 2400x1800 images is minutes of silent waiting and is almost never
# what you want mid-pipeline, so it is opt-in: `MPDMS_RENDER=1 python -m mpdms run ...`.
# The .pdb/.pml/.cxc are always written, so the figures can be made later on a desktop.
RENDER = os.environ.get("MPDMS_RENDER", "").strip() not in ("", "0", "false", "no")
RENDER_TIMEOUT = 300


def write_structures(cfg, sites: pd.DataFrame, outdir) -> list:
    """Two coloured structures per protein, each as a PDB (B-factor carries the value) plus
    PyMOL (.pml) and ChimeraX (.cxc) scripts that load it, colour it and save a PNG:
      a15_zscore     : site median z, red (constrained) - white - blue; untested residues grey
      a15_functional : functional sites red, everything else white
    """
    import shutil
    import subprocess
    from pathlib import Path

    from Bio.PDB import PDBIO, PDBParser
    src = cfg.resolve(cfg.get_path("structure.path"))
    if src is None or not src.exists():
        return []
    sdir = Path(outdir) / "structures"
    sdir.mkdir(parents=True, exist_ok=True)
    off = int(cfg.get_path("structure.numbering_offset", 0) or 0)
    z = dict(zip(sites.pos, sites.median_z))
    func = set(sites.loc[sites.functional, "pos"])
    tested = {p_ - off for p_ in z}
    funcres = sorted(p_ - off for p_ in func)
    # frame the view on confidently predicted residues (AlphaFold pLDDT in the B-factor column)
    conf = sorted({r.id[1] for r in PDBParser(QUIET=True).get_structure("s", str(src)).get_residues()
                   if "CA" in r and r["CA"].bfactor >= 70})
    orient_sel = f" and resi {_ranges(conf)}" if conf else ""
    out = []
    for name, value in (("a15_zscore", lambda p_: z.get(p_, np.nan)),
                        ("a15_functional", lambda p_: 1.0 if p_ in func else 0.0)):
        st = PDBParser(QUIET=True).get_structure(cfg.id, str(src))
        for res in st.get_residues():
            v = value(res.id[1] + off)
            for a in res:
                a.bfactor = float(np.clip(v, -99, 99)) if np.isfinite(v) else 0.0
        pdb = sdir / f"{name}.pdb"
        io = PDBIO(); io.set_structure(st); io.save(str(pdb))
        out.append(pdb)
        obj = name  # same name PyMOL gives the object when the PDB is opened via File > Open
        sel_t = _ranges(tested) or "none"
        sel_f = _ranges(funcres)
        if name == "a15_zscore":
            pml_col = [f"color grey70, {obj}",
                       f"spectrum b, red_white_blue, {obj} and resi {sel_t}, "
                       f"minimum=-{Z_COLOR_RANGE}, maximum={Z_COLOR_RANGE}"]
            cxc_col = ["color #1 #b3b3b3",
                       f"color byattribute bfactor #1:{_ranges(tested, ',') or '0'} "
                       f"palette -{Z_COLOR_RANGE},red:0,white:{Z_COLOR_RANGE},blue"]
        else:
            pml_col = [f"color white, {obj}"] + ([f"color red, {obj} and resi {sel_f}"] if sel_f else [])
            cxc_col = ["color #1 white"] + ([f"color #1:{_ranges(funcres, ',')} red"] if funcres else [])
        pml = [f"# run with:  cd <folder containing {pdb.name}>  then  @{name}.pml   (not 'run')",
               f"delete {obj}", f"load {pdb.name}, {obj}", "bg_color white", "hide everything", f"show cartoon, {obj}",
               "set cartoon_transparency, 0", "set ray_opaque_background, 0",
               "set ray_trace_mode, 1", "set ray_trace_color, black", "set antialias, 2", *pml_col,
               f"orient {obj}{orient_sel}", f"save {name}.pse",
               f"png {name}.png, width=2400, height=1800, dpi=300, ray=1"]
        cxc = [f"open {pdb.name}", "set bgColor white", "hide atoms", "show cartoons", *cxc_col,
               "lighting soft", "graphics silhouettes true", "view", f"save {name}_chimerax.png width 2400 height 1800 supersample 3"]
        (sdir / f"{name}.pml").write_text("\n".join(pml) + "\n")
        (sdir / f"{name}.cxc").write_text("\n".join(cxc) + "\n")
        out += [sdir / f"{name}.pml", sdir / f"{name}.cxc"]
        import importlib.util
        import sys
        if not RENDER:
            continue  # scripts are written; rendering is opt-in (see RENDER above)
        exe = shutil.which("pymol")
        cmd = [exe, "-cq", f"{name}.pml"] if exe else (
            [sys.executable, "-m", "pymol", "-cq", f"{name}.pml"] if importlib.util.find_spec("pymol") else None)
        if cmd:  # render a PNG if PyMOL is available (headless)
            print(f"    rendering {name}.png with PyMOL (MPDMS_RENDER=1; up to {RENDER_TIMEOUT}s)", flush=True)
            try:
                subprocess.run(cmd, cwd=sdir, timeout=RENDER_TIMEOUT, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for ext in ("png", "pse"):
                    if (sdir / f"{name}.{ext}").exists():
                        out.append(sdir / f"{name}.{ext}")
            except Exception as e:
                print(f"    PyMOL render failed ({e.__class__.__name__}) - open {name}.pml by hand", flush=True)
    return out



def ranking_figure(cfg, sites: pd.DataFrame, func: pd.DataFrame, figdir):
    """The functional sites ordered by how constrained they are, not by position.

    Panel (d) of A15 lists them in sequence order, which is the right way to find a residue
    but the wrong way to decide which to work on. Here they are ranked by median z, with the
    abundance each site actually shows beside it: a site can be strongly constrained in
    ESM-1v and barely move in abundance, and those are the interesting ones, so they are
    marked rather than left to be read off two panels.
    """
    if not len(func):
        return []
    r = func.sort_values("median_z").reset_index(drop=True)
    r["rank"] = np.arange(1, len(r) + 1)
    h = max(2.4, 0.19 * len(r) + 1.4)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, h), layout="constrained",
                             gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(r))[::-1]

    ax = axes[0]
    cm = P.sequential_cmap()
    lo = float(min(r.median_z.min(), Z_SITE))
    cols = [cm(0.15 + 0.7 * (1 - (z - lo) / (Z_SITE - lo))) if Z_SITE != lo else cm(0.5)
            for z in r.median_z]
    ax.barh(y, r.median_z, color=cols, height=0.68, zorder=2)
    ax.axvline(Z_SITE, color=FUNC_COLOR, lw=0.9, ls=(0, (3, 3)), zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i:>3}. {w}{int(p_)}" for i, w, p_ in zip(r["rank"], r.wt, r.pos)],
                       fontsize=6, family="monospace")
    # a highlighted residue that the statistics also called functional is worth finding
    # in the ranking at a glance; one that was not called never appears here at all,
    # which the terminal says rather than the figure implying it does not exist
    hl = set(highlight_positions(cfg))
    if hl:
        for tick, p_ in zip(ax.get_yticklabels(), r.pos):
            if int(p_) in hl:
                tick.set_fontweight("bold")
                tick.set_color(HIGHLIGHT_COLOR)
        for yy, p_, z in zip(y, r.pos, r.median_z):
            if int(p_) in hl:
                ax.barh([yy], [z], height=0.68, color="none",
                        edgecolor=HIGHLIGHT_COLOR, lw=1.1, zorder=4)
    ax.set_xlabel(f"Site median z (ESM-1v − expected);  dashed = {Z_SITE} threshold")
    ax.set_title(f"(a) {len(r)} functional sites, most constrained first", loc="left",
                 fontsize=9)
    # the q values sat on the bars, dark grey on dark viridis; put them past the bar ends
    # instead, where the panel background is always light
    pad = 0.02 * abs(lo)
    for yy, z, q in zip(y, r.median_z, r.q):
        ax.text(z - pad, yy, f"q = {q:.1e}", ha="right", va="center", fontsize=5.2,
                color=P.MUTED)
    ax.set_xlim(lo * 1.22, 0)

    ax = axes[1]
    tol = r.abundance_tolerant.to_numpy(bool) if "abundance_tolerant" in r else np.zeros(len(r), bool)
    ax.barh(y, r.median_abundance, height=0.68,
            color=np.where(tol, "#B8912F", P.SERIES[0]), zorder=2)
    ax.axvline(TOLERANT, color=P.MUTED, lw=0.9, ls=(0, (2, 2)))
    ax.axvline(0, color=P.INK, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("Site median abundance")
    ax.set_title(f"(b) Abundance at the same sites; gold = tolerant (≥ {TOLERANT}), "
                 f"n = {int(tol.sum())}", loc="left", fontsize=9)
    P.title(fig, cfg, "A15b functional sites ranked by constraint")
    out = P.save(fig, figdir, "a15b_functional_ranking", cfg, "A15")
    plt.close(fig)
    return out


def run(df, cfg, outdir):
    p = cfg.resolve(cfg.get_path("evolution.esm_scores"))
    if p is None or not p.exists():
        return skipped("a15", cfg, "evolution.esm_scores not set - run `python -m mpdms esm <config>`")
    esm = read_esm(p)
    if esm is None:
        return skipped("a15", cfg, f"{p.name}: need per-variant columns pos, mut and esm1v/llr/score")
    figdir, tabdir = dirs(outdir)
    d = missense(df)[["pos", "wt", "mut", "score_z", "segment", "seg_type"]].merge(
        esm, on=["pos", "mut"], how="inner", suffixes=("", "_esm"))
    caveats = []
    if "wt_esm" in d:
        bad = (d.wt != d.wt_esm).mean()
        if bad > 0.01:
            caveats.append(f"WT residue differs between DMS and ESM table for {bad:.0%} of variants - numbering?")
        d = d[d.wt == d.wt_esm].drop(columns="wt_esm")
    d = d.dropna(subset=["score_z", "esm1v"])
    if len(d) < 100:
        return skipped("a15", cfg, f"only {len(d)} variants with both DMS and ESM-1v scores")

    x, y = d.score_z.to_numpy(), d.esm1v.to_numpy()
    f = loess_fit(x, y)
    d["esm_expected"] = f(x)
    d["residual"] = y - d.esm_expected
    scale = 1.4826 * ss.median_abs_deviation(d.residual)
    d["z"] = d.residual / scale
    rho = ss.spearmanr(x, y)[0]

    # site-level
    rows = []
    for pos, g in d.groupby("pos"):
        if len(g) < MIN_VARIANTS:
            continue
        pv = ss.wilcoxon(g.residual, alternative="less").pvalue if (g.residual != 0).any() else 1.0
        rows.append({"pos": pos, "wt": g.wt.iloc[0], "segment": g.segment.iloc[0], "seg_type": g.seg_type.iloc[0],
                     "n_variants": len(g), "median_abundance": g.score_z.median(), "median_esm1v": g.esm1v.median(),
                     "median_residual": g.residual.median(), "median_z": g.z.median(),
                     f"frac_variants_z_lt_{Z_SITE:g}": float((g.z < Z_SITE).mean()), "p": pv})
    sites = pd.DataFrame(rows)
    sites["q"] = bh_fdr(sites.p)
    sites["functional"] = (sites.q < Q_SITE) & (sites.median_z <= Z_SITE)
    sites["abundance_tolerant"] = sites.median_abundance >= TOLERANT
    sites["functional_and_tolerant"] = sites.functional & sites.abundance_tolerant
    func = sites[sites.functional].sort_values("q")
    d = d.merge(sites[["pos", "functional"]], on="pos", how="left")
    d["functional"] = d["functional"].astype("boolean").fillna(False).astype(bool)

    # bootstrap band for the LOESS (resample positions)
    grid = np.linspace(np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5), 120)
    rng = np.random.default_rng(15)
    posidx = [g.index.to_numpy() for _, g in d.reset_index(drop=True).groupby("pos")]
    dx, dy = d.score_z.to_numpy(), d.esm1v.to_numpy()
    boots = []
    for _ in range(60):
        pick = np.concatenate([posidx[k] for k in rng.integers(0, len(posidx), len(posidx))])
        boots.append(loess_fit(dx[pick], dy[pick])(grid))
    boots = np.array(boots)

    hl = highlight_positions(cfg)
    sites["highlighted"] = sites.pos.isin(hl)
    hls = sites[sites.highlighted]
    if hl:
        missing = [p for p in hl if p not in set(sites.pos)]
        print(f"    highlighting {len(hls)} of {len(hl)} requested positions"
              + (f"; no site-level estimate for {missing} "
                 f"(fewer than {MIN_VARIANTS} scored variants)" if missing else ""),
              flush=True)

    lines = functional_list(func)
    list_h = 0.085 + 0.034 * len(lines)         # fraction of the base figure height for the list
    fig = plt.figure(figsize=(10.5, 7.2 + list_h * 7.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.62, list_h], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    nf, fv = d[~d.functional], d[d.functional]
    # small black dots with the fit in red and the statistic as plain corner text
    P.scatter_fit(ax, nf.score_z, nf.esm1v, fit=(grid, f(grid)),
                  stat=f"Spearman's ρ = {rho:.2f}\nn = {len(d):,}", s=1.6, alpha=0.30)
    ax.fill_between(grid, np.quantile(boots, .025, 0), np.quantile(boots, .975, 0),
                    color=P.FIT_RED, alpha=0.18, lw=0, zorder=2)
    ax.scatter(fv.score_z, fv.esm1v, s=5, color=FUNC_COLOR, lw=0, alpha=0.8, zorder=4,
               label=f"functional sites (n = {len(fv):,})")
    if hl:
        hv = d[d.pos.isin(hl)]
        ax.scatter(hv.score_z, hv.esm1v, s=22, facecolor="none", edgecolor=HIGHLIGHT_COLOR,
                   lw=0.8, zorder=6, label=f"highlighted ({len(hv):,} variants)")
    ax.set_xlabel("Abundance (DMS normalised fitness)")
    ax.set_ylabel("ESM-1v score (masked marginal)")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.set_title("(a) Variants", loc="left")
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(sites.median_abundance, sites.median_esm1v, c=sites.median_z.clip(-3, 3), cmap=P.diverging_cmap(),
                    vmin=-3, vmax=3, s=14, edgecolor=P.INK, lw=0.2)
    fs = sites[sites.functional]
    ax.scatter(fs.median_abundance, fs.median_esm1v, s=40, facecolor="none", edgecolor=FUNC_COLOR, lw=1.1,
               label=f"functional sites (n={len(fs)})")
    for r in fs.nsmallest(12, "q").itertuples():
        ax.annotate(f"{r.wt}{r.pos}", (r.median_abundance, r.median_esm1v), fontsize=6, xytext=(3, 2),
                    textcoords="offset points", color=FUNC_COLOR)
    if len(hls):
        ax.scatter(hls.median_abundance, hls.median_esm1v, s=90, facecolor="none",
                   edgecolor=HIGHLIGHT_COLOR, lw=1.5, zorder=6,
                   label=f"highlighted (n={len(hls)})")
        P.label_points(ax, hls.median_abundance.to_numpy(), hls.median_esm1v.to_numpy(),
                       [f"{r.wt}{r.pos}" for r in hls.itertuples()],
                       fontsize=7, colors=[HIGHLIGHT_COLOR] * len(hls))
    ax.axvline(TOLERANT, color=P.MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("Site median abundance")
    ax.set_ylabel("Site median ESM-1v")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.colorbar(sc, ax=ax, fraction=0.04).set_label("Site median z (ESM-1v − expected)", fontsize=6.5)
    ax.set_title("(b) Sites", loc="left")
    ax = fig.add_subplot(gs[1, :])
    P.shade_topology(ax, cfg)
    ax.bar(sites.pos, sites.median_z, width=1, lw=0, color=np.where(sites.functional, FUNC_COLOR, "#9DB8DB"))
    ax.axhline(Z_SITE, color=FUNC_COLOR, lw=0.6, ls=(0, (2, 2)))
    ax.axhline(0, color=P.INK, lw=0.5)
    for r in fs.nsmallest(15, "q").itertuples():
        ax.text(r.pos, r.median_z - 0.15, f"{r.wt}{r.pos}", rotation=90, ha="center", va="top", fontsize=5.5,
                color=FUNC_COLOR)
    lo = min(sites.median_z.min(), Z_SITE) if len(sites) else Z_SITE
    ax.set_ylim(lo - 1.2, max(1.0, sites.median_z.max() + 0.2) if len(sites) else 1.0)
    # after set_ylim, so the markers sit at the top of the panel as drawn
    for r in hls.itertuples():
        ax.axvline(r.pos, color=HIGHLIGHT_COLOR, lw=0.7, alpha=0.55, zorder=1)
        ax.annotate(f"{r.wt}{r.pos}", (r.pos, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 2), textcoords="offset points", rotation=90,
                    ha="center", va="bottom", fontsize=6, fontweight="bold",
                    color=HIGHLIGHT_COLOR, annotation_clip=False)
    P.no_square(ax)                 # a 600-residue track does not belong in a square
    ax.set_xlabel("Position")
    ax.set_ylabel("Site median z")
    ax.set_title(f"(c) Residual along the sequence: red = functional (q < {Q_SITE}, median z ≤ {Z_SITE}); grey = TM",
                 loc="left")
    axl = fig.add_subplot(gs[2, :])
    axl.axis("off")
    P.no_square(axl)
    n_tol = int(func.abundance_tolerant.sum()) if len(func) else 0
    axl.text(0, 1.0, f"(d) All {len(func)} functional residues (q < {Q_SITE}, median z ≤ {Z_SITE}), "
                     f"in sequence order; * = abundance-tolerant (median ≥ {TOLERANT}), n = {n_tol}",
             transform=axl.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
    # one text artist, so matplotlib handles line spacing: a per-line offset in axes
    # fractions overflows the panel as soon as the list runs to several lines
    axl.text(0, 0.74, "\n".join(lines), transform=axl.transAxes, ha="left", va="top",
             fontsize=7, family="monospace", linespacing=1.5,
             color=FUNC_COLOR if lines != ["none"] else P.MUTED)
    P.title(fig, cfg, "A15 abundance vs ESM-1v: functional sites")
    figs = P.save(fig, figdir, "a15_esm_functional", cfg, "A15")

    if hl:
        called = sorted(set(func.pos) & set(hl)) if len(func) else []
        print(f"    of the highlighted positions, {len(called)} were called functional"
              + (f": {called}" if called else
                 " - they appear on the A15 panels but not in the A15b ranking, which "
                 "only ranks sites the statistics called"), flush=True)
    figs = figs + ranking_figure(cfg, sites, func, figdir)

    t1, t2 = tabdir / "a15_variants.csv", tabdir / "a15_sites.csv"
    d.to_csv(t1, index=False)
    # ranked by how constrained the site is, which is the order the shortlist is read in
    ranked = sites.sort_values(["functional", "median_z"], ascending=[False, True]).copy()
    ranked.insert(0, "rank", np.where(ranked.functional,
                                      np.arange(1, len(ranked) + 1), np.nan))
    ranked.to_csv(t2, index=False)
    t3 = tabdir / "a15_functional_ranked.csv"
    (ranked[ranked.functional][["rank", "pos", "wt", "segment", "seg_type", "n_variants",
                                "median_z", "median_abundance", "median_esm1v", "q",
                                "abundance_tolerant"]]
     .to_csv(t3, index=False))
    off = int(cfg.get_path("structure.numbering_offset", 0) or 0)
    pml = (f"select {cfg.id}_functional, resi " + "+".join(str(int(p_) - off) for p_ in func.pos)) if len(func) else ""
    (tabdir / "a15_functional_sites.pml").write_text(pml + "\n")
    tabs_extra = [t3]
    structs = write_structures(cfg, sites, outdir)
    if not structs:
        caveats.append("no structure in config - coloured structures not written")
    elif not RENDER:
        caveats.append("structure PDB/PyMOL/ChimeraX scripts written but not rendered; "
                       "re-run with MPDMS_RENDER=1 for the PNGs and .pse sessions, or open the .pml yourself")
    if len(sites) and sites.functional.mean() > 0.3:
        caveats.append(f"{sites.functional.mean():.0%} of sites called functional - check ESM numbering/scale")
    caveats.append("ESM-1v scores conservation for any reason (function, folding in other contexts, interactions); "
                   "a 'functional' call means constraint beyond this abundance readout, not proof of function")
    head = (f"{len(func)} functional sites (ESM-1v more constrained than abundance explains, q<{Q_SITE}); "
            f"{int(sites.functional_and_tolerant.sum())} of them abundance-tolerant; variant ρ(abundance, ESM-1v) = {rho:.2f}"
            + (f"; most constrained: "
               + ", ".join(f"{r.wt}{int(r.pos)} (z {r.median_z:+.1f})"
                           for r in func.nsmallest(5, "median_z").itertuples())
               if len(func) else ""))
    return result("a15", cfg, head, {"spearman": rho, "n_variants": len(d), "n_sites_tested": len(sites),
                                     "n_functional": int(len(func)), "residual_scale": scale,
                                     "n_functional_tolerant": int(sites.functional_and_tolerant.sum()),
                                     "functional_sites": [f"{r.wt}{r.pos}" for r in func.itertuples()],
                                     "pymol": pml, "best_p": fmt_p(sites.p.min())},
                  caveats, figs, [t1, t2] + tabs_extra + [str(x) for x in structs])
