"""A15 - DMS abundance vs ESM-1v: sites more constrained in evolution than abundance explains.

x = DMS normalised fitness (abundance), y = ESM-1v masked-marginal score. A LOESS of y on x
gives the ESM-1v score expected from the abundance effect. For each variant
    residual = ESM-1v - LOESS(abundance),  z = residual / (1.4826 * MAD of all residuals).
Site-level test: one-sided Wilcoxon signed-rank of the site's variant residuals < 0, BH-FDR.
Functional site: q < 0.05 and median z <= -1.5 (ESM-1v predicts it markedly worse than its
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

Z_SITE = -1.5
Q_SITE = 0.05
TOLERANT = -0.5
MIN_VARIANTS = 5
FUNC_COLOR = "#D62728"


def read_esm(path) -> pd.DataFrame | None:
    e = pd.read_csv(path)
    e.columns = [c.strip().lower() for c in e.columns]
    col = next((c for c in ("esm1v", "esm1v_score", "llr", "esm_llr", "score", "esm_score") if c in e.columns), None)
    if col is None or not {"pos", "mut"} <= set(e.columns):
        return None
    out = e[["pos", "mut"] + (["wt"] if "wt" in e.columns else []) + [col]].rename(columns={col: "esm1v"})
    out["pos"] = out.pos.astype(int)
    return out


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

    lines = functional_list(func)
    list_h = 0.085 + 0.034 * len(lines)         # fraction of the base figure height for the list
    fig = plt.figure(figsize=(10.5, 7.2 + list_h * 7.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.62, list_h], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    nf = d[~d.functional]
    ax.hexbin(nf.score_z, nf.esm1v, gridsize=55, cmap="Greys", bins="log", mincnt=1, linewidths=0, rasterized=True)
    fv = d[d.functional]
    ax.scatter(fv.score_z, fv.esm1v, s=6, color=FUNC_COLOR, lw=0, alpha=0.7, label=f"variants at functional sites (n={len(fv)})")
    ax.fill_between(grid, np.quantile(boots, .025, 0), np.quantile(boots, .975, 0), color="#2F6DB5", alpha=0.25, lw=0)
    ax.plot(grid, f(grid), color="#2F6DB5", lw=1.8, label="LOESS (± position bootstrap)")
    ax.set_xlabel("Abundance (DMS normalised fitness)")
    ax.set_ylabel("ESM-1v score (masked marginal)")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.set_title(f"(a) Variants: Spearman ρ = {rho:.2f}, n = {len(d):,}", loc="left")
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(sites.median_abundance, sites.median_esm1v, c=sites.median_z.clip(-3, 3), cmap=P.diverging_cmap(),
                    vmin=-3, vmax=3, s=14, edgecolor=P.INK, lw=0.2)
    fs = sites[sites.functional]
    ax.scatter(fs.median_abundance, fs.median_esm1v, s=40, facecolor="none", edgecolor=FUNC_COLOR, lw=1.1,
               label=f"functional sites (n={len(fs)})")
    for r in fs.nsmallest(12, "q").itertuples():
        ax.annotate(f"{r.wt}{r.pos}", (r.median_abundance, r.median_esm1v), fontsize=6, xytext=(3, 2),
                    textcoords="offset points", color=FUNC_COLOR)
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
    ax.set_xlabel("Position")
    ax.set_ylabel("Site median z")
    ax.set_title(f"(c) Residual along the sequence: red = functional (q < {Q_SITE}, median z ≤ {Z_SITE}); grey = TM",
                 loc="left")
    axl = fig.add_subplot(gs[2, :])
    axl.axis("off")
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

    t1, t2 = tabdir / "a15_variants.csv", tabdir / "a15_sites.csv"
    d.to_csv(t1, index=False)
    sites.sort_values("q").to_csv(t2, index=False)
    off = int(cfg.get_path("structure.numbering_offset", 0) or 0)
    pml = (f"select {cfg.id}_functional, resi " + "+".join(str(int(p_) - off) for p_ in func.pos)) if len(func) else ""
    (tabdir / "a15_functional_sites.pml").write_text(pml + "\n")
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
            + (f"; top: {', '.join(f'{r.wt}{r.pos}' for r in func.head(5).itertuples())}" if len(func) else ""))
    return result("a15", cfg, head, {"spearman": rho, "n_variants": len(d), "n_sites_tested": len(sites),
                                     "n_functional": int(len(func)), "residual_scale": scale,
                                     "n_functional_tolerant": int(sites.functional_and_tolerant.sum()),
                                     "functional_sites": [f"{r.wt}{r.pos}" for r in func.itertuples()],
                                     "pymol": pml, "best_p": fmt_p(sites.p.min())},
                  caveats, figs, [t1, t2] + [str(x) for x in structs])
