"""A08 - depth-dependent tolerance."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from statsmodels.nonparametric.smoothers_lowess import lowess

from .. import plotting as P
from ..structure import membrane_frame, residue_table
from .base import dirs, missense, result, skipped

HYDROPHOBIC = set("AVLIMFWC")
POLAR = set("DEKRNQHST")
ZMAX = 30.0


def _lowess_curve(x, y, grid, frac=0.5):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.full(len(grid), np.nan)
    f = lowess(y[ok], x[ok], frac=frac, return_sorted=True)
    xs, idx = np.unique(f[:, 0], return_index=True)
    return np.interp(grid, xs, f[idx, 1], left=np.nan, right=np.nan)


def run(df, cfg, outdir):
    p = cfg.resolve(cfg.get_path("structure.path"))
    if p is None or not p.exists():
        return skipped("a08", cfg, "structure.path is null or missing")
    if cfg.get_path("structure.membrane_normal", "none") == "none":
        return skipped("a08", cfg, "structure.membrane_normal is none")
    figdir, tabdir = dirs(outdir)
    rt = residue_table(p, cfg.get_path("structure.chain", "A"), int(cfg.get_path("structure.numbering_offset", 0) or 0))
    rt, info = membrane_frame(rt, cfg)
    if info["method"] == "none":
        return skipped("a08", cfg, f"could not define membrane frame: {info.get('reason', '')}")
    rt["absz"] = rt.zdepth.abs()
    mis = missense(df).merge(rt[["pos", "zdepth", "absz", "radial"]], on="pos", how="inner")
    mis = mis[mis.absz <= ZMAX]
    polar = mis[mis.wt.isin(list(HYDROPHOBIC)) & mis.mut.isin(list(POLAR))]
    hh = mis[mis.wt.isin(list(HYDROPHOBIC)) & mis.mut.isin(list(HYDROPHOBIC))]
    grid = np.linspace(0, ZMAX, 121)
    rng = np.random.default_rng(3)

    def pos_means(d):
        return d.groupby("pos").agg(score=("score_z", "mean"), absz=("absz", "first"))

    curves, metrics = {}, {"membrane_frame": info}
    for name, d in (("polar", polar), ("hydrophobic", hh)):
        pm = pos_means(d)
        c = _lowess_curve(pm.absz.to_numpy(), pm.score.to_numpy(), grid)
        boots = []
        for _ in range(300):
            b = pm.sample(len(pm), replace=True, random_state=int(rng.integers(1e9)))
            boots.append(_lowess_curve(b.absz.to_numpy(), b.score.to_numpy(), grid))
        boots = np.array(boots)
        curves[name] = (pm, c, np.nanquantile(boots, 0.025, axis=0), np.nanquantile(boots, 0.975, axis=0), boots)
        if np.isfinite(c).any():
            metrics[f"{name}_peak_sensitivity_absz"] = float(grid[np.nanargmin(c)])
            peaks = [grid[np.nanargmin(b)] for b in boots if np.isfinite(b).any()]
            metrics[f"{name}_peak_absz_ci"] = [float(np.quantile(peaks, 0.025)), float(np.quantile(peaks, 0.975))]
        metrics[f"{name}_interface_12_18"] = float(pm.loc[pm.absz.between(12, 18), "score"].mean())
        metrics[f"{name}_core_lt6"] = float(pm.loc[pm.absz < 6, "score"].mean())

    # residuals from the polar depth trend
    pm, c, *_ = curves["polar"]
    pm = pm.assign(expected=np.interp(pm.absz, grid, np.nan_to_num(c, nan=np.nanmean(c) if np.isfinite(c).any() else 0)))
    pm["residual"] = pm.score - pm.expected
    tree = cKDTree(rt[["x", "y", "z"]].to_numpy())
    xyz = rt.set_index("pos")[["x", "y", "z"]]
    top = pm.nsmallest(15, "residual").copy()
    seqmap = dict(zip(rt.pos, rt.aa))
    top["wt"] = [seqmap.get(p_, "") for p_ in top.index]
    top["z"] = [float(rt.set_index("pos").loc[p_, "zdepth"]) for p_ in top.index]
    top["neighbors_8A"] = [",".join(f"{seqmap[rt.pos.iloc[j]]}{rt.pos.iloc[j]}" for j in tree.query_ball_point(xyz.loc[p_].to_numpy(), 8.0)
                                    if rt.pos.iloc[j] != p_) for p_ in top.index]
    metrics["n_positions_in_slab"] = int(pm.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for name, col, lab in (("polar", "#C0392B", "hydrophobic→polar/charged"), ("hydrophobic", "#1B4B80", "hydrophobic→hydrophobic")):
        pm_, c_, lo_, hi_, _ = curves[name]
        ax.scatter(pm_.absz, pm_.score, s=5, color=col, alpha=0.35, lw=0)
        ax.fill_between(grid, lo_, hi_, color=col, alpha=0.18, lw=0)
        ax.plot(grid, c_, color=col, lw=1.4, label=lab)
    ax.axvspan(12, 18, color=P.TM_GREY, alpha=0.6, lw=0, zorder=0)
    ax.text(15, 0.98, "interface", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=P.MUTED)
    ax.axhline(0, color=P.MUTED, lw=0.5)
    ax.set_xlabel("|z| from membrane centre (Å)")
    ax.set_ylabel("Mean " + cfg.get_path("plotting.score_label").lower())
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.08), ncol=2, borderaxespad=0)
    ax.set_title("(a) Tolerance vs depth (LOESS ± bootstrap)", loc="left", pad=18)
    ax = axes[1]
    sens = missense(df).groupby("pos").score_z.mean()
    r2 = rt[rt.absz <= ZMAX].assign(s=lambda t: t.pos.map(sens))
    vmin, vmax = cfg.get_path("plotting.heatmap_vmin", P.VMIN), cfg.get_path("plotting.heatmap_vmax", P.VMAX)
    sc = ax.scatter(r2.radial, r2.zdepth, c=r2.s, cmap=P.fitness_cmap(vmin, vmax), norm=P.fitness_norm(vmin, vmax),
                    s=9, edgecolor=P.INK, lw=0.2)
    for zz in (-15, 15):
        ax.axhline(zz, color=P.MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("Radial distance from bundle axis (Å)")
    ax.set_ylabel("z (Å, + = lumenal)")
    ax.set_title("(b) Residues in membrane frame", loc="left")
    fig.colorbar(sc, ax=ax, fraction=0.05, extend="both").set_label("Mean score", fontsize=6)
    P.title(fig, cfg, f"A08 depth dependence ({info['method']})")
    fig.tight_layout(rect=(0, 0.01, 1, 0.9))
    figs = P.save(fig, figdir, "a08_depth_dependence", cfg, "A08")
    t1, t2 = tabdir / "a08_top_residuals.csv", tabdir / "a08_position_depth.csv"
    top.reset_index().to_csv(t1, index=False)
    rt.to_csv(t2, index=False)
    caveats = [f"membrane frame from {info['method']}" + (" (approximate; supply a PPM/OPM-oriented model for exact z)"
                                                          if info["method"] == "pca_tm_axes" else "")]
    head = (f"Polar introductions are least tolerated at |z| ≈ {metrics.get('polar_peak_sensitivity_absz', np.nan):.0f} Å; "
            f"core (<6 Å) {metrics['polar_core_lt6']:.2f} vs interface (12–18 Å) {metrics['polar_interface_12_18']:.2f}; "
            f"top residual: {top.index[0] if len(top) else 'NA'}")
    return result("a08", cfg, head, metrics, caveats, figs, [t1, t2])
