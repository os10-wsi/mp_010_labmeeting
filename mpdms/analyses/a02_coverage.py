"""A02 - coverage, depth and missingness structure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..annot import AA20, HEATMAP_ORDER
from ..config import protein_sequence
from ..io import position_matrix
from ..stats import runs_test
from .base import dirs, result

ROWS = list(HEATMAP_ORDER) + ["*"]


def longest_run(mask: np.ndarray, positions) -> tuple[int, int | None, int | None]:
    best, cur, start, bstart = 0, 0, None, None
    for i, m in enumerate(mask):
        if m:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, bstart = cur, start
        else:
            cur = 0
    if best == 0:
        return 0, None, None
    return best, int(positions[bstart]), int(positions[bstart + best - 1])


def run(df, cfg, outdir):
    figdir, tabdir = dirs(outdir)
    seq = protein_sequence(cfg)
    lo, hi = cfg.get_path("protein.region") or (int(df.pos.min()), int(df.pos.max()))
    positions = np.arange(lo, hi + 1)
    L = len(positions)
    mis_all = df[df.vclass == "missense"]
    mis_pass = mis_all[mis_all.pass_filter]
    possible = 19 * L
    cov_pre = mis_all.drop_duplicates(["pos", "mut"]).shape[0] / possible
    cov_post = mis_pass.drop_duplicates(["pos", "mut"]).shape[0] / possible
    per_pos = mis_pass.groupby("pos")["mut"].nunique().reindex(positions, fill_value=0) / 19
    per_aa = mis_pass.groupby("mut")["pos"].nunique().reindex(list(AA20), fill_value=0) / L
    nr = df["n_reads"].dropna()
    metrics = {
        "coverage_missense_prefilter": cov_pre, "coverage_missense_postfilter": cov_post,
        "n_reads_median": float(nr.median()) if len(nr) else np.nan,
        "n_reads_iqr": [float(nr.quantile(.25)), float(nr.quantile(.75))] if len(nr) else None,
        "positions_below_50pct": int((per_pos < 0.5).sum()),
    }
    # missingness structure
    low = (per_pos < 0.5).to_numpy()
    rt = runs_test(low)
    lr, lr_a, lr_b = longest_run(low, positions)
    metrics.update(runs_test_z=rt["z"], runs_test_p=rt["p"], longest_lowcov_run=lr,
                   longest_lowcov_run_start=lr_a, longest_lowcov_run_end=lr_b)
    # a circular shift leaves autocorrelation unchanged, so clustering of missingness is tested
    # against a position-shuffle null on the lag-1 autocorrelation of the coverage profile
    cov = per_pos.to_numpy()
    ac1 = np.corrcoef(cov[:-1], cov[1:])[0, 1] if cov.std() > 0 else np.nan
    rng = np.random.default_rng(0)
    null = [np.corrcoef(cov[:-1], rng.permutation(cov)[1:])[0, 1] for _ in range(2000)] if cov.std() > 0 else []
    metrics["coverage_lag1_autocorr"] = ac1
    metrics["coverage_lag1_p_vs_shuffle"] = float((1 + np.sum(np.array(null) >= ac1)) / (1 + len(null))) if len(null) else np.nan

    # filter sensitivity
    sens_rows, profiles = [], {}
    reps = [c for c in df.columns if c.startswith("input") and c[5:].isdigit()]
    syn = df[df.pass_filter & (df.vclass == "synonymous")]["score_z"]
    p025 = np.nanpercentile(syn, 2.5) if len(syn) >= 5 else np.nan
    for t in (5, 10, 20, 50):
        if reps:
            npass = sum((df[c].fillna(0) >= t).astype(int) for c in reps)
            keep = npass >= min(int(cfg.get_path("filters.min_replicates", 2)), len(reps))
        else:
            keep = df["n_reads"].fillna(np.inf) >= t
        m = df[keep & (df.vclass == "missense") & df.score_z.notna()]
        profiles[t] = m.groupby("pos")["score_z"].median().reindex(positions)
        sens_rows.append({"min_reads": t, "n_missense": len(m),
                          "deleterious_frac": float(np.mean(m.score_z < p025)) if len(m) else np.nan})
    sens = pd.DataFrame(sens_rows)
    a, b = profiles[5], profiles[50]
    ok = a.notna() & b.notna()
    rho = ss.spearmanr(a[ok], b[ok])[0] if ok.sum() > 5 else np.nan
    sens["spearman_profile_vs_min5"] = [ss.spearmanr(profiles[5][profiles[5].notna() & profiles[t].notna()],
                                                     profiles[t][profiles[5].notna() & profiles[t].notna()])[0]
                                        if (profiles[5].notna() & profiles[t].notna()).sum() > 5 else np.nan
                                        for t in sens.min_reads]
    metrics["profile_spearman_min5_vs_min50"] = rho
    caveats = []
    if np.isfinite(rho) and rho < 0.8:
        caveats.append(f"DEPTH-SENSITIVE: positional profile Spearman {rho:.2f} between min_reads 5 and 50")
    metrics["depth_sensitive"] = bool(np.isfinite(rho) and rho < 0.8)
    if lr >= 5:
        caveats.append(f"contiguous low-coverage block {lr_a}-{lr_b} ({lr} positions) - tiling/primer failure?")

    # figure: presence heatmap (measured score coloured, unmeasured hatched) + marginal coverage
    vmin, vmax = cfg.get_path("plotting.heatmap_vmin", P.VMIN), cfg.get_path("plotting.heatmap_vmax", P.VMAX)
    mat = position_matrix(df).reindex(index=positions, columns=ROWS).T.to_numpy(float)
    width = min(max(6.5, L * 0.018), 16)
    fig = plt.figure(figsize=(width, 3.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.12, 1.0, 0.45], width_ratios=[1, 0.012], hspace=0.08, wspace=0.01)
    at = fig.add_subplot(gs[0, 0])
    P.topology_track(at, cfg, (lo - 0.5, hi + 0.5))
    ax = fig.add_subplot(gs[1, 0])
    ax.add_patch(plt.Rectangle((lo - 0.5, -0.5), L, len(ROWS), facecolor="white", hatch="//////",
                               edgecolor="#C8C8C8", lw=0, zorder=0))
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=P.fitness_cmap(vmin, vmax), norm=P.fitness_norm(vmin, vmax),
                   aspect="auto", interpolation="nearest", extent=[lo - 0.5, hi + 0.5, len(ROWS) - 0.5, -0.5], zorder=1)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS, fontsize=4.5, fontfamily="monospace")
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_subplot(gs[1, 1])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_ticks([vmin, -0.5, 0, vmax])
    cb.ax.tick_params(labelsize=5)
    ab = fig.add_subplot(gs[2, 0])
    P.shade_topology(ab, cfg, loops=False)
    ab.bar(positions, per_pos.to_numpy(), width=1.0, color=P.CLASS_COLORS["missense"], lw=0)
    ab.axhline(0.5, color=P.MUTED, lw=0.5, ls=(0, (2, 2)))
    if lr >= 5:
        ab.axvspan(lr_a - 0.5, lr_b + 0.5, color=P.CLASS_COLORS["nonsense"], alpha=0.15, lw=0)
    ab.set_ylim(0, 1)
    ab.set_ylabel("Coverage")
    ab.set_xlabel("Position")
    ab.set_xlim(lo - 0.5, hi + 0.5)
    P.title(fig, cfg, f"A02 coverage — {cov_post:.0%} of 19×L missense measured after filters")
    figs = P.save(fig, figdir, "a02_coverage", cfg, "A02")
    t1 = tabdir / "a02_coverage_per_position.csv"
    per_pos.rename("coverage").to_frame().assign(wt=[seq[p - 1] if seq and p <= len(seq) else "" for p in positions]).to_csv(t1)
    t2 = tabdir / "a02_coverage_per_aa.csv"
    per_aa.rename("coverage").to_csv(t2)
    t3 = tabdir / "a02_filter_sensitivity.csv"
    sens.to_csv(t3, index=False)
    head = (f"{cov_post:.0%} of possible missense variants measured after filters "
            f"({cov_pre:.0%} before); longest low-coverage block {lr} positions")
    return result("a02", cfg, head, metrics, caveats, figs, [t1, t2, t3])
