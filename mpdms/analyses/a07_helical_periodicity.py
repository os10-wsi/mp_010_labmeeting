"""A07 - helical periodicity within TMDs.

Null model note: a circular shift of the positional profile leaves its power spectrum
(essentially) unchanged, so a circular-shift permutation cannot test spectral power.
We instead use AR(1) surrogates matched to each segment's mean, variance and lag-1
autocorrelation - this preserves the profile's own short-range autocorrelation, which is
the property the circular-shift test was meant to protect.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lombscargle

from .. import plotting as P
from ..annot import segments_df
from ..stats import bh_fdr
from .base import dirs, missense, result, skipped

PERIODS = np.linspace(2, 8, 241)
BAND = (3.4, 3.8)
ARC = 140


def spectrum(x_pos, y):
    ok = np.isfinite(y)
    xp, yy = x_pos[ok].astype(float), y[ok] - np.nanmean(y[ok])
    if len(yy) < 6 or np.std(yy) == 0:
        return np.full(len(PERIODS), np.nan)
    w = 2 * np.pi / PERIODS
    p = lombscargle(xp, yy, w, normalize=True)
    return p


def ar1_surrogates(y, n, rng):
    y = y[np.isfinite(y)]
    mu, sd = y.mean(), y.std()
    phi = np.corrcoef(y[:-1], y[1:])[0, 1] if len(y) > 3 and sd > 0 else 0.0
    phi = float(np.clip(np.nan_to_num(phi), -0.95, 0.95))
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), (n, len(y)))
    out = np.empty_like(e)
    out[:, 0] = rng.normal(0, sd, n)
    for t in range(1, len(y)):
        out[:, t] = phi * out[:, t - 1] + e[:, t]
    return out + mu, phi


def arc_means(angles, vals):
    starts = np.arange(0, 360, 5)
    out = []
    for s in starts:
        rel = (angles - s) % 360
        m = rel <= ARC
        out.append(np.nanmean(vals[m]) if m.any() else np.nan)
    return starts, np.array(out)


def run(df, cfg, outdir):
    segs = segments_df(cfg)
    tms = segs[segs.type == "TM"]
    if tms.empty:
        return skipped("a07", cfg, "no TM segments")
    figdir, tabdir = dirs(outdir)
    mis = missense(df)
    med = mis.groupby("pos").score_z.median()
    rng = np.random.default_rng(1)
    rows, panels, caveats = [], [], []
    for s in tms.itertuples():
        pos = np.arange(s.start, s.end + 1)
        y = med.reindex(pos).to_numpy(float)
        L = len(pos)
        pw = spectrum(pos, y)
        band = (PERIODS >= BAND[0]) & (PERIODS <= BAND[1])
        if L < 14 or np.all(np.isnan(pw)):
            rows.append({"segment": s.name, "start": s.start, "end": s.end, "length": L, "flag": "too short/sparse (<14)"})
            panels.append((s, pos, y, pw, None))
            continue
        obs = np.nanmax(pw[band])
        sur, phi = ar1_surrogates(y, 2000, rng)
        ok = np.isfinite(y)
        null = np.array([np.nanmax(spectrum(pos[ok], sv)[band]) for sv in sur])
        p = (1 + np.sum(null >= obs)) / (1 + len(null))
        # helical wheel
        ang = ((pos - s.start) * 100.0) % 360
        starts, am = arc_means(ang, y)
        i_min = int(np.nanargmin(am))
        face = (starts[i_min] + ARC / 2) % 360
        asym = float(np.nanmax(am) - np.nanmin(am))
        # bootstrap CI on asymmetry, resampling variants within position
        g = {p_: v.to_numpy() for p_, v in mis[mis.pos.between(s.start, s.end)].groupby("pos").score_z}
        boots = []
        for _ in range(500):
            yb = np.array([np.median(rng.choice(g[p_], len(g[p_]))) if p_ in g and len(g[p_]) else np.nan for p_ in pos])
            _, amb = arc_means(ang, yb)
            boots.append(np.nanmax(amb) - np.nanmin(amb))
        acf = [np.corrcoef(y[ok][:-k], y[ok][k:])[0, 1] if ok.sum() > k + 2 else np.nan for k in range(1, 9)]
        rows.append({"segment": s.name, "start": s.start, "end": s.end, "length": L,
                     "fitted_period": float(PERIODS[np.nanargmax(pw)]), "band_power": float(obs), "p": float(p),
                     "ar1_phi": phi, "sensitive_face_deg": float(face), "face_asymmetry": asym,
                     "face_asym_lo": float(np.nanquantile(boots, 0.025)), "face_asym_hi": float(np.nanquantile(boots, 0.975)),
                     **{f"acf_lag{k}": a for k, a in enumerate(acf, 1)}, "flag": ""})
        panels.append((s, pos, y, pw, (ang, face)))
    tab = pd.DataFrame(rows)
    if "p" in tab:
        tab["q"] = bh_fdr(tab["p"])
        tab["periodic"] = tab["q"] < 0.05
    short = tab[tab.flag != ""]
    if len(short):
        caveats.append(f"{len(short)} TM segment(s) <14 residues or too sparse - spectrum not reported")
    caveats.append("significance from AR(1) surrogates (circular shift leaves the spectrum invariant)")
    n_per = int(tab.get("periodic", pd.Series(dtype=bool)).fillna(False).sum())

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(max(2.0 * n, 4), 4.2), squeeze=False)
    vmin, vmax = cfg.get_path("plotting.heatmap_vmin", P.VMIN), cfg.get_path("plotting.heatmap_vmax", P.VMAX)
    cmap, norm = P.fitness_cmap(vmin, vmax), P.fitness_norm(vmin, vmax)
    cover = mis.groupby("pos").size()
    seq = None
    from ..config import protein_sequence
    seq = protein_sequence(cfg)
    for j, (s, pos, y, pw, wheel) in enumerate(panels):
        ax = axes[0, j]
        ax.axvspan(*BAND, color=P.TM_GREY, lw=0)
        ax.plot(PERIODS, pw, color=P.INK, lw=1)
        ax.axvline(3.6, color=P.CLASS_COLORS["nonsense"], lw=0.6, ls=(0, (2, 2)))
        r = tab[tab.segment == s.name].iloc[0]
        q = r.get("q", np.nan)
        ax.set_title(f"{s.name}\nq = {q:.2g}" if np.isfinite(q) else f"{s.name}\n{r.flag}", fontsize=7)
        ax.set_xlabel("Period (res)")
        if j == 0:
            ax.set_ylabel("LS power")
        ax = axes[1, j]
        ax.set_aspect("equal"); ax.axis("off")
        if wheel is None:
            continue
        ang, face = wheel
        th = np.deg2rad(90 - ang)
        rad = 1 + 0.35 * (pos - pos[0]) / 18  # slight spiral to avoid overlap
        sizes = 8 + 3 * cover.reindex(pos).fillna(0).to_numpy()
        ax.plot(np.cos(th) * rad, np.sin(th) * rad, color="#DDDDDD", lw=0.4, zorder=0)
        ax.scatter(np.cos(th) * rad, np.sin(th) * rad, c=np.clip(np.nan_to_num(y, nan=0), vmin, vmax), cmap=cmap,
                   norm=norm, s=sizes, edgecolor=P.INK, lw=0.3, zorder=2)
        for k, p_ in enumerate(pos):
            if seq:
                ax.text(np.cos(th[k]) * (rad[k] + 0.28), np.sin(th[k]) * (rad[k] + 0.28), seq[p_ - 1], fontsize=4.5,
                        ha="center", va="center")
        fa = np.deg2rad(90 - face)
        ax.annotate("", xy=(0.75 * np.cos(fa), 0.75 * np.sin(fa)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=P.CLASS_COLORS["nonsense"], lw=1))
        ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9)
        ax.set_title(f"asym {r.face_asymmetry:.2f}", fontsize=6.5)
    P.title(fig, cfg, f"A07 helical periodicity — {n_per}/{len(tms)} TMDs periodic (q<0.05)")
    fig.tight_layout(rect=(0, 0.01, 1, 0.9))
    figs = P.save(fig, figdir, "a07_helical_periodicity", cfg, "A07")
    t = tabdir / "a07_periodicity.csv"
    tab.to_csv(t, index=False)
    best = tab.sort_values("face_asymmetry", ascending=False).iloc[0] if "face_asymmetry" in tab and tab.face_asymmetry.notna().any() else None
    head = (f"{n_per}/{len(tms)} TMDs show ~3.6-residue periodicity in mutational sensitivity (q<0.05)"
            + (f"; strongest face asymmetry in {best.segment} ({best.face_asymmetry:.2f})" if best is not None else ""))
    return result("a07", cfg, head, {"n_tm": len(tms), "n_periodic": n_per,
                                     "median_face_asymmetry": float(tab.get("face_asymmetry", pd.Series(dtype=float)).median())},
                  caveats, figs, [t])
