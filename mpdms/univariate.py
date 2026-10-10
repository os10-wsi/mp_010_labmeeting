"""`python -m mpdms univariate configs/*.yaml`

How much does each feature predict on its own, before any model combines them?

One figure per protein ranks every feature by its rank correlation with the measured
effect, and one figure puts all the proteins together: the same ranking computed on the
pooled data, beside a feature x protein grid showing where the proteins agree and where
they do not. A feature that is strong in one protein and absent in the others is not a
finding about membrane proteins; it is a finding about that protein.

Intervals come from resampling POSITIONS, not variants. Nineteen substitutions at one
site are nineteen measurements of that site, so a variant-level interval would be about
three times too narrow and every feature would look significant.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .config import REPO_ROOT, load_config
from .features import FEATURES, POSITION_FEATURES, build
from .io import load_dataset
from .stats import meta_analysis

OUT = REPO_ROOT / "outputs" / "_univariate"
MIN_N = 100
N_BOOT = 400

LABELS = {
    "delta_biological": "Δ hydrophobicity (biological)", "delta_kd": "Δ hydrophobicity (KD)",
    "delta_volume": "Δ volume", "delta_charge": "Δ charge", "delta_helix": "Δ helix propensity",
    "wt_biological": "WT hydrophobicity", "wt_volume": "WT volume",
    "mut_biological": "mutant hydrophobicity", "mut_volume": "mutant volume",
    "is_pro": "mutant is proline", "is_gly": "mutant is glycine",
    "wt_is_hydrophobic": "WT is hydrophobic", "mut_is_charged": "mutant is charged",
    "is_tm": "in a TM helix", "is_loop": "in a loop", "rel_pos": "position in segment",
    "abs_rel": "distance from segment centre", "dist_end": "distance from segment end",
    "frac_in_protein": "position in protein", "rsa": "solvent accessibility (RSA)",
    "contacts": "contact number", "plddt": "AlphaFold pLDDT", "abs_z": "depth in membrane |z|",
    "helix_rsa": "helix-face RSA", "hyd_window9": "local hydrophobicity (9 aa)",
    "dist_to_tm": "distance to nearest TM", "esm1v": "ESM-1v score",
}


def spearman_ci(x, y, groups, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Rank correlation with an interval from resampling positions, not variants."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    g = np.asarray(groups)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    if len(x) < MIN_N or np.ptp(x) == 0:
        return {"n": int(len(x)), "rho": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan}
    rho = float(ss.spearmanr(x, y)[0])
    idx = pd.Series(np.arange(len(g))).groupby(g).apply(lambda s: s.to_numpy())
    blocks = list(idx)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
        xb, yb = x[pick], y[pick]
        boots[b] = ss.spearmanr(xb, yb)[0] if np.ptp(xb) > 0 else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"n": int(len(x)), "n_positions": int(len(blocks)), "rho": rho,
            "lo": float(lo), "hi": float(hi), "se": float(np.nanstd(boots, ddof=1))}


def table(feats: pd.DataFrame, columns: list[str], seed: int = 0,
          n_boot: int = N_BOOT) -> pd.DataFrame:
    """One row per (protein, feature)."""
    rows = []
    for ds, g in feats.groupby("dataset_id"):
        for c in columns:
            if c not in g or g[c].notna().sum() < MIN_N:
                continue
            r = spearman_ci(g[c], g.score_z, g.pos, n_boot=n_boot, seed=seed)
            r.update(dataset_id=ds, feature=c, label=LABELS.get(c, c))
            rows.append(r)
    return pd.DataFrame(rows)


def _order(t: pd.DataFrame) -> list[str]:
    """Features ranked by how strongly they predict, averaged over the proteins that have them."""
    s = t.assign(a=t.rho.abs()).groupby("feature").a.mean().sort_values(ascending=False)
    return s.index.tolist()


def figure_one(t: pd.DataFrame, ds: str, outdir: Path):
    d = t[t.dataset_id == ds].copy()
    d = d.dropna(subset=["rho"]).sort_values("rho")
    if not len(d):
        return None
    fig, ax = plt.subplots(figsize=(5.6, 0.22 * len(d) + 1.3), layout="constrained")
    y = np.arange(len(d))
    col = np.where(d.rho > 0, "#2F6DB5", "#C0392B")
    ax.barh(y, d.rho, color=col, height=0.66, zorder=2)
    ax.errorbar(d.rho, y, xerr=[d.rho - d.lo, d.hi - d.rho], fmt="none",
                ecolor=P.INK, elinewidth=0.8, capsize=0, zorder=3)
    ax.axvline(0, color=P.INK, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d.label, fontsize=6.5)
    ax.set_xlabel("Spearman ρ with normalised fitness\n(95% CI from resampling positions)")
    ax.set_title(f"{ds}: what each feature predicts on its own", loc="left", fontsize=9)
    ax.set_xlim(-1, 1)
    best = d.iloc[d.rho.abs().argmax()]
    ax.text(0.98, 0.02, f"n = {int(best.n):,} variants at {int(best.n_positions)} positions",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=P.MUTED)
    paths = P.save(fig, outdir, f"u01_{ds}_single_features", None, "U01")
    plt.close(fig)
    return paths


def figure_all(t: pd.DataFrame, pooled: pd.DataFrame, outdir: Path):
    order = _order(t)[::-1]
    ids = sorted(t.dataset_id.unique())
    fig = plt.figure(figsize=(11.0, 0.26 * len(order) + 2.0), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(order))
    pm = pooled.set_index("feature").reindex(order)
    ax.barh(y, pm.rho, color=np.where(pm.rho > 0, "#2F6DB5", "#C0392B"), height=0.62, zorder=2)
    ax.errorbar(pm.rho, y, xerr=[(pm.rho - pm.lo).clip(lower=0), (pm.hi - pm.rho).clip(lower=0)],
                fmt="none", ecolor=P.INK, elinewidth=0.8, capsize=0, zorder=3)
    for ds in ids:                       # every protein's own value over the pooled bar
        s = t[t.dataset_id == ds].set_index("feature").reindex(order)
        ax.plot(s.rho, y, "o", ms=2.6, color=P.MUTED, alpha=0.75, zorder=4)
    ax.axvline(0, color=P.INK, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(f, f) for f in order], fontsize=6.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ, all proteins pooled\n(grey dots = the individual proteins)")
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_title("(a) Pooled", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    M = (t.pivot_table(index="feature", columns="dataset_id", values="rho")
         .reindex(index=order, columns=ids))
    # origin="lower" so row i sits at y = i, the same place barh puts it in (a); with the
    # default the two panels read as mirror images of each other
    im = ax.imshow(np.ma.masked_invalid(M.to_numpy(float)), cmap=P.diverging_cmap(),
                   vmin=-0.8, vmax=0.8, aspect="auto", interpolation="nearest",
                   origin="lower")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([i.replace("_fitness", "") for i in ids], rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([])
    ax.set_ylim(-0.5, len(order) - 0.5)
    fig.colorbar(im, ax=ax, fraction=0.035).set_label("Spearman ρ", fontsize=6.5)
    miss = M.isna().to_numpy()
    for i, j in zip(*np.where(miss)):
        ax.text(j, i, "·", ha="center", va="center", fontsize=6, color=P.MUTED)
    ax.set_title("(b) Per protein  (· = not available)", loc="left", fontsize=9)
    paths = P.save(fig, outdir, "u02_all_proteins_single_features", None, "U02")
    plt.close(fig)
    return paths


def pooled_table(feats: pd.DataFrame, t: pd.DataFrame, columns: list[str], seed: int = 0,
                 n_boot: int = N_BOOT):
    """Pooled rho, and the meta-analytic combination of the per-protein values beside it."""
    rows = []
    for c in columns:
        sub = t[t.feature == c].dropna(subset=["rho", "se"])
        g = feats.dropna(subset=[c, "score_z"]) if c in feats else pd.DataFrame()
        r = (spearman_ci(g[c], g.score_z, g.dataset_id + ":" + g.pos.astype(str),
                         n_boot=n_boot, seed=seed)
             if len(g) >= MIN_N else {"rho": np.nan, "lo": np.nan, "hi": np.nan,
                                      "se": np.nan, "n": 0})
        # Fisher z keeps the meta-analysis on a scale where the variance does not depend
        # on the correlation itself, then back-transform for reporting
        if len(sub) >= 2:
            z = np.arctanh(sub.rho.clip(-0.999, 0.999))
            vz = (sub.se / (1 - sub.rho.clip(-0.999, 0.999) ** 2)) ** 2
            m = meta_analysis(z, vz)
            r.update(meta_rho=float(np.tanh(m["mu"])), meta_lo=float(np.tanh(m["lo"])),
                     meta_hi=float(np.tanh(m["hi"])), meta_I2=m["I2"], k=m["k"])
        r.update(feature=c, label=LABELS.get(c, c), n_proteins=int(len(sub)))
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms univariate",
        description="What each feature predicts on its own, per protein and pooled.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap resamples (default 400)")
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)

    loaded = []
    for c in a.configs:
        cfg = load_config(Path(c))
        loaded.append((load_dataset(cfg), cfg))
        print(f"  loaded {cfg.id}", flush=True)
    feats = build(loaded)
    feats = feats[feats.score_z.notna()]
    cols = [c for c in list(FEATURES) + ["esm1v"]
            if c in feats.columns and feats[c].notna().sum() >= MIN_N]
    print(f"  {len(feats):,} variants, {len(cols)} features with data\n", flush=True)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t = table(feats, cols, n_boot=a.boot)
    t.to_csv(outdir / "per_protein.csv", index=False)
    for ds in sorted(t.dataset_id.unique()):
        figure_one(t, ds, outdir)
        top = t[t.dataset_id == ds].dropna(subset=["rho"]).sort_values("rho", key=abs)
        if len(top):
            b = top.iloc[-1]
            print(f"  {ds:<16} strongest single feature: {b.label} (ρ = {b.rho:+.2f})", flush=True)
    pooled = pooled_table(feats, t, cols, n_boot=a.boot)
    pooled.to_csv(outdir / "pooled.csv", index=False)
    figure_all(t, pooled, outdir)

    p = pooled.dropna(subset=["rho"]).sort_values("rho", key=abs, ascending=False)
    print("\n  pooled over all proteins:")
    print(f"  {'feature':<32}{'rho':>7}{'95% CI':>18}{'meta rho':>10}{'I2':>7}")
    for r in p.head(12).itertuples():
        ci = f"[{r.lo:+.2f}, {r.hi:+.2f}]"
        mr = f"{r.meta_rho:+.2f}" if np.isfinite(getattr(r, "meta_rho", np.nan)) else "    -"
        i2 = f"{r.meta_I2:.0f}%" if np.isfinite(getattr(r, "meta_I2", np.nan)) else "   -"
        print(f"  {r.label:<32}{r.rho:+7.2f}{ci:>18}{mr:>10}{i2:>7}")
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
