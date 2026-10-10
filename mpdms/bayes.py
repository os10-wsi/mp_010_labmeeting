"""Bayesian pooling across the members of a protein family.

Every quantity here is estimated once per protein and then combined with a hierarchical
normal model (Gelman, BDA3 section 5.4):

    y_j | theta_j ~ Normal(theta_j, s_j^2)        what protein j measured, with its own error
    theta_j | mu, tau ~ Normal(mu, tau^2)         the family's spread around a common value
    p(mu, tau) propto 1                           uniform on tau, flat on mu given tau

mu is what the family shares, tau is how much its members genuinely differ, and the
posterior predictive Normal(mu, tau^2) is what the family says about a member nobody has
measured. That last distribution is the honest meaning of "aggregate predictive power":
not how well the model fits the proteins it was given, but how tightly it constrains the
next one.

The posterior is computed exactly on a grid over tau rather than sampled. With one scalar
nuisance parameter that is cheaper and has no convergence to check, which matters when the
model is refitted for every alignment column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAU_GRID = 400


def _tau_grid(y, s, n: int = TAU_GRID) -> np.ndarray:
    """From 0 to comfortably beyond any plausible between-protein spread."""
    spread = float(np.std(y, ddof=1)) if len(y) > 1 else float(np.mean(s))
    hi = max(3.0 * spread, 3.0 * float(np.max(s)), 1e-3)
    return np.linspace(0.0, hi, n)


def hierarchical_normal(y, s, tau_grid=None, n_draws: int = 0, seed: int = 0) -> dict:
    """Exact posterior for the family mean, the family spread, and the next member.

    `y` are per-protein estimates and `s` their standard errors. Returns posterior
    summaries for mu (what the family shares), tau (how much members differ), the shrunken
    theta_j, and the posterior predictive for an unmeasured member of the same family.
    """
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    y, s = y[ok], s[ok]
    J = len(y)
    if J == 0:
        return {"J": 0}
    if J == 1:
        return {"J": 1, "mu_mean": float(y[0]), "mu_sd": float(s[0]),
                "mu_lo": float(y[0] - 1.96 * s[0]), "mu_hi": float(y[0] + 1.96 * s[0]),
                "tau_mean": np.nan, "tau_median": np.nan,
                "pred_mean": float(y[0]), "pred_sd": float(s[0]),
                "pred_lo": float(y[0] - 1.96 * s[0]), "pred_hi": float(y[0] + 1.96 * s[0]),
                "p_positive": float(y[0] > 0), "theta_mean": y.copy(), "theta_sd": s.copy(),
                "note": "one protein: no pooling is possible, the prior does all the work"}

    tau = _tau_grid(y, s) if tau_grid is None else np.asarray(tau_grid, float)
    v = s[None, :] ** 2 + tau[:, None] ** 2          # (n_tau, J)
    w = 1.0 / v
    Vmu = 1.0 / w.sum(1)
    mu_hat = Vmu * (w * y[None, :]).sum(1)
    # log p(tau | y), marginalising mu analytically
    logp = 0.5 * np.log(Vmu) - 0.5 * np.log(v).sum(1) - 0.5 * (w * (y[None, :] - mu_hat[:, None]) ** 2).sum(1)
    logp -= logp.max()
    p = np.exp(logp)
    if tau[0] == 0:            # trapezoid weights; the grid endpoints count half
        p[0] *= 0.5
    p[-1] *= 0.5
    p /= p.sum()

    tau_mean = float((p * tau).sum())
    c = np.cumsum(p)
    tau_med = float(np.interp(0.5, c, tau))
    mu_mean = float((p * mu_hat).sum())
    mu_var = float((p * (Vmu + mu_hat ** 2)).sum() - mu_mean ** 2)

    # theta_j: shrunk toward mu by how much the family actually agrees. Written as a
    # weighted average rather than a sum of precisions, so tau = 0 (complete pooling,
    # theta_j = mu) is an ordinary value of the formula and not a division by zero.
    s2 = s[None, :] ** 2
    t2 = (tau[:, None] ** 2) * np.ones((1, J))
    denom = s2 + t2
    th_mean = (t2 * y[None, :] + s2 * mu_hat[:, None]) / denom
    # Var(theta_j | tau) must also carry the uncertainty in mu: theta_j leans on mu by a
    # factor s2/(s2+t2), so mu's own variance enters squared through that weight. Leaving
    # it out understates every theta_j sd, most of all where the family pools hardest.
    th_var = s2 * t2 / denom + (s2 / denom) ** 2 * Vmu[:, None]
    theta_mean = (p[:, None] * th_mean).sum(0)
    theta_var = (p[:, None] * (th_var + th_mean ** 2)).sum(0) - theta_mean ** 2

    # posterior predictive for a member of this family that has not been measured
    pred_mean = mu_mean
    pred_var = float((p * (Vmu + tau ** 2 + mu_hat ** 2)).sum() - mu_mean ** 2)

    rng = np.random.default_rng(seed)
    k = rng.choice(len(tau), size=max(n_draws, 4000), p=p)
    mu_d = rng.normal(mu_hat[k], np.sqrt(Vmu[k]))
    pred_d = rng.normal(mu_d, tau[k])
    out = {"J": J, "mu_mean": mu_mean, "mu_sd": float(np.sqrt(max(mu_var, 0))),
           "mu_lo": float(np.quantile(mu_d, 0.025)), "mu_hi": float(np.quantile(mu_d, 0.975)),
           "tau_mean": tau_mean, "tau_median": tau_med,
           "tau_lo": float(np.interp(0.025, c, tau)), "tau_hi": float(np.interp(0.975, c, tau)),
           "pred_mean": float(pred_mean), "pred_sd": float(np.sqrt(max(pred_var, 0))),
           "pred_lo": float(np.quantile(pred_d, 0.025)), "pred_hi": float(np.quantile(pred_d, 0.975)),
           "p_positive": float((mu_d > 0).mean()),
           "theta_mean": theta_mean, "theta_sd": np.sqrt(np.maximum(theta_var, 0)),
           "shrinkage": float(1 - np.mean(np.abs(theta_mean - mu_mean)) /
                              max(np.mean(np.abs(y - mu_mean)), 1e-12))}
    if n_draws:
        out["mu_draws"], out["pred_draws"] = mu_d[:n_draws], pred_d[:n_draws]
    return out


def log_predictive_density(y_new, s_new, mu_mean, mu_sd, tau) -> np.ndarray:
    """log p(y_new | family), the score a Bayesian uses to compare predictions.

    The held-out protein's own measurement error widens the predictive distribution, so a
    noisily measured protein is not penalised for being noisy.
    """
    y_new = np.asarray(y_new, float)
    s_new = np.asarray(s_new, float)
    var = mu_sd ** 2 + tau ** 2 + s_new ** 2
    return -0.5 * (np.log(2 * np.pi * var) + (y_new - mu_mean) ** 2 / var)


def coverage(y_new, s_new, mu_mean, mu_sd, tau, level: float = 0.9) -> float:
    """Fraction of held-out values inside the central predictive interval of that level.

    A model whose 90% intervals contain 60% of what happens is overconfident, however good
    its correlation looks; this is the check that catches it.
    """
    from scipy import stats as ss
    y_new = np.asarray(y_new, float)
    sd = np.sqrt(mu_sd ** 2 + tau ** 2 + np.asarray(s_new, float) ** 2)
    z = ss.norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(y_new - mu_mean) <= z * sd))


# ===================================================================== pipeline
import argparse                                                       # noqa: E402
import warnings                                                       # noqa: E402
from pathlib import Path                                              # noqa: E402

import matplotlib.pyplot as plt                                       # noqa: E402
from scipy import stats as ss                                         # noqa: E402

from . import plotting as P                                           # noqa: E402
from .analyses.base import missense                                   # noqa: E402
from .config import REPO_ROOT, load_config, protein_sequence          # noqa: E402
from .family import align_sequences                                   # noqa: E402
from .io import load_dataset                                          # noqa: E402

OUT = REPO_ROOT / "outputs" / "_bayes"
MIN_VARIANTS = 6          # per position, before its mean is worth a standard error
MIN_PROTEINS = 3          # per alignment column, before the family can be pooled


def position_estimates(df: pd.DataFrame, standardise: bool = False) -> pd.DataFrame:
    """Each position's mean missense effect, its standard error, and its segment type.

    `standardise` divides a protein's position effects by their own spread before pooling.
    Proteins measured on assays of different dynamic range otherwise enter the family model
    on different scales, and the pooled mean ends up between them rather than describing
    any of them.
    """
    d = missense(df).dropna(subset=["score_z"])
    g = d.groupby("pos").score_z.agg(["mean", "std", "size"])
    g = g[g["size"] >= MIN_VARIANTS]
    seg = (d.groupby("pos").seg_type.agg(lambda v: v.mode().iloc[0] if len(v.mode()) else None)
           if "seg_type" in d.columns else pd.Series(index=g.index, dtype=object))
    y = g["mean"].to_numpy()
    se = (g["std"] / np.sqrt(g["size"])).to_numpy()
    scale = 1.0
    if standardise:
        scale = float(np.nanstd(y, ddof=1)) or 1.0
        y, se = y / scale, se / scale
    out = pd.DataFrame({"pos": g.index, "y": y, "s": se, "n": g["size"].to_numpy(),
                        "seg_type": seg.reindex(g.index).to_numpy()})
    out.attrs["scale"] = scale
    return out.query("s > 0")


def column_frame(aln: pd.DataFrame, est: dict) -> pd.DataFrame:
    """Long table of (alignment column, protein, y, s) for every measured position."""
    rows = []
    for g, e in est.items():
        c = f"{g}_pos"
        if c not in aln.columns:
            continue
        m = aln[["col", c]].dropna()
        m = m.assign(pos=m[c].astype(int)).merge(e, on="pos", how="inner")
        m["protein"] = g
        if "seg_type" not in m:
            m["seg_type"] = None
        rows.append(m[["col", "protein", "pos", "y", "s", "n", "seg_type"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pool_columns(long: pd.DataFrame, min_proteins: int = MIN_PROTEINS) -> pd.DataFrame:
    """One hierarchical fit per alignment column."""
    rows = []
    for col, g in long.groupby("col"):
        if len(g) < min_proteins:
            continue
        r = hierarchical_normal(g.y.to_numpy(), g.s.to_numpy(), seed=int(col))
        seg = g.seg_type.dropna() if "seg_type" in g else pd.Series(dtype=object)
        rows.append({"col": int(col), "seg_type": (seg.mode().iloc[0] if len(seg) else None),
                     "J": r["J"], "mu": r["mu_mean"], "mu_sd": r["mu_sd"],
                     "mu_lo": r["mu_lo"], "mu_hi": r["mu_hi"], "tau": r["tau_median"],
                     "tau_hi": r.get("tau_hi", np.nan), "pred_sd": r["pred_sd"],
                     "shrinkage": r.get("shrinkage", np.nan),
                     "proteins": ",".join(sorted(g.protein))})
    return pd.DataFrame(rows)


def predict_held_out(long: pd.DataFrame, held: str, min_proteins: int = 2) -> pd.DataFrame:
    """Fit each column on the rest of the family, then score the protein left out.

    This is the aggregate predictive power the family actually has: what the other members
    say about a member they have never seen.
    """
    rest = long[long.protein != held]
    test = long[long.protein == held].set_index("col")
    rows = []
    for col, g in rest.groupby("col"):
        if len(g) < min_proteins or col not in test.index:
            continue
        t = test.loc[col]
        t = t.iloc[0] if isinstance(t, pd.DataFrame) else t
        r = hierarchical_normal(g.y.to_numpy(), g.s.to_numpy(), seed=int(col))
        rows.append({"col": int(col), "pos": int(t.pos), "observed": float(t.y),
                     "obs_se": float(t.s), "pred": r["mu_mean"], "mu_sd": r["mu_sd"],
                     "tau": r["tau_median"], "n_train": r["J"],
                     "seg_type": (t.seg_type if "seg_type" in t.index else None)})
    out = pd.DataFrame(rows)
    if len(out):
        out["lpd"] = log_predictive_density(out.observed, out.obs_se, out.pred,
                                            out.mu_sd, out.tau)
        # what the rest of the family says about this column's segment type alone: knowing
        # only "transmembrane or not" is a much harder baseline than one number, and it is
        # the one that decides whether the alignment carries POSITIONAL information.
        # Topology is not always in the config, so this baseline is optional.
        if "seg_type" in rest and rest.seg_type.notna().any():
            seg_mu = rest.dropna(subset=["seg_type"]).groupby("seg_type").y.mean()
            out["pred_segment"] = out.seg_type.map(seg_mu).fillna(float(rest.y.mean()))
    return out


def score_predictions(pred: pd.DataFrame, baseline: float) -> dict:
    """Correlation, density, calibration, and whether the predictions are compressed.

    The slope of observed on predicted is the diagnostic that separates two very different
    failures. A slope near 1 with a low correlation means the model is simply uncertain. A
    slope well below 1 means it ranks positions correctly but squashes them toward the
    family average, which is what shrinkage does when the per-position measurements are
    noisy, and it destroys the predictive density while leaving the rank correlation
    looking respectable.
    """
    if len(pred) < 10:
        return {"n": int(len(pred))}
    spread = float(np.std(pred.observed, ddof=1))
    base_lpd = log_predictive_density(pred.observed, pred.obs_se, baseline, 0.0, spread)
    slope = np.nan
    if np.ptp(pred.pred) > 0:
        slope = float(np.polyfit(pred.pred, pred.observed, 1)[0])
    seg = {}
    if "pred_segment" in pred and pred.pred_segment.notna().any():
        seg_lpd = log_predictive_density(pred.observed, pred.obs_se, pred.pred_segment,
                                         0.0, spread)
        seg["elpd_segment"] = float(np.mean(seg_lpd))
        seg["elpd_gain_over_segment"] = float(pred.lpd.mean() - np.mean(seg_lpd))
        if np.ptp(pred.pred_segment) > 0:
            seg["spearman_segment"] = float(ss.spearmanr(pred.observed, pred.pred_segment)[0])
    return {"n": int(len(pred)), "calibration_slope": slope,
            "pred_spread": float(np.std(pred.pred, ddof=1)), "obs_spread": spread, **seg,
            "spearman": float(ss.spearmanr(pred.observed, pred.pred)[0]),
            "pearson": float(ss.pearsonr(pred.observed, pred.pred)[0]),
            "elpd": float(pred.lpd.mean()),
            "elpd_baseline": float(np.mean(base_lpd)),
            "elpd_gain": float(pred.lpd.mean() - np.mean(base_lpd)),
            "coverage_50": coverage(pred.observed, pred.obs_se, pred.pred, pred.mu_sd,
                                    pred.tau, 0.50),
            "coverage_90": coverage(pred.observed, pred.obs_se, pred.pred, pred.mu_sd,
                                    pred.tau, 0.90),
            "rmse": float(np.sqrt(np.mean((pred.observed - pred.pred) ** 2)))}


def figure_family(name: str, cols: pd.DataFrame, long: pd.DataFrame, outdir: Path):
    """What the family shares along the alignment, and where its members disagree."""
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 6.2), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.5, 0.8, 0.5]})
    c = cols.sort_values("col")
    for ax in axes:
        if "seg_type" in c and c.seg_type.notna().any():   # the membrane, on every panel
            tm = (c.seg_type == "TM").to_numpy()
            ax.fill_between(c.col, 0, 1, where=tm, transform=ax.get_xaxis_transform(),
                            color=P.TM_GREY, lw=0, zorder=0, step="mid")
    ax = axes[0]
    ax.fill_between(c.col, c.mu_lo, c.mu_hi, color=P.SERIES[0], alpha=0.22, lw=0,
                    label="95% credible interval for the family mean")
    ax.plot(c.col, c.mu, lw=1.0, color=P.SERIES[0], label="family mean (posterior)")
    ax.axhline(0, color=P.MUTED, lw=0.6)
    ax.set_ylabel("Pooled position effect")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", ncol=2)
    ax.set_title(f"(a) {name}: what the family shares at each aligned position "
                 f"(grey = transmembrane)", loc="left", fontsize=9)
    ax = axes[1]
    cm = P.sequential_cmap()
    hi = float(c.tau.max()) or 1.0
    ax.fill_between(c.col, 0, c.tau, color=cm(0.55), alpha=0.30, lw=0)
    ax.plot(c.col, c.tau, lw=0.8, color=cm(0.25))
    # the columns where the family genuinely disagrees are the interesting ones
    top = c.nlargest(min(8, len(c)), "tau")
    ax.scatter(top.col, top.tau, s=10, color=cm(0.85), zorder=4, lw=0)
    ax.set_ylabel("τ  (between-protein SD)")
    ax.set_title("(b) Where the family members genuinely differ", loc="left", fontsize=9)
    ax = axes[2]
    # not grey: the transmembrane band behind it is grey, and two greys read as one
    ax.fill_between(c.col, 0, c.J, color=P.SERIES[1], alpha=0.55, lw=0, step="mid")
    ax.set_ylabel("proteins")
    ax.set_xlabel("Alignment column")
    ax.set_title("(c) How many proteins measured each column", loc="left", fontsize=9)
    P.save(fig, outdir, f"b01_{name}_family_posterior", None, "B01")
    plt.close(fig)


def figure_predictive(name: str, scores: pd.DataFrame, outdir: Path):
    """How well the family predicts a member it has not seen."""
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), layout="constrained")
    s = scores.sort_values("spearman")
    y = np.arange(len(s))
    ax = axes[0]
    cm = P.sequential_cmap()
    ax.barh(y, s.spearman, color=[cm(0.15 + 0.7 * max(v, 0)) for v in s.spearman], height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(s.held_out, fontsize=7)
    ax.set_xlabel("Spearman, held-out protein")
    ax.axvline(0, color=P.INK, lw=0.7)
    ax.set_title("(a) Rank agreement", loc="left", fontsize=9)
    ax = axes[1]
    ax.barh(y, s.elpd_gain, color=np.where(s.elpd_gain > 0, "#1B9E77", "#C0392B"), height=0.6)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.axvline(0, color=P.INK, lw=0.7)
    ax.set_xlabel("elpd gain over one number for the whole family\n(positive = the alignment helps)")
    ax.set_title("(b) Predictive density", loc="left", fontsize=9)
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=P.MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.scatter([0.5] * len(s), s.coverage_50, s=22, color="#B8912F", label="50% interval")
    ax.scatter([0.9] * len(s), s.coverage_90, s=22, color="#1B4B80", label="90% interval")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal")
    ax.set_ylabel("Actual coverage")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title("(c) Calibration", loc="left", fontsize=9)
    P.save(fig, outdir, f"b02_{name}_predictive_power", None, "B02")
    plt.close(fig)


def figure_diagnostic(name: str, preds: pd.DataFrame, scores: pd.DataFrame, outdir: Path):
    """Predicted against observed for each held-out protein, with the compression visible."""
    held = sorted(preds.held_out.unique())
    n = len(held)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 0.6, 2.9), layout="constrained",
                             squeeze=False)
    sc = scores.set_index("held_out")
    lo = float(min(preds.observed.min(), preds.pred.min()))
    hi = float(max(preds.observed.max(), preds.pred.max()))
    for ax, h in zip(axes[0], held):
        g = preds[preds.held_out == h]
        ax.scatter(g.pred, g.observed, s=5, color=P.INK, alpha=0.35, lw=0)
        ax.plot([lo, hi], [lo, hi], color=P.MUTED, lw=0.8, ls=(0, (3, 3)))
        if np.ptp(g.pred) > 0:
            b, a = np.polyfit(g.pred, g.observed, 1)
            xs = np.array([g.pred.min(), g.pred.max()])
            ax.plot(xs, a + b * xs, color=P.FIT_RED, lw=1.4)
        r = sc.loc[h] if h in sc.index else {}
        ax.set_title(f"{h}\nρ = {r.get('spearman', float('nan')):.2f}, "
                     f"slope = {r.get('calibration_slope', float('nan')):.2f}",
                     loc="left", fontsize=7.5)
        ax.set_xlabel("Predicted by the rest of the family", fontsize=7)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    axes[0][0].set_ylabel("Observed")
    fig.suptitle("Dashed = perfect prediction; red = the fit. A red line flatter than the "
                 "dashed one means the family ranks the positions but squashes them.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, f"b03_{name}_calibration", None, "B03")
    plt.close(fig)


def run_family(name: str, members: list[str], outdir: Path, min_proteins: int,
               standardise: bool = False) -> dict:
    cfgs = {}
    for c in members:
        cfg = load_config(Path(c))
        cfgs[cfg.id] = cfg
    seqs = {i: (protein_sequence(c) or "") for i, c in cfgs.items()}
    missing = [i for i, s in seqs.items() if not s]
    if missing:
        return {"status": "skipped", "reason": f"no sequence for {', '.join(missing)}"}
    if len(seqs) < 2:
        return {"status": "skipped", "reason": "a family needs at least two proteins"}
    aln = align_sequences(seqs)
    est = {i: position_estimates(load_dataset(c), standardise) for i, c in cfgs.items()}
    long = column_frame(aln, est)
    if not len(long):
        return {"status": "skipped", "reason": "no measured position maps onto the alignment"}
    cols = pool_columns(long, min_proteins)
    if not len(cols):
        return {"status": "skipped",
                "reason": f"no alignment column has {min_proteins}+ proteins measured"}
    cols.to_csv(outdir / f"{name}_columns.csv", index=False)
    long.to_csv(outdir / f"{name}_positions.csv", index=False)
    figure_family(name, cols, long, outdir)

    rows, preds = [], []
    grand = float(long.y.mean())
    for held in sorted(long.protein.unique()):
        pr = predict_held_out(long, held)
        if not len(pr):
            continue
        sc = score_predictions(pr, grand)
        sc.update(held_out=held)
        rows.append(sc)
        pr.insert(0, "held_out", held)
        preds.append(pr)
    scores = pd.DataFrame(rows)
    if len(scores) and "spearman" in scores:
        allp = pd.concat(preds, ignore_index=True)
        scores.to_csv(outdir / f"{name}_predictive.csv", index=False)
        allp.to_csv(outdir / f"{name}_predictions.csv", index=False)
        figure_predictive(name, scores, outdir)
        figure_diagnostic(name, allp, scores, outdir)
    return {"status": "ok", "method": aln.attrs.get("method", "?"),
            "n_proteins": len(seqs), "n_columns": int(len(cols)),
            "median_tau": float(cols.tau.median()),
            "median_mu_sd": float(cols.mu_sd.median()),
            "scores": scores.to_dict("records") if len(scores) else [],
            "headline": (f"{len(seqs)} proteins, {len(cols)} pooled columns; "
                         f"median τ = {cols.tau.median():.2f}"
                         + (f"; held-out Spearman {scores.spearman.mean():.2f}, "
                            f"elpd gain {scores.elpd_gain.mean():+.3f}"
                            if len(scores) and "spearman" in scores else ""))}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms bayes",
        description="Hierarchical pooling across a protein family, on its alignment.")
    ap.add_argument("--family", action="append", required=True, metavar="NAME=cfg1,cfg2,...",
                    help="one family per flag, e.g. PF07690=configs/AQR1.yaml,configs/QDR2.yaml")
    ap.add_argument("--min-proteins", type=int, default=MIN_PROTEINS,
                    help="proteins a column needs before it is pooled (default 3)")
    ap.add_argument("--standardise", action="store_true",
                    help="scale each protein's position effects by their own spread before "
                         "pooling, so assays of different dynamic range are comparable")
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = {}
    for spec in a.family:
        name, _, members = spec.partition("=")
        members = [m.strip() for m in members.split(",") if m.strip()]
        if not name or not members:
            raise SystemExit(f"--family {spec}: expected NAME=cfg1.yaml,cfg2.yaml,...")
        # say which entry is wrong, rather than failing deep inside the loader on it
        bad = [m for m in members if not Path(m).is_file()]
        if bad:
            raise SystemExit(
                f"--family {name}: not a config file: {', '.join(bad)}\n"
                f"        each member must be a path to one .yaml, e.g. configs/HXT1.yaml")
        dup = sorted({m for m in members if members.count(m) > 1})
        if dup:
            raise SystemExit(f"--family {name}: listed twice: {', '.join(dup)}")
        print(f"\n== {name}: {len(members)} proteins", flush=True)
        try:
            r = run_family(name, members, outdir, a.min_proteins, a.standardise)
        except Exception as e:
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        out[name] = r
        print(f"  {r['status']}: {r.get('headline') or r.get('reason', '')}", flush=True)
        for s in r.get("scores", []):
            print(f"    held out {s['held_out']:<14} ρ {s.get('spearman', float('nan')):+.2f}"
                  f"  slope {s.get('calibration_slope', float('nan')):+.2f}"
                  f"  elpd gain {s.get('elpd_gain', float('nan')):+.2f} vs one number,"
                  f" {s.get('elpd_gain_over_segment', float('nan')):+.2f} vs TM/loop"
                  f"  cover {s.get('coverage_50', float('nan')):.0%}/"
                  f"{s.get('coverage_90', float('nan')):.0%}", flush=True)
        sl = [s.get("calibration_slope", np.nan) for s in r.get("scores", [])]
        if sl and np.nanmean(sl) < 0.75:
            print(f"    NOTE: mean calibration slope {np.nanmean(sl):.2f} - the family ranks "
                  "positions better than it places them; the predictions are compressed "
                  "toward the family average, which is what costs the predictive density")
    (outdir / "bayes.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0
