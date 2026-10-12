"""`python -m mpdms famstruct --family hexose=cfg,cfg,... [--human SLC2A1=P11166]`

A family model whose prior mean comes from structure, not just from the other members.

The plain family model says a column's effect is drawn around one number shared by the
family. That has nothing to say about a column no one measured, and at a column where the
members disagree it falls back to the family average. Structure can do better, because the
things that make a position in a membrane protein intolerant are measurable from a model
of it:

    burial in the bundle      a residue packed against other helices cannot be changed freely
    depth across the membrane the centre of the bilayer is not the interface
    distance from the axis    small means lining the central cavity, large means facing
                              lipid - RSA cannot separate these, since it calls both exposed
    solvent accessibility     the usual burial measure, which in a membrane means lipid too
    model confidence          pLDDT, so a disordered loop is not read as a real measurement

So the prior mean becomes a regression on those, fitted across the family:

    theta_jc ~ Normal(m_c, tau^2),   m_c = beta0 + beta . x_c

Everything else is the model in `bayes`. beta is estimated from the data (empirical Bayes),
and when a protein is held out, beta is refitted without it, so the comparison between this
and the intercept-only model is honest.
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .bayes import (column_frame, hierarchical_normal, log_predictive_density,
                    pool_columns, position_estimates)
from .config import REPO_ROOT, load_config, protein_sequence
from .family import align_sequences
from .features import position_table
from .io import load_dataset

OUT = REPO_ROOT / "outputs" / "_famstruct"
COVARIATES = ["rsa", "abs_z", "radial", "contacts", "plddt", "is_tm"]
LABELS = {"rsa": "solvent accessibility", "abs_z": "depth |z| across the membrane",
          "radial": "distance from the bundle axis", "contacts": "contact number",
          "plddt": "AlphaFold pLDDT", "is_tm": "in a TM helix"}


def structural_columns(aln: pd.DataFrame, cfgs: dict, dfs: dict) -> pd.DataFrame:
    """Per alignment column, each covariate averaged over the proteins that have a structure.

    Averaging is the honest summary: the proteins share a fold, so a column's burial is a
    property of that position in the fold rather than of any one member, and a column where
    only some members have a structure still gets a value from those that do.
    """
    rows = []
    for g, cfg in cfgs.items():
        c = f"{g}_pos"
        if c not in aln.columns:
            continue
        pt = position_table(dfs[g], cfg)
        keep = ["pos"] + [k for k in COVARIATES if k in pt.columns]
        m = aln[["col", c]].dropna()
        m = m.assign(pos=m[c].astype(int)).merge(pt[keep], on="pos", how="inner")
        m["protein"] = g
        rows.append(m[["col", "protein"] + [k for k in COVARIATES if k in m.columns]])
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    have = [k for k in COVARIATES if k in long.columns and long[k].notna().any()]
    out = long.groupby("col")[have].mean().reset_index()
    out["n_structures"] = long.groupby("col").protein.nunique().to_numpy()
    return out


def hierarchical_prior_mean(y, s, m: float, k: float, seed: int = 0) -> dict:
    """The same hierarchical normal model, but with a proper prior mu ~ Normal(m, k^2).

    The model in `bayes` puts a flat prior on mu. Under a flat prior a column-specific
    prior MEAN does nothing at all - shifting the data by m and shifting the answer back
    returns exactly the original posterior - so structure can only enter through a prior
    with finite width. `k` is that width: the residual spread of the structural
    regression, i.e. how well structure predicted a column mean on the proteins it was
    fitted on. A regression that explains little gives a wide prior and the model reverts
    to the flat-prior answer, which is the behaviour we want from an honest prior.

    mu is marginalised analytically at each tau, exactly as in the flat case:
        p(y | tau) = p_flat(y | tau) * sqrt(2 pi Vmu) * Normal(mu_hat | m, Vmu + k^2)
    """
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    y, s = y[ok], s[ok]
    J = len(y)
    if J == 0:
        return {"J": 0}
    if not (np.isfinite(m) and np.isfinite(k) and k > 0):
        return hierarchical_normal(y, s, seed=seed)
    spread = float(np.std(y, ddof=1)) if J > 1 else float(np.mean(s))
    tau = np.linspace(0.0, max(3.0 * spread, 3.0 * float(np.max(s)), 1e-3), 400)
    v = s[None, :] ** 2 + tau[:, None] ** 2
    w = 1.0 / v
    Vmu = 1.0 / w.sum(1)
    mu_hat = Vmu * (w * y[None, :]).sum(1)
    logp = (0.5 * np.log(Vmu) - 0.5 * np.log(v).sum(1)
            - 0.5 * (w * (y[None, :] - mu_hat[:, None]) ** 2).sum(1)
            - 0.5 * np.log(Vmu + k ** 2) - 0.5 * (mu_hat - m) ** 2 / (Vmu + k ** 2))
    logp -= logp.max()
    pt = np.exp(logp)
    if tau[0] == 0:
        pt[0] *= 0.5
    pt[-1] *= 0.5
    pt /= pt.sum()
    Vpost = 1.0 / (1.0 / Vmu + 1.0 / k ** 2)
    mu_post = Vpost * (mu_hat / Vmu + m / k ** 2)
    mu_mean = float((pt * mu_post).sum())
    mu_var = float((pt * (Vpost + mu_post ** 2)).sum() - mu_mean ** 2)
    pred_var = float((pt * (Vpost + tau ** 2 + mu_post ** 2)).sum() - mu_mean ** 2)
    c = np.cumsum(pt)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(tau), size=4000, p=pt)
    mu_d = rng.normal(mu_post[i], np.sqrt(Vpost[i]))
    return {"J": J, "mu_mean": mu_mean, "mu_sd": float(np.sqrt(max(mu_var, 0))),
            "mu_lo": float(np.quantile(mu_d, 0.025)),
            "mu_hi": float(np.quantile(mu_d, 0.975)),
            "tau_mean": float((pt * tau).sum()),
            "tau_median": float(np.interp(0.5, c, tau)),
            "pred_mean": mu_mean, "pred_sd": float(np.sqrt(max(pred_var, 0)))}


def figure_pairgrid(long: pd.DataFrame, outdir: Path, name: str):
    """Protein against protein at shared alignment columns: does the family agree at all?"""
    piv = long.pivot_table(index="col", columns="protein", values="y")
    prots = sorted(piv.columns)
    n = len(prots)
    if n < 2:
        return []
    lim = np.nanpercentile(piv.to_numpy(), [0.5, 99.5])
    pad = 0.08 * (lim[1] - lim[0])
    lim = (lim[0] - pad, lim[1] + pad)
    size = 1.8
    fig, axes = plt.subplots(n, n, figsize=(size * n + 1.3, size * n + 0.9), squeeze=False,
                             layout="constrained")
    mappable, rows = None, []
    for i, j in itertools.product(range(n), range(n)):
        ax = axes[i, j]
        if i == j:
            ax.text(0.5, 0.5, prots[i].replace("_fitness", ""), transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue
        d = piv[[prots[j], prots[i]]].dropna()
        x, y = d[prots[j]].to_numpy(), d[prots[i]].to_numpy()
        r = ss.pearsonr(x, y)[0] if len(d) > 2 else np.nan
        rho = ss.spearmanr(x, y)[0] if len(d) > 2 else np.nan
        if j > i:
            ax.text(0.5, 0.5, f"r = {r:.2f}", transform=ax.transAxes, ha="center",
                    va="center", fontsize=10, color="#4C5BA8", style="italic")
            ax.set_xticks([]); ax.set_yticks([])
            rows.append({"protein_x": prots[j], "protein_y": prots[i], "pearson": r,
                         "spearman": rho, "n_columns": int(len(d))})
        else:
            im = P.density_scatter(ax, x, y, s=4.0)
            mappable = mappable or im
            ax.plot(lim, lim, ls=(0, (5, 4)), color=P.INK, lw=0.9)
            ax.axhline(0, color=P.INK, lw=0.6, ls=(0, (1, 2)), zorder=0)
            ax.axvline(0, color=P.INK, lw=0.6, ls=(0, (1, 2)), zorder=0)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.set_title(f"ρ = {rho:.2f}, n = {len(d):,}", fontsize=6.5)
            if i != n - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(P.INK); sp.set_linewidth(0.8)
        ax.tick_params(labelsize=6)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes, fraction=0.022, pad=0.01).set_label(
            "Density", fontsize=8, fontweight="bold")
    fig.supxlabel("Position effect at the aligned column", fontsize=8, fontweight="bold")
    fig.supylabel("Position effect at the aligned column", fontsize=8, fontweight="bold")
    paths = P.save(fig, outdir, f"f01_{name}_column_correlations", None, "F01")
    plt.close(fig)
    return paths, pd.DataFrame(rows)


def figure_depth(long: pd.DataFrame, feats: pd.DataFrame, outdir: Path, name: str):
    """Effect against the two membrane axes: how deep, and how far from the bundle axis."""
    d = long.merge(feats, on="col", how="inner")
    panels = [(c, LABELS[c]) for c in ("abs_z", "radial", "rsa")
              if c in d.columns and d[c].notna().sum() > 50]
    if not panels:
        return [], pd.DataFrame()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels) + 0.4, 3.2),
                             layout="constrained", squeeze=False)
    rows = []
    cols = dict(zip(sorted(d.protein.unique()), P.series_colors(d.protein.nunique())))
    for ax, (c, lab) in zip(axes[0], panels):
        sub = d.dropna(subset=[c, "y"])
        for prot, g in sub.groupby("protein"):
            b = pd.cut(g[c], 8)
            m = g.groupby(b, observed=True).agg(x=(c, "mean"), y=("y", "median")).dropna()
            ax.plot(m.x, m.y, "-", lw=0.9, color=cols[prot], alpha=0.75,
                    label=prot.replace("_fitness", ""))
            rows.append({"covariate": c, "protein": prot,
                         "spearman": float(ss.spearmanr(g[c], g.y)[0]), "n": int(len(g))})
        b = pd.cut(sub[c], 10)
        m = sub.groupby(b, observed=True).agg(x=(c, "mean"), y=("y", "median"),
                                              lo=("y", lambda v: v.quantile(.25)),
                                              hi=("y", lambda v: v.quantile(.75))).dropna()
        ax.fill_between(m.x, m.lo, m.hi, color=P.FIT_RED, alpha=0.15, lw=0)
        ax.plot(m.x, m.y, "-o", ms=4, lw=2.0, color=P.FIT_RED, zorder=5, label="all proteins")
        rho = float(ss.spearmanr(sub[c], sub.y)[0])
        ax.text(0.97, 0.03, f"ρ = {rho:+.2f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.5)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.set_xlabel(lab)
        rows.append({"covariate": c, "protein": "ALL", "spearman": rho, "n": int(len(sub))})
    axes[0][0].set_ylabel("Position effect")
    axes[0][-1].legend(frameon=False, fontsize=6, ncol=2, loc="lower left")
    paths = P.save(fig, outdir, f"f02_{name}_structure_effects", None, "F02")
    plt.close(fig)
    return paths, pd.DataFrame(rows)


# --------------------------------------------------- structure-informed prior
def fit_prior(long: pd.DataFrame, feats: pd.DataFrame, covars: list[str]):
    """Regress the column means on structure. Returns (predict_fn, coefficient table).

    Fitted on precision-weighted column means rather than on every variant, so a column
    measured in four proteins does not count four times, and standardised so the
    coefficients can be read against one another.
    """
    import statsmodels.api as sm
    w = 1.0 / long.s.to_numpy() ** 2
    cm = (long.assign(w=w, wy=long.y * w).groupby("col")
          .agg(wy=("wy", "sum"), w=("w", "sum")).reset_index())
    cm["ybar"] = cm.wy / cm.w
    d = cm.merge(feats, on="col", how="inner").dropna(subset=covars + ["ybar"])
    if len(d) < 30:
        return None, pd.DataFrame()
    mu = d[covars].mean()
    sd = d[covars].std().replace(0, 1.0)
    X = sm.add_constant(((d[covars] - mu) / sd).to_numpy())
    m = sm.OLS(d.ybar.to_numpy(), X).fit()
    coefs = pd.DataFrame({"covariate": ["intercept"] + covars, "coef": m.params,
                          "se": m.bse, "p": m.pvalues,
                          "lo": m.conf_int()[:, 0], "hi": m.conf_int()[:, 1]})

    def predict(f: pd.DataFrame) -> pd.Series:
        """Prior mean per column; NaN where a covariate is missing."""
        z = (f.reindex(columns=covars) - mu) / sd
        ok = z.notna().all(axis=1)
        out = pd.Series(np.nan, index=f.index)
        out[ok] = sm.add_constant(z[ok].to_numpy(), has_constant="add") @ m.params
        return out

    # the residual spread IS the prior width: it is how far a column mean typically sits
    # from what structure predicted for it on the proteins the regression was fitted on
    predict.sigma = float(np.sqrt(m.mse_resid))
    return predict, coefs.assign(r2=m.rsquared, n_columns=len(d), sigma=predict.sigma)


def pool_with_prior(long: pd.DataFrame, prior: pd.Series, sigma: float,
                    min_proteins: int = 3):
    """The hierarchical fit again, but each column shrinks toward its own structural mean."""
    rows = []
    pr = prior.to_dict() if prior is not None else {}
    for col, g in long.groupby("col"):
        if len(g) < min_proteins:
            continue
        m = pr.get(col)
        y, s = g.y.to_numpy(), g.s.to_numpy()
        if m is not None and np.isfinite(m):
            r = hierarchical_prior_mean(y, s, float(m), sigma, seed=int(col))
        else:
            r = hierarchical_normal(y, s, seed=int(col))
        rows.append({"col": int(col), "J": r["J"], "mu": r["mu_mean"], "mu_lo": r["mu_lo"],
                     "mu_hi": r["mu_hi"], "mu_sd": r["mu_sd"], "tau": r["tau_median"],
                     "pred_mean": r["pred_mean"], "pred_sd": r["pred_sd"],
                     "prior_mean": m if m is not None else np.nan})
    return pd.DataFrame(rows)


def figure_coefficients(coefs: pd.DataFrame, outdir: Path, name: str):
    """The fitted prior, as something readable: which structural features predict intolerance."""
    d = coefs[coefs.covariate != "intercept"].copy()
    if not len(d):
        return []
    d["label"] = d.covariate.map(LABELS).fillna(d.covariate)
    d = d.sort_values("coef")
    fig, ax = plt.subplots(figsize=(6.4, 0.46 * len(d) + 1.5), layout="constrained")
    y = np.arange(len(d))
    sig = d.p < 0.05
    ax.hlines(y, d.lo, d.hi, color=P.INK, lw=1.4)
    ax.scatter(d.coef, y, s=46, zorder=5, color=np.where(sig, P.FIT_RED, P.MUTED),
               edgecolor=P.INK, linewidth=0.6)
    ax.axvline(0, color=P.MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(y); ax.set_yticklabels(d.label, fontsize=8)
    ax.set_xlabel("Change in the prior mean effect per 1 SD of the feature")
    r2 = float(coefs.r2.iloc[0]) if "r2" in coefs else np.nan
    n = int(coefs.n_columns.iloc[0]) if "n_columns" in coefs else 0
    ax.set_title(f"(a) structure as a prior for {name}: R² = {r2:.3f} over {n:,} columns\n"
                 "negative = the family is less tolerant there; filled red = p < 0.05",
                 loc="left", fontsize=8.5)
    paths = P.save(fig, outdir, f"f03_{name}_prior_coefficients", None, "F03")
    plt.close(fig)
    return paths


def predict_held_out(long: pd.DataFrame, held: str, prior: pd.Series | None,
                     sigma: float = np.nan, min_proteins: int = 2) -> pd.DataFrame:
    """Fit each column on the rest of the family and score the protein left out.

    The same leave-one-protein-out loop as in `bayes`, except that the column's prior mean
    may come from structure. `prior` must already have been fitted WITHOUT the held-out
    protein, otherwise the comparison is circular.
    """
    rest = long[long.protein != held]
    test = long[long.protein == held].set_index("col")
    pr = prior.to_dict() if prior is not None else {}
    rows = []
    for col, g in rest.groupby("col"):
        if len(g) < min_proteins or col not in test.index:
            continue
        t = test.loc[col]
        t = t.iloc[0] if isinstance(t, pd.DataFrame) else t
        m = pr.get(col)
        use = m is not None and np.isfinite(m) and np.isfinite(sigma) and sigma > 0
        r = (hierarchical_prior_mean(g.y.to_numpy(), g.s.to_numpy(), float(m), sigma,
                                     seed=int(col)) if use
             else hierarchical_normal(g.y.to_numpy(), g.s.to_numpy(), seed=int(col)))
        rows.append({"col": int(col), "pos": int(t.pos), "observed": float(t.y),
                     "obs_se": float(t.s), "pred": r["mu_mean"],
                     "mu_sd": r["mu_sd"], "tau": r["tau_median"], "n_train": r["J"],
                     "used_prior": bool(use),
                     "seg_type": (t.seg_type if "seg_type" in t.index else None)})
    out = pd.DataFrame(rows)
    if len(out):
        out["lpd"] = log_predictive_density(out.observed, out.obs_se, out.pred,
                                            out.mu_sd, out.tau)
    return out


def esm_columns(aln: pd.DataFrame, cfgs: dict) -> pd.DataFrame:
    """Mean ESM-1v masked-marginal score per alignment column, per protein.

    ESM is the obvious external yardstick: it is a zero-shot predictor of the same thing,
    trained on sequences alone, and it needs no family data at all. Averaging over the
    substitutions at a position gives a per-position constraint score on the same footing
    as a position's mean measured effect.
    """
    from .analyses.a15_esm_functional import read_esm
    rows = []
    for g, cfg in cfgs.items():
        c = f"{g}_pos"
        p = cfg.resolve(cfg.get_path("evolution.esm_scores"))
        if c not in aln.columns or not (p and p.exists()):
            continue
        e = read_esm(p)
        if e is None:
            continue
        e = e.groupby("pos").esm1v.mean().rename("esm").reset_index()
        m = aln[["col", c]].dropna()
        m = m.assign(pos=m[c].astype(int)).merge(e, on="pos", how="inner")
        rows.append(m.assign(protein=g)[["col", "protein", "esm"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calibrate_esm(train: pd.DataFrame):
    """ESM score -> measured effect, fitted on the training proteins only.

    ESM is on an arbitrary log-likelihood-ratio scale, so it has to be put on the scale of
    the assay before it can be compared to the family model on anything but rank. The
    residual spread of that fit is the honest uncertainty to give an ESM prediction, and it
    is what lets ESM be weighed against the family posterior.
    """
    d = train.dropna(subset=["esm", "y"])
    if len(d) < 30 or np.ptp(d.esm) == 0:
        return None
    b, a = np.polyfit(d.esm.to_numpy(), d.y.to_numpy(), 1)
    resid = d.y.to_numpy() - (a + b * d.esm.to_numpy())
    sd = float(np.std(resid, ddof=2)) or np.nan
    return lambda x: a + b * np.asarray(x, dtype=float), sd


def blend(pred: pd.DataFrame) -> pd.DataFrame:
    """Combine the family posterior and the calibrated ESM prediction by precision.

    Two predictions of the same quantity with roughly independent errors - one from what
    the relatives measure at the aligned column, one from what a language model makes of
    the sequence - so the simple thing is the right thing: weight each by one over its
    variance. No new parameters are fitted on the held-out protein.
    """
    vf = pred.mu_sd ** 2 + pred.tau ** 2
    ve = pred.esm_sd ** 2
    ok = pred.esm_pred.notna() & np.isfinite(ve) & (ve > 0)
    wf, we = 1.0 / vf, 1.0 / ve
    out = pred.copy()
    out["pred_blend"] = np.where(ok, (wf * pred.pred + we * pred.esm_pred) / (wf + we),
                                 pred.pred)
    v = np.where(ok, 1.0 / (wf + we), vf)
    out["blend_sd"] = np.sqrt(v)
    return out


def compare_models(long: pd.DataFrame, feats: pd.DataFrame, covars: list[str],
                   esm: pd.DataFrame | None = None) -> tuple:
    """Leave-one-protein-out across four models.

        intercept only      the plain family model: shrink toward one number per column
        structure           shrink toward what the membrane geometry predicts there
        ESM-1v              no family data at all, calibrated on the training proteins
        structure + ESM-1v  the two combined by precision

    Everything a model needs is refitted without the held-out protein each time, including
    the structural regression and the ESM calibration, so none of the four has seen it.
    """
    rows, preds = [], []
    grand = float(long.y.mean())
    e = esm if esm is not None and len(esm) else None
    for held in sorted(long.protein.unique()):
        rest = long[long.protein != held]
        fn, _ = fit_prior(rest, feats, covars)
        pr_map = fn(feats.set_index("col")) if fn is not None else None
        sig = getattr(fn, "sigma", np.nan) if fn is not None else np.nan
        cal = None
        if e is not None:
            tr = rest.merge(e, on=["col", "protein"], how="inner")
            cal = calibrate_esm(tr)
        emap = (e[e.protein == held].groupby("col").esm.mean() if e is not None
                else pd.Series(dtype=float))
        for kind, pmap in (("intercept only", None), ("structure", pr_map)):
            p = predict_held_out(long, held, pmap, sig if pmap is not None else np.nan)
            if len(p) < 10:
                continue
            sc = score(p, grand); sc.update(held_out=held, model=kind)
            rows.append(sc)
            preds.append(p.assign(held_out=held, model=kind))
            if cal is None or not len(emap):
                continue
            f, sd = cal
            p = p.assign(esm_raw=p.col.map(emap))
            p["esm_pred"] = np.where(p.esm_raw.notna(), f(p.esm_raw.fillna(0)), np.nan)
            p["esm_sd"] = sd
            if kind == "intercept only":          # ESM alone, scored once
                q = p.dropna(subset=["esm_pred"]).copy()
                if len(q) >= 10:
                    q["pred"], q["mu_sd"], q["tau"] = q.esm_pred, 0.0, q.esm_sd
                    q["lpd"] = log_predictive_density(q.observed, q.obs_se, q.pred,
                                                      q.mu_sd, q.tau)
                    sc = score(q, grand); sc.update(held_out=held, model="ESM-1v")
                    rows.append(sc)
                    preds.append(q.assign(held_out=held, model="ESM-1v"))
            if kind == "structure":
                b = blend(p)
                b["pred"], b["mu_sd"], b["tau"] = b.pred_blend, 0.0, b.blend_sd
                b["lpd"] = log_predictive_density(b.observed, b.obs_se, b.pred,
                                                  b.mu_sd, b.tau)
                sc = score(b, grand); sc.update(held_out=held, model="structure + ESM-1v")
                rows.append(sc)
                preds.append(b.assign(held_out=held, model="structure + ESM-1v"))
    return pd.DataFrame(rows), (pd.concat(preds, ignore_index=True) if preds
                                else pd.DataFrame())


def score(pred: pd.DataFrame, baseline: float) -> dict:
    spread = float(np.std(pred.observed, ddof=1))
    base = log_predictive_density(pred.observed, pred.obs_se, baseline, 0.0, spread)
    slope = float(np.polyfit(pred.pred, pred.observed, 1)[0]) if np.ptp(pred.pred) > 0 else np.nan
    return {"n": int(len(pred)),
            "spearman": float(ss.spearmanr(pred.observed, pred.pred)[0]),
            "pearson": float(ss.pearsonr(pred.observed, pred.pred)[0]),
            "calibration_slope": slope,
            "rmse": float(np.sqrt(np.mean((pred.observed - pred.pred) ** 2))),
            "elpd": float(pred.lpd.mean()),
            "elpd_gain": float(pred.lpd.mean() - np.mean(base)),
            "frac_columns_with_prior": float(pred.used_prior.mean())}


MODELS = ["intercept only", "structure", "ESM-1v", "structure + ESM-1v"]
MARKERS = {"intercept only": "o", "structure": "D", "ESM-1v": "s",
           "structure + ESM-1v": "^"}


def figure_comparison(scores: pd.DataFrame, outdir: Path, name: str):
    """Held-out performance of each model, protein by protein."""
    if not len(scores):
        return []
    models = [m for m in MODELS if m in set(scores.model)]
    cols = dict(zip(models, P.series_colors(len(models))))
    cols[models[0]] = P.MUTED
    metrics = [("spearman", "Spearman ρ"), ("elpd_gain", "elpd gain over one number"),
               ("calibration_slope", "calibration slope"), ("rmse", "RMSE")]
    prots = sorted(scores.held_out.unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.4),
                             layout="constrained")
    y = np.arange(len(prots))
    for ax, (m, lab) in zip(axes, metrics):
        d = scores.pivot_table(index="held_out", columns="model", values=m).reindex(prots)
        have = [c for c in models if c in d.columns]
        if len(have) > 1:
            ax.hlines(y, d[have].min(axis=1), d[have].max(axis=1), color=P.MUTED, lw=0.9,
                      zorder=2)
        for i, c in enumerate(have):
            ax.scatter(d[c], y, s=34, color=cols[c], marker=MARKERS[c], zorder=4 + i,
                       label=c, edgecolor=P.INK, linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([p.replace("_fitness", "") for p in prots], fontsize=7)
        ax.set_xlabel(lab, fontsize=8)
        if m == "elpd_gain":
            ax.axvline(0, color=P.INK, lw=0.7, ls=(0, (3, 3)))
        if m == "calibration_slope":
            ax.axvline(1, color=P.INK, lw=0.7, ls=(0, (3, 3)))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=len(l), loc="outside lower center")
    fig.suptitle("Leave-one-protein-out. The structural regression and the ESM calibration "
                 "are refitted without the held-out protein each time.",
                 fontsize=7.5, x=0.01, ha="left")
    paths = P.save(fig, outdir, f"f04_{name}_model_comparison", None, "F04")
    plt.close(fig)
    return paths


def figure_esm(scores: pd.DataFrame, preds: pd.DataFrame, outdir: Path, name: str):
    """The family model against ESM-1v directly, and what the two together buy."""
    if not len(preds) or "ESM-1v" not in set(preds.model):
        return []
    sp = scores.pivot_table(index="held_out", columns="model", values="spearman")
    el = scores.pivot_table(index="held_out", columns="model", values="elpd")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), layout="constrained")
    ax = axes[0]
    have = [c for c in ("structure", "structure + ESM-1v") if c in sp.columns]
    cols = dict(zip(have, [P.FIT_RED, P.SERIES[0]]))
    lim = [0, max(0.05, np.nanmax(sp.to_numpy()) * 1.15)]
    for c in have:
        ax.scatter(sp["ESM-1v"], sp[c], s=44, color=cols[c], marker=MARKERS[c],
                   edgecolor=P.INK, linewidth=0.5, label=c, zorder=4)
    ax.plot(lim, lim, ls=(0, (5, 4)), color=P.INK, lw=0.9)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Spearman ρ, ESM-1v"); ax.set_ylabel("Spearman ρ, family model")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title("(a) above the line, the family beats ESM", loc="left", fontsize=8.5)

    ax = axes[1]
    d = preds[preds.model == "ESM-1v"].dropna(subset=["esm_raw", "observed"])
    if len(d):
        P.density_scatter(ax, d.esm_raw.to_numpy(), d.observed.to_numpy(), s=5)
        ax.set_xlabel("ESM-1v, mean over substitutions")
        ax.set_ylabel("Observed position effect")
        ax.set_title(f"(b) ESM against the measurement, ρ = "
                     f"{ss.spearmanr(d.esm_raw, d.observed)[0]:.2f}", loc="left",
                     fontsize=8.5)

    ax = axes[2]
    w = preds[preds.model == "structure"].set_index(["held_out", "col"]).pred
    v = preds[preds.model == "ESM-1v"].set_index(["held_out", "col"]).esm_pred
    j = pd.concat([w.rename("family"), v.rename("esm")], axis=1).dropna()
    if len(j):
        P.density_scatter(ax, j.esm.to_numpy(), j.family.to_numpy(), s=5)
        ax.set_xlabel("ESM-1v, calibrated to the assay scale")
        ax.set_ylabel("Family model prediction")
        ax.set_title(f"(c) do they say the same thing? ρ = "
                     f"{ss.spearmanr(j.esm, j.family)[0]:.2f} - the lower this is, the more "
                     "there is to gain\n     from combining them", loc="left", fontsize=8.5)
    paths = P.save(fig, outdir, f"f06_{name}_versus_esm", None, "F06")
    plt.close(fig)
    return paths


# --------------------------------------------------------------- query proteins
def figure_queries(res: dict, outdir: Path, name: str):
    """For each unmeasured protein: how the two models call its residues."""
    if not res:
        return []
    genes = sorted(res)
    fig, axes = plt.subplots(len(genes), 1, figsize=(10.0, 1.9 * len(genes) + 0.8),
                             squeeze=False, layout="constrained", sharex=False)
    for ax, g in zip(axes[:, 0], genes):
        d = res[g]
        m = d[d.mapped]
        ax.fill_between(m.pos, 0, 1, where=(m.seg_type == "TM").to_numpy(),
                        transform=ax.get_xaxis_transform(), color=P.TM_GREY, lw=0, zorder=0,
                        step="mid")
        if "p_loss_plain" in d:
            ax.plot(m.pos, m.p_loss_plain, lw=0.8, color=P.MUTED, label="intercept only")
        ax.plot(m.pos, m.p_loss, lw=0.9, color=P.FIT_RED, label="structure-informed")
        ax.axhline(0.8, color=P.INK, lw=0.6, ls=(0, (3, 3)))
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(loss)", fontsize=8)
        ax.set_title(f"{g}: {int(d.mapped.sum()):,}/{len(d):,} residues mapped, "
                     f"{int((d.call == 'loss').sum()):,} called loss "
                     f"(dashed = the 0.8 calling threshold)", loc="left", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    axes[-1, 0].set_xlabel("Residue in the query protein")
    paths = P.save(fig, outdir, f"f05_{name}_query_predictions", None, "F05")
    plt.close(fig)
    return paths


def run(name: str, members: list[str], queries: dict[str, str], outdir: Path,
        min_proteins: int = 3, covars: list[str] | None = None,
        mafft_args: list[str] | None = None) -> dict:
    from .predict import predict_residues, summarise
    covars = covars or COVARIATES
    cfgs = {}
    for c in members:
        cfg = load_config(Path(c))
        cfgs[cfg.id] = cfg
    seqs = {i: (protein_sequence(c) or "") for i, c in cfgs.items()}
    bad = [i for i, s in seqs.items() if not s]
    if bad:
        return {"status": "skipped", "reason": f"no sequence for {', '.join(bad)}"}
    if len(seqs) < 2:
        return {"status": "skipped", "reason": "a family needs at least two proteins"}
    aln = align_sequences({**seqs, **queries}, mafft_args)
    dfs = {i: load_dataset(c) for i, c in cfgs.items()}
    est = {i: position_estimates(d) for i, d in dfs.items()}
    long = column_frame(aln, est)
    if not len(long):
        return {"status": "skipped", "reason": "no measured position maps onto the alignment"}
    feats = structural_columns(aln, cfgs, dfs)
    if not len(feats):
        return {"status": "skipped", "reason": "no member has a structure"}
    long.to_csv(outdir / f"{name}_positions.csv", index=False)
    feats.to_csv(outdir / f"{name}_column_structure.csv", index=False)

    _, grid = figure_pairgrid(long, outdir, name)
    if len(grid):
        grid.to_csv(outdir / f"{name}_column_correlations.csv", index=False)
    _, depth = figure_depth(long, feats, outdir, name)
    if len(depth):
        depth.to_csv(outdir / f"{name}_structure_effects.csv", index=False)

    fn, coefs = fit_prior(long, feats, [c for c in covars if c in feats.columns])
    if fn is None:
        return {"status": "skipped", "reason": "too few columns with complete structure"}
    coefs.to_csv(outdir / f"{name}_prior_coefficients.csv", index=False)
    figure_coefficients(coefs, outdir, name)
    prior = fn(feats.set_index("col"))

    esm = esm_columns(aln, cfgs)
    scores, preds = compare_models(long, feats, [c for c in covars if c in feats.columns],
                                   esm)
    if len(scores):
        scores.to_csv(outdir / f"{name}_model_comparison.csv", index=False)
        preds.to_csv(outdir / f"{name}_heldout_predictions.csv", index=False)
        figure_comparison(scores, outdir, name)
        figure_esm(scores, preds, outdir, name)

    plain = pool_columns(long, min_proteins)
    struct = pool_with_prior(long, prior, getattr(fn, 'sigma', np.nan),
                             min_proteins)
    seg = plain.set_index("col").seg_type if "seg_type" in plain else pd.Series(dtype=object)
    struct["seg_type"] = struct.col.map(seg)
    struct.to_csv(outdir / f"{name}_columns_structure_prior.csv", index=False)
    plain.to_csv(outdir / f"{name}_columns_plain.csv", index=False)

    qres, qsum = {}, []
    for g, qseq in queries.items():
        c = f"{g}_pos"
        if c not in aln.columns:
            continue
        m = aln[["col", c]].dropna()
        posmap = dict(zip(m[c].astype(int), m["col"].astype(int)))
        a = predict_residues(qseq, posmap, struct)
        b = predict_residues(qseq, posmap, plain)
        a["p_loss_plain"] = b.p_loss
        a["call_plain"] = b.call
        a.to_csv(outdir / f"{name}_{g}_residues.csv", index=False)
        qres[g] = a
        s = summarise(a, g, -0.5); s["model"] = "structure"
        t = summarise(b, g, -0.5); t["model"] = "intercept only"
        # how often the structural prior changes the call at all
        s["n_calls_changed"] = int((a.call != b.call).sum())
        qsum += [t, s]
    if qres:
        figure_queries(qres, outdir, name)
        pd.DataFrame(qsum).to_csv(outdir / f"{name}_query_summary.csv", index=False)

    gain = {}
    if len(scores):
        p = scores.pivot_table(index="held_out", columns="model", values="elpd_gain")
        r = scores.pivot_table(index="held_out", columns="model", values="spearman")
        if {"intercept only", "structure"} <= set(p.columns):
            gain = {"mean_elpd_gain_from_structure":
                    float((p["structure"] - p["intercept only"]).mean()),
                    "n_proteins_improved":
                    int((p["structure"] > p["intercept only"]).sum()),
                    "mean_spearman_plain": float(r["intercept only"].mean()),
                    "mean_spearman_structure": float(r["structure"].mean())}
        for k in ("ESM-1v", "structure + ESM-1v"):
            if k in r.columns:
                key = k.replace(" + ", "_plus_").replace("-", "").replace(" ", "_")
                gain[f"mean_spearman_{key}"] = float(r[k].mean())
                gain[f"mean_elpd_gain_{key}"] = float(p[k].mean())
        if {"ESM-1v", "structure"} <= set(r.columns):
            gain["n_proteins_family_beats_esm"] = int((r["structure"] > r["ESM-1v"]).sum())
    sig = coefs[(coefs.covariate != "intercept") & (coefs.p < 0.05)]
    return {"status": "ok", "method": aln.attrs.get("method", "?"),
            "n_proteins": len(seqs), "n_columns": int(len(plain)),
            "n_columns_with_structure": int(len(feats)),
            "prior_r2": float(coefs.r2.iloc[0]),
            "prior_significant": sig.covariate.tolist(),
            "queries": qsum, **gain,
            "scores": scores.to_dict("records") if len(scores) else [],
            "headline": (f"{len(seqs)} proteins, {len(plain):,} pooled columns, "
                         f"structure explains R² = {coefs.r2.iloc[0]:.3f} of the column means"
                         + (f" ({', '.join(LABELS.get(c, c) for c in sig.covariate)})"
                            if len(sig) else "")
                         + (f"; ESM-1v ρ {gain['mean_spearman_ESM1v']:.2f} vs family "
                            f"{gain['mean_spearman_structure']:.2f}, together "
                            f"{gain.get('mean_spearman_structure_plus_ESM1v', float('nan')):.2f}"
                            if "mean_spearman_ESM1v" in gain else "")
                         + (f"; structure changes held-out elpd by "
                            f"{gain['mean_elpd_gain_from_structure']:+.3f} and helps "
                            f"{gain['n_proteins_improved']}/{len(scores.held_out.unique())} "
                            "proteins" if gain else ""))}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms famstruct",
        description="A family model whose prior mean comes from the structure, compared "
                    "against the plain family model and against ESM-1v.")
    ap.add_argument("--family", action="append", required=True, metavar="NAME=cfg1,cfg2,...")
    ap.add_argument("--human", action="append", default=[], metavar="GENE=ACCESSION",
                    help="a protein nobody measured, to predict: e.g. SLC2A1=P11166")
    ap.add_argument("--covariates", default=",".join(COVARIATES),
                    help=f"structural features for the prior (default {','.join(COVARIATES)})")
    ap.add_argument("--min-proteins", type=int, default=3)
    ap.add_argument("--mafft-args", default=None,
                    help="MAFFT options in place of --auto, e.g. "
                         "\"--maxiterate 1000 --localpair\"")
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    covars = [c.strip() for c in a.covariates.split(",") if c.strip()]
    bad = [c for c in covars if c not in COVARIATES]
    if bad:
        raise SystemExit(f"--covariates: not a structural feature: {', '.join(bad)}\n"
                         f"        choose from {', '.join(COVARIATES)}")

    from .clinvar import fetch_sequence
    queries = {}
    for h in a.human:
        gene, _, acc = h.partition("=")
        if not gene or not acc:
            raise SystemExit(f"--human {h}: expected GENE=ACCESSION, e.g. SLC2A1=P11166")
        queries[gene] = fetch_sequence(acc)

    out = {}
    for spec in a.family:
        name, _, members = spec.partition("=")
        members = [m.strip() for m in members.split(",") if m.strip()]
        if not name or not members:
            raise SystemExit(f"--family {spec}: expected NAME=cfg1.yaml,cfg2.yaml,...")
        bad = [m for m in members if not Path(m).is_file()]
        if bad:
            raise SystemExit(f"--family {name}: not a config file: {', '.join(bad)}\n"
                             "        each member must be a path to one .yaml")
        print(f"\n== {name}: {len(members)} proteins, "
              f"{len(queries)} query protein(s)", flush=True)
        try:
            r = run(name, members, queries, outdir, a.min_proteins, covars,
                    (a.mafft_args.split() if a.mafft_args else None))
        except Exception as e:
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        out[name] = r
        print(f"  {r['status']}: {r.get('headline') or r.get('reason', '')}", flush=True)
        sc = pd.DataFrame(r.get("scores", []))
        if len(sc):
            print(f"    {'held out':<14} " + "".join(f"{m:>22}" for m in MODELS
                                                     if m in set(sc.model)), flush=True)
            for held, g in sc.groupby("held_out"):
                g = g.set_index("model")
                cells = "".join(
                    f"{g.loc[m, 'spearman']:>10.2f} /{g.loc[m, 'elpd_gain']:>+9.2f}"
                    if m in g.index else f"{'-':>22}"
                    for m in MODELS if m in set(sc.model))
                print(f"    {held:<14} {cells}", flush=True)
            print("    (Spearman / elpd gain over predicting one number everywhere)",
                  flush=True)
        for q in r.get("queries", []):
            print(f"    {q['gene']:<10} {q['model']:<16} "
                  f"{q['n_mapped']:,}/{q['n_residues']:,} mapped, "
                  f"{q['n_loss']:,} loss, {q['n_tolerated']:,} tolerated, "
                  f"{q['n_uncertain']:,} uncertain"
                  + (f", {q['n_calls_changed']:,} calls changed"
                     if "n_calls_changed" in q else ""), flush=True)
    (outdir / "famstruct.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0
