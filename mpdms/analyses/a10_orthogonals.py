"""A10 - structural and evolutionary orthogonals, and the residual."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..annot import AA20

from .base import dirs, missense, result, skipped

# Robinson & Robinson 1991 background amino-acid frequencies
BACKGROUND = dict(A=.078, R=.051, N=.045, D=.054, C=.019, Q=.043, E=.063, G=.074, H=.022, I=.051, L=.090,
                  K=.057, M=.022, F=.039, P=.052, S=.071, T=.058, W=.013, Y=.032, V=.064)


def read_msa(path) -> list[str]:
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
            elif not line.startswith("#"):
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    # a3m: drop insertions (lower case) so every row aligns to the query
    return ["".join(c for c in s if not c.islower() and c != ".") for s in seqs]


def conservation(msa: list[str]) -> pd.DataFrame:
    q = msa[0]
    cols = [i for i, c in enumerate(q) if c != "-"]
    bg = np.array([BACKGROUND[a] for a in AA20]); bg /= bg.sum()
    rows = []
    for k, i in enumerate(cols):
        col = [s[i] for s in msa if i < len(s) and s[i] in AA20]
        cnt = np.array([col.count(a) for a in AA20], float) + 0.5  # pseudocount
        f = cnt / cnt.sum()
        H = -np.sum(f * np.log2(f))
        m = 0.5 * (f + bg)
        jsd = 0.5 * np.sum(f * np.log2(f / m)) + 0.5 * np.sum(bg * np.log2(bg / m))
        rows.append({"pos": k + 1, "entropy": H, "jsd": jsd, "msa_depth": len(col)})
    return pd.DataFrame(rows)


def read_esm(path) -> pd.Series:
    e = pd.read_csv(path)
    e.columns = [c.lower() for c in e.columns]
    col = next(c for c in e.columns if c in ("llr", "esm_llr", "score", "esm_score"))
    return e.groupby("pos")[col].mean().rename("esm_llr")


def run(df, cfg, outdir):
    figdir, tabdir = dirs(outdir)
    mis = missense(df)
    y = mis.groupby("pos").score_z.mean().rename("mean_score_z")
    feats = pd.DataFrame(index=y.index)
    used, skipped_f = [], []
    sp = cfg.resolve(cfg.get_path("structure.path"))
    if sp and sp.exists():
        from ..structure import membrane_frame, residue_table
        rt = residue_table(sp, cfg.get_path("structure.chain", "A"), int(cfg.get_path("structure.numbering_offset", 0) or 0))
        rt, info = membrane_frame(rt, cfg)
        rt = rt.set_index("pos")
        feats = feats.join(rt[["rsa", "contacts", "plddt"]])
        used += ["rsa", "contacts", "plddt"]
        if rt["zdepth"].notna().any():
            feats["abs_z"] = rt["zdepth"].abs()
            used.append("abs_z")
    else:
        skipped_f.append("structure (rsa, contacts, z)")
    msa_p = cfg.resolve(cfg.get_path("evolution.msa"))
    if msa_p and msa_p.exists():
        cons = conservation(read_msa(msa_p)).set_index("pos")
        feats = feats.join(cons[["entropy", "jsd"]])
        used += ["entropy", "jsd"]
    else:
        skipped_f.append("MSA conservation")
    esm_p = cfg.resolve(cfg.get_path("evolution.esm_scores"))
    if esm_p and esm_p.exists():
        feats = feats.join(read_esm(esm_p))
        used.append("esm_llr")
    else:
        skipped_f.append("ESM LLR")
    if not used:
        return skipped("a10", cfg, "no orthogonal features available (structure, MSA, ESM all null)")

    data = feats.join(y).join(df.drop_duplicates("pos").set_index("pos")[["segment", "seg_type", "wt"]])
    data = data.dropna(subset=used + ["mean_score_z"])
    corr = pd.DataFrame([{"feature": f, "spearman": ss.spearmanr(data[f], data.mean_score_z)[0],
                          "pearson": ss.pearsonr(data[f], data.mean_score_z)[0]} for f in used])

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, Y, groups = data[used].to_numpy(), data.mean_score_z.to_numpy(), data.segment.astype(str).to_numpy()
    n_groups = len(set(groups))
    cv = GroupKFold(n_splits=min(5, n_groups)) if n_groups >= 3 else 5
    kw = {"groups": groups} if n_groups >= 3 else {}
    lin = make_pipeline(StandardScaler(), LinearRegression())
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
    pred_lin = cross_val_predict(lin, X, Y, cv=cv, **kw)
    pred_gbr = cross_val_predict(gbr, X, Y, cv=cv, **kw)
    # contrast: contiguous 20-residue blocks (less extrapolation than holding out whole segments)
    blocks = (data.index.to_numpy() // 20)
    bcv = GroupKFold(n_splits=min(5, len(set(blocks))))
    pred_blk = cross_val_predict(lin, X, Y, cv=bcv, groups=blocks)
    metrics = {"features_used": used, "features_skipped": skipped_f, "n_positions": len(data),
               "r2_linear_cv": float(r2_score(Y, pred_lin)), "r2_gbr_cv": float(r2_score(Y, pred_gbr)),
               "r2_linear_blockcv": float(r2_score(Y, pred_blk)),
               "cv": f"GroupKFold by segment ({n_groups} groups)" if n_groups >= 3 else "KFold(5)"}
    gbr.fit(X, Y)
    pi = permutation_importance(gbr, X, Y, n_repeats=20, random_state=0)
    imp = pd.DataFrame({"feature": used, "importance": pi.importances_mean, "sd": pi.importances_std})
    data["predicted"] = pred_lin
    data["residual"] = data.mean_score_z - data.predicted

    # candidate biogenesis-specific sites
    pct = lambda s: s.rank(pct=True)  # noqa: E731
    cand = data.copy()
    cand["score_pct"] = pct(cand.mean_score_z)
    cand["conservation_pct"] = pct(cand["jsd"]) if "jsd" in cand else np.nan
    cand["esm_pct"] = pct(-cand["esm_llr"]) if "esm_llr" in cand else np.nan  # high = predicted intolerant
    m = cand.score_pct <= 0.25
    if "rsa" in cand:
        m &= cand.rsa >= 0.25
    if "contacts" in cand:
        m &= cand.contacts <= cand.contacts.median()
    if "jsd" in cand:
        m &= cand.conservation_pct <= 0.5
    if "esm_llr" in cand:
        m &= cand.esm_pct <= 0.5
    top = cand[m].sort_values("residual").head(20).copy()

    def why(r):
        bits = [f"mean score {r.mean_score_z:.2f} (bottom {r.score_pct:.0%})"]
        if "rsa" in r and np.isfinite(r.rsa):
            bits.append(f"exposed (RSA {r.rsa:.2f})")
        if "contacts" in r and np.isfinite(r.contacts):
            bits.append(f"{int(r.contacts)} long-range contacts")
        if np.isfinite(r.get("conservation_pct", np.nan)):
            bits.append(f"conservation pct {r.conservation_pct:.0%}")
        if np.isfinite(r.get("esm_pct", np.nan)):
            bits.append(f"ESM intolerance pct {r.esm_pct:.0%}")
        bits.append(f"{r.residual:+.2f} below model")
        return "; ".join(bits)
    top["rationale"] = [why(r) for _, r in top.iterrows()] if len(top) else []
    off = int(cfg.get_path("structure.numbering_offset", 0) or 0)
    pymol = f"select {cfg.id}_biogenesis_hits, resi " + "+".join(str(int(p) - off) for p in top.index) if len(top) else ""
    metrics["n_candidates"] = len(top)
    metrics["pymol_selection"] = pymol

    fig = plt.figure(figsize=(8.6, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.7], hspace=0.5, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    ax.barh(corr.feature, corr.spearman, color=np.where(corr.spearman > 0, P.CLASS_COLORS["missense"], P.MUTED))
    ax.axvline(0, color=P.INK, lw=0.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ with mean score")
    ax.set_title("(a) Feature correlations", loc="left")
    ax = fig.add_subplot(gs[0, 1])
    cols = {"TM": "#1B4B80", "loop": "#E69F00", "soluble": "#2A9D55"}
    for t, g in data.groupby("seg_type"):
        ax.scatter(g.predicted, g.mean_score_z, s=6, color=cols.get(t, P.MUTED), label=t, lw=0, alpha=0.8)
    lims = [min(data.predicted.min(), Y.min()), max(data.predicted.max(), Y.max())]
    ax.plot(lims, lims, color=P.MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("Predicted (linear, grouped CV)")
    ax.set_ylabel("Observed mean score")
    ax.legend(loc="upper left")
    ax.set_title(f"(b) R² linear {metrics['r2_linear_cv']:.2f} / boosted {metrics['r2_gbr_cv']:.2f}", loc="left")
    ax = fig.add_subplot(gs[1, :])
    P.shade_topology(ax, cfg)
    ax.bar(data.index, data.residual, width=1, lw=0, color=np.where(data.residual < 0, P.CLASS_COLORS["nonsense"], P.MUTED))
    ax.scatter(top.index, top.residual, s=14, facecolor="none", edgecolor=P.INK, lw=0.6, zorder=3, label="candidates")
    ax.axhline(0, color=P.INK, lw=0.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Residual")
    ax.legend(loc="lower right")
    ax.set_title("(c) DMS signal unexplained by orthogonals", loc="left")
    P.title(fig, cfg, "A10 orthogonals and residuals")
    figs = P.save(fig, figdir, "a10_orthogonals", cfg, "A10")
    t1, t2, t3, t4 = (tabdir / f"a10_{n}.csv" for n in ("features", "correlations", "importance", "candidates"))
    data.to_csv(t1)
    corr.to_csv(t2, index=False)
    imp.to_csv(t3, index=False)
    keep = [c for c in ["wt", "segment", "mean_score_z", "conservation_pct", "esm_pct", "rsa", "contacts", "abs_z",
                        "residual", "rationale"] if c in top]
    top[keep].to_csv(t4)
    (tabdir / "a10_pymol_selection.pml").write_text(pymol + "\n")
    caveats = []
    if metrics["r2_linear_cv"] < 0:
        caveats.append(f"segment-grouped CV R² is negative ({metrics['r2_linear_cv']:.2f}; 20-residue block CV "
                       f"{metrics['r2_linear_blockcv']:.2f}) - the model does not transfer between segment types, "
                       "typical with few TMs; residuals are from this out-of-segment prediction")
    if skipped_f:
        caveats.append("features skipped: " + ", ".join(skipped_f)
                       + " - candidate list is NOT filtered for evolutionary tolerance" * ("MSA conservation" in skipped_f and "ESM LLR" in skipped_f))
    head = (f"Orthogonal features explain R² = {metrics['r2_linear_cv']:.2f} (linear) / {metrics['r2_gbr_cv']:.2f} (boosted) "
            f"of position-level sensitivity; {len(top)} candidate biogenesis-specific sites")
    return result("a10", cfg, head, metrics, caveats, figs, [t1, t2, t3, t4])
