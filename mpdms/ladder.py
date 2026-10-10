"""`python -m mpdms ladder configs/*.yaml`

How few features does a pooled predictor actually need?

Features are added one at a time, and at every step the feature chosen is the one that
most improves prediction on a protein family the model has not seen. The curve flattens
where extra features stop buying anything, and that is the model worth writing down.

Two things decide how this has to be done. The features are badly collinear - "in a TM
helix", "WT is hydrophobic" and "local hydrophobicity" are largely one fact told three
ways - so ranking them by their separate correlations says almost nothing about what each
adds. And the single-feature correlations disagree strongly between proteins, so a
coefficient fitted once on the pool describes no protein in particular; every coefficient
here is therefore shown with its per-protein refits beside it.

Selection is scored on held-out families, never on the data the step was chosen from.
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
from .benchmark import assign_families, split_folds
from .config import REPO_ROOT, load_config, protein_sequence
from .features import FEATURES, build
from .io import load_dataset
from .univariate import LABELS

OUT = REPO_ROOT / "outputs" / "_ladder"
MAX_STEPS = 8
MIN_GAIN = 0.005          # a step that buys less held-out Spearman than this is not taken


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], alpha_grid=None):
    """Standardised ridge: simple, additive, and its coefficients can be read."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    med = train[cols].median(numeric_only=True)
    Xtr = train[cols].astype(float).fillna(med).to_numpy()
    Xte = test[cols].astype(float).fillna(med).to_numpy()
    m = make_pipeline(StandardScaler(),
                      RidgeCV(alphas=alpha_grid if alpha_grid is not None else np.logspace(-3, 3, 13)))
    m.fit(Xtr, train.score_z.to_numpy())
    return m, m.predict(Xte)


def cv_score(feats: pd.DataFrame, cols: list[str], split: str = "family") -> dict:
    """Mean held-out Spearman over the folds, and the per-fold values behind it."""
    per = {}
    for name, tr, te in split_folds(feats.reset_index(drop=True), split):
        d = feats.reset_index(drop=True)
        train, test = d.loc[tr], d.loc[te]
        if len(train) < 200 or len(test) < 50:
            continue
        try:
            _, pred = _fit_predict(train, test, cols)
        except Exception:
            continue
        y = test.score_z.to_numpy()
        per[name] = float(ss.spearmanr(y, pred)[0]) if np.ptp(pred) > 0 else 0.0
    if not per:
        return {"mean": np.nan, "per_fold": {}}
    return {"mean": float(np.mean(list(per.values()))), "per_fold": per,
            "min": float(np.min(list(per.values())))}


def forward_ladder(feats: pd.DataFrame, candidates: list[str], split: str = "family",
                   max_steps: int = MAX_STEPS, min_gain: float = MIN_GAIN):
    """Add the feature that most improves held-out prediction, until none does.

    Returns (table of every step tried, list of the features actually kept). The kept list
    is returned rather than recovered from the table: a missing flag column reads back as
    NaN, and `not NaN` is False, which silently empties the model.
    """
    chosen, rows, prev = [], [], 0.0
    pool = list(candidates)
    for step in range(1, max_steps + 1):
        best, best_s = None, None
        for c in pool:
            s = cv_score(feats, chosen + [c], split)
            if np.isfinite(s["mean"]) and (best_s is None or s["mean"] > best_s["mean"]):
                best, best_s = c, s
        if best is None:
            break
        gain = best_s["mean"] - prev
        rows.append({"step": step, "feature": best, "label": LABELS.get(best, best),
                     "cv_spearman": best_s["mean"], "gain": gain,
                     "worst_fold": best_s.get("min", np.nan),
                     **{f"fold_{k}": v for k, v in best_s["per_fold"].items()}})
        if gain < min_gain and step > 1:
            rows[-1]["stopped_here"] = True
            break
        chosen.append(best)
        pool.remove(best)
        prev = best_s["mean"]
    return pd.DataFrame(rows), chosen


def final_model(feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Standardised coefficients from the pool, and from each protein on its own.

    With the single-feature correlations disagreeing as much as they do, a pooled
    coefficient is an average of numbers that are not the same; the per-protein column is
    there so that can be seen rather than taken on trust.
    """
    m, _ = _fit_predict(feats, feats, cols)
    coefs = m[-1].coef_
    rows = [{"feature": c, "label": LABELS.get(c, c), "pooled": float(b)} for c, b in zip(cols, coefs)]
    out = pd.DataFrame(rows)
    for ds, g in feats.groupby("dataset_id"):
        if len(g) < 200:
            continue
        try:
            mm, _ = _fit_predict(g, g, cols)
            out[ds] = mm[-1].coef_
        except Exception:
            out[ds] = np.nan
    per = [c for c in out.columns if c not in ("feature", "label", "pooled")]
    if per:
        out["per_protein_sd"] = out[per].std(axis=1)
        out["sign_agreement"] = (np.sign(out[per]).eq(np.sign(out.pooled), axis=0)
                                 .mean(axis=1))
    return out


def figure(ladder: pd.DataFrame, coefs: pd.DataFrame, baselines: dict, outdir: Path,
           split: str):
    fig = plt.figure(figsize=(11.0, 4.4), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 0.95])

    ax = fig.add_subplot(gs[0, 0])
    k = np.arange(1, len(ladder) + 1)
    ax.plot(k, ladder.cv_spearman, "-o", ms=5, lw=1.6, color="#1B4B80", zorder=3)
    folds = [c for c in ladder.columns if c.startswith("fold_")]
    for c in folds:
        ax.plot(k, ladder[c], lw=0.7, color=P.MUTED, alpha=0.65, zorder=2)
    for name, v in baselines.items():
        if np.isfinite(v):
            ax.axhline(v, lw=0.9, ls=(0, (4, 2)), color="#C0392B")
            # above the line and inside the axes: at the right edge it sat on the line
            # itself and ran off the panel
            ax.text(0.015, v, f"{name} ({v:.2f})", transform=ax.get_yaxis_transform(),
                    va="bottom", ha="left", fontsize=6.5, color="#C0392B")
    ax.set_xticks(k)
    ax.set_xticklabels([f"+{l}" for l in ladder.label], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel(f"Spearman on a held-out {split}")
    ax.set_title("(a) What each added feature buys", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    c = coefs.reindex(coefs.pooled.abs().sort_values().index)
    y = np.arange(len(c))
    ax.barh(y, c.pooled, color=np.where(c.pooled > 0, "#2F6DB5", "#C0392B"),
            height=min(0.6, 0.12 * len(c) + 0.2))
    per = [x for x in coefs.columns if x not in ("feature", "label", "pooled",
                                                 "per_protein_sd", "sign_agreement")]
    for ds in per:
        ax.plot(c[ds], y, "o", ms=2.6, color=P.MUTED, alpha=0.75)
    ax.axvline(0, color=P.INK, lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels(c.label, fontsize=6.5)
    ax.set_xlabel("Standardised coefficient\n(grey dots = refit on one protein)")
    ax.set_title("(b) The model, and how much it moves", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    if folds:
        last = ladder.iloc[-1]
        vals = pd.Series({c[5:]: last[c] for c in folds}).sort_values()
        yy = np.arange(len(vals))
        ax.barh(yy, vals.to_numpy(), color="#1B4B80", height=0.6)
        ax.set_yticks(yy); ax.set_yticklabels(vals.index, fontsize=6.5)
        ax.axvline(0, color=P.INK, lw=0.7)
        ax.set_xlabel("Spearman, that fold held out")
    ax.set_title(f"(c) Per held-out {split}", loc="left", fontsize=9)
    P.save(fig, outdir, "l01_feature_ladder", None, "L01")
    plt.close(fig)


def equation(coefs: pd.DataFrame, cols: list[str]) -> str:
    c = coefs.set_index("feature").reindex(cols)
    parts = [f"{c.loc[f, 'pooled']:+.3f}·z({f})" for f in cols]
    return "score_z ≈ " + " ".join(parts)


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms ladder",
        description="Build the smallest pooled predictor that still predicts.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--split", default="family", choices=["family", "protein"])
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--min-gain", type=float, default=MIN_GAIN)
    ap.add_argument("--no-esm", action="store_true",
                    help="leave ESM-1v out, to see what the biophysics carries alone")
    ap.add_argument("--family-threshold", type=float, default=0.40)
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)

    loaded = []
    for c in a.configs:
        cfg = load_config(Path(c))
        loaded.append((load_dataset(cfg), cfg))
        print(f"  loaded {cfg.id}", flush=True)
    feats = build(loaded)
    feats = feats[feats.score_z.notna()].reset_index(drop=True)
    seqs = {cfg.id: (protein_sequence(cfg) or "") for _, cfg in loaded}
    fam, _ = assign_families(seqs, a.family_threshold)
    feats["family"] = feats.dataset_id.map(fam).fillna(feats.dataset_id)
    groups = sorted(set(feats.family))
    print(f"  {len(feats):,} variants, {feats.dataset_id.nunique()} proteins, "
          f"{len(groups)} families: {', '.join(groups)}\n", flush=True)

    cands = [c for c in list(FEATURES) + ["esm1v"]
             if c in feats.columns and feats[c].notna().mean() > 0.5]
    if a.no_esm:
        cands = [c for c in cands if c != "esm1v"]

    base = {}
    if "esm1v" in feats.columns and feats.esm1v.notna().mean() > 0.5 and not a.no_esm:
        base["ESM-1v alone"] = cv_score(feats, ["esm1v"], a.split)["mean"]
    base["all features"] = cv_score(feats, cands, a.split)["mean"]

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("  building the ladder (each step is scored on held-out families) ...", flush=True)
    lad, chosen = forward_ladder(feats, cands, a.split, a.max_steps, a.min_gain)
    if not len(lad) or not chosen:
        raise SystemExit("no feature could be scored; too few folds for this split?")
    lad.to_csv(outdir / "ladder.csv", index=False)
    coefs = final_model(feats, chosen)
    coefs.to_csv(outdir / "coefficients.csv", index=False)
    figure(lad, coefs, base, outdir, a.split)

    print(f"\n  {'step':<5}{'feature':<34}{'CV ρ':>7}{'gain':>8}{'worst fold':>12}")
    for r in lad.itertuples():
        mark = "  (stopped: gain too small)" if r.feature not in chosen else ""
        print(f"  {r.step:<5}{r.label:<34}{r.cv_spearman:>7.3f}{r.gain:>+8.3f}"
              f"{r.worst_fold:>12.3f}{mark}")
    print("\n  baselines on the same split:")
    for k, v in base.items():
        print(f"    {k:<16}{v:>7.3f}")
    print(f"\n  the model ({len(chosen)} features):\n    {equation(coefs, chosen)}")
    if "sign_agreement" in coefs:
        flip = coefs[coefs.sign_agreement < 0.75]
        if len(flip):
            print("\n  WARNING: these coefficients change sign between proteins, so the pooled "
                  "value is an average of disagreeing numbers:")
            for r in flip.itertuples():
                print(f"    {r.label:<32} pooled {r.pooled:+.3f}, "
                      f"sign agrees in {r.sign_agreement:.0%} of proteins")
    (outdir / "ladder.json").write_text(json.dumps(
        {"chosen": chosen, "baselines": base, "equation": equation(coefs, chosen),
         "split": a.split, "families": groups}, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
