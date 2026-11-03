"""`python -m mpdms transfer --measured outputs/_dual/d02_sites.csv --prediction ...`

Does what yeast knows about abundance carry over to a human transporter measured both
ways?

`bayes` pools a yeast family column by column and `predict` turns that into one number
per residue of a human query. Both are built entirely out of **abundance** assays. This
asks what those numbers are worth against a human protein where abundance and function
were measured in the same library, which is the only setting where the question has a
clean answer.

    t01  every predictor against every measured quantity, one panel each
    t02  the same correlations side by side, with bootstrap intervals

The three measured quantities are not interchangeable and the comparison between them
is the result:

    abundance       what the yeast assays also measured. A family abundance model that
                    cannot predict this is not working at all.
    function        abundance plus everything else the protein has to do. A model built
                    from abundance should predict it, but only through abundance.
    residual        function with abundance regressed out - D02's quantity. An
                    abundance model has no access to this by construction, so a
                    correlation here is the interesting failure: it would mean the
                    yeast assays were reading something beyond abundance all along.

So the prediction the module states before it looks is rho(abundance) > rho(function)
>> rho(residual) ~ 0, and t02 is drawn to make that ordering, or its absence, visible.
Correlations are Spearman over positions, not variants: these predictors are
position-scored, every substitution at a residue gets the same number, and treating
variants as independent would inflate n by twenty and shrink every interval to nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .config import REPO_ROOT

OUT = REPO_ROOT / "outputs" / "_transfer"
N_BOOT = 2000
MIN_N = 20

# measured column in d02_sites.csv -> how it is labelled and what is expected of it
MEASURED = [
    ("median_abundance", "measured abundance",
     "what the yeast assays also measured"),
    ("median_function", "measured function",
     "abundance plus everything else"),
    ("median_z", "residual (function | abundance)",
     "what an abundance model cannot see"),
]


def boot_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                  seed: int = 0) -> dict:
    """Spearman with a percentile bootstrap over positions."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < MIN_N:
        return {"n": int(len(x)), "rho": np.nan, "p": np.nan,
                "lo": np.nan, "hi": np.nan}
    r, p = ss.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = np.array([ss.spearmanr(x[i], y[i]).statistic for i in idx])
    bs = bs[np.isfinite(bs)]
    return {"n": int(len(x)), "rho": float(r), "p": float(p),
            "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def read_measured(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    missing = [c for c, _, _ in MEASURED if c not in t.columns]
    if "pos" not in t.columns or missing:
        raise SystemExit(f"{path}: wants the d02_sites.csv written by `mpdms dual` "
                         f"(missing {['pos'] * ('pos' not in t.columns) + missing})")
    return t


def read_prediction(path: Path) -> pd.DataFrame:
    """A `*_residues.csv` from `mpdms predict`: mu is the pooled family effect."""
    d = pd.read_csv(path)
    if "pos" not in d.columns or "mu" not in d.columns:
        raise SystemExit(f"{path}: wants a *_residues.csv from `mpdms predict` "
                         f"(columns pos, mu); found {list(d.columns)}")
    return d[["pos", "mu"]].dropna()


def read_donor(path: Path) -> pd.DataFrame:
    """One yeast protein's own site scores, already on the target's numbering.

    Produced by joining that protein's site medians onto an alignment column map, e.g.
    data/alignments/aqr1_slc22a1_columns.csv. Kept as a plain two-column file because
    the mapping is the part worth inspecting by hand, and burying it inside this module
    would hide the thing most likely to be wrong.
    """
    d = pd.read_csv(path)
    cols = {c.lower(): c for c in d.columns}
    pos = cols.get("pos") or cols.get("target_pos") or cols.get("slc22a1_pos")
    sc = cols.get("score") or cols.get("median_score") or cols.get("site_median")
    if not pos or not sc:
        raise SystemExit(f"{path}: wants columns pos (target numbering) and score; "
                         f"found {list(d.columns)}")
    return d[[pos, sc]].rename(columns={pos: "pos", sc: "mu"}).dropna()


# ------------------------------------------------------- t01 / t02, the comparison
def correlate(t: pd.DataFrame, predictors: dict) -> pd.DataFrame:
    rows = []
    for pname, col in predictors.items():
        for mcol, mlabel, _ in MEASURED:
            s = boot_spearman(t[col].to_numpy(float), t[mcol].to_numpy(float))
            rows.append({"predictor": pname, "measured": mlabel,
                         "measured_col": mcol, **s})
    return pd.DataFrame(rows)


def t01_panels(t: pd.DataFrame, predictors: dict, stats: pd.DataFrame, name: str,
               outdir: Path) -> dict:
    nrow, ncol = len(predictors), len(MEASURED)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.75 * ncol, 2.6 * nrow + 0.4),
                             squeeze=False, layout="constrained")
    for i, (pname, col) in enumerate(predictors.items()):
        for j, (mcol, mlabel, note) in enumerate(MEASURED):
            ax = axes[i][j]
            d = t[[col, mcol]].dropna()
            called = t.loc[t.functional.astype(bool), [col, mcol]].dropna() \
                if "functional" in t.columns else d.iloc[:0]
            ax.scatter(d[col], d[mcol], s=7, color=P.NA_COLOR, lw=0, alpha=0.7,
                       rasterized=True)
            if len(called):
                ax.scatter(called[col], called[mcol], s=14, color="#B2182B", lw=0.25,
                           edgecolor=P.INK, zorder=3,
                           label=f"called functional (n = {len(called)})")
            r = stats[(stats.predictor == pname) &
                      (stats.measured_col == mcol)].iloc[0]
            ax.set_title(
                "too few mapped positions" if not np.isfinite(r.rho) else
                f"ρ = {r.rho:+.2f}  [{r.lo:+.2f}, {r.hi:+.2f}]  n = {r.n}",
                loc="left", fontsize=7.5)
            ax.axhline(0, color=P.MUTED, lw=0.5)
            if j == 0:
                ax.set_ylabel(f"{pname}\n\n{mlabel}", fontsize=7)
            else:
                ax.set_ylabel(mlabel, fontsize=7)
            ax.set_xlabel(f"{pname} predicted effect", fontsize=7)
            if i == 0 and j == ncol - 1 and len(called):
                ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.suptitle(f"{name}: a family model built from yeast abundance assays, against "
                 f"abundance, function and the part of function abundance does not "
                 f"explain.", fontsize=7.5, x=0.01, ha="left")
    P.save(fig, outdir, "t01_predictor_against_measured", None, "T01")
    return {"status": "ok"}


def t02_summary(stats: pd.DataFrame, name: str, outdir: Path) -> dict:
    fig, ax = plt.subplots(figsize=(5.6, 0.52 * len(stats) + 1.3),
                           layout="constrained")
    y = np.arange(len(stats))[::-1]
    colors = {m[1]: c for m, c in zip(MEASURED, ("#2166AC", "#4D4D4D", "#B2182B"))}
    for yy, r in zip(y, stats.itertuples()):
        if not np.isfinite(r.rho):
            continue
        ax.plot([r.lo, r.hi], [yy, yy], color=colors[r.measured], lw=1.4,
                solid_capstyle="round")
        ax.scatter([r.rho], [yy], s=26, color=colors[r.measured], lw=0.3,
                   edgecolor=P.INK, zorder=3)
    ax.axvline(0, color=P.MUTED, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.predictor}  →  {r.measured}"
                        for r in stats.itertuples()], fontsize=6.5)
    ax.set_xlabel("Spearman ρ over positions (95% bootstrap interval)",
                  fontsize=7.5)
    ax.set_title(f"{name}: an abundance model should rank abundance best and the "
                 f"residual not at all.", loc="left", fontsize=7.5)
    P.save(fig, outdir, "t02_correlation_summary", None, "T02")
    return {"status": "ok"}


def verdict(stats: pd.DataFrame) -> dict:
    """Is the ordering the one an abundance-only model has to produce?"""
    out = {}
    for pname, g in stats.groupby("predictor"):
        g = g.set_index("measured_col")
        a, f, z = (g.rho.get("median_abundance", np.nan),
                   g.rho.get("median_function", np.nan),
                   g.rho.get("median_z", np.nan))
        res = g.loc["median_z"] if "median_z" in g.index else None
        residual_null = bool(res is not None and np.isfinite(res.lo)
                             and res.lo <= 0 <= res.hi)
        out[pname] = {
            "rho_abundance": None if not np.isfinite(a) else float(a),
            "rho_function": None if not np.isfinite(f) else float(f),
            "rho_residual": None if not np.isfinite(z) else float(z),
            "abundance_beats_function": bool(np.isfinite(a) and np.isfinite(f)
                                             and abs(a) > abs(f)),
            "residual_interval_covers_zero": residual_null,
        }
    return out


def run(measured: Path, predictions: dict, donors: dict, name: str,
        outdir: Path) -> dict:
    t = read_measured(measured)
    predictors = {}
    for pname, path in predictions.items():
        d = read_prediction(Path(path)).rename(columns={"mu": f"pred_{pname}"})
        t = t.merge(d, on="pos", how="left")
        predictors[pname] = f"pred_{pname}"
    for dname, path in donors.items():
        d = read_donor(Path(path)).rename(columns={"mu": f"pred_{dname}"})
        t = t.merge(d, on="pos", how="left")
        predictors[dname] = f"pred_{dname}"
    if not predictors:
        raise SystemExit("give at least one --prediction or --donor")

    outdir.mkdir(parents=True, exist_ok=True)
    stats = correlate(t, predictors)
    t.to_csv(outdir / "t01_merged.csv", index=False)
    stats.to_csv(outdir / "t02_correlations.csv", index=False)
    res = {"t01": t01_panels(t, predictors, stats, name, outdir),
           "t02": t02_summary(stats, name, outdir),
           "verdict": verdict(stats),
           "correlations": stats.to_dict("records"),
           "n_positions": int(len(t))}
    (outdir / "transfer.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def _pairs(items: list[str], flag: str) -> dict:
    out = {}
    for s in items:
        if "=" not in s:
            raise SystemExit(f"{flag} wants NAME=FILE, got {s!r}")
        k, v = s.split("=", 1)
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mpdms transfer", description=__doc__.strip())
    ap.add_argument("--measured", default=str(REPO_ROOT / "outputs" / "_dual"
                                              / "d02_sites.csv"))
    ap.add_argument("--prediction", action="append", default=[], metavar="NAME=FILE",
                    help="a *_residues.csv from `mpdms predict`. Repeatable.")
    ap.add_argument("--donor", action="append", default=[], metavar="NAME=FILE",
                    help="one yeast protein's site scores on the target's numbering "
                         "(columns pos,score). Repeatable.")
    ap.add_argument("--id", default="SLC22A1")
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)

    res = run(Path(a.measured), _pairs(a.prediction, "--prediction"),
              _pairs(a.donor, "--donor"), a.id, Path(a.outdir))
    print(f"\n  {a.id}  {res['n_positions']} positions\n")
    for r in res["correlations"]:
        if not np.isfinite(r["rho"]):
            print(f"  {r['predictor']:>12s} -> {r['measured']:<32s} "
                  f"too few mapped positions (n {r['n']})")
            continue
        print(f"  {r['predictor']:>12s} -> {r['measured']:<32s} "
              f"rho {r['rho']:+.2f}  [{r['lo']:+.2f}, {r['hi']:+.2f}]  n {r['n']}")
    print()
    for pname, v in res["verdict"].items():
        print(f"  {pname}: abundance ranked above function: "
              f"{v['abundance_beats_function']};  residual interval covers zero: "
              f"{v['residual_interval_covers_zero']}")
    print(f"\n  -> {a.outdir}/")
    return 0
