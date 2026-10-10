"""`python -m mpdms predict --family hexose=... --human SLC2A1=P11166 [--table clinvar.txt.gz]`

What the family predicts, residue by residue, for one protein nobody measured.

The useful quantity is not the pooled mean on its own. A column where the family agrees on
a mild effect and a column where it disagrees wildly can share a mean, and they say very
different things about an unmeasured member. So every residue is scored by the posterior
predictive for a new member of the family,

    effect at this position ~ Normal(mu, tau^2 + var(mu))

and the call is P(effect < threshold): the probability that a protein the family has never
seen loses abundance there. A column the family is unsure about produces a probability near
one half, which is the honest answer rather than a confident one.
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
from .clinvar import family_with_queries, fetch_sequence, read_clinvar
from .config import REPO_ROOT
from .stats import bh_fdr

OUT = REPO_ROOT / "outputs" / "_predict"
LOSS = -0.5        # score_z below this is a clear loss of abundance (as in the benchmark)
CALL = 0.80        # posterior probability at which a position is called


def predict_residues(seq: str, posmap: dict, cols: pd.DataFrame,
                     threshold: float = LOSS, call_at: float = CALL) -> pd.DataFrame:
    """One row per residue of the query protein."""
    by = cols.set_index("col")
    rows = []
    for i, wt in enumerate(seq, start=1):
        col = posmap.get(i)
        rec = {"pos": i, "wt": wt, "col": col}
        if col is not None and col in by.index:
            c = by.loc[col]
            sd = float(c.pred_sd) if np.isfinite(c.pred_sd) and c.pred_sd > 0 else np.nan
            rec.update(mu=float(c.mu), mu_lo=float(c.mu_lo), mu_hi=float(c.mu_hi),
                       tau=float(c.tau), pred_sd=sd, n_proteins=int(c.J),
                       seg_type=c.get("seg_type"),
                       p_loss=float(ss.norm.cdf(threshold, float(c.mu), sd))
                       if np.isfinite(sd) else np.nan)
        rows.append(rec)
    d = pd.DataFrame(rows)
    # a protein where nothing maps must still come back with the full set of columns,
    # or every downstream read of it raises instead of reporting zero coverage
    for c in ("mu", "mu_lo", "mu_hi", "tau", "pred_sd", "p_loss", "n_proteins", "seg_type"):
        if c not in d:
            d[c] = np.nan
    d["mapped"] = d.mu.notna()
    d["call"] = np.where(d.p_loss.isna(), "unmapped",
                         np.where(d.p_loss >= call_at, "loss",
                                  np.where(d.p_loss <= 1 - call_at, "tolerated", "uncertain")))
    return d


def summarise(d: pd.DataFrame, gene: str, threshold: float, call_at: float = CALL) -> dict:
    m = d[d.mapped]
    out = {"gene": gene, "n_residues": int(len(d)), "n_mapped": int(len(m)),
           "frac_mapped": float(len(m) / max(len(d), 1)),
           "threshold": threshold, "call_at": call_at}
    for k in ("loss", "tolerated", "uncertain"):
        out[f"n_{k}"] = int((d.call == k).sum())
        out[f"frac_{k}_of_mapped"] = float((d.call == k).sum() / max(len(m), 1))
    out["n_p_loss_over_half"] = int((m.p_loss > 0.5).sum())
    out["expected_losses"] = float(m.p_loss.sum())   # the sum of probabilities, not a count
    if "seg_type" in m and m.seg_type.notna().any():
        for seg, g in m.dropna(subset=["seg_type"]).groupby("seg_type"):
            out[f"n_loss_{seg}"] = int((g.call == "loss").sum())
            out[f"n_mapped_{seg}"] = int(len(g))
            out[f"frac_loss_{seg}"] = float((g.call == "loss").mean())
    return out


def clinvar_crosstab(d: pd.DataFrame, cv: pd.DataFrame, gene: str) -> dict:
    """Are the predicted loss positions where the pathogenic variants are?"""
    v = cv[cv.gene == gene].merge(d[["pos", "call", "p_loss", "mu"]], on="pos", how="inner")
    v = v[v.call != "unmapped"]
    if not len(v):
        return {"reason": "no classified variant lands on a predicted position"}
    tab = pd.crosstab(v.call == "loss", v.label == "pathogenic")
    out = {"n_variants": int(len(v)),
           "n_pathogenic_at_loss": int(((v.call == "loss") & (v.label == "pathogenic")).sum()),
           "n_benign_at_loss": int(((v.call == "loss") & (v.label == "benign")).sum()),
           "n_pathogenic_elsewhere": int(((v.call != "loss") & (v.label == "pathogenic")).sum()),
           "n_benign_elsewhere": int(((v.call != "loss") & (v.label == "benign")).sum())}
    if tab.shape == (2, 2):
        odds, p = ss.fisher_exact(tab.to_numpy())
        out["fisher_p"] = float(p)
        out["odds_ratio"] = float(odds)
        if not np.isfinite(odds):
            # an empty cell sends the odds ratio to infinity, which cannot be reported or
            # plotted; Haldane's correction adds a half to every cell so it stays finite
            a_, b_, c_, d_ = (tab.to_numpy() + 0.5).ravel()
            out["odds_ratio_haldane"] = float((a_ * d_) / (b_ * c_))
    at = v[v.call == "loss"]
    other = v[v.call != "loss"]
    if len(at):
        out["frac_pathogenic_at_loss"] = float((at.label == "pathogenic").mean())
    if len(other):
        out["frac_pathogenic_elsewhere"] = float((other.label == "pathogenic").mean())
    return out


def figure(gene: str, d: pd.DataFrame, cv: pd.DataFrame | None, outdir: Path, name: str,
           threshold: float = LOSS, call_at: float = CALL):
    fig = plt.figure(figsize=(11.0, 5.6), layout="constrained")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 0.75, 0.9],
                          width_ratios=[1, 1, 1])
    cm = P.sequential_cmap()

    ax = fig.add_subplot(gs[0, :])
    m = d[d.mapped]
    if "seg_type" in d and d.seg_type.notna().any():
        tm = (d.seg_type == "TM").to_numpy()
        ax.fill_between(d.pos, 0, 1, where=tm, transform=ax.get_xaxis_transform(),
                        color=P.TM_GREY, lw=0, zorder=0, step="mid")
    ax.fill_between(m.pos, m.mu_lo, m.mu_hi, color=P.SERIES[0], alpha=0.20, lw=0)
    ax.plot(m.pos, m.mu, lw=0.9, color=P.SERIES[0])
    loss = d[d.call == "loss"]
    ax.scatter(loss.pos, loss.mu, s=9, color="#C0392B", zorder=4, lw=0,
               label=f"predicted loss (P ≥ {call_at:.0%}), n = {len(loss)}")
    ax.axhline(threshold, color="#C0392B", lw=0.7, ls=(0, (3, 3)))
    ax.axhline(0, color=P.MUTED, lw=0.6)
    ax.set_ylabel("Predicted effect")
    ax.set_xlabel(f"{gene} residue")
    # the whole protein, not just the part that mapped: a gap in the trace is the honest
    # picture of coverage, and an axis zoomed to the mapped region hides it
    ax.set_xlim(1, len(d))
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title(f"(a) {gene}: what the family predicts along the sequence "
                 "(grey = transmembrane)", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[1, :])
    ax.fill_between(m.pos, 0, m.p_loss, color=cm(0.45), alpha=0.55, lw=0, step="mid")
    ax.axhline(call_at, color="#C0392B", lw=0.7, ls=(0, (3, 3)))
    ax.axhline(0.5, color=P.MUTED, lw=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlim(1, len(d))
    ax.set_ylabel("P(loss)")
    ax.set_xlabel(f"{gene} residue")
    ax.set_title("(b) Posterior probability of losing abundance", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[2, 0])
    ax.hist(m.p_loss.dropna(), bins=25, color=cm(0.45), lw=0)
    ax.axvline(call_at, color="#C0392B", lw=0.9, ls=(0, (3, 3)))
    ax.set_xlabel("P(loss)"); ax.set_ylabel("Residues")
    ax.set_title("(c) How confident", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[2, 1])
    if "seg_type" in m and m.seg_type.notna().any():
        g = (m.dropna(subset=["seg_type"]).groupby("seg_type")
             .agg(frac=("call", lambda v: float((v == "loss").mean())), n=("call", "size"))
             .sort_values("frac"))
        y = np.arange(len(g))
        ax.barh(y, g.frac, color=[cm(0.2 + 0.6 * v) for v in g.frac], height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{i}  (n = {int(r)})" for i, r in zip(g.index, g.n)], fontsize=6.5)
        ax.set_xlabel("Fraction called loss")
    ax.set_title("(d) By topology", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[2, 2])
    if cv is not None and len(cv):
        v = cv.merge(d[["pos", "call"]], on="pos", how="inner")
        v = v[v.call != "unmapped"]
        if len(v):
            tab = (pd.crosstab(v.call == "loss", v.label, normalize="index")
                   .reindex([False, True]).fillna(0))
            lab = ["elsewhere", "predicted loss"]
            bot = np.zeros(len(tab))
            for cls, col in (("benign", "#1B9E77"), ("pathogenic", "#C0392B")):
                if cls in tab:
                    ax.bar(range(len(tab)), tab[cls], bottom=bot, color=col, width=0.62,
                           label=cls)
                    bot = bot + tab[cls].to_numpy()
            ax.set_xticks(range(len(tab)))
            ax.set_xticklabels(lab, fontsize=6.5)
            ax.set_ylabel("Fraction of variants")
            ax.legend(frameon=False, fontsize=6)
            for k, n in enumerate(pd.crosstab(v.call == "loss", v.label)
                                  .reindex([False, True]).fillna(0).sum(axis=1)):
                ax.text(k, 1.01, f"n = {int(n)}", ha="center", va="bottom", fontsize=6,
                        color=P.MUTED, transform=ax.get_xaxis_transform())
    ax.set_title("(e) ClinVar at predicted positions", loc="left", fontsize=9)
    P.save(fig, outdir, f"p01_{name}_{gene}", None, "P01")
    plt.close(fig)


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms predict",
        description="Per-residue predictions for a protein the family has not measured.")
    ap.add_argument("--family", required=True, metavar="NAME=cfg1,cfg2,...")
    ap.add_argument("--human", action="append", required=True, metavar="GENE=ACCESSION")
    ap.add_argument("--table", help="optional ClinVar table, to cross-tabulate the calls")
    ap.add_argument("--threshold", type=float, default=LOSS,
                    help=f"effect below which a position counts as lost (default {LOSS})")
    ap.add_argument("--call-at", type=float, default=CALL,
                    help=f"posterior probability needed to call a position (default {CALL})")
    ap.add_argument("--min-proteins", type=int, default=3)
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)

    name, _, members = a.family.partition("=")
    members = [m.strip() for m in members.split(",") if m.strip()]
    bad = [m for m in members if not Path(m).is_file()]
    if bad:
        raise SystemExit(f"--family {name}: not a config file: {', '.join(bad)}")
    seqs = {}
    for h in a.human:
        g, _, acc = h.partition("=")
        if not acc:
            raise SystemExit(f"--human {h}: expected GENE=ACCESSION")
        seqs[g.strip().upper()] = fetch_sequence(acc.strip())
        print(f"  {g.strip().upper():<10} {acc.strip()}  {len(seqs[g.strip().upper()])} residues")

    cols, maps, aln, long = family_with_queries(members, seqs, a.min_proteins)
    print(f"  {len(cols):,} pooled columns from {long.protein.nunique()} measured proteins "
          f"({aln.attrs.get('method', '?')})", flush=True)
    cv = read_clinvar(Path(a.table), set(seqs)) if a.table else None

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for gene, seq in seqs.items():
        d = predict_residues(seq, maps.get(gene, {}), cols, a.threshold, a.call_at)
        gcv = cv[cv.gene == gene] if cv is not None else None
        if gcv is not None and len(gcv):
            # keep only variants whose wild type agrees with the sequence we predicted on
            gcv = gcv[[0 < p <= len(seq) and seq[p - 1] == w
                       for p, w in zip(gcv.pos, gcv.wt)]]
        d.to_csv(outdir / f"{name}_{gene}_residues.csv", index=False)
        s = summarise(d, gene, a.threshold, a.call_at)
        if gcv is not None and len(gcv):
            s["clinvar"] = clinvar_crosstab(d, gcv, gene)
        out[gene] = s
        figure(gene, d, gcv, outdir, name, a.threshold, a.call_at)

        print(f"\n  == {gene}")
        print(f"    {s['n_mapped']:,} of {s['n_residues']:,} residues map onto a pooled "
              f"column ({s['frac_mapped']:.0%})")
        print(f"    predicted loss of abundance (P ≥ {a.call_at:.0%} that the effect is below "
              f"{a.threshold:+.2f}): {s['n_loss']:,} positions "
              f"({s['frac_loss_of_mapped']:.0%} of mapped)")
        print(f"    tolerated: {s['n_tolerated']:,}   uncertain: {s['n_uncertain']:,}")
        print(f"    positions more likely than not to be lost: {s['n_p_loss_over_half']:,}; "
              f"expected number lost, summing the probabilities: {s['expected_losses']:.0f}")
        for seg in ("TM", "loop", "soluble"):
            if f"n_mapped_{seg}" in s:
                print(f"      {seg:<9} {s['n_loss_' + seg]:>4} of {s['n_mapped_' + seg]:>4} "
                      f"({s['frac_loss_' + seg]:.0%})")
        c = s.get("clinvar", {})
        if "n_variants" in c:
            print(f"    ClinVar on these positions ({c['n_variants']} variants): "
                  f"{c['n_pathogenic_at_loss']}P/{c['n_benign_at_loss']}B at predicted-loss "
                  f"positions vs {c['n_pathogenic_elsewhere']}P/{c['n_benign_elsewhere']}B "
                  "elsewhere")
            if "odds_ratio" in c:
                o = (f"{c['odds_ratio']:.2f}" if np.isfinite(c["odds_ratio"]) else
                     f"infinite (a cell is empty); {c.get('odds_ratio_haldane', float('nan')):.2f} "
                     "with Haldane's correction")
                print(f"      odds ratio {o}, Fisher p = {c['fisher_p']:.3g}")
            if "frac_pathogenic_at_loss" in c and "frac_pathogenic_elsewhere" in c:
                print(f"      pathogenic fraction: {c['frac_pathogenic_at_loss']:.0%} at "
                      f"predicted-loss vs {c['frac_pathogenic_elsewhere']:.0%} elsewhere")
        elif c:
            print(f"    ClinVar: {c.get('reason')}")
    (outdir / f"{name}_predictions.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
