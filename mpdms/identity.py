"""`python -m mpdms identity configs/*.yaml [--human SLC2A1=P11166 ...]`

Pairwise sequence identity across the proteins, as the context for every pooled result.

A hierarchical family model borrows strength between its members. That helps while the
members really do share constraint, and hurts once they do not: adding a distant relative
raises the between-protein spread at every column, which shrinks the pooled mean toward the
family average and flattens exactly the contrast a predictor needs. The identity matrix is
how you tell those two regimes apart before reading anything into a pooled number.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P
from .config import REPO_ROOT, load_config, protein_sequence

OUT = REPO_ROOT / "outputs" / "_identity"


def pairwise_identity(seqs: dict[str, str]) -> pd.DataFrame:
    """Global alignment identity, as a fraction of the shorter sequence.

    Normalising by the shorter sequence rather than the alignment length keeps a long
    extension from reading as divergence when the shared region is in fact well conserved.
    """
    from Bio import Align
    ids = sorted(seqs)
    al = Align.PairwiseAligner(scoring="blastp", mode="global")
    M = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if not seqs[a] or not seqs[b]:
                M.loc[a, b] = M.loc[b, a] = np.nan
                continue
            aln = al.align(seqs[a], seqs[b])[0]
            sa, sb = str(aln[0]), str(aln[1])
            same = sum(x == y for x, y in zip(sa, sb) if x != "-")
            M.loc[a, b] = M.loc[b, a] = same / max(1, min(len(seqs[a]), len(seqs[b])))
    return M


def cluster_order(M: pd.DataFrame) -> list[str]:
    """Order rows so relatives sit together; the block structure is the point of the plot."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    D = 1.0 - M.to_numpy(float)
    D = np.nan_to_num(D, nan=float(np.nanmax(D)))
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    try:
        return [M.index[i] for i in leaves_list(linkage(squareform(D, checks=False), "average"))]
    except Exception:
        return list(M.index)


def figure(M: pd.DataFrame, order: list[str], groups: dict[str, str], outdir: Path,
           name: str = "identity"):
    n = len(order)
    has_q = any(groups.get(g) == "query" for g in order)
    fig, axes = plt.subplots(1, 2 if has_q else 1,
                             figsize=(0.42 * n + (7.6 if has_q else 4.6), 0.42 * n + 2.0),
                             layout="constrained",
                             gridspec_kw={"width_ratios": [1, 0.85]} if has_q else None,
                             squeeze=False)
    ax = axes[0][0]
    V = M.loc[order, order].to_numpy(float) * 100
    im = ax.imshow(V, cmap=P.sequential_cmap(), vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=6.5)
    ax.set_yticklabels(order, fontsize=6.5)
    for i in range(n):
        for j in range(n):
            v = V[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.2,
                        color="white" if v < 62 else "#111111")
    # a query protein is not one of the measured set; mark it so no one reads it as data
    for k, g in enumerate(order):
        if groups.get(g) == "query":
            for a in (ax.get_xticklabels()[k], ax.get_yticklabels()[k]):
                a.set_style("italic"); a.set_color(P.SERIES[0])
    fig.colorbar(im, ax=ax, fraction=0.035).set_label("% identity", fontsize=7)
    ax.set_title("(a) Pairwise identity (italic = not measured)", loc="left", fontsize=9)
    ax.tick_params(length=0)

    if has_q:
        ax = axes[0][1]
        meas = [g for g in order if groups.get(g) != "query"]
        qs = [g for g in order if groups.get(g) == "query"]
        y = np.arange(len(qs))
        cm = P.sequential_cmap()
        for k, q in enumerate(qs):
            v = M.loc[q, meas].astype(float) * 100
            ax.scatter(v, np.full(len(v), k), s=26, color=[cm(x / 100) for x in v],
                       edgecolor=P.INK, lw=0.3, zorder=3)
            ax.plot([v.min(), v.max()], [k, k], color=P.MUTED, lw=0.8, zorder=2)
            ax.text(v.max() + 1.6, k, f"max {v.max():.0f}%  median {np.median(v):.0f}%",
                    va="center", fontsize=6, color=P.MUTED)
        ax.set_yticks(y); ax.set_yticklabels(qs, fontsize=7)
        ax.set_xlabel("% identity to each measured protein")
        ax.set_xlim(0, max(60, float(np.nanmax(M.loc[qs, meas].to_numpy(float)) * 100) + 14))
        ax.set_title("(b) How far the query sits from the data", loc="left", fontsize=9)
    P.save(fig, outdir, f"i01_{name}", None, "I01")
    plt.close(fig)


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms identity",
        description="Pairwise sequence identity across the measured proteins.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--human", action="append", default=[], metavar="GENE=ACCESSION",
                    help="an unmeasured protein to place against them, e.g. SLC2A1=P11166")
    ap.add_argument("--name", default="identity")
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)

    seqs, groups = {}, {}
    for c in a.configs:
        cfg = load_config(Path(c))
        s = protein_sequence(cfg)
        if not s:
            print(f"  skipping {cfg.id}: no sequence in the config")
            continue
        seqs[cfg.id] = s
        groups[cfg.id] = "measured"
    from .clinvar import fetch_sequence
    for h in a.human:
        g, _, acc = h.partition("=")
        if not acc:
            raise SystemExit(f"--human {h}: expected GENE=ACCESSION")
        seqs[g.strip().upper()] = fetch_sequence(acc.strip())
        groups[g.strip().upper()] = "query"
    if len(seqs) < 2:
        raise SystemExit("need at least two sequences")

    print(f"  {len(seqs)} sequences, {len(seqs) * (len(seqs) - 1) // 2} pairs ...", flush=True)
    M = pairwise_identity(seqs)
    order = cluster_order(M)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (M * 100).round(1).to_csv(outdir / f"{a.name}_percent.csv")
    figure(M, order, groups, outdir, a.name)

    meas = [g for g in order if groups[g] == "measured"]
    print("\n  measured set:")
    sub = M.loc[meas, meas].to_numpy(float)
    off = sub[~np.eye(len(meas), dtype=bool)]
    print(f"    identity between members: median {np.median(off) * 100:.0f}%, "
          f"range {off.min() * 100:.0f}-{off.max() * 100:.0f}%")
    for g in order:
        if groups[g] == "query":
            v = M.loc[g, meas].astype(float) * 100
            print(f"    {g:<10} closest measured protein: {v.idxmax()} at {v.max():.0f}%, "
                  f"median {np.median(v):.0f}%")
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
