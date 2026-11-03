"""`python -m mpdms transfer --family hexose=configs/HXT1.yaml,... --query SLC22A1=O15245`

Does what a measured family knows transfer to a protein nobody measured in it?

The family model in `bayes` estimates, per alignment column, what the family shares.
This asks whether that number says anything about a different protein's own
measurements at the residues that align to those columns - and, beside it, whether one
chosen member on its own says as much.

    x01  how good is the alignment, really - the question everything else depends on
    x02  the family's column mean against the query's measured score, per readout
    x03  one named member against the query, at the same columns

**x01 comes first and is not a formality.** Position-level transfer is only as real as
the column assignment underneath it, and two MFS transporters from different families
sit at 20-25% identity, which is the twilight zone: an aligner returns columns, they
look like data, and a register slip of three residues turns a real relationship into
noise. A null result at that identity is uninterpretable - it cannot be told apart from
a bad alignment - so x01 reports each query's identity to each member against the
family's own internal identity, and the run says so loudly when a query is no closer to
the family than an unrelated protein would be.

The honest design is a **positive control**: pass a second `--query` that is a genuine
ortholog of the family. If the family transfers to the ortholog and not to the distant
one, that is a result. If it transfers to neither, the method is what failed, not the
biology.
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
from .clinvar import family_with_queries, fetch_sequence
from .config import REPO_ROOT
from .environs import grid, panel

OUT = REPO_ROOT / "outputs" / "_transfer"
MIN_SITES = 25            # aligned positions with a score, before a correlation is drawn
N_BOOT = 2000
TWILIGHT = 0.25           # identity at or below which an alignment cannot be trusted


# --------------------------------------------------------------------- loading
def read_scores(path: Path, cols: list[str]) -> pd.DataFrame:
    """Per-position medians of each named score column, from a tidy variant table.

    The table `mpdms dual` writes (`variants.csv`) is the expected input, but any csv
    with a `pos` column and one column per readout will do. Medians, not means: these
    distributions have a dead tail and a mean slides into it.
    """
    d = pd.read_csv(path)
    if "pos" not in d.columns:
        raise SystemExit(f"{path}: no `pos` column; found {list(d.columns)}")
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise SystemExit(f"{path}: no column(s) {missing}; found {list(d.columns)}")
    if "kind" in d.columns:          # keep only substitutions, never deletions
        d = d[d.kind == "missense"]
    d = d[d.pos.notna()].copy()
    d["pos"] = d.pos.astype(int)
    g = d.groupby("pos")[cols].median()
    g["n_variants"] = d.groupby("pos").size()
    return g.reset_index()


def boot_spearman(x, y, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Spearman with a percentile interval. Each row is already one position."""
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    if len(d) < MIN_SITES:
        return {"n": int(len(d)), "rho": np.nan}
    obs = ss.spearmanr(d.x, d.y)
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        if s.x.nunique() > 2 and s.y.nunique() > 2:
            draws[i] = ss.spearmanr(s.x, s.y).statistic
    draws = draws[np.isfinite(draws)]
    return {"n": int(len(d)), "rho": float(obs.statistic), "p": float(obs.pvalue),
            "lo": float(np.quantile(draws, 0.025)) if len(draws) else np.nan,
            "hi": float(np.quantile(draws, 0.975)) if len(draws) else np.nan}


# ------------------------------------------------- x01 is the alignment worth anything
def pair_identity(aln: pd.DataFrame, a: str, b: str) -> dict:
    """Identity and coverage over the columns where both proteins have a residue."""
    both = aln.dropna(subset=[f"{a}_pos", f"{b}_pos"])
    if not len(both):
        return {"pair": f"{a}–{b}", "a": a, "b": b, "aligned": 0, "identity": np.nan}
    ident = float((both[f"{a}_aa"].str.upper() == both[f"{b}_aa"].str.upper()).mean())
    return {"pair": f"{a}–{b}", "a": a, "b": b, "aligned": int(len(both)),
            "identity": ident}


def x01_alignment(aln: pd.DataFrame, members: list[str], queries: list[str],
                  outdir: Path) -> dict:
    """Every pairwise identity, with the family's own as the yardstick."""
    rows = [pair_identity(aln, a, b) for a, b in itertools.combinations(
        list(members) + list(queries), 2)]
    t = pd.DataFrame(rows)
    t["kind"] = np.where(t.a.isin(members) & t.b.isin(members), "within family",
                         np.where(t.a.isin(queries) & t.b.isin(queries),
                                  "query–query", "family–query"))
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(t) + 1.4), layout="constrained")
    s = t.sort_values("identity")
    colors = {"within family": "#2166AC", "family–query": "#B2182B",
              "query–query": "#6B6B6B"}
    ax.barh(np.arange(len(s)), s.identity, color=[colors[k] for k in s.kind], height=0.7)
    ax.axvline(TWILIGHT, color=P.INK, lw=0.9, ls=(0, (3, 3)))
    ax.text(TWILIGHT, len(s) - 0.2, "  twilight zone", fontsize=6.5, va="top",
            color=P.INK)
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels([f"{r.pair}  ({r.aligned})" for r in s.itertuples()], fontsize=6)
    ax.set_xlabel("identity over columns where both have a residue", fontsize=8)
    for k, c in colors.items():
        ax.barh([], [], color=c, label=k)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle("Blue is the family against itself - the yardstick. Red is each query "
                 "against each member. Below the dashed line an alignment returns "
                 "columns but cannot be trusted to have got the register right, and a "
                 "null result there says nothing about biology.",
                 fontsize=7, x=0.01, ha="left", wrap=True)
    P.save(fig, outdir, "x01_alignment_identity", None, "X01")
    plt.close(fig)
    t.to_csv(outdir / "x01_alignment_identity.csv", index=False)
    fam = t[t.kind == "within family"].identity
    worst = {}
    for q in queries:
        sub = t[(t.kind == "family–query") & ((t.a == q) | (t.b == q))]
        worst[q] = float(sub.identity.max()) if len(sub) else np.nan
    note = "; ".join(f"{q} best {v:.0%}" for q, v in worst.items())
    flag = [q for q, v in worst.items() if np.isfinite(v) and v <= TWILIGHT]
    return {"status": "ok", "within_family_median": float(fam.median()) if len(fam) else np.nan,
            "best_identity_to_family": worst, "below_twilight": flag,
            "headline": f"family itself {fam.median():.0%}; " + note
                        + (f"  — AT OR BELOW THE TWILIGHT ZONE: {', '.join(flag)}"
                           if flag else ""),
            "table": t.to_dict("records")}


# --------------------------------------- x02 the family's column mean against the query
def x02_family_vs_query(cols: pd.DataFrame, maps: dict, scores: dict,
                        score_cols: list[str], outdir: Path) -> dict:
    """Family mu at a column against what the query measured at the residue there."""
    rows, panels = [], []
    for q, site in scores.items():
        m = maps.get(q, {})
        if not m:
            continue
        s = site.assign(col=site.pos.map(m)).dropna(subset=["col"])
        s["col"] = s.col.astype(int)
        j = s.merge(cols[["col", "mu", "mu_sd", "tau", "J"]], on="col", how="inner")
        for c in score_cols:
            if c not in j:
                continue
            r = boot_spearman(j.mu, j[c], seed=abs(hash((q, c))) % 2**31)
            rows.append({"query": q, "score": c, **r})
            panels.append((q, c, j))
    if not panels:
        return {"status": "skipped", "reason": "no query residue mapped onto a pooled column"}
    names = [f"{q} · {c}" for q, c, _ in panels]
    fig, pairs, _ = grid(names, w=3.0, h=2.8, extra=0.3)
    for i, ((q, c, j), (_, ax)) in enumerate(zip(panels, pairs)):
        r = [x for x in rows if x["query"] == q and x["score"] == c][0]
        ax.scatter(j.mu, j[c], s=11, color="#2166AC", lw=0.2, edgecolor=P.INK, alpha=0.7)
        if np.isfinite(r.get("rho", np.nan)) and j.mu.nunique() > 2:
            b, a_ = np.polyfit(j.mu.astype(float), j[c].astype(float), 1)
            xs = np.array([j.mu.min(), j.mu.max()])
            ax.plot(xs, a_ + b * xs, color=P.INK, lw=1.2, ls=(0, (4, 2)))
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6)
        extra = (f"ρ {r['rho']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]  n {r['n']}"
                 if np.isfinite(r.get("rho", np.nan)) else
                 f"only {r['n']} aligned positions with a score")
        panel(ax, i, f"{q} · {c}", extra, size=7)
        if i % 2 == 0:
            ax.set_ylabel("the query's own site median", fontsize=7.5)
    fig.supxlabel("family μ at that alignment column (lower = more constrained)",
                  fontsize=8)
    fig.suptitle("Each point is one residue of the query that aligns to a column the "
                 "family pooled. Read this against x01: at twilight-zone identity a "
                 "flat cloud is as likely to be a misaligned register as an absence of "
                 "transfer.", fontsize=7, x=0.01, ha="left", wrap=True)
    P.save(fig, outdir, "x02_family_vs_query", None, "X02")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "x02_family_vs_query.csv", index=False)
    return {"status": "ok", "by_readout": t.to_dict("records"),
            "headline": "; ".join(f"{r.query}·{r.score} ρ {r.rho:+.2f} "
                                  f"[{r.lo:+.2f}, {r.hi:+.2f}] n {r.n}"
                                  for r in t.itertuples()
                                  if np.isfinite(r.rho))}


# ------------------------------------------- x03 one member against the query, directly
def x03_member_vs_query(long: pd.DataFrame, maps: dict, scores: dict,
                        score_cols: list[str], against: str, outdir: Path) -> dict:
    """One measured protein's own site effects against the query's, column by column.

    The family model pools and shrinks; this is the raw pairwise comparison the pooling
    is meant to improve on, and it is the one to quote when the question is about two
    proteins rather than about a family.
    """
    mem = long[long.protein == against]
    if not len(mem):
        return {"status": "skipped", "reason": f"{against} is not one of the pooled members"}
    rows, panels = [], []
    for q, site in scores.items():
        m = maps.get(q, {})
        if not m:
            continue
        s = site.assign(col=site.pos.map(m)).dropna(subset=["col"])
        s["col"] = s.col.astype(int)
        j = s.merge(mem[["col", "y", "s"]], on="col", how="inner")
        for c in score_cols:
            if c not in j:
                continue
            r = boot_spearman(j.y, j[c], seed=abs(hash((q, c, against))) % 2**31)
            rows.append({"query": q, "against": against, "score": c, **r})
            panels.append((q, c, j))
    if not panels:
        return {"status": "skipped", "reason": "no shared column carried both scores"}
    names = [f"{q} · {c}" for q, c, _ in panels]
    fig, pairs, _ = grid(names, w=3.0, h=2.8, extra=0.3)
    for i, ((q, c, j), (_, ax)) in enumerate(zip(panels, pairs)):
        r = [x for x in rows if x["query"] == q and x["score"] == c][0]
        ax.scatter(j.y, j[c], s=11, color="#B2182B", lw=0.2, edgecolor=P.INK, alpha=0.7)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6)
        extra = (f"ρ {r['rho']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]  n {r['n']}"
                 if np.isfinite(r.get("rho", np.nan)) else
                 f"only {r['n']} aligned positions with a score")
        panel(ax, i, f"{q} · {c}", extra, size=7)
        if i % 2 == 0:
            ax.set_ylabel("the query's own site median", fontsize=7.5)
    fig.supxlabel(f"{against} mean effect at that alignment column", fontsize=8)
    fig.suptitle(f"{against} on its own, at the same columns: the pairwise comparison "
                 f"the family model is meant to improve on.", fontsize=7, x=0.01,
                 ha="left", wrap=True)
    P.save(fig, outdir, "x03_member_vs_query", None, "X03")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "x03_member_vs_query.csv", index=False)
    return {"status": "ok", "by_readout": t.to_dict("records"),
            "headline": "; ".join(f"{r.query}·{r.score} ρ {r.rho:+.2f} "
                                  f"[{r.lo:+.2f}, {r.hi:+.2f}] n {r.n}"
                                  for r in t.itertuples() if np.isfinite(r.rho))}


ANALYSES = {"x01": "alignment identity, family against queries",
            "x02": "family μ against the query's measurements",
            "x03": "one member against the query"}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "divide by zero|Degrees of freedom|"
                                              "All-NaN|tight_layout).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms transfer",
        description="Does a measured family's constraint transfer to a protein measured "
                    "separately? Pass a true ortholog as a second --query for a control.")
    ap.add_argument("--family", required=True, metavar="NAME=cfg1,cfg2,...")
    ap.add_argument("--query", action="append", default=[], metavar="GENE=ACCESSION",
                    help="a protein to map onto the family's columns, by UniProt "
                         "accession; repeatable. Pass a genuine ortholog too, as the "
                         "positive control the comparison needs")
    ap.add_argument("--query-fasta", action="append", default=[], metavar="GENE=FILE",
                    help="the same, from a local fasta instead of an accession")
    ap.add_argument("--scores", action="append", default=[], metavar="GENE=FILE",
                    help="that protein's own measurements: a csv with `pos` and one "
                         "column per readout, such as the variants.csv `mpdms dual` "
                         "writes")
    ap.add_argument("--score-cols", default="abundance,function",
                    help="comma-separated columns to correlate (default "
                         "abundance,function)")
    ap.add_argument("--against", default=None,
                    help="the family member for x03 (default: the first one)")
    ap.add_argument("--min-proteins", type=int, default=3)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)
    globals()["N_BOOT"] = int(a.boot)
    keys = [k.strip() for k in (a.only.split(",") if a.only else ANALYSES)]
    bad = [k for k in keys if k not in ANALYSES]
    if bad:
        raise SystemExit(f"unknown analysis {bad}; available: {', '.join(ANALYSES)}")

    name, _, mem = a.family.partition("=")
    members = [m.strip() for m in mem.split(",") if m.strip()]
    if not name or not members:
        raise SystemExit("--family wants NAME=cfg1.yaml,cfg2.yaml,...")
    missing = [m for m in members if not Path(m).is_file()]
    if missing:
        raise SystemExit(f"--family {name}: not a config file: {', '.join(missing)}")
    queries = {}
    for q in a.query:
        g, _, acc = q.partition("=")
        if not g or not acc:
            raise SystemExit(f"--query wants GENE=ACCESSION, got {q!r}")
        queries[g] = fetch_sequence(acc)
    for q in a.query_fasta:
        g, _, f = q.partition("=")
        if not g or not f:
            raise SystemExit(f"--query-fasta wants GENE=FILE, got {q!r}")
        if not Path(f).is_file():
            raise SystemExit(f"--query-fasta {g}: no such file: {f}")
        queries[g] = "".join(l.strip() for l in Path(f).read_text().splitlines()
                             if not l.startswith(">"))
    if not queries:
        raise SystemExit("--query is required: there is nothing to transfer to")
    score_cols = [c.strip() for c in a.score_cols.split(",") if c.strip()]
    scores = {}
    for s in a.scores:
        g, _, f = s.partition("=")
        if g not in queries:
            raise SystemExit(f"--scores {g}: no matching --query {g}=ACCESSION")
        if not Path(f).is_file():
            raise SystemExit(f"--scores {g}: no such file: {f}")
        scores[g] = read_scores(Path(f), score_cols)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"== {name}: {len(members)} measured, {len(queries)} query "
          f"({', '.join(queries)})", flush=True)
    cols, maps, aln, long = family_with_queries(members, queries, a.min_proteins)
    ids = sorted(set(long.protein)) if len(long) else []
    print(f"   alignment: {len(aln)} columns, {len(cols)} pooled "
          f"(>= {a.min_proteins} proteins), members {', '.join(ids)}", flush=True)
    for g, s in scores.items():
        mapped = int(s.pos.isin(maps.get(g, {})).sum())
        print(f"   {g}: {len(s)} scored positions, {mapped} land on an aligned column",
              flush=True)
    against = a.against or (ids[0] if ids else None)

    calls = {"x01": lambda: x01_alignment(aln, ids, list(queries), outdir),
             "x02": lambda: x02_family_vs_query(cols, maps, scores, score_cols, outdir),
             "x03": lambda: x03_member_vs_query(long, maps, scores, score_cols,
                                                against, outdir)}
    out = {}
    print()
    for k in keys:
        try:
            r = calls[k]()
        except Exception as e:
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        out[k] = r
        print(f"  {k}  {r.get('status'):<8} {r.get('headline') or r.get('reason', '')}",
              flush=True)
    if out.get("x01", {}).get("below_twilight"):
        print("\n  READ x01 FIRST: " + ", ".join(out["x01"]["below_twilight"])
              + " is no closer to this family than an unrelated protein would be. A flat"
                " x02 there cannot be told apart from a misaligned register, so it is not"
                " evidence that constraint does not transfer.", flush=True)
    (outdir / "transfer.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
