"""A19 - published functional residues vs this abundance screen.

Asks the one question a DMS abundance screen can answer about a residue someone
already mutated by hand: does the screen see it?

Three outcomes per residue, and all three are informative:
  abundance-tolerant + ESM-constrained  the residue matters for transport, not
                                        for biogenesis. The screen is RIGHT to
                                        leave it alone, and A15 should flag it.
  abundance-dead                        the published loss of function may be a
                                        folding/trafficking defect rather than a
                                        mechanistic one - reinterpret the paper.
  neither                               the screen has nothing to say here; with
                                        a handful of residues this is weak
                                        evidence against anything.

The curated input is configs/_literature.yaml. Entries whose stated wild-type
residue does not match the protein sequence are REFUSED, not silently used:
numbering drift between a paper, UniProt and a DMS construct is the single most
likely way to get this wrong.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats as ss

from .. import plotting as P
from ..config import REPO_ROOT, protein_sequence
from .a15_esm_functional import Q_SITE, TOLERANT, Z_SITE
from .base import dirs, fmt_p, result, skipped

LIT_PATH = REPO_ROOT / "configs" / "_literature.yaml"
LIT_COLOR = "#B8860B"
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def load_literature(gene: str, path: Path = LIT_PATH) -> dict | None:
    """Curated entries for a gene, matched case-insensitively on the symbol."""
    if not Path(path).exists():
        return None
    book = yaml.safe_load(Path(path).read_text()) or {}
    for key, val in book.items():
        if str(key).strip().upper() == str(gene).strip().upper():
            return val or {}
    return None


def check_numbering(entries: list[dict], seq: str | None) -> tuple[list[dict], list[str]]:
    """Keep entries whose stated wt matches the sequence; report every rejection."""
    ok, problems = [], []
    for e in entries:
        pos, wt = int(e["pos"]), str(e.get("wt", "")).strip().upper()
        wt = AA3.get(wt, wt)
        if seq is None:
            e = {**e, "wt_checked": False}
            ok.append(e)
            continue
        if pos < 1 or pos > len(seq):
            problems.append(f"{wt}{pos}: outside the {len(seq)}-residue sequence")
            continue
        actual = seq[pos - 1]
        if wt and actual != wt:
            problems.append(f"{wt}{pos}: sequence has {actual} at {pos} - numbering mismatch, entry dropped")
            continue
        ok.append({**e, "wt_checked": True})
    return ok, problems


def residue_table(entries: list[dict], sites: pd.DataFrame) -> pd.DataFrame:
    """One row per published residue, joined to what the screen measured there."""
    s = sites.set_index("pos") if len(sites) else pd.DataFrame()
    rows = []
    for e in entries:
        pos = int(e["pos"])
        r = {"pos": pos, "wt": e.get("wt"), "substitution": e.get("substitution"),
             "lit_function": e.get("function", "unknown"), "lit_abundance": e.get("abundance", "unknown"),
             "confidence": e.get("confidence", "unverified"), "doi": e.get("doi"),
             "wt_checked": e.get("wt_checked", False)}
        if pos in s.index:
            g = s.loc[pos]
            r.update(tested=True, segment=g.get("segment"), n_variants=int(g.n_variants),
                     median_abundance=float(g.median_abundance), median_z=float(g.median_z),
                     q=float(g.q), functional=bool(g.functional),
                     abundance_tolerant=bool(g.abundance_tolerant))
            # the prediction this project cares about
            r["verdict"] = ("functional, abundance-tolerant" if g.functional and g.abundance_tolerant
                            else "functional, abundance-dead" if g.functional
                            else "abundance-dead only" if g.median_abundance < TOLERANT
                            else "not detected")
        else:
            r.update(tested=False, verdict="not covered by the screen")
        rows.append(r)
    return pd.DataFrame(rows)


def enrichment(t: pd.DataFrame, sites: pd.DataFrame) -> dict:
    """Fisher exact: are published residues over-represented among functional calls?"""
    cov = t[t.tested]
    if not len(cov) or not len(sites):
        return {"testable": False, "note": "no published residue is covered by the screen"}
    a = int(cov.functional.sum())                              # published & functional
    b = len(cov) - a                                           # published & not
    c = int(sites.functional.sum()) - a                        # other & functional
    d = len(sites) - len(cov) - c                              # other & not
    if min(a + b, c + d) == 0:
        return {"testable": False, "note": "degenerate contingency table"}
    odds, p = ss.fisher_exact([[a, b], [c, d]], alternative="greater")
    # smallest p this table could ever produce, i.e. if every published residue hit
    best = ss.fisher_exact([[a + b, 0], [c + a - (a + b), d + b]], alternative="greater")[1] \
        if (c + a - (a + b)) >= 0 else np.nan
    return {"testable": True, "n_published_covered": len(cov), "n_hit": a,
            "base_rate": float(sites.functional.mean()), "odds_ratio": float(odds), "p": float(p),
            "min_attainable_p": float(best) if np.isfinite(best) else None,
            "underpowered": bool(len(cov) < 8)}


def figure(cfg, t: pd.DataFrame, sites: pd.DataFrame, figdir):
    """Published residues placed on the abundance x ESM-constraint plane."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6), layout="constrained")
    ax.scatter(sites.median_abundance, sites.median_z, s=9, c="#C9D3DD", lw=0, label="all tested sites", zorder=1)
    f = sites[sites.functional]
    ax.scatter(f.median_abundance, f.median_z, s=11, c="#D62728", lw=0, label="A15 functional", zorder=2)
    cov = t[t.tested]
    ax.scatter(cov.median_abundance, cov.median_z, s=70, facecolor="none", edgecolor=LIT_COLOR,
               lw=1.6, label="published residue", zorder=3)
    for r in cov.itertuples():
        ax.annotate(f"{r.wt}{r.pos}", (r.median_abundance, r.median_z), textcoords="offset points",
                    xytext=(7, 4), fontsize=8, color=LIT_COLOR, zorder=4)
    ax.axhline(Z_SITE, color="#D62728", lw=0.6, ls=(0, (2, 2)))
    ax.axvline(TOLERANT, color="#555555", lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("median abundance (normalised fitness)")
    ax.set_ylabel("median ESM-1v residual z")
    ax.set_title("Published functional residues against the screen\n"
                 f"bottom-right quadrant = constrained in evolution (z ≤ {Z_SITE}) but "
                 f"abundance-tolerant (≥ {TOLERANT})", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    P.title(fig, cfg, "A19 published residues vs this screen")
    return P.save(fig, figdir, "a19_literature", cfg, "A19")


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    gene = cfg.get_path("protein.gene", cfg.id)
    lit = load_literature(gene)
    if lit is None:
        return skipped("a19", cfg, f"no entry for {gene} in configs/_literature.yaml")
    entries = list(lit.get("measured") or [])
    if not entries:
        return skipped("a19", cfg, f"{gene} has no published residue-level mutagenesis "
                                   f"({str(lit.get('note', '')).strip().splitlines()[0][:80]})")

    sp = Path(outdir) / "tables" / "a15_sites.csv"
    if not sp.exists():
        return skipped("a19", cfg, "a15_sites.csv not found - run a15 first (needs ESM-1v scores)")
    sites = pd.read_csv(sp)

    entries, problems = check_numbering(entries, protein_sequence(cfg))
    if not entries:
        return skipped("a19", cfg, "every published residue failed the wild-type check: " + "; ".join(problems))

    t = residue_table(entries, sites)
    enr = enrichment(t, sites)
    tp = tabdir / "a19_literature_residues.csv"
    t.to_csv(tp, index=False)
    figs = figure(cfg, t, sites, figdir)

    cov = t[t.tested]
    hits = cov[cov.functional]
    gold = cov[cov.verdict == "functional, abundance-tolerant"]
    caveats = [f"wild-type check: {p}" for p in problems]
    unver = t[t.confidence != "verified"]
    if len(unver):
        caveats.append(f"{len(unver)}/{len(t)} entries are marked unverified in _literature.yaml - "
                       "the primary text has not been read; confirm residue numbers before presenting")
    if enr.get("underpowered"):
        caveats.append(f"only {enr.get('n_published_covered', 0)} published residues are covered: the "
                       "enrichment test is a descriptive statistic, not evidence. Read the per-residue table")
    caveats.append("a published residue that this screen leaves alone is the expected result for a "
                   "mechanistic site - absence of an abundance effect is not a failure of the assay")

    head = (f"{len(cov)}/{len(t)} published residues covered; {len(hits)} called functional by A15, "
            f"{len(gold)} of those abundance-tolerant"
            + (f" ({', '.join(f'{r.wt}{r.pos}' for r in gold.itertuples())})" if len(gold) else "")
            + (f"; Fisher {fmt_p(enr['p'])}, OR = {enr['odds_ratio']:.1f}" if enr.get("testable") else ""))
    return result("a19", cfg, head,
                  {"n_published": len(t), "n_covered": len(cov), "n_functional": len(hits),
                   "n_functional_tolerant": len(gold), "enrichment": enr,
                   "residues": [{"res": f"{r.wt}{r.pos}", "verdict": r.verdict} for r in t.itertuples()],
                   "wt_check_failures": problems},
                  caveats, figs, [tp])
