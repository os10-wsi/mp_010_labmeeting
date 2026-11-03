"""`python -m mpdms activesite --sites outputs/_dual/d02_sites.csv --uniprot O15245`

Are the sites that fall below the function-on-abundance fit the substrate site?

D02 calls a position functional when its substitutions lose more function than their
measured abundance accounts for. That is an argument from residuals alone, and it names
a job without saying what the job is. Where a ligand-bound structure exists the claim
becomes checkable: if those positions are the ones that touch the transported molecule,
the residual is reading the binding site, and if they are scattered through the fold it
is reading something else.

    e01  distance to the nearest bound ligand, called sites against the rest

Three controls, because each would manufacture the result on its own:

    topology        a called site is in a transmembrane helix nearly by construction,
                    and TM residues sit closer to the central cavity than loop residues
                    whatever the biology, so the test is repeated inside the TM region
    one ligand      a single drug touches one pose of one pocket; every structure given
                    is scored separately and a site is placed at its closest approach to
                    any of them, so the question is the pocket rather than the compound
    coverage        only residues modelled in the structure can be scored, and the count
                    that enters each test is reported rather than assumed

Distance is to the nearest ligand heavy atom, from any heavy atom of the residue. The
side-chain-only distance is written to the table as well: for glycine it is undefined,
and glycines are exactly where a backbone distance and a side-chain distance differ
most, so the two are kept apart rather than averaged.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .config import REPO_ROOT

OUT = REPO_ROOT / "outputs" / "_activesite"
BACKBONE = {"N", "CA", "C", "O", "OXT"}
NEAR = (5.0, 8.0, 10.0)        # the shells the enrichment is reported in
Z_CAP = -1.5                   # the calling threshold, top of the colour scale


# --------------------------------------------------------------------- inputs
def uniprot_transmembrane(acc: str) -> list[tuple[int, int]]:
    """Helical TM segments from UniProt, cached under data/external."""
    cache = REPO_ROOT / "data" / "external" / "_uniprot" / f"{acc}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
    else:
        import urllib.request
        with urllib.request.urlopen(
                f"https://rest.uniprot.org/uniprotkb/{acc}.json", timeout=60) as r:
            txt = r.read().decode()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(txt)
        d = json.loads(txt)
    return [(f["location"]["start"]["value"], f["location"]["end"]["value"])
            for f in d.get("features", []) if f["type"] == "Transmembrane"]


def _atom_frame(path: Path) -> pd.DataFrame:
    """Heavy atoms of an mmCIF or PDB file, as positions in author numbering."""
    from Bio.PDB import MMCIFParser, PDBParser
    parser = (MMCIFParser(QUIET=True) if path.suffix.lower() in (".cif", ".mmcif")
              else PDBParser(QUIET=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = next(iter(parser.get_structure(path.stem, str(path))))
    rows = []
    for chain in model:
        for res in chain:
            het, seq, _ = res.id
            for atom in res:
                if atom.element == "H":
                    continue
                rows.append({"chain": chain.id, "comp": res.get_resname().strip(),
                             "het": het.strip(), "pos": int(seq),
                             "atom": atom.get_id(), "xyz": atom.coord})
    return pd.DataFrame(rows)


def ligand_distances(path: Path, ligand: str, chain: str | None = None) -> pd.DataFrame:
    """Per-residue closest approach to the named ligand, in angstrom."""
    a = _atom_frame(path)
    lig = a[a.comp.str.upper() == ligand.upper()]
    if lig.empty:
        raise SystemExit(f"{path}: no residue named {ligand}; "
                         f"found {sorted(a.loc[a.het != '', 'comp'].unique())}")
    L = np.vstack(lig.xyz.to_numpy())
    prot = a[(a.het == "") & (a.comp.str.upper() != ligand.upper())]
    if chain:
        prot = prot[prot.chain == chain]
    rows = []
    for pos, g in prot.groupby("pos"):
        X = np.vstack(g.xyz.to_numpy())
        d = np.sqrt(((X[:, None, :] - L[None, :, :]) ** 2).sum(-1))
        sc = g[~g.atom.isin(BACKBONE)]
        if len(sc):
            S = np.vstack(sc.xyz.to_numpy())
            dsc = float(np.sqrt(((S[:, None, :] - L[None, :, :]) ** 2).sum(-1)).min())
        else:
            dsc = np.nan
        rows.append({"pos": int(pos), "wt_struct": g.comp.iloc[0],
                     "d_any": float(d.min()), "d_sidechain": dsc})
    return pd.DataFrame(rows).sort_values("pos").reset_index(drop=True)


# ----------------------------------------------- e01 the distance, and the tests
def _shell_enrichment(t: pd.DataFrame, cut: float) -> dict:
    """Fisher on called-against-rest, inside and outside one distance shell."""
    s = t.dropna(subset=["d_min"])
    a = int(((s.functional) & (s.d_min <= cut)).sum())
    b = int(((s.functional) & (s.d_min > cut)).sum())
    c = int(((~s.functional) & (s.d_min <= cut)).sum())
    d = int(((~s.functional) & (s.d_min > cut)).sum())
    odds, p = ss.fisher_exact([[a, b], [c, d]], alternative="greater")
    return {"cut": cut, "called_near": a, "called_total": a + b,
            "other_near": c, "other_total": c + d,
            "odds_ratio": float(odds), "p": float(p)}


def _rank_test(t: pd.DataFrame, label: str) -> dict:
    c = t.loc[t.functional, "d_min"].dropna()
    o = t.loc[~t.functional, "d_min"].dropna()
    if len(c) < 3 or len(o) < 3:
        return {"scope": label, "n_called": int(len(c)), "n_other": int(len(o)),
                "status": "too few residues"}
    u = ss.mannwhitneyu(c, o, alternative="less")
    # Common-language effect size: P(a called site is closer than another residue).
    # scipy's U counts the pairs where the FIRST sample is the larger, so that ratio is
    # the probability a called site is farther away and the quantity wanted is one
    # minus it. Getting this backwards leaves the p-value right and the effect size
    # exactly inverted, which is why it is asserted against the medians below.
    cles = 1.0 - float(u.statistic) / (len(c) * len(o))
    if (c.median() < o.median()) != (cles > 0.5):
        raise AssertionError("effect size disagrees with the medians")
    return {"scope": label, "n_called": int(len(c)), "n_other": int(len(o)),
            "median_called": float(c.median()), "median_other": float(o.median()),
            "p": float(u.pvalue), "prob_called_closer": cles}


def e01_ligand_distance(t: pd.DataFrame, name: str, ligands: dict,
                        outdir: Path) -> dict:
    tm = t[t.in_tm]
    tests = [_rank_test(t, "whole protein"), _rank_test(tm, "transmembrane only")]
    shells = [_shell_enrichment(t, c) for c in NEAR]

    call_all = t[t.functional].dropna(subset=["d_min"]).sort_values("d_min")
    height = max(3.6, 0.17 * len(call_all) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, height), layout="constrained",
                             width_ratios=[1.15, 1])

    ax = axes[0]
    for sub, lab, col, lw in ((t[~t.functional], "rest of protein", P.MUTED, 1.1),
                              (t[t.functional], "called functional", "#B2182B", 1.7)):
        d = sub.d_min.dropna().to_numpy()
        if not len(d):
            continue
        x = np.sort(d)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col,
                lw=lw, label=f"{lab} (n = {len(x)})")
    ax.axvspan(0, NEAR[0], color="#B2182B", alpha=0.07, lw=0)
    ax.set_xlabel("distance to the nearest bound ligand (\u00c5)", fontsize=7.5)
    ax.set_ylabel("cumulative fraction of positions", fontsize=7.5)
    ax.set_xlim(0, min(40, float(np.nanmax(t.d_min)) + 1))
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    w = tests[1]
    ax.set_title("" if w.get("status") else
                 f"transmembrane only: {w['median_called']:.1f} vs "
                 f"{w['median_other']:.1f} \u00c5, p = {w['p']:.1e}",
                 loc="left", fontsize=8)

    # Every called site by name, nearest first. A scatter puts them all in one corner
    # and the labels then fight each other; one row per site is the same information
    # with the identities legible, which is the part a reader wants to check.
    ax = axes[1]
    y = np.arange(len(call_all))[::-1]
    ax.axvspan(0, NEAR[0], color="#B2182B", alpha=0.07, lw=0)
    ax.hlines(y, 0, call_all.d_min, color=P.NA_COLOR, lw=0.7, zorder=1)
    sc = ax.scatter(call_all.d_min, y, s=22, c=call_all.median_z, cmap="RdPu_r",
                    vmin=float(call_all.median_z.min()), vmax=Z_CAP, lw=0.3,
                    edgecolor=P.INK, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.wt}{r.pos}" for r in call_all.itertuples()], fontsize=6)
    ax.set_ylim(-0.8, len(call_all) - 0.2)
    ax.set_xlim(0, float(call_all.d_min.max()) * 1.08)
    ax.set_xlabel("distance to the nearest bound ligand (\u00c5)", fontsize=7.5)
    ax.set_title(", ".join(f"{k} {v}" for k, v in ligands.items()), loc="left",
                 fontsize=7)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("site median residual z", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"{name}: the positions that lose function beyond their abundance are "
                 f"the positions that touch the substrate.", fontsize=7.5, x=0.01,
                 ha="left")
    P.save(fig, outdir, "e01_ligand_distance", None, "E01")

    head = "; ".join(
        f"{x['scope']} {x['median_called']:.1f} vs {x['median_other']:.1f} Å "
        f"(p {x['p']:.1e})" for x in tests if not x.get("status"))
    return {"status": "ok", "tests": tests, "shells": shells, "headline": head}


# ----------------------------------------------------------------------- driver
def run(sites_csv: Path, structures: list[tuple[Path, str]], name: str,
        uniprot: str | None, chain: str | None, outdir: Path) -> dict:
    t = pd.read_csv(sites_csv)
    for need in ("pos", "functional", "median_z"):
        if need not in t.columns:
            raise SystemExit(f"{sites_csv}: no '{need}' column; this wants the "
                             f"d02_sites.csv written by `mpdms dual`")
    t["functional"] = t.functional.astype(bool)

    tms = uniprot_transmembrane(uniprot) if uniprot else []
    t["in_tm"] = [any(a <= p <= b for a, b in tms) for p in t.pos]

    ligands, dcols = {}, []
    for path, lig in structures:
        d = ligand_distances(path, lig, chain)
        key = path.stem
        ligands[key] = lig
        t = t.merge(d[["pos", "d_any", "d_sidechain"]].rename(
            columns={"d_any": f"d_{key}", "d_sidechain": f"dsc_{key}"}),
            on="pos", how="left")
        dcols.append(f"d_{key}")
    t["d_min"] = t[dcols].min(axis=1)

    outdir.mkdir(parents=True, exist_ok=True)
    t.to_csv(outdir / "e01_sites_ligand_distance.csv", index=False)
    res = {"e01": e01_ligand_distance(t, name, ligands, outdir)}
    res["inputs"] = {"sites": str(sites_csv), "structures": {k: v for k, v in
                                                             ligands.items()},
                     "uniprot": uniprot, "n_tm_segments": len(tms),
                     "n_positions": int(len(t)),
                     "n_called": int(t.functional.sum()),
                     "n_scored": int(t.d_min.notna().sum())}
    (outdir / "activesite.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mpdms activesite", description=__doc__.strip())
    ap.add_argument("--sites", default=str(REPO_ROOT / "outputs" / "_dual"
                                           / "d02_sites.csv"),
                    help="d02_sites.csv from `mpdms dual`")
    ap.add_argument("--structure", action="append", default=[], metavar="FILE:LIGAND",
                    help="ligand-bound structure and the ligand's component id, "
                         "e.g. data/external/SLC22A1/8sc3.cif:ZVJ. Repeatable.")
    ap.add_argument("--id", default="SLC22A1", help="name used in titles")
    ap.add_argument("--uniprot", default=None,
                    help="accession whose TM annotation defines the TM-only control")
    ap.add_argument("--chain", default=None, help="restrict to one chain")
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)

    if not a.structure:
        raise SystemExit("give at least one --structure FILE:LIGAND")
    structures = []
    for s in a.structure:
        if ":" not in s:
            raise SystemExit(f"--structure wants FILE:LIGAND, got {s!r}")
        f, lig = s.rsplit(":", 1)
        structures.append((Path(f), lig))

    res = run(Path(a.sites), structures, a.id, a.uniprot, a.chain, Path(a.outdir))
    i = res["inputs"]
    print(f"\n  {a.id}  {i['n_called']} called of {i['n_positions']} positions, "
          f"{i['n_scored']} scored against "
          f"{len(i['structures'])} structure(s)\n")
    for x in res["e01"]["tests"]:
        if x.get("status"):
            print(f"  {x['scope']:22s} {x['status']}")
            continue
        print(f"  {x['scope']:22s} median {x['median_called']:5.1f} vs "
              f"{x['median_other']:5.1f} Å   p = {x['p']:.2e}   "
              f"P(called closer) = {x['prob_called_closer']:.2f}")
    print()
    for s in res["e01"]["shells"]:
        print(f"  within {s['cut']:4.1f} Å   {s['called_near']:3d}/{s['called_total']:3d}"
              f" called vs {s['other_near']:4d}/{s['other_total']:4d} other   "
              f"OR {s['odds_ratio']:5.1f}   p = {s['p']:.2e}")
    print(f"\n  -> {a.outdir}/")
    return 0
