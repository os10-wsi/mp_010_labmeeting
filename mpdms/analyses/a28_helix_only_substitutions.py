"""A28 - the substitution comparison restricted to predicted alpha helices.

A05b contrasts transmembrane positions with everything else, which confounds two things
at once: TM positions sit in a bilayer AND are almost all helical, while the rest is
largely loop and coil. Some of that difference is secondary structure, not environment.

Here both subsets are helical, so secondary structure is held constant and only the
environment varies:

    TM helix        helical (DSSP H) and inside a transmembrane segment
    non-TM helix    helical and outside every transmembrane segment

Helix assignment comes from the structure, so unlike A05b this needs a model, not just a
gff3. The non-TM helical set is usually small - an amphipathic helix in a loop, or a
terminal one - so its size is reported and the analysis refuses rather than draws a
matrix from a handful of positions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..annot import HYDROPHOBICITY_ORDER, assign_topology
from ..structure import residue_table
from .a05_substitution_physchem import substitution_panels
from .base import dirs, missense, result, skipped

MIN_POSITIONS = 10      # per subset, before a 20x20 matrix is worth drawing
HELIX_SS = "H"


def helix_subsets(df: pd.DataFrame, cfg) -> tuple[dict, pd.DataFrame, dict]:
    p = cfg.resolve(cfg.get_path("structure.path"))
    if p is None or not Path(p).exists():
        raise FileNotFoundError("structure.path is null or missing - helix assignment needs a model")
    rt = residue_table(p, cfg.get_path("structure.chain", "A"),
                       int(cfg.get_path("structure.numbering_offset", 0) or 0))
    d = missense(assign_topology(df, cfg)).merge(rt[["pos", "ss"]], on="pos", how="inner")
    d["is_tm"] = d.seg_type == "TM"
    d["helix"] = d.ss == HELIX_SS
    h = d[d.helix]
    subs = {"TM helix": h[h.is_tm], "non-TM helix": h[~h.is_tm]}
    counts = {k: int(v.pos.nunique()) for k, v in subs.items()}
    counts["helical positions"] = int(h.pos.nunique())
    counts["all positions on the model"] = int(d.pos.nunique())
    counts["non-helical TM"] = int(d[(~d.helix) & d.is_tm].pos.nunique())
    return subs, d, counts


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    if not [s for s in (cfg.get_path("topology.segments") or []) if s.get("type") == "TM"]:
        return skipped("a28", cfg, "no TM segments - run `mpdms tmhmm` first")
    try:
        subs, d, counts = helix_subsets(df, cfg)
    except FileNotFoundError as e:
        return skipped("a28", cfg, str(e))
    small = {k: counts[k] for k in ("TM helix", "non-TM helix") if counts[k] < MIN_POSITIONS}
    if small:
        return skipped("a28", cfg, f"too few helical positions to compare: {small} "
                                   f"(need {MIN_POSITIONS}); helical overall {counts['helical positions']}")

    figs, tabs, mstats = substitution_panels(
        cfg, d, list(HYDROPHOBICITY_ORDER), figdir, tabdir, subs=subs,
        name="a28_helix_substitution_matrices", analysis="A28",
        title="A28 substitution matrices, helices in the membrane vs helices outside it")

    tp = tabdir / "a28_helix_positions.csv"
    (d.drop_duplicates("pos")[["pos", "wt", "segment", "seg_type", "ss", "is_tm", "helix"]]
     .sort_values("pos").to_csv(tp, index=False))

    frac_tm_helical = (counts["TM helix"] /
                       max(counts["TM helix"] + counts["non-helical TM"], 1))
    caveats = ["both subsets are helical, so this comparison holds secondary structure constant "
               "and varies only the environment; A05b's TM-vs-rest contrast confounds the two",
               "helix assignment is DSSP on a predicted model, so a low-confidence region may be "
               "called helical on geometry alone - check pLDDT for any position that matters",
               f"{frac_tm_helical:.0%} of transmembrane positions are assigned helical; a much "
               "lower number would mean the model or the topology disagree",
               "non-TM helices are usually few and are often amphipathic interfacial or terminal "
               "helices, which are not the same thing as a soluble protein's helices"]
    head = (f"{counts['TM helix']} TM-helical vs {counts['non-TM helix']} non-TM-helical positions; "
            f"{mstats['n_cells_compared']} shared (wt, mut) cells, "
            f"{mstats['n_shared_wt_residues']} shared wild-type residues"
            + (f"; ρ between matrices {mstats['spearman_between_matrices']:.2f}"
               if np.isfinite(mstats["spearman_between_matrices"]) else ""))
    return result("a28", cfg, head,
                  {"counts": counts, "fraction_of_TM_helical": frac_tm_helical,
                   "substitution_matrices": mstats},
                  caveats, figs, list(tabs) + [tp])
