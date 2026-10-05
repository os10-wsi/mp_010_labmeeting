"""Reference soluble-protein DMS abundance data, for comparison against the membrane set.

The comparator this is written for is an aPCA abundance screen of small soluble domains
(the human domainome). That readout is the right control for this project: it measures
cellular abundance of a folded protein in yeast, as these screens do, so the difference
between the two is about the protein being in a bilayer rather than about the assay.

Normalisation is PER DOMAIN. Each domain has its own dynamic range, and pooling before
normalising would let a domain with a wide range dominate every comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "dataset": ["dataset", "domain", "domain_id", "protein", "gene", "pdb", "id"],
    "pos": ["pos", "position", "wt_pos", "aa_pos", "site"],
    "wt": ["wt", "wt_aa", "wild_type", "aa_wt", "wt_codon_aa"],
    "mut": ["mut", "mut_aa", "mutant", "aa_mut", "alt"],
    "score": ["score_z", "normalized_fitness", "normalised_fitness", "fitness", "score",
              "nscore", "gr_norm", "apca_fitness", "mean_fitness"],
    "vclass": ["vclass", "mut_type", "variant_class", "class", "type"],
    "rsa": ["rsa", "rel_sasa", "relative_sasa", "sasa_rel", "rsasa", "scHAmin_ligand"],
}
SYN_WORDS = {"synonymous", "syn", "silent", "wt", "wildtype"}
STOP_WORDS = {"nonsense", "stop", "stop_codon", "nonsense_mutation", "ter"}


def _pick(cols: list[str], names: list[str]) -> str | None:
    low = {c.strip().lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def classify(mut: pd.Series, wt: pd.Series, raw: pd.Series | None) -> pd.Series:
    """missense / synonymous / nonsense, from an explicit column when there is one."""
    if raw is not None:
        s = raw.astype(str).str.strip().str.lower()
        out = pd.Series("missense", index=s.index, dtype=object)
        out[s.isin(SYN_WORDS)] = "synonymous"
        out[s.isin(STOP_WORDS)] = "nonsense"
        return out
    m = mut.astype(str).str.strip()
    return pd.Series(np.where(m.isin(["*", "X", "Ter"]), "nonsense",
                              np.where(m == wt.astype(str).str.strip(), "synonymous", "missense")),
                     index=mut.index)


def load_soluble(path: str | Path, min_variants: int = 50) -> pd.DataFrame:
    """Long table: dataset, pos, wt, mut, vclass, score_z (syn = 0, nonsense = -1), rsa.

    A domain is dropped unless it carries both synonymous and nonsense controls, because
    without them its scores cannot be put on the same scale as the membrane data and a
    comparison of distributions would be meaningless.
    """
    p = Path(path)
    e = pd.read_csv(p, low_memory=False)
    cols = list(e.columns)
    need = {k: _pick(cols, v) for k, v in ALIASES.items()}
    missing = [k for k in ("pos", "mut", "score") if need[k] is None]
    if missing:
        raise ValueError(f"{p.name}: could not find column(s) for {missing}. "
                         f"Columns present: {cols[:25]}")
    d = pd.DataFrame({
        "dataset": (e[need["dataset"]].astype(str) if need["dataset"] else "soluble"),
        "pos": pd.to_numeric(e[need["pos"]], errors="coerce"),
        "wt": (e[need["wt"]].astype(str).str.strip() if need["wt"] else ""),
        "mut": e[need["mut"]].astype(str).str.strip(),
        "score_raw": pd.to_numeric(e[need["score"]], errors="coerce"),
    })
    d["rsa"] = pd.to_numeric(e[need["rsa"]], errors="coerce") if need["rsa"] else np.nan
    d["vclass"] = classify(d.mut, d.wt, e[need["vclass"]] if need["vclass"] else None)
    d = d.dropna(subset=["pos", "score_raw"])
    d["pos"] = d.pos.astype(int)

    kept, dropped = [], []
    for name, g in d.groupby("dataset", observed=True):
        syn = g.loc[g.vclass == "synonymous", "score_raw"]
        stop = g.loc[g.vclass == "nonsense", "score_raw"]
        if len(g) < min_variants or len(syn) < 3 or len(stop) < 3:
            dropped.append((name, len(g), len(syn), len(stop)))
            continue
        sm, pm = float(syn.median()), float(stop.median())
        if not np.isfinite(sm - pm) or abs(sm - pm) < 1e-9:
            dropped.append((name, len(g), len(syn), len(stop)))
            continue
        g = g.copy()
        g["score_z"] = (g.score_raw - sm) / (sm - pm)
        kept.append(g)
    if not kept:
        raise ValueError(f"{p.name}: no domain had enough variants plus synonymous and nonsense "
                         f"controls (checked {d.dataset.nunique()})")
    out = pd.concat(kept, ignore_index=True)
    out.attrs["n_datasets_kept"] = out.dataset.nunique()
    out.attrs["n_datasets_dropped"] = len(dropped)
    out.attrs["dropped"] = dropped[:20]
    out.attrs["source"] = p.name
    return out
