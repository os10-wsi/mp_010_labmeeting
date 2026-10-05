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
    "rsa": ["rsa", "rel_sasa", "relative_sasa", "sasa_rel", "rsasa"],
    "quality": ["quality_rank", "quality", "rank"],
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


def _sep(p: Path) -> str:
    """Sniff the delimiter from the header, since these tables ship as .txt or .tsv."""
    head = p.open(errors="replace").readline()
    return "\t" if head.count("\t") > head.count(",") else ","


def load_soluble(path: str | Path, min_variants: int = 50, max_quality_rank: int | None = None,
                 min_nonsense_depth: float = 0.2) -> pd.DataFrame:
    """Long table: dataset, pos, wt, mut, vclass, score_z (syn/WT = 0, nonsense = -1), rsa.

    Only SINGLE amino-acid variants are kept. Published tables of this kind also carry
    multi-mutant rows, which have no single (position, wt, mut) and would not be
    comparable with a single-mutant screen; they are counted and dropped.

    A domain reaches the same scale by either of two routes, because not every published
    table has synonymous variants:

      renormalised           it has >= 3 synonymous and >= 3 nonsense of its own, so the
                             score is rescaled on their medians exactly as the membrane
                             data is, putting the two on the same footing by construction.
      rescaled on nonsense   it has no synonymous variants but its zero is already the
                             wild type, as published tables on this convention are, so
                             only the lower anchor needs setting: divide by |median
                             nonsense|. Near-identity for a table already on this scale,
                             and it brings domains whose stops sit at -1.4 or -1.7 onto
                             the same scale instead of discarding them.

    A domain is dropped only when it has no usable lower anchor at all, i.e. its nonsense
    median is not meaningfully below zero, since then nothing distinguishes a dead variant
    from a neutral one.
    """
    p = Path(path)
    e = pd.read_csv(p, sep=_sep(p), low_memory=False)
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
    d["quality"] = pd.to_numeric(e[need["quality"]], errors="coerce") if need["quality"] else np.nan
    d["vclass"] = classify(d.mut, d.wt, e[need["vclass"]] if need["vclass"] else None)

    # single variants only: multi-mutant rows carry no single (position, wt, mut)
    n_all = len(d)
    d = d[d.pos.notna() & d.score_raw.notna() & d.wt.ne("") & d.wt.ne("nan") & d.mut.ne("")]
    n_multi = n_all - len(d)
    d = d.copy()
    d["pos"] = d.pos.astype(int)
    if max_quality_rank is not None and d.quality.notna().any():
        d = d[d.quality <= max_quality_rank]

    kept, dropped, modes = [], [], {}
    for name, g in d.groupby("dataset", observed=True):
        syn = g.loc[g.vclass == "synonymous", "score_raw"]
        stop = g.loc[g.vclass == "nonsense", "score_raw"]
        if len(g) < min_variants or len(stop) < 3:
            dropped.append({"dataset": name, "n": len(g), "n_syn": len(syn), "n_stop": len(stop),
                            "reason": "too few variants or nonsense controls"})
            continue
        g = g.copy()
        sm, pm = (float(syn.median()) if len(syn) >= 3 else np.nan), float(stop.median())
        if np.isfinite(sm) and abs(sm - pm) > 1e-9:
            g["score_z"] = (g.score_raw - sm) / (sm - pm)
            modes[name] = "renormalised"
        elif pm <= -min_nonsense_depth:
            g["score_z"] = g.score_raw / -pm    # zero is already WT; set the lower anchor
            modes[name] = "rescaled on nonsense"
        else:
            dropped.append({"dataset": name, "n": len(g), "n_syn": len(syn), "n_stop": len(stop),
                            "reason": f"no synonymous controls and nonsense median {pm:.2f} is not "
                                      f"below -{min_nonsense_depth}: no usable lower anchor"})
            continue
        kept.append(g)
    if not kept:
        raise ValueError(f"{p.name}: no domain could be put on the syn/WT = 0, nonsense = -1 scale "
                         f"(checked {d.dataset.nunique()}). First reasons: {dropped[:3]}")
    out = pd.concat(kept, ignore_index=True)
    out.attrs.update(source=p.name, n_datasets_kept=out.dataset.nunique(),
                     n_datasets_dropped=len(dropped), dropped=dropped[:20],
                     n_multi_dropped=int(n_multi), modes=modes,
                     mode_counts=pd.Series(modes).value_counts().to_dict() if modes else {})
    return out
