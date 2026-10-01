"""Per-variant feature table shared by the validation battery (A16) and the benchmark harness.

Only features that exist for *any* yeast membrane protein are included, so a model trained
on one protein can be applied to another: topology, position within a TM helix, burial and
depth from the AlphaFold model, and physicochemical deltas. Nothing protein-specific
(absolute position, identity of the protein) is used, which is what makes leave-one-
protein-out transfer a fair test of generalisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .annot import BIOLOGICAL, CHARGE, HELIX, KYTE_DOOLITTLE, VOLUME, add_deltas
from .config import protein_sequence

VARIANT_FEATURES = ["delta_biological", "delta_kd", "delta_volume", "delta_charge", "delta_helix",
                    "wt_biological", "wt_volume", "mut_biological", "mut_volume", "is_pro", "is_gly",
                    "wt_is_hydrophobic", "mut_is_charged"]
POSITION_FEATURES = ["is_tm", "is_loop", "rel_pos", "abs_rel", "dist_end", "frac_in_protein",
                     "rsa", "contacts", "plddt", "abs_z", "helix_rsa", "hyd_window9", "dist_to_tm"]
FEATURES = VARIANT_FEATURES + POSITION_FEATURES
HYDROPHOBIC = set("AVLIFMWC")
CHARGED = set("DEKR")


def hydrophobicity_window(seq: str, w: int = 9) -> dict[int, float]:
    h = np.array([KYTE_DOOLITTLE.get(a, 0.0) for a in seq])
    pad = w // 2
    sm = np.convolve(np.pad(h, pad, mode="edge"), np.ones(w) / w, mode="valid")
    return {i + 1: float(v) for i, v in enumerate(sm)}


def position_table(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """One row per position: topology, helix geometry, burial, depth."""
    lo, hi = int(df.pos.min()), int(df.pos.max())
    pos = np.arange(lo, hi + 1)
    t = pd.DataFrame({"pos": pos})
    segs = cfg.get_path("topology.segments", []) or []
    tms = [s for s in segs if s.get("type") == "TM"]
    seg_type, rel, dist_end, helix = {}, {}, {}, {}
    for s in segs:
        for p in range(s["start"], s["end"] + 1):
            seg_type[p] = s.get("type", "unannotated")
    for s in tms:
        L = s["end"] - s["start"] + 1
        in_out = (s.get("orientation") or "in_out") == "in_out"
        for p in range(s["start"], s["end"] + 1):
            f = (p - s["start"]) / max(L - 1, 1)
            rel[p] = (2 * f - 1) if in_out else (1 - 2 * f)
            dist_end[p] = min(p - s["start"], s["end"] - p)
            helix[p] = s["name"]
    t["seg_type"] = t.pos.map(seg_type).fillna("unannotated")
    t["helix"] = t.pos.map(helix)
    t["is_tm"] = (t.seg_type == "TM").astype(float)
    t["is_loop"] = t.seg_type.isin(["loop", "soluble"]).astype(float)
    t["rel_pos"] = t.pos.map(rel)
    t["abs_rel"] = t.rel_pos.abs()
    t["dist_end"] = t.pos.map(dist_end)
    t["frac_in_protein"] = (t.pos - lo) / max(hi - lo, 1)
    # distance (residues) to the nearest TM segment; 0 inside one
    if tms:
        starts = np.array([s["start"] for s in tms]); ends = np.array([s["end"] for s in tms])
        d = np.min(np.maximum.reduce([starts[None, :] - pos[:, None], pos[:, None] - ends[None, :],
                                      np.zeros((len(pos), len(tms)))]), axis=1)
        t["dist_to_tm"] = d
    else:
        t["dist_to_tm"] = np.nan

    seq = protein_sequence(cfg)
    if seq:
        hw = hydrophobicity_window(seq)
        t["hyd_window9"] = t.pos.map(hw)
    else:
        t["hyd_window9"] = np.nan

    sp = cfg.resolve(cfg.get_path("structure.path"))
    if sp and sp.exists():
        from .structure import membrane_frame, residue_table
        rt = residue_table(sp, cfg.get_path("structure.chain", "A"),
                           int(cfg.get_path("structure.numbering_offset", 0) or 0))
        rt, _ = membrane_frame(rt, cfg)
        rt["abs_z"] = rt.zdepth.abs()
        t = t.merge(rt[["pos", "rsa", "contacts", "plddt", "abs_z"]], on="pos", how="left")
        hx = t[t.plddt >= 70].groupby("helix").rsa.median().rename("helix_rsa")
        t = t.join(hx, on="helix")
    else:
        for c in ("rsa", "contacts", "plddt", "abs_z", "helix_rsa"):
            t[c] = np.nan
    return t


def variant_features(df: pd.DataFrame, cfg, include_esm: bool = True) -> pd.DataFrame:
    """Missense variants that pass filters, with features and the target `score_z`."""
    d = df[df.pass_filter & (df.vclass == "missense")].copy()
    d = add_deltas(d)
    d["wt_biological"] = d.wt.map(BIOLOGICAL)
    d["mut_biological"] = d.mut.map(BIOLOGICAL)
    d["wt_volume"] = d.wt.map(VOLUME)
    d["mut_volume"] = d.mut.map(VOLUME)
    d["wt_helix"] = d.wt.map(HELIX)
    d["wt_charge"] = d.wt.map(CHARGE)
    d["is_pro"] = (d.mut == "P").astype(float)
    d["is_gly"] = (d.mut == "G").astype(float)
    d["wt_is_hydrophobic"] = d.wt.isin(list(HYDROPHOBIC)).astype(float)
    d["mut_is_charged"] = d.mut.isin(list(CHARGED)).astype(float)
    pt = position_table(df, cfg)
    keep = ["pos", "seg_type", "helix"] + [c for c in POSITION_FEATURES if c in pt.columns]
    out = d.merge(pt[keep], on="pos", how="left", suffixes=("", "_pos"))
    if "seg_type_pos" in out:
        out["seg_type"] = out.seg_type_pos.fillna(out.seg_type)
        out = out.drop(columns="seg_type_pos")
    if include_esm:
        p = cfg.resolve(cfg.get_path("evolution.esm_scores"))
        if p and p.exists():
            from .analyses.a15_esm_functional import read_esm
            e = read_esm(p)
            if e is not None:
                out = out.merge(e[["pos", "mut", "esm1v"]], on=["pos", "mut"], how="left")
    if "esm1v" not in out:
        out["esm1v"] = np.nan
    out["dataset_id"] = cfg.id
    out["gene"] = cfg.get_path("protein.gene") or cfg.id
    out["uniprot"] = cfg.get_path("protein.uniprot")
    cols = ["dataset_id", "gene", "uniprot", "pos", "wt", "mut", "score_z", "se", "n_reads",
            "seg_type", "helix", "esm1v"] + FEATURES
    return out[[c for c in cols if c in out.columns]]


def build_external(csv: str, cfg, include_esm: bool = True) -> pd.DataFrame:
    """Feature table for a DMS the pipeline did not produce (e.g. a published human membrane
    protein). `csv` needs pos, wt, mut and a score column; `cfg` supplies the sequence,
    topology and structure, so the same features can be computed. Scores are rescaled to the
    syn=0 / stop=-1 convention when a `vclass` column marks the controls, otherwise they are
    used as given and a warning is printed."""
    e = pd.read_csv(csv)
    e.columns = [c.strip().lower() for c in e.columns]
    sc = next((c for c in ("score_z", "score", "dms_score", "fitness") if c in e.columns), None)
    if sc is None or not {"pos", "mut"} <= set(e.columns):
        raise ValueError(f"{csv}: need pos, mut and one of score_z/score/dms_score/fitness")
    d = pd.DataFrame({"pos": e.pos.astype(int), "mut": e.mut.astype(str).str.strip(),
                      "score_raw": pd.to_numeric(e[sc], errors="coerce")})
    d["wt"] = e.wt.astype(str).str.strip() if "wt" in e.columns else None
    d["vclass"] = e.vclass if "vclass" in e.columns else np.where(d.mut == "*", "nonsense", "missense")
    d["pass_filter"] = True
    d["se"] = np.nan
    d["n_reads"] = np.nan
    syn = d.loc[d.vclass == "synonymous", "score_raw"]
    stop = d.loc[d.vclass == "nonsense", "score_raw"]
    if len(syn) >= 5 and len(stop) >= 5:
        from .io import normalise
        d["score_z"] = normalise(d.score_raw, float(syn.median()), float(stop.median()))
    else:
        d["score_z"] = d.score_raw
        print(f"  note: {csv} has no synonymous/stop controls - scores used unscaled; "
              "rank metrics (Spearman, AUROC) are unaffected, RMSE is not comparable")
    if d.wt.isna().all():
        seq = protein_sequence(cfg)
        if seq:
            d["wt"] = [seq[p - 1] if 0 < p <= len(seq) else None for p in d.pos]
    return variant_features(d, cfg, include_esm)


def build(configs, include_esm: bool = True) -> pd.DataFrame:
    """Feature table across several datasets. `configs` is an iterable of (df, cfg)."""
    return pd.concat([variant_features(df, cfg, include_esm) for df, cfg in configs], ignore_index=True)
