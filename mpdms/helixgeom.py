"""Per-residue table for TM helices: position along the helix, burial (RSA) on the
AlphaFold model, helix-level exposure class, and per-position fitness split into
Pro / Gly / rest. Shared by a13_helix_burial and a14_hydrophobic_facing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .structure import residue_table

SUBS = {"rest": "Missense (excl. Pro, Gly)", "P": "→ Pro", "G": "→ Gly"}
SUB_COLORS = {"rest": "#2F6DB5", "P": "#D62728", "G": "#E69F00"}
CLASS_COLORS = {"surface": "#4FA3C7", "core": "#7A3E9D", "mid": "#BBBBBB"}
HYDROPHOBIC = set("AVLIFMWC")
POLAR = set("DEKRNQHST")
PLDDT_MIN = 70.0
RSA_LIPID, RSA_BURIED = 0.25, 0.10


def _helix_segments(cfg, rt: pd.DataFrame, boundaries: str) -> list[dict]:
    """TM helices from UniProt (config) or re-drawn from the AlphaFold model's DSSP helix
    that overlaps each TM segment most."""
    tms = [s for s in cfg.get_path("topology.segments", []) or [] if s.get("type") == "TM"]
    if boundaries == "uniprot":
        return tms
    ss = dict(zip(rt.pos, rt.ss))
    out = []
    for s in tms:
        best = None
        p = s["start"] - 8
        while p <= s["end"] + 8:
            if ss.get(p) == "H":
                q = p
                while ss.get(q + 1) == "H":
                    q += 1
                ov = max(0, min(q, s["end"]) - max(p, s["start"]) + 1)
                if ov > 0 and (best is None or ov > best[2]):
                    best = (p, q, ov)
                p = q + 1
            else:
                p += 1
        if best and best[1] - best[0] + 1 >= 10:
            out.append(dict(s, start=int(best[0]), end=int(best[1])))
        else:
            out.append(s)
    return out


def _partner_helices(rt: pd.DataFrame, helices: list[dict], cutoff: float = 8.0, min_pairs: int = 3) -> dict:
    """Number of other TM helices each helix packs against (>= min_pairs CA-CA pairs < cutoff)."""
    xyz = rt.set_index("pos")[["x", "y", "z"]]
    coords = {h["name"]: xyz.reindex(range(h["start"], h["end"] + 1)).dropna().to_numpy() for h in helices}
    out = {}
    for a in helices:
        n = 0
        for b in helices:
            if a["name"] == b["name"] or not len(coords[a["name"]]) or not len(coords[b["name"]]):
                continue
            d = np.linalg.norm(coords[a["name"]][:, None, :] - coords[b["name"]][None, :, :], axis=2)
            n += int((d < cutoff).sum() >= min_pairs)
        out[a["name"]] = n
    return out


def tm_residue_table(df: pd.DataFrame, cfg, boundaries: str = "uniprot",
                     rsa_lipid: float = RSA_LIPID, rsa_buried: float = RSA_BURIED) -> pd.DataFrame:
    """One row per TM-helix residue with geometry, burial classes and fitness by substitution type."""
    p = cfg.resolve(cfg.get_path("structure.path"))
    if p is None or not p.exists():
        raise FileNotFoundError("structure.path missing - needed for RSA")
    rt = residue_table(p, cfg.get_path("structure.chain", "A"), int(cfg.get_path("structure.numbering_offset", 0) or 0))
    helices = _helix_segments(cfg, rt, boundaries)
    partners = _partner_helices(rt, helices)
    rti = rt.set_index("pos")

    d = df[df.pass_filter & (df.vclass == "missense")]
    fit = pd.DataFrame({
        "fit_rest": d[~d.mut.isin(["P", "G"])].groupby("pos").score_z.mean(),
        "n_rest": d[~d.mut.isin(["P", "G"])].groupby("pos").score_z.size(),
        "fit_P": d[d.mut == "P"].groupby("pos").score_z.mean(),
        "fit_G": d[d.mut == "G"].groupby("pos").score_z.mean(),
    })
    rows = []
    for h in helices:
        L = h["end"] - h["start"] + 1
        in_out = (h.get("orientation") or "in_out") == "in_out"
        for pos in range(h["start"], h["end"] + 1):
            f = (pos - h["start"]) / max(L - 1, 1)          # 0 at N-term end, 1 at C-term end
            rel = (2 * f - 1) if in_out else (1 - 2 * f)    # -1 cytosolic end ... +1 lumenal end
            r = rti.loc[pos] if pos in rti.index else None
            rows.append({
                "dataset_id": cfg.id, "helix": h["name"], "helix_start": h["start"], "helix_end": h["end"],
                "helix_len": L, "pos": pos, "rel_pos": rel, "abs_rel": abs(rel),
                "dist_end": min(pos - h["start"], h["end"] - pos),
                "wt": r["aa"] if r is not None else None,
                "rsa": float(r["rsa"]) if r is not None else np.nan,
                "plddt": float(r["plddt"]) if r is not None else np.nan,
                "contacts": float(r["contacts"]) if r is not None else np.nan,
                "partner_helices": partners.get(h["name"], np.nan),
            })
    t = pd.DataFrame(rows)
    t = t.join(fit, on="pos")
    t["confident"] = t.plddt >= PLDDT_MIN
    # helix-level exposure from confident residues
    hx = t[t.confident].groupby("helix").rsa.median().rename("helix_rsa")
    t = t.join(hx, on="helix")
    med = hx.median()
    lo, hi = hx.quantile(1 / 3), hx.quantile(2 / 3)
    t["class_median"] = np.where(t.helix_rsa > med, "surface", "core")
    t["class_tertile"] = np.select([t.helix_rsa >= hi, t.helix_rsa <= lo], ["surface", "core"], "mid")
    t["facing"] = np.select([t.rsa > rsa_lipid, t.rsa < rsa_buried], ["lipid", "buried"], "intermediate")
    t.attrs["boundaries"] = boundaries
    return t


def variant_table(df: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """Variant-level rows for TM positions with residue annotations and substitution type."""
    d = df[df.pass_filter & (df.vclass == "missense")][["pos", "wt", "mut", "score_z"]]
    keep = ["pos", "helix", "rel_pos", "abs_rel", "rsa", "confident", "facing", "class_median",
            "class_tertile", "helix_rsa", "dataset_id"]
    v = d.merge(tm[keep], on="pos", how="inner")
    v["subtype"] = np.select(
        [v.mut == "P", v.mut == "G", v.mut.isin(list(POLAR)), v.mut.isin(list(HYDROPHOBIC))],
        ["P", "G", "to_polar", "to_hydrophobic"], "other")
    return v
