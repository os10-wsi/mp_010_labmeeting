"""Topology assignment, amino-acid property scales and sequence motifs."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Kyte & Doolittle 1982, J Mol Biol 157:105
KYTE_DOOLITTLE = dict(I=4.5, V=4.2, L=3.8, F=2.8, C=2.5, M=1.9, A=1.8, G=-0.4, T=-0.7, S=-0.8,
                      W=-0.9, Y=-1.3, P=-1.6, H=-3.2, E=-3.5, Q=-3.5, D=-3.5, N=-3.5, K=-3.9, R=-4.5)
# Hessa et al. 2005, Nature 433:377 - biological hydrophobicity scale, dGapp (kcal/mol) for
# insertion via Sec61 at the TM centre. Lower = more favourable; we use -dGapp so that, like
# KD, higher = more hydrophobic.
HESSA_DGAPP = dict(I=-0.60, L=-0.55, F=-0.32, V=-0.31, M=-0.10, W=0.30, A=0.11, Y=0.68, G=0.74,
                   C=-0.13, T=0.52, S=0.84, P=2.23, H=2.06, N=2.05, Q=2.36, E=2.68, D=3.49,
                   K=2.71, R=2.58)
BIOLOGICAL = {a: -v for a, v in HESSA_DGAPP.items()}
# Zamyatnin 1972 residue volumes (A^3)
VOLUME = dict(G=60.1, A=88.6, S=89.0, C=108.5, D=111.1, P=112.7, N=114.1, T=116.1, E=138.4,
              V=140.0, Q=143.8, H=153.2, M=162.9, I=166.7, L=166.7, K=168.6, R=173.4, F=189.9,
              Y=193.6, W=227.8)
# Pace & Scholtz 1998 helix propensity, ddG (kcal/mol) relative to Ala; we use -ddG
PACE_SCHOLTZ = dict(A=0.0, L=0.21, R=0.21, M=0.24, K=0.26, Q=0.39, E=0.40, I=0.41, W=0.49, S=0.50,
                    Y=0.53, F=0.54, H=0.61, V=0.61, N=0.65, T=0.66, C=0.68, D=0.69, G=1.00, P=3.16)
HELIX = {a: -v for a, v in PACE_SCHOLTZ.items()}
CHARGE = {a: 0.0 for a in AA20} | dict(D=-1.0, E=-1.0, K=1.0, R=1.0, H=0.1)
# Tien et al. 2013 theoretical max ASA (A^2), for RSA
MAX_ASA = dict(A=129, R=274, N=195, D=193, C=167, E=223, Q=225, G=104, H=224, I=197, L=201,
               K=236, M=224, F=240, P=159, S=155, T=172, W=285, Y=263, V=174)

SCALE_SOURCES = {
    "kd": "Kyte & Doolittle 1982 J Mol Biol 157:105",
    "biological": "Hessa et al. 2005 Nature 433:377 (-dGapp, kcal/mol)",
    "volume": "Zamyatnin 1972 Prog Biophys Mol Biol 24:107 (A^3)",
    "helix": "Pace & Scholtz 1998 Biophys J 75:422 (-ddG, kcal/mol)",
    "charge": "formal charge at pH 7 (His = +0.1)",
    "max_asa": "Tien et al. 2013 PLoS One 8:e80635 (theoretical)",
}
HYDROPHOBICITY_ORDER = "IVLFCMAGTSWYPHEQDNKR"  # KD order, used for matrices
HEATMAP_ORDER = "GAVLMIFYWKRHDESTCNQP"


def property_table() -> pd.DataFrame:
    return pd.DataFrame({
        "aa": list(AA20),
        "kd": [KYTE_DOOLITTLE[a] for a in AA20],
        "biological": [BIOLOGICAL[a] for a in AA20],
        "volume": [VOLUME[a] for a in AA20],
        "helix": [HELIX[a] for a in AA20],
        "charge": [CHARGE[a] for a in AA20],
        "max_asa": [MAX_ASA[a] for a in AA20],
    })


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta_* property columns for missense rows (mut - wt)."""
    out = df.copy()
    for name, scale in [("kd", KYTE_DOOLITTLE), ("biological", BIOLOGICAL), ("volume", VOLUME),
                        ("helix", HELIX), ("charge", CHARGE)]:
        out[f"delta_{name}"] = out["mut"].map(scale) - out["wt"].map(scale)
    return out


# ------------------------------------------------------------------ topology
def assign_topology(df: pd.DataFrame, cfg) -> pd.DataFrame:
    segs = cfg.get_path("topology.segments", []) or []
    out = df.copy()
    out["segment"] = "NA"
    out["seg_type"] = "unannotated"
    out["seg_side"] = "NA"
    for s in segs:
        m = out["pos"].between(s["start"], s["end"])
        out.loc[m, "segment"] = s["name"]
        out.loc[m, "seg_type"] = s.get("type", "unannotated")
        out.loc[m, "seg_side"] = s.get("side") or s.get("orientation") or "membrane"
    return out


def segments_df(cfg) -> pd.DataFrame:
    segs = cfg.get_path("topology.segments", []) or []
    cols = ["name", "start", "end", "type", "side", "orientation"]
    return pd.DataFrame([{c: s.get(c) for c in cols} for s in segs], columns=cols)


def tm_mask(positions, cfg) -> np.ndarray:
    positions = np.asarray(positions)
    m = np.zeros(len(positions), bool)
    for s in cfg.get_path("topology.segments", []) or []:
        if s.get("type") == "TM":
            m |= (positions >= s["start"]) & (positions <= s["end"])
    return m


def segments_from_features(features: list[dict], length: int, n_terminus_hint: str | None = None
                           ) -> tuple[list[dict], str]:
    """UniProt features (Transmembrane / Topological domain / Intramembrane) -> contiguous
    segment list covering 1..length. Returns (segments, n_terminus)."""
    tms, topo = [], []
    for f in features:
        loc = f.get("location", {})
        try:
            a, b = int(loc["start"]["value"]), int(loc["end"]["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if f.get("type") == "Transmembrane":
            tms.append((a, b, "TM"))
        elif f.get("type") == "Intramembrane":
            tms.append((a, b, "intramembrane"))
        elif f.get("type") == "Topological domain":
            desc = (f.get("description") or "").lower()
            side = "cytosolic" if "cytopl" in desc else ("lumenal" if desc else None)
            topo.append((a, b, side))
    return build_segments(sorted(tms), length, topo, n_terminus_hint)


def build_segments(tms: list[tuple[int, int, str]], length: int, topo=None,
                   n_terminus_hint: str | None = None) -> tuple[list[dict], str]:
    topo = topo or []

    def side_at(a, b):
        for ta, tb, s in topo:
            if s and ta <= (a + b) // 2 <= tb:
                return s
        return None

    n_term = side_at(1, max(1, (tms[0][0] - 1) if tms else length)) or n_terminus_hint or "cytosolic"
    segs, cur, side, nt, nl = [], 1, n_term, 0, 0
    for a, b, kind in tms:
        if a > cur:
            nl += 1
            s = side_at(cur, a - 1) or side
            segs.append(dict(name=f"{'N' if cur == 1 else 'L'}{'term' if cur == 1 else nl - 1}",
                             start=cur, end=a - 1,
                             type="soluble" if (a - cur) >= 60 else "loop", side=s))
            side = s
        nt += 1
        if kind == "TM":
            orient = "in_out" if side == "cytosolic" else "out_in"
            segs.append(dict(name=f"TM{nt}", start=a, end=b, type="TM", orientation=orient))
            side = "lumenal" if side == "cytosolic" else "cytosolic"
        else:
            segs.append(dict(name=f"IM{nt}", start=a, end=b, type="intramembrane", side=side))
        cur = b + 1
    if cur <= length:
        segs.append(dict(name="Cterm", start=cur, end=length,
                         type="soluble" if (length - cur) >= 60 else "loop", side=side_at(cur, length) or side))
    if not tms:
        segs = [dict(name="soluble", start=1, end=length, type="soluble", side=n_term)]
    return segs, n_term


def predict_tm_hydropathy(seq: str, window: int = 19, threshold: float = 1.6,
                          min_len: int = 17, max_len: int = 25) -> list[tuple[int, int, str]]:
    """Kyte-Doolittle sliding window TM prediction - a fallback when UniProt has nothing."""
    h = np.array([KYTE_DOOLITTLE.get(a, 0.0) for a in seq])
    if len(seq) < window:
        return []
    avg = np.convolve(h, np.ones(window) / window, mode="valid")
    above = avg >= threshold
    tms, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            centre = (i + j) // 2 + window // 2
            half = min(max(min_len, (j - i) + window), max_len) // 2
            a, b = max(1, centre - half + 1), min(len(seq), centre + half)
            if tms and a <= tms[-1][1] + 3:
                a = tms[-1][1] + 4
            if b - a + 1 >= 14:
                tms.append((a, b, "TM"))
            i = j + 1
        else:
            i += 1
    return tms


def positive_inside_n_terminus(seq: str, tms: list[tuple[int, int, str]]) -> str:
    """Pick the N-terminal side by the positive-inside rule (K+R in flanking 15 aa)."""
    if not tms:
        return "cytosolic"
    odd, even = 0, 0
    bounds = [0] + [x for t in tms for x in (t[0] - 1, t[1])] + [len(seq)]
    for k in range(0, len(bounds) - 1, 2):
        a, b = bounds[k], bounds[k + 1]
        flank = seq[max(a, b - 15):b] + seq[a:min(b, a + 15)]
        n = sum(c in "KR" for c in flank)
        if (k // 2) % 2 == 0:
            odd += n
        else:
            even += n
    return "cytosolic" if odd >= even else "lumenal"


# -------------------------------------------------------------------- motifs
DEFAULT_MOTIFS = {
    "GxxxG": r"G...G",
    "small_xxx_small": r"[GAS]...[GAS]",
    "N_glyc_sequon": r"N[^P][ST]",
}


def load_motifs(path: Path | None = None) -> dict[str, str]:
    motifs = dict(DEFAULT_MOTIFS)
    if path and Path(path).exists():
        with open(path) as fh:
            extra = yaml.safe_load(fh) or {}
        motifs.update({k: v["regex"] if isinstance(v, dict) else v for k, v in extra.items()})
    return motifs


def motif_hits(seq: str, regex: str) -> set[tuple[int, int]]:
    """All (start, end) 1-based spans matching regex (overlapping allowed)."""
    pat = re.compile(f"(?=({regex}))")
    return {(m.start() + 1, m.start() + len(m.group(1))) for m in pat.finditer(seq)}


def motif_effects(seq: str, variants: pd.DataFrame, motifs: dict[str, str]) -> pd.DataFrame:
    """For each missense variant and motif class: destroyed / created / rearranged / unaffected."""
    base = {name: motif_hits(seq, rx) for name, rx in motifs.items()}
    rows = []
    for idx, r in variants.iterrows():
        p, m = int(r["pos"]), r["mut"]
        if not (0 < p <= len(seq)) or m not in AA20:
            continue
        mseq = seq[: p - 1] + m + seq[p:]
        for name, rx in motifs.items():
            new = motif_hits(mseq, rx)
            lost, gained = base[name] - new, new - base[name]
            eff = ("destroyed" if lost and not gained else "created" if gained and not lost
                   else "rearranged" if lost and gained else "unaffected")
            rows.append({"index": idx, "motif": name, "effect": eff})
    return pd.DataFrame(rows)
