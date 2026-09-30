"""load_dataset(): source table -> canonical long table with score_z.

Normalisation (per protein, per replicate and for the mean score):
    score_z = (x - median(synonymous)) / (median(synonymous) - median(nonsense))
so that median(synonymous) -> 0 and median(nonsense) -> -1. Anchors are
computed on variants that pass the read/replicate filters.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .annot import assign_topology
from .config import Config, protein_sequence

AA20 = "ACDEFGHIKLMNPQRSTVWY"
THREE2ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Stop": "*", "*": "*", "=": "=",
}
CODON_TABLE = {
    a + b + c: aa for (a, b, c), aa in zip(
        [(x, y, z) for x in "TCAG" for y in "TCAG" for z in "TCAG"],
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    )
}

# candidate column names, in priority order (case-insensitive match)
CANDIDATES = {
    "position": ["pos", "position", "aa_pos", "residue", "residue_number", "resnum", "site"],
    "wt_aa": ["wt_aa", "wt", "wtaa", "aa_wt", "ref_aa", "wildtype", "wild_type", "wt_residue"],
    "mut_aa": ["mut_aa", "mut", "mutaa", "aa_mut", "alt_aa", "mt", "mutant", "mut_residue"],
    "variant": ["variant", "hgvs", "hgvs_pro", "mutation", "aa_substitution", "aa_change", "mutant_id", "id"],
    "aa_seq": ["aa_seq", "aa_sequence", "protein_seq"],
    "nt_seq": ["nt_seq", "nt_sequence", "dna_seq"],
    "score": ["rescaled_fitness", "fitness", "mean_fitness", "fitness_mean", "score", "mean_score",
              "rescaled_fitness_mean", "mean_rescaled_fitness"],
    "se": ["rescaled_sigma", "sigma", "se", "fitness_se", "std_error", "sem", "sd", "fitness_sd"],
    "n_reads": ["n_reads", "mean_count", "input_count", "depth", "count_input", "reads"],
    "codon": ["codon", "mut_codon", "alt_codon", "variant_codon"],
    "variant_class": ["variant_class", "vclass", "mutation_type", "consequence", "class"],
    "aa_ham": ["aa_ham", "nham_aa", "n_aa_mut", "aa_hamming"],
    "nt_ham": ["nt_ham", "nham_nt", "n_nt_mut", "nt_hamming"],
    "wt_flag": ["wt", "is_wt", "wildtype_flag"],
    "stop_flag": ["stop", "is_stop", "nonsense"],
}
REP_PATTERNS = [
    r"^rescaled_fitness_rep_?(\d+)$", r"^fitness_rep_?(\d+)$", r"^fitness(\d+)_uncorr$",
    r"^rep_?(\d+)_fitness$", r"^fitness_?(\d+)$", r"^score_rep_?(\d+)$", r"^rep_?(\d+)$",
]
REP_SE_PATTERNS = [r"^rescaled_sigma_rep_?(\d+)$", r"^sigma_rep_?(\d+)$", r"^sigma(\d+)_uncorr$"]
INPUT_PATTERNS = [r"^input_?(\d+)(?:_.*)?$", r"^count_input_?(\d+)$", r"^in_?(\d+)_count$"]
OUTPUT_PATTERNS = [r"^output_?(\d+)(?:_.*)?$", r"^count_output_?(\d+)$", r"^out_?(\d+)_count$"]


# ------------------------------------------------------------ source reading
def read_source(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():  # "folder called fitness_estimation.tsv" - take the table inside
        hits = sorted(path.glob("*.tsv")) + sorted(path.glob("*.txt")) + sorted(path.glob("*.csv"))
        if not hits:
            raise FileNotFoundError(f"no .tsv/.txt/.csv inside {path}")
        path = hits[0]
    sep = "," if path.suffix == ".csv" else None
    return pd.read_csv(path, sep=sep, engine="python", comment=None)


def _norm(name: str) -> str:
    return re.sub(r"[\s\-.]+", "_", str(name).strip().lower())


def _find(cols: list[str], names: list[str]) -> str | None:
    low = {_norm(c): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def _find_numbered(cols: list[str], patterns: list[str]) -> list[str]:
    for pat in patterns:
        hits = []
        for c in cols:
            m = re.match(pat, _norm(c), flags=re.I)
            if m:
                hits.append((int(m.group(1)), c))
        if hits:
            return [c for _, c in sorted(hits)]
    return []


def detect_columns(raw: pd.DataFrame, cfg: Config) -> dict:
    cols = list(raw.columns)
    spec = dict(cfg.get("columns") or {})
    out: dict = {}
    for key in ["position", "wt_aa", "mut_aa", "variant", "aa_seq", "nt_seq", "score", "se",
                "n_reads", "codon", "variant_class", "aa_ham", "nt_ham", "wt_flag", "stop_flag"]:
        v = spec.get(key, "auto")
        if v in (None, "null"):
            out[key] = None
        elif v == "auto":
            out[key] = _find(cols, CANDIDATES[key])
        else:
            if v not in cols:
                raise KeyError(f"columns.{key} = {v!r} not in source columns {cols}")
            out[key] = v
    for key in ("wt_flag", "stop_flag"):
        c = out.get(key)
        if c is not None and (c in (out.get("wt_aa"), out.get("mut_aa")) or not _is_flag(raw[c])):
            out[key] = None
    # a DiMSum "WT" column is a boolean flag, not the WT residue
    for key in ("wt_aa", "mut_aa"):
        c = out.get(key)
        if c is not None:
            vals = raw[c].dropna().astype(str)
            if not vals.str.fullmatch(r"[A-Za-z*=_]{1,4}").mean() > 0.9 or vals.str.upper().isin(["TRUE", "FALSE"]).mean() > 0.5:
                out[key] = None
    for key, pats in [("replicates", REP_PATTERNS), ("replicate_se", REP_SE_PATTERNS),
                      ("input_counts", INPUT_PATTERNS), ("output_counts", OUTPUT_PATTERNS)]:
        v = spec.get(key, "auto")
        if v == "auto" or v is None and key == "replicate_se":
            out[key] = _find_numbered(cols, pats)
        else:
            out[key] = list(v or [])
            missing = [c for c in out[key] if c not in cols]
            if missing:
                raise KeyError(f"columns.{key}: {missing} not in source")
    # never treat the mean column as a replicate
    out["replicates"] = [c for c in out["replicates"] if c != out["score"]]
    if out["score"] is None and not out["replicates"]:
        raise KeyError(f"no fitness/score column found in {cols}; set columns.score in the config")
    return out


def _is_flag(col: pd.Series) -> bool:
    v = col.dropna().astype(str).str.strip().str.upper()  # noqa: PD011
    v = v[v != ""]
    return len(v) == 0 or v.isin(["TRUE", "FALSE", "T", "F", "1", "0", "1.0", "0.0", "YES", "NO"]).all()


def _truthy(col: pd.Series) -> pd.Series:
    return col.fillna("").astype(str).str.strip().str.upper().isin(["TRUE", "T", "1", "1.0", "YES"])


# ---------------------------------------------------------- variant parsing
_SINGLE_RE = re.compile(
    r"^(?:p\.)?\(?(?P<wt>[A-Z][a-z]{2}|[A-Z*])(?P<pos>-?\d+)(?P<mut>[A-Z][a-z]{2}|[A-Z*=]|=|del|fs)\)?$"
)


def _aa(tok: str) -> str:
    return THREE2ONE.get(tok, tok) if len(tok) > 1 else tok


def parse_variant_string(s) -> tuple[float, str, str] | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.lower() in ("wt", "wildtype", "_wt", "p.="):
        return None
    m = _SINGLE_RE.match(s)
    if not m:
        return None
    return float(m.group("pos")), _aa(m.group("wt")), _aa(m.group("mut"))


def _parse_from_aa_seq(raw: pd.DataFrame, colmap: dict, wt_seq: str | None) -> pd.DataFrame:
    seqs = raw[colmap["aa_seq"]].astype(str).str.replace("_", "*")
    wt_flag = None
    for c in raw.columns:
        if c.lower() == "wt":
            wt_flag = raw[c].astype(str).str.upper().isin(["TRUE", "T", "1"])
    if wt_flag is not None and wt_flag.any():
        ref = seqs[wt_flag].iloc[0]
    elif wt_seq:
        ref = wt_seq
    else:
        ref = seqs.mode().iloc[0]
    L = len(ref)
    nt_ref = None
    if colmap.get("nt_seq") and wt_flag is not None and wt_flag.any():
        nt_ref = raw.loc[wt_flag, colmap["nt_seq"]].astype(str).iloc[0].upper()
    pos, wt, mut, codon = [], [], [], []
    for i, s in enumerate(seqs):
        if len(s) != L:
            pos.append(np.nan); wt.append(None); mut.append(None); codon.append(None)
            continue
        diffs = [j for j in range(L) if s[j] != ref[j]]
        if len(diffs) == 1:
            j = diffs[0]
            pos.append(j + 1); wt.append(ref[j]); mut.append(s[j]); codon.append(None)
        elif len(diffs) == 0:
            # synonymous (or WT): locate the codon from the nt sequence if we can
            p, cod = np.nan, None
            if nt_ref is not None:
                nt = str(raw[colmap["nt_seq"]].iloc[i]).upper()
                ndiff = [k for k in range(min(len(nt), len(nt_ref))) if nt[k] != nt_ref[k]]
                if ndiff:
                    p = ndiff[0] // 3 + 1
                    cod = nt[(p - 1) * 3:(p - 1) * 3 + 3]
                else:
                    p = -1  # the WT itself
            pos.append(p); wt.append(ref[int(p) - 1] if p and p > 0 else None)
            mut.append("=" if p != -1 else "WT_ROW"); codon.append(cod)
        else:
            pos.append(np.nan); wt.append(None); mut.append(None); codon.append(None)
    out = pd.DataFrame({"pos": pos, "wt": wt, "mut": mut, "codon_parsed": codon}, index=raw.index)
    return out


def parse_variants(raw: pd.DataFrame, colmap: dict, cfg: Config, wt_seq: str | None) -> pd.DataFrame:
    syn = set(cfg.get_path("variant_encoding.synonymous_symbols", []))
    stop = set(cfg.get_path("variant_encoding.nonsense_symbols", []))
    if colmap["position"] and colmap["wt_aa"] and colmap["mut_aa"]:
        v = pd.DataFrame({
            "pos": pd.to_numeric(raw[colmap["position"]], errors="coerce"),
            "wt": raw[colmap["wt_aa"]].fillna("").astype(str).str.strip(),
            "mut": raw[colmap["mut_aa"]].fillna("").astype(str).str.strip(),
        }, index=raw.index)
        v["wt"] = v["wt"].map(lambda t: (_aa(t.capitalize()) if len(t) == 3 else t.upper()) or None)
        v["mut"] = v["mut"].map(lambda t: (_aa(t.capitalize()) if len(t) == 3 else t) or None)
    elif colmap["variant"]:
        parsed = raw[colmap["variant"]].map(parse_variant_string)
        v = pd.DataFrame(
            [p if p else (np.nan, None, None) for p in parsed], columns=["pos", "wt", "mut"], index=raw.index
        )
    elif colmap["aa_seq"]:
        v = _parse_from_aa_seq(raw, colmap, wt_seq)
    else:
        raise KeyError("cannot identify variants: need pos/wt/mut, a variant/hgvs column, or aa_seq")
    v["mut"] = v["mut"].map(lambda m: "*" if m in stop else ("=" if m in syn else m))
    same = v["mut"].notna() & (v["mut"] == v["wt"])
    v.loc[same, "mut"] = "="
    if "codon_parsed" not in v:
        v["codon_parsed"] = None
    v = _recover_synonymous(raw, colmap, v, wt_seq)
    if colmap.get("stop_flag"):
        v.loc[_truthy(raw[colmap["stop_flag"]]) & v["pos"].notna(), "mut"] = "*"
    return v


def _first_seq(x) -> str:
    return str(x).split(",")[0].strip().upper() if isinstance(x, str) else ""


def wt_nt_reference(raw: pd.DataFrame, colmap: dict) -> str | None:
    """WT coding sequence: the nt_ham == 0 / WT-flagged row if present, else the per-base
    consensus of all variant sequences (every variant differs from WT at only a few bases)."""
    nt = colmap.get("nt_seq")
    if not nt:
        return None
    seqs = raw[nt].map(_first_seq)
    for key in ("nt_ham", "wt_flag"):
        c = colmap.get(key)
        if c:
            m = (pd.to_numeric(raw[c], errors="coerce") == 0) if key == "nt_ham" else _truthy(raw[c])
            cand = seqs[m & (seqs != "")]
            if len(cand):
                return cand.iloc[0]
    seqs = seqs[seqs != ""]
    if not len(seqs):
        return None
    L = seqs.str.len().mode().iloc[0]
    arr = np.array([list(x) for x in seqs[seqs.str.len() == L].iloc[:5000]])
    out = []
    for j in range(arr.shape[1]):
        vals, cnt = np.unique(arr[:, j], return_counts=True)
        out.append(vals[np.argmax(cnt)])
    return "".join(out)


def _recover_synonymous(raw: pd.DataFrame, colmap: dict, v: pd.DataFrame, wt_seq: str | None) -> pd.DataFrame:
    """Rows with no amino-acid change often carry no pos/wt/mut (only aa_ham = 0 and an
    nt_seq). Locate the changed codon from the nucleotide sequence so they count as
    synonymous at the right position (and as the 0 anchor); an unchanged row is the WT."""
    missing = v["pos"].isna()
    if colmap.get("aa_ham"):
        missing &= pd.to_numeric(raw[colmap["aa_ham"]], errors="coerce") == 0
    else:
        return v
    if not missing.any():
        return v
    if colmap.get("wt_flag"):
        wtm = missing & _truthy(raw[colmap["wt_flag"]])
        v.loc[wtm, "mut"] = "WT_ROW"
        missing &= ~wtm
    ref = wt_nt_reference(raw, colmap)
    if ref is None:
        v.loc[missing, "mut"] = "="  # synonymous but position unknown -> dropped later
        return v
    v = v.copy()
    for i in v.index[missing]:
        nt = _first_seq(raw.at[i, colmap["nt_seq"]])
        diffs = [k for k in range(min(len(nt), len(ref))) if nt[k] != ref[k]]
        if not diffs:
            v.at[i, "mut"] = "WT_ROW"
            continue
        p = diffs[0] // 3 + 1
        cod = nt[(p - 1) * 3:(p - 1) * 3 + 3]
        wt_res = wt_seq[p - 1] if wt_seq and p <= len(wt_seq) else CODON_TABLE.get(ref[(p - 1) * 3:(p - 1) * 3 + 3], "X")
        v.at[i, "pos"] = p
        v.at[i, "wt"] = wt_res
        v.at[i, "mut"] = "="
        v.at[i, "codon_parsed"] = cod
    return v


# ------------------------------------------------------------ normalisation
def normalise(x: pd.Series, syn_med: float, stop_med: float,
              syn_anchor: float = 0.0, stop_anchor: float = -1.0) -> pd.Series:
    scale = (syn_anchor - stop_anchor) / (syn_med - stop_med)
    return syn_anchor + (x - syn_med) * scale


def anchors(df: pd.DataFrame, col: str, cfg: Config) -> dict:
    ok = df["pass_filter"] & df[col].notna()
    syn = df.loc[ok & (df.vclass == "synonymous"), col]
    stop_mask = ok & (df.vclass == "nonsense")
    n_excl = int(cfg.get_path("normalization.stop_exclude_last_n", 0) or 0)
    if n_excl > 0:
        stop_mask &= df["pos"] <= df["pos"].max() - n_excl
    stop = df.loc[stop_mask, col]
    info = {"column": col, "syn_median": float(syn.median()) if len(syn) else np.nan,
            "n_syn": int(len(syn)), "n_stop": int(len(stop)), "stop_source": "nonsense"}
    if len(stop) >= 5:
        info["stop_median"] = float(stop.median())
    else:
        mis = df.loc[ok & (df.vclass == "missense"), col]
        info["stop_median"] = float(np.nanpercentile(mis, 1)) if len(mis) else np.nan
        info["stop_source"] = "missense_p1_FALLBACK"
    if len(syn) < 5:
        mis = df.loc[ok & (df.vclass == "missense"), col]
        info["syn_median"] = float(mis.median())
        info["syn_source"] = "missense_median_FALLBACK"
    return info


# -------------------------------------------------------------------- main
def load_dataset(cfg: Config, use_cache: bool = False) -> pd.DataFrame:
    proc = cfg.resolve(f"data/processed/{cfg.id}.parquet")
    if use_cache and proc.exists():
        df = pd.read_parquet(proc)
        meta_p = proc.with_suffix(".meta.json")
        if meta_p.exists():
            df.attrs.update(json.loads(meta_p.read_text()))
        return df

    raw = read_source(cfg.resolve(cfg.source.path))
    colmap = detect_columns(raw, cfg)
    wt_seq = protein_sequence(cfg)
    var = parse_variants(raw, colmap, cfg, wt_seq)

    df = pd.DataFrame(index=raw.index)
    df["pos"] = var["pos"]
    df["wt"], df["mut"] = var["wt"], var["mut"]
    sign = 1.0 if cfg.get_path("assay.higher_is_better", True) else -1.0

    reps = colmap["replicates"]
    for i, c in enumerate(reps, 1):
        df[f"rep{i}_raw"] = sign * pd.to_numeric(raw[c], errors="coerce")
    for i, c in enumerate(colmap["replicate_se"][: len(reps)], 1):
        df[f"rep{i}_se_raw"] = pd.to_numeric(raw[c], errors="coerce")
    for i, c in enumerate(colmap["input_counts"], 1):
        df[f"input{i}"] = pd.to_numeric(raw[c], errors="coerce")
    for i, c in enumerate(colmap["output_counts"], 1):
        df[f"output{i}"] = pd.to_numeric(raw[c], errors="coerce")
    if colmap["score"]:
        df["score_raw"] = sign * pd.to_numeric(raw[colmap["score"]], errors="coerce")
    else:
        df["score_raw"] = df[[f"rep{i}_raw" for i in range(1, len(reps) + 1)]].mean(axis=1)
    df["se_raw"] = pd.to_numeric(raw[colmap["se"]], errors="coerce") if colmap["se"] else np.nan
    if colmap["n_reads"]:
        df["n_reads"] = pd.to_numeric(raw[colmap["n_reads"]], errors="coerce")
    elif colmap["input_counts"]:
        df["n_reads"] = df[[f"input{i}" for i in range(1, len(colmap["input_counts"]) + 1)]].mean(axis=1)
    else:
        df["n_reads"] = np.nan
    codon_col = colmap["codon"]
    df["codon"] = raw[codon_col].astype(str) if codon_col else var["codon_parsed"]

    # single amino-acid variants only; keep WT row aside
    n_in = len(df)
    wt_rows = df[df["mut"] == "WT_ROW"]
    df = df[df["pos"].notna() & df["pos"].gt(0) & df["mut"].isin(list(AA20) + ["*", "="])].copy()
    df["pos"] = df["pos"].astype(int) + int(cfg.get_path("source.numbering_offset", 0) or 0)
    df["vclass"] = np.select([df["mut"] == "*", df["mut"] == "="], ["nonsense", "synonymous"], "missense")
    df["wt"] = df["wt"].astype(str).str.upper()

    # collapse exact duplicates (e.g. codon-level rows of the same aa variant)
    key = ["pos", "wt", "mut"] + (["codon"] if df["codon"].notna().any() and (df["vclass"] == "synonymous").any() else [])
    num = [c for c in df.columns if c not in key + ["vclass", "codon"]]
    agg = {c: ("sum" if c.startswith(("input", "output")) else "mean") for c in num}
    agg["vclass"] = "first"
    if "codon" not in key:
        agg["codon"] = "first"
    df = df.groupby(key, as_index=False, dropna=False).agg(agg)

    # ------------------------------------------------------------ filters
    min_reads = float(cfg.get_path("filters.min_reads", 0) or 0)
    min_reps = int(cfg.get_path("filters.min_replicates", 1) or 1)
    nrep = len(reps)
    for i in range(1, nrep + 1):
        inp = f"input{i}"
        df[f"rep{i}_pass"] = df[f"rep{i}_raw"].notna()
        if inp in df and min_reads > 0:
            df[f"rep{i}_pass"] &= df[inp].fillna(0) >= min_reads
    if nrep:
        df["n_reps"] = df[[f"rep{i}_pass" for i in range(1, nrep + 1)]].sum(axis=1)
    else:
        df["n_reps"] = 1
    drop = set(cfg.get_path("filters.drop_positions", []) or [])
    df["pass_filter"] = (df["n_reps"] >= min(min_reps, max(nrep, 1))) & df["score_raw"].notna() & ~df["pos"].isin(drop)
    if nrep == 0 and min_reads > 0 and df["n_reads"].notna().any():
        df["pass_filter"] &= df["n_reads"] >= min_reads

    # ------------------------------------------------------ normalisation
    sa = float(cfg.get_path("normalization.syn_anchor", 0.0))
    ta = float(cfg.get_path("normalization.stop_anchor", -1.0))
    norm = {"score": anchors(df, "score_raw", cfg)}
    a = norm["score"]
    df["score_z"] = normalise(df["score_raw"], a["syn_median"], a["stop_median"], sa, ta)
    df["se"] = df["se_raw"] * abs((sa - ta) / (a["syn_median"] - a["stop_median"]))
    for i in range(1, nrep + 1):
        col = f"rep{i}_raw"
        tmp = df.assign(pass_filter=df["pass_filter"] & df[f"rep{i}_pass"])
        ai = anchors(tmp, col, cfg)
        norm[f"rep{i}"] = ai
        df[f"rep{i}"] = normalise(df[col].where(df[f"rep{i}_pass"]), ai["syn_median"], ai["stop_median"], sa, ta)
        if f"rep{i}_se_raw" in df:
            df[f"rep{i}_se"] = df[f"rep{i}_se_raw"] * abs((sa - ta) / (ai["syn_median"] - ai["stop_median"]))

    df = assign_topology(df, cfg)
    df.insert(0, "dataset_id", cfg.id)
    df = df.sort_values(["pos", "vclass", "mut"]).reset_index(drop=True)

    meta = {
        "dataset_id": cfg.id,
        "columns_detected": {k: v for k, v in colmap.items()},
        "n_source_rows": int(n_in),
        "n_single_variants": int(len(df)),
        "n_pass_filter": int(df["pass_filter"].sum()),
        "n_replicates": nrep,
        "wt_row_score_raw": float(wt_rows["score_raw"].mean()) if len(wt_rows) else None,
        "normalization": norm,
        "normalization_fallback": any("FALLBACK" in str(v.get("stop_source", "")) + str(v.get("syn_source", ""))
                                      for v in norm.values()),
        "sign_flipped": sign < 0,
    }
    df.attrs.update(meta)
    proc.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc, index=False)
    proc.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return df


def replicate_cols(df: pd.DataFrame, raw: bool = False) -> list[str]:
    out = [c for c in df.columns if re.fullmatch(r"rep\d+", c)]
    out = sorted(out, key=lambda c: int(c[3:]))
    return [c + "_raw" for c in out] if raw else out


def position_matrix(df: pd.DataFrame, col: str = "score_z", only_pass: bool = True) -> pd.DataFrame:
    """pos x mut (20 aa + '*') matrix; synonymous averaged into the WT cell."""
    d = df[df["pass_filter"]] if only_pass else df
    d = d.copy()
    d.loc[d["mut"] == "=", "mut"] = d.loc[d["mut"] == "=", "wt"]
    m = d.pivot_table(index="pos", columns="mut", values=col, aggfunc="mean")
    return m.reindex(columns=[c for c in list(AA20) + ["*"]])
