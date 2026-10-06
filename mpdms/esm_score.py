"""`python -m mpdms esm configs/QDR2.yaml configs/AQR1.yaml [--models 1,2,3,4,5] [--device cuda]`
`python -m mpdms esm configs/QDR2.yaml configs/AQR1.yaml --from-table all_esm1v_predictions_with_mean.csv`

ESM-1v variant effect scores by the masked-marginal method (Meier et al. 2021, NeurIPS):
    score(pos, wt->mut) = log p(mut | seq masked at pos) - log p(wt | seq masked at pos)
averaged over the chosen ESM-1v models. Writes data/external/<GENE>/<ID>_esm1v.csv
(pos, wt, mut, esm1v, esm1v_model1..N) and points the config's evolution.esm_scores at it.

Needs `pip install fair-esm torch`. Weights (~2.6 GB per model) download once to
$TORCH_HOME (set it to scratch space, home quotas are small). On CPU one model takes
~5-20 min for a 600-residue protein; a GPU node is much faster.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT, load_config, protein_sequence

AA20 = "ACDEFGHIKLMNPQRSTVWY"
MAX_LEN = 1022  # ESM-1v positional limit (tokens excluding BOS/EOS)


def windows(L: int, size: int = MAX_LEN, overlap: int = 256) -> list[tuple[int, int]]:
    """Overlapping windows covering 0..L; each position is scored in the window where it is most central."""
    if L <= size:
        return [(0, L)]
    out, start = [], 0
    while True:
        end = min(start + size, L)
        out.append((start, end))
        if end == L:
            return out
        start = end - overlap


def masked_marginals(seq: str, logprob_fn, batch: int = 8) -> np.ndarray:
    """L x 20 array of log p(aa | masked at pos) - log p(wt | masked at pos).

    logprob_fn(list_of_(window_seq, masked_index)) -> array (n, 20) of log-probabilities over AA20
    at the masked index. Kept model-agnostic so it can be tested without ESM weights."""
    L = len(seq)
    out = np.full((L, 20), np.nan)
    wins = windows(L)
    # assign each position to the window where it is furthest from an edge
    best = {}
    for ws, we in wins:
        for p in range(ws, we):
            margin = min(p - ws, we - 1 - p)
            if p not in best or margin > best[p][1]:
                best[p] = ((ws, we), margin)
    jobs = [(p, best[p][0]) for p in range(L)]
    for k in range(0, L, batch):
        chunk = jobs[k:k + batch]
        lp = logprob_fn([(seq[ws:we], p - ws) for p, (ws, we) in chunk])
        for (p, _), row in zip(chunk, lp):
            wt = seq[p]
            if wt in AA20:
                out[p] = row - row[AA20.index(wt)]
    return out


def esm_logprob_fn(model_idx: int, device: str = "auto"):
    import esm
    import torch
    model, alphabet = getattr(esm.pretrained, f"esm1v_t33_650M_UR90S_{model_idx}")()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    conv = alphabet.get_batch_converter()
    aa_idx = [alphabet.get_idx(a) for a in AA20]

    def fn(items):
        data = [(str(i), s) for i, (s, _) in enumerate(items)]
        _, _, toks = conv(data)
        for i, (_, m) in enumerate(items):
            toks[i, m + 1] = alphabet.mask_idx  # +1 for BOS
        with torch.no_grad():
            logits = model(toks.to(device))["logits"]
        lp = torch.log_softmax(logits, dim=-1)
        rows = [lp[i, m + 1, aa_idx].cpu().numpy() for i, (_, m) in enumerate(items)]
        return np.array(rows)
    return fn, device


def score_sequence(seq: str, models: list[int], device: str = "auto", batch: int = 8) -> pd.DataFrame:
    mats = []
    for k in models:
        fn, dev = esm_logprob_fn(k, device)
        print(f"    ESM-1v model {k} on {dev} ...", flush=True)
        mats.append(masked_marginals(seq, fn, batch))
    return to_long(seq, mats, models)


def to_long(seq: str, mats: list[np.ndarray], models: list[int]) -> pd.DataFrame:
    rows = []
    for p, wt in enumerate(seq):
        for j, aa in enumerate(AA20):
            if aa == wt:
                continue
            r = {"pos": p + 1, "wt": wt, "mut": aa}
            vals = [m[p, j] for m in mats]
            for k, v in zip(models, vals):
                r[f"esm1v_model{k}"] = v
            r["esm1v"] = float(np.nanmean(vals))
            rows.append(r)
    return pd.DataFrame(rows)


def set_config_esm(cfg_path: Path, csv_rel: str) -> None:
    """Point evolution.esm_scores at the CSV, editing the YAML text in place (keeps comments)."""
    txt = Path(cfg_path).read_text()
    if re.search(r"^\s*esm_scores:", txt, flags=re.M):
        txt = re.sub(r"^(\s*esm_scores:).*$", rf"\1 {csv_rel}", txt, count=1, flags=re.M)
    elif re.search(r"^evolution:", txt, flags=re.M):
        txt = re.sub(r"^(evolution:.*)$", rf"\1\n  esm_scores: {csv_rel}", txt, count=1, flags=re.M)
    else:
        txt += f"\nevolution:\n  esm_scores: {csv_rel}\n"
    Path(cfg_path).write_text(txt)


def set_config_uniprot(cfg_path: Path, acc: str) -> None:
    """Record the accession under protein.uniprot so later runs skip the sequence search."""
    txt = Path(cfg_path).read_text()
    if re.search(r"^\s*uniprot:\s*\S", txt, flags=re.M):
        return
    if re.search(r"^\s*uniprot:", txt, flags=re.M):          # present but null/blank
        txt = re.sub(r"^(\s*uniprot:).*$", rf"\1 {acc}", txt, count=1, flags=re.M)
    elif re.search(r"^protein:", txt, flags=re.M):
        txt = re.sub(r"^(protein:.*)$", rf"\1\n  uniprot: {acc}", txt, count=1, flags=re.M)
    else:
        return
    Path(cfg_path).write_text(txt)


MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


CHUNK = 1_000_000


def _columns(path: Path) -> dict:
    """Which columns of the table carry the accession, the mutation and the scores."""
    head = pd.read_csv(path, nrows=0)
    names = [c for c in head.columns if not str(c).startswith("Unnamed")]
    cols = {str(c).lower().strip(): c for c in names}
    idc = next((cols[k] for k in ("id", "uniprot", "accession", "uniprot_id", "protein") if k in cols), None)
    mc = next((cols[k] for k in ("mutation", "mutant", "variant", "mut") if k in cols), None)
    sc = next((cols[k] for k in ("mean", "esm1v", "score", "esm1v_mean") if k in cols), None)
    if not (idc and mc and sc):
        raise ValueError(f"{path}: need id, mutation and mean/esm1v columns; found {names}")
    models = {c: int(k.group(1)) for c in names
              if (k := re.match(r"^delta_logp_(\d+)$", str(c)))}
    return {"id": idc, "mut": mc, "score": sc, "models": models}


def parse_mutations(s: pd.Series) -> pd.DataFrame:
    """Split M1A-style strings into wt, pos, mut.

    Slicing the ends off is five times quicker than a regex over tens of millions of rows,
    which is the difference between a scan and a coffee break on a proteome-wide table.
    Anything that does not look like letter-digits-letter gets a NaN position and is dropped
    by the caller, exactly as the regex used to leave it."""
    s = s.astype(str)
    wt, mut = s.str[0], s.str[-1]
    pos = pd.to_numeric(s.str[1:-1], errors="coerce")
    a, b = wt.to_numpy().astype("U1"), mut.to_numpy().astype("U1")
    good = (a >= "A") & (a <= "Z") & (b >= "A") & (b <= "Z") & (s.str.len().to_numpy() >= 3)
    return pd.DataFrame({"wt": wt, "pos": pos.where(pd.Series(good, index=s.index)), "mut": mut})


def _normalise(t: pd.DataFrame, c: dict, quiet: bool = False) -> pd.DataFrame:
    m = parse_mutations(t[c["mut"]])
    out = pd.DataFrame({"acc": t[c["id"]].astype(str).str.strip(), "wt": m["wt"],
                        "pos": m["pos"], "mut": m["mut"]})
    if c["score"] in t.columns:
        out["esm1v"] = pd.to_numeric(t[c["score"]], errors="coerce")
    for col, k in c["models"].items():
        if col in t.columns:
            out[f"esm1v_model{k}"] = pd.to_numeric(t[col], errors="coerce")
    bad = int(out.pos.isna().sum())
    if bad and not quiet:
        print(f"  note: {bad} rows with unparseable mutation strings skipped (expected like M1A)")
    out = out.dropna(subset=["pos"])
    out["pos"] = out.pos.astype(int)
    return out


def iter_bulk_table(path: Path, c: dict | None = None, usecols: list | None = None,
                    chunksize: int = CHUNK):
    """Stream the table in chunks of normalised rows, never holding all of it in memory."""
    c = c or _columns(path)
    for raw in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        yield _normalise(raw, c, quiet=True)


def read_bulk_table(path: Path) -> pd.DataFrame:
    """Whole precomputed ESM-1v table, e.g.
        id,mutation,delta_logp_1..5,mean     (rows may carry an extra leading row number)
    Columns: acc, pos, wt, mut, esm1v (+ esm1v_model1..N when present). Proteome-wide tables
    are tens of millions of rows, so prefer the streaming helpers below."""
    c = _columns(path)
    return _normalise(pd.read_csv(path), c)


def _rank(stats: pd.DataFrame, seq: str, min_agreement: float, min_positions: int):
    """stats: one row per accession with ok, n, positions. -> (accession or None, info)."""
    g = stats[stats.positions >= min_positions].sort_values("agreement", ascending=False)
    if not len(g):
        return None, {"reason": f"no accession covers {min_positions}+ positions in range",
                      "n_accessions_searched": int(len(stats))}
    best = g.index[0]
    info = {"accession": str(best), "agreement": float(g.agreement.iloc[0]),
            "n_variants": int(g.n.iloc[0]), "positions": int(g.positions.iloc[0]),
            "runner_up": (str(g.index[1]) if len(g) > 1 else None),
            "runner_up_agreement": (float(g.agreement.iloc[1]) if len(g) > 1 else None),
            "n_accessions_searched": int(len(stats))}
    if info["agreement"] < min_agreement:
        info["reason"] = (f"best match {best} agrees at only {info['agreement']:.1%}, "
                          f"below {min_agreement:.0%}")
        return None, info
    return str(best), info


def _agreement(acc, pos, wt, seq: str) -> pd.DataFrame:
    """Per-accession agreement between the table's wild-type residues and the sequence."""
    target = np.array(list(seq))
    pos = np.asarray(pos, dtype=int)
    inr = (pos >= 1) & (pos <= len(seq))
    if not inr.any():
        return pd.DataFrame(columns=["ok", "n", "positions"])
    d = pd.DataFrame({"acc": np.asarray(acc)[inr], "pos": pos[inr],
                      "ok": target[pos[inr] - 1] == np.asarray(wt)[inr]})
    return d.groupby("acc").agg(ok=("ok", "sum"), n=("ok", "size"), positions=("pos", "nunique"))


def match_accession(bulk: pd.DataFrame, seq: str, min_agreement: float = 0.9,
                    min_positions: int = 20) -> tuple[str | None, dict]:
    """Find which accession in an in-memory table is this protein, from the sequence."""
    b = bulk.dropna(subset=["pos", "wt"])
    stats = _agreement(b.acc.to_numpy(), b.pos.to_numpy(int), b.wt.to_numpy(), seq)
    if not len(stats):
        return None, {"reason": "no rows fall inside the sequence"}
    stats["agreement"] = stats.ok / stats.n
    return _rank(stats, seq, min_agreement, min_positions)


def match_accession_streaming(path: Path, seq: str, min_agreement: float = 0.9,
                              min_positions: int = 20, chunksize: int = CHUNK):
    """Identify the protein by sequence, reading only the accession and mutation columns.

    Every mutation string carries the wild-type residue it was made from, so an accession
    can be identified by how well those residues agree with the sequence at the same
    positions. That is more reliable than guessing an accession and silently importing
    another protein's scores, which would look perfectly normal downstream. Only two string
    columns are parsed and only per-accession counters are kept, so a proteome-wide table
    costs a scan rather than its full size in memory.
    """
    c = _columns(path)
    counts, pairs = [], []
    for ch in iter_bulk_table(path, c, usecols=[c["id"], c["mut"]], chunksize=chunksize):
        part = _agreement(ch.acc.to_numpy(), ch.pos.to_numpy(int), ch.wt.to_numpy(), seq)
        if len(part):
            counts.append(part[["ok", "n"]])
            inr = ch.loc[(ch.pos >= 1) & (ch.pos <= len(seq)), ["acc", "pos"]]
            pairs.append(inr.drop_duplicates())   # only distinct (acc, pos) survive a chunk
    if not counts:
        return None, {"reason": "no rows fall inside the sequence"}
    stats = pd.concat(counts).groupby(level=0).sum()
    seen = pd.concat(pairs).drop_duplicates().groupby("acc").pos.nunique()
    stats["positions"] = seen.reindex(stats.index).fillna(0).astype(int)
    stats["agreement"] = stats.ok / stats.n
    return _rank(stats, seq, min_agreement, min_positions)


def rows_for_accession(path: Path, acc: str, chunksize: int = CHUNK) -> pd.DataFrame:
    """Stream the table and keep only this accession's rows."""
    c = _columns(path)
    keep = [ch[ch.acc == acc] for ch in iter_bulk_table(path, c, chunksize=chunksize)]
    keep = [k for k in keep if len(k)]
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(
        columns=["acc", "wt", "pos", "mut", "esm1v"])


def import_from_table(table: Path, cfg_path: Path, acc_override: str | None = None,
                      bulk: pd.DataFrame | None = None) -> None:
    """Pull one protein's scores out of a precomputed table.

    The table may cover a whole proteome (tens of millions of rows), so it is streamed:
    an accession named in the config goes straight to a single filtering pass, and an
    unknown one costs one extra pass over two string columns to identify it by sequence.
    """
    cfg = load_config(cfg_path)
    acc = acc_override or cfg.get_path("protein.uniprot")
    seq = protein_sequence(cfg)
    table = Path(table)

    if bulk is not None:                                   # already in memory
        if (not acc or acc not in set(bulk.acc)) and seq:
            why = "none given" if not acc else f"{acc} is not in the table"
            acc = _report_match(cfg.id, why, *match_accession(bulk, seq))
            if not acc:
                return
        sub = bulk[bulk.acc == acc].drop(columns="acc") if acc else None
    else:
        if not acc and not seq:
            print(f"{cfg.id}: no protein.uniprot in config and no sequence to match on "
                  "- pass --id ACCESSION")
            return
        if not acc:
            print(f"{cfg.id}: scanning {table.name} for the matching accession ...", flush=True)
            acc = _report_match(cfg.id, "none given", *match_accession_streaming(table, seq))
            if not acc:
                return
        sub = rows_for_accession(table, acc)
        if not len(sub) and seq:
            acc = _report_match(cfg.id, f"{acc} is not in the table",
                                *match_accession_streaming(table, seq))
            if not acc:
                return
            sub = rows_for_accession(table, acc)
        sub = sub.drop(columns="acc")

    if acc is None or sub is None or not len(sub):
        print(f"{cfg.id}: {acc} has no rows in {table.name} - use --id")
        return
    if seq:
        pos = sub.pos.to_numpy(int)
        inr = (pos >= 1) & (pos <= len(seq))
        ok = np.zeros(len(sub), dtype=bool)
        ok[inr] = np.array(list(seq))[pos[inr] - 1] == sub.wt.to_numpy()[inr]
        print(f"{cfg.id}: {acc}: {len(sub):,} variants, {sub.pos.nunique()} positions; "
              f"WT matches config sequence for {ok.mean():.1%}")
        if ok.mean() < 0.95:
            print("  WARNING: WT residues disagree with the config sequence - different isoform/numbering?")
    gene = cfg.get_path("protein.gene") or cfg.id
    out = REPO_ROOT / "data" / "external" / str(gene).upper() / f"{cfg.id}_esm1v.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    set_config_esm(Path(cfg_path), str(out.relative_to(REPO_ROOT)))
    set_config_uniprot(Path(cfg_path), str(acc))   # next run skips the search
    print(f"  -> {out.relative_to(REPO_ROOT)}; evolution.esm_scores set in {cfg_path}")


def _report_match(cfg_id: str, why: str, found: str | None, info: dict) -> str | None:
    if not found:
        print(f"{cfg_id}: accession {why} and no sequence match "
              f"({info.get('reason', '')}) - pass --id ACCESSION")
        return None
    print(f"{cfg_id}: accession {why}; matched {found} by sequence "
          f"({info['agreement']:.1%} of {info['n_variants']:,} wild-type residues agree "
          f"over {info['positions']} positions, searched {info['n_accessions_searched']:,})")
    if info.get("runner_up_agreement") is not None:
        print(f"  next best {info['runner_up']} at {info['runner_up_agreement']:.1%}")
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mpdms esm")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--models", default="1,2,3,4,5", help="ESM-1v model numbers to ensemble (default all 5)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="recompute even if the CSV exists")
    ap.add_argument("--from-table", type=Path,
                    help="take scores from a precomputed multi-protein table (id, mutation, ..., mean) instead of running ESM")
    ap.add_argument("--id", help="UniProt accession to pick from --from-table (default: protein.uniprot in the config)")
    a = ap.parse_args(argv)
    if a.from_table:
        for c in a.configs:
            import_from_table(a.from_table, Path(c), a.id)
        return
    models = [int(x) for x in a.models.split(",")]
    for c in a.configs:
        cfg = load_config(Path(c))
        seq = protein_sequence(cfg)
        if not seq:
            print(f"{cfg.id}: no protein sequence in config - run init first")
            continue
        gene = cfg.get_path("protein.gene") or cfg.id
        out = REPO_ROOT / "data" / "external" / str(gene).upper() / f"{cfg.id}_esm1v.csv"
        if out.exists() and not a.force:
            print(f"{cfg.id}: {out.relative_to(REPO_ROOT)} exists (use --force to recompute)")
        else:
            print(f"{cfg.id}: scoring {len(seq)} residues with ESM-1v models {models}", flush=True)
            tab = score_sequence(seq, models, a.device, a.batch)
            out.parent.mkdir(parents=True, exist_ok=True)
            tab.to_csv(out, index=False)
            print(f"  -> {out.relative_to(REPO_ROOT)}")
        set_config_esm(Path(c), str(out.relative_to(REPO_ROOT)))
        print(f"  config {c}: evolution.esm_scores set")


if __name__ == "__main__":
    main()
