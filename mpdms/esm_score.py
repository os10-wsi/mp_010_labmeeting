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


MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def read_bulk_table(path: Path) -> pd.DataFrame:
    """Precomputed ESM-1v table covering many proteins, e.g.
        id,mutation,delta_logp_1..5,mean     (rows may carry an extra leading row number)
    Returns columns: acc, pos, wt, mut, esm1v (+ esm1v_model1..N when present)."""
    t = pd.read_csv(path)
    t = t.loc[:, [c for c in t.columns if not str(c).startswith("Unnamed")]]
    cols = {c.lower().strip(): c for c in t.columns}
    idc = next((cols[k] for k in ("id", "uniprot", "accession", "uniprot_id", "protein") if k in cols), None)
    mc = next((cols[k] for k in ("mutation", "mutant", "variant", "mut") if k in cols), None)
    sc = next((cols[k] for k in ("mean", "esm1v", "score", "esm1v_mean") if k in cols), None)
    if not (idc and mc and sc):
        raise ValueError(f"{path}: need id, mutation and mean/esm1v columns; found {list(t.columns)}")
    m = t[mc].astype(str).str.extract(MUT_RE)
    out = pd.DataFrame({"acc": t[idc].astype(str).str.strip(), "wt": m[0], "pos": pd.to_numeric(m[1]),
                        "mut": m[2], "esm1v": pd.to_numeric(t[sc], errors="coerce")})
    for c in t.columns:
        k = re.match(r"^delta_logp_(\d+)$", str(c))
        if k:
            out[f"esm1v_model{k.group(1)}"] = pd.to_numeric(t[c], errors="coerce")
    bad = out.pos.isna().sum()
    if bad:
        print(f"  note: {bad} rows with unparseable mutation strings skipped (expected like M1A)")
    out = out.dropna(subset=["pos"])
    out["pos"] = out.pos.astype(int)
    return out


def match_accession(bulk: pd.DataFrame, seq: str, min_agreement: float = 0.9,
                    min_positions: int = 20) -> tuple[str | None, dict]:
    """Find which accession in a proteome-wide table is this protein, from the sequence.

    Every mutation string carries the wild-type residue it was made from, so an accession
    can be identified by how well those residues agree with the sequence at the same
    positions. That is more reliable than guessing an accession and silently importing
    another protein's scores, which would look perfectly normal downstream.
    """
    target = np.array(list(seq))
    b = bulk[(bulk.pos >= 1) & (bulk.pos <= len(seq))].dropna(subset=["pos", "wt"])
    if not len(b):
        return None, {"reason": "no rows fall inside the sequence"}
    agree = target[b.pos.to_numpy(int) - 1] == b.wt.to_numpy()
    g = (pd.DataFrame({"acc": b.acc.to_numpy(), "ok": agree, "pos": b.pos.to_numpy(int)})
         .groupby("acc").agg(agreement=("ok", "mean"), n=("ok", "size"),
                             positions=("pos", "nunique")))
    g = g[g.positions >= min_positions].sort_values("agreement", ascending=False)
    if not len(g):
        return None, {"reason": f"no accession covers {min_positions}+ positions in range"}
    best = g.index[0]
    info = {"accession": best, "agreement": float(g.agreement.iloc[0]),
            "n_variants": int(g.n.iloc[0]), "positions": int(g.positions.iloc[0]),
            "runner_up": (str(g.index[1]) if len(g) > 1 else None),
            "runner_up_agreement": (float(g.agreement.iloc[1]) if len(g) > 1 else None),
            "n_accessions_searched": int(bulk.acc.nunique())}
    if info["agreement"] < min_agreement:
        info["reason"] = (f"best match {best} agrees at only {info['agreement']:.1%}, "
                          f"below {min_agreement:.0%}")
        return None, info
    return best, info


def import_from_table(table: Path, cfg_path: Path, acc_override: str | None = None, bulk=None) -> None:
    cfg = load_config(cfg_path)
    acc = acc_override or cfg.get_path("protein.uniprot")
    bulk = read_bulk_table(table) if bulk is None else bulk
    seq = protein_sequence(cfg)
    if (not acc or acc not in set(bulk.acc)) and seq:
        why = "none given" if not acc else f"{acc} is not in the table"
        found, info = match_accession(bulk, seq)
        if found:
            acc = found
            print(f"{cfg.id}: accession {why}; matched {found} by sequence "
                  f"({info['agreement']:.1%} of {info['n_variants']:,} wild-type residues agree "
                  f"over {info['positions']} positions, searched {info['n_accessions_searched']:,})")
            if info.get("runner_up_agreement") is not None:
                print(f"  next best {info['runner_up']} at {info['runner_up_agreement']:.1%}")
        else:
            print(f"{cfg.id}: accession {why} and no sequence match "
                  f"({info.get('reason', '')}) - pass --id ACCESSION")
            return
    if not acc:
        print(f"{cfg.id}: no protein.uniprot in config and no sequence to match on "
              "- pass --id ACCESSION")
        return
    sub = bulk[bulk.acc == acc].drop(columns="acc")
    if not len(sub):
        print(f"{cfg.id}: {acc} has no rows in {Path(table).name}")
        return
    if sub.empty:
        print(f"{cfg.id}: accession {acc} not in {Path(table).name} "
              f"({bulk.acc.nunique()} proteins there, e.g. {', '.join(bulk.acc.unique()[:6])}) - use --id")
        return
    if seq:
        ok = sub.apply(lambda r: r.pos <= len(seq) and seq[r.pos - 1] == r.wt, axis=1)
        print(f"{cfg.id}: {acc}: {len(sub):,} variants, {sub.pos.nunique()} positions; "
              f"WT matches config sequence for {ok.mean():.1%}")
        if ok.mean() < 0.95:
            print("  WARNING: WT residues disagree with the config sequence - different isoform/numbering?")
    gene = cfg.get_path("protein.gene") or cfg.id
    out = REPO_ROOT / "data" / "external" / str(gene).upper() / f"{cfg.id}_esm1v.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    set_config_esm(Path(cfg_path), str(out.relative_to(REPO_ROOT)))
    print(f"  -> {out.relative_to(REPO_ROOT)}; evolution.esm_scores set in {cfg_path}")


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
        bulk = read_bulk_table(a.from_table)
        for c in a.configs:
            import_from_table(a.from_table, Path(c), a.id, bulk)
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
