"""`python -m mpdms esm configs/QDR2.yaml configs/AQR1.yaml [--models 1,2,3,4,5] [--device cuda]`

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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mpdms esm")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--models", default="1,2,3,4,5", help="ESM-1v model numbers to ensemble (default all 5)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="recompute even if the CSV exists")
    a = ap.parse_args(argv)
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
