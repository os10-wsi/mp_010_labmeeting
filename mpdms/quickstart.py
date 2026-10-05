"""One command from a DiMSum fitness file plus a DeepTMHMM gff3 to a finished report.

    python -m mpdms quickstart fitness_singles_hxt2.txt --gff3 hxt2.gff3

Everything else is derived. The wild-type protein sequence is recovered from the file
itself - aa_seq is the full mutant sequence, so reverting the one substitution each row
describes gives the wild type - so no sequence has to be fetched or supplied.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT
from .io import CANDIDATES, _find, read_source

TICK, WARN, CROSS = "  [ok]  ", "  [!]   ", "  [x]   "


def say(mark: str, msg: str) -> None:
    print(mark + msg, flush=True)


def gene_from_filename(path: Path) -> str:
    """fitness_singles_hxt2.txt -> HXT2; falls back to the stem."""
    stem = Path(path).stem
    for pat in (r"fitness[_-]singles?[_-](.+)", r"(.+?)[_-]fitness", r"fitness[_-](.+)"):
        m = re.fullmatch(pat, stem, flags=re.I)
        if m:
            return re.sub(r"[^A-Za-z0-9]+", "_", m.group(1)).strip("_").upper()
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").upper()


def wt_sequence(df: pd.DataFrame, pos_c: str, wt_c: str, mut_c: str, seq_c: str) -> tuple[str, dict]:
    """Reconstruct the wild type by reverting each row's own substitution, then take the
    consensus. Disagreement between rows means the table is not internally consistent."""
    votes: Counter = Counter()
    checked = 0
    for r in df.dropna(subset=[seq_c, pos_c, wt_c]).head(500).itertuples():
        s, p = str(getattr(r, seq_c)), int(getattr(r, pos_c))
        wt = str(getattr(r, wt_c))
        if not (1 <= p <= len(s)) or len(wt) != 1:
            continue
        votes[s[:p - 1] + wt + s[p:]] += 1
        checked += 1
    if not votes:
        return "", {"ok": False, "reason": "no usable aa_seq / Pos / WT_AA rows"}
    seq, n = votes.most_common(1)[0]
    return seq, {"ok": True, "length": len(seq), "rows_checked": checked,
                 "agreement": n / max(checked, 1), "n_distinct": len(votes)}


def classify_counts(df: pd.DataFrame, cols: dict) -> dict:
    mut, ham, stop = cols.get("mut_aa"), cols.get("aa_ham"), cols.get("stop_flag")
    out = {"missense": 0, "synonymous": 0, "nonsense": 0}
    m = df[mut].astype(str).str.strip() if mut else pd.Series(dtype=str)
    is_stop = m.isin(["*", "X"]) if len(m) else pd.Series(False, index=df.index)
    if stop:
        is_stop = is_stop | df[stop].astype(str).str.upper().eq("TRUE")
    is_syn = (pd.to_numeric(df[ham], errors="coerce").eq(0) if ham
              else (m == df[cols["wt_aa"]].astype(str).str.strip() if cols.get("wt_aa")
                    else pd.Series(False, index=df.index)))
    out["nonsense"] = int(is_stop.sum())
    out["synonymous"] = int((is_syn & ~is_stop).sum())
    out["missense"] = int(len(df) - out["nonsense"] - out["synonymous"])
    return out


CONFIG = """# written by `python -m mpdms quickstart`
id: {gene}
display_name: {gene}
source:
  path: {source}
  citation: ''
  numbering: uniprot
  numbering_offset: 0
protein:
  gene: {gene}
  organism_id: 559292
  uniprot: null
  sequence: {fasta}
  region: null
assay:
  type: abundance
  readout: growth
  higher_is_better: true
  n_bins: null
columns:
  position: auto
  wt_aa: auto
  mut_aa: auto
  aa_seq: auto
  score: auto
  se: auto
  n_reads: auto
  replicates: auto
  input_counts: auto
  output_counts: auto
  codon: auto
  variant_class: null
  aa_ham: auto
  nt_ham: auto
  wt_flag: auto
  stop_flag: auto
variant_encoding:
  synonymous_symbols: ['=']
filters:
  min_reads: {min_reads}
  min_replicates: 1
topology:
  n_terminus: cytosolic
  source: none
  segments: []
structure:
  path: {structure}
  chain: A
  membrane_normal: {membrane_normal}
  numbering_offset: 0
plotting:
  score_label: Normalised fitness
"""


def build(src: Path, gene: str, structure: Path | None, min_reads: int) -> tuple[Path, dict]:
    raw = read_source(src)
    cols = {k: _find(list(raw.columns), v) for k, v in CANDIDATES.items()}
    need = [k for k in ("position", "wt_aa", "mut_aa", "score") if not cols.get(k)]
    if need:
        raise SystemExit(f"{CROSS}{src.name}: no column found for {need}.\n"
                         f"        columns present: {list(raw.columns)}")
    say(TICK, f"{len(raw):,} rows, columns mapped: "
              + ", ".join(f"{k}={cols[k]}" for k in ("position", "wt_aa", "mut_aa", "score")
                          if cols.get(k)))

    counts = classify_counts(raw, cols)
    say(TICK, f"variants: {counts['missense']:,} missense, {counts['synonymous']:,} synonymous, "
              f"{counts['nonsense']:,} nonsense")
    if counts["synonymous"] < 3 or counts["nonsense"] < 3:
        say(WARN, "fewer than 3 synonymous or nonsense variants. The syn = 0 / nonsense = -1 "
                  "scale needs both; without them scores are anchored on the missense "
                  "distribution and are NOT comparable with other datasets")

    ext = REPO_ROOT / "data" / "external" / gene
    ext.mkdir(parents=True, exist_ok=True)
    fasta = ext / f"{gene}.fasta"
    if cols.get("aa_seq"):
        seq, info = wt_sequence(raw, cols["position"], cols["wt_aa"], cols["mut_aa"], cols["aa_seq"])
        if info["ok"]:
            fasta.write_text(f">{gene}\n{seq}\n")
            say(TICK, f"wild-type sequence recovered from aa_seq: {info['length']} residues, "
                      f"{info['agreement']:.0%} of {info['rows_checked']} rows agree")
            if info["agreement"] < 0.99:
                say(WARN, f"{info['n_distinct']} distinct reconstructions - check {fasta}")
        else:
            fasta = None
            say(WARN, f"could not recover the sequence ({info['reason']}); "
                      "set protein.sequence by hand for the structure-dependent analyses")
    else:
        fasta = None
        say(WARN, "no aa_seq column, so no sequence; set protein.sequence by hand")

    cfg_path = REPO_ROOT / "configs" / f"{gene}.yaml"
    cfg_path.write_text(CONFIG.format(
        gene=gene, source=src.resolve(), fasta=(fasta.resolve() if fasta else "null"),
        structure=(Path(structure).resolve() if structure else "null"),
        membrane_normal=("pca_tm_axes" if structure else "none"), min_reads=min_reads))
    say(TICK, f"config written: {cfg_path}")
    return cfg_path, counts


# what to add to unlock the analyses that most often skip, in the order worth doing
UNLOCKS = [
    ("structure", "an AlphaFold model: --structure model.pdb",
     ("structure.path", "membrane_normal", "helix assignment needs a model", "depth")),
    ("ESM-1v", "ESM-1v scores: python -m mpdms esm <config> --from-table <csv>",
     ("esm_scores", "a15_variants.csv",)),
    ("controls", "synonymous and nonsense variants in the source table",
     ("nonsense variants", "synonymous",)),
    ("topology", "a DeepTMHMM gff3: --gff3 <file>", ("no TM segments", "topology")),
]


def summarise(gene: str, style: str, had_structure: bool) -> None:
    """Say what ran, what did not, and what would unlock it - the skips are the useful part."""
    import json
    sp = REPO_ROOT / "outputs" / gene / "stats.json"
    if not sp.exists():
        return
    res = json.loads(sp.read_text())
    ok = [r for r in res if r.get("status") == "ok"]
    skipped = [r for r in res if r.get("status") == "skipped"]
    errors = [r for r in res if r.get("status") == "error"]
    figdir = f"outputs/{gene}/{'figures_paper' if style == 'paper' else 'figures'}"
    print(f"\n== {gene}: {len(ok)} analyses ran, {len(skipped)} skipped"
          + (f", {len(errors)} ERRORED" if errors else ""))
    for r in errors:
        say(CROSS, f"{r['analysis']}: {r.get('reason', '')[:90]}")
    for _, advice, needles in UNLOCKS:
        hit = [r["analysis"] for r in skipped
               if any(n in str(r.get("reason", "")) for n in needles)]
        if hit:
            say(WARN, f"{len(hit)} skipped for want of {advice}\n"
                      f"          ({', '.join(sorted(hit))})")
    other = [r["analysis"] for r in skipped
             if not any(any(n in str(r.get("reason", "")) for n in nd) for _, _, nd in UNLOCKS)]
    if other:
        say(WARN, f"{len(other)} skipped for other reasons: {', '.join(sorted(other))} "
                  "(see outputs/%s/stats.json)" % gene)
    print(f"\n{TICK}figures: {figdir}/")
    print(f"{TICK}report:  outputs/{gene}/report/{gene}_report.pdf\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m mpdms quickstart",
        description="One command: a DiMSum fitness table plus a DeepTMHMM gff3 -> a full report.")
    ap.add_argument("fitness", help="fitness_singles_<gene>.txt (or any DiMSum variant table)")
    ap.add_argument("--gff3", help="DeepTMHMM .gff3 for the same protein")
    ap.add_argument("--gene", help="override the name inferred from the filename")
    ap.add_argument("--structure", help="AlphaFold/PDB model; without it the structural "
                                        "analyses skip and everything else still runs")
    ap.add_argument("--min-reads", type=int, default=0)
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--only", help="comma-separated analysis keys, default: all")
    ap.add_argument("--no-run", action="store_true", help="write the config and stop")
    a = ap.parse_args(argv)

    src = Path(a.fitness).expanduser()
    if not src.exists():
        raise SystemExit(f"{CROSS}{src} not found")
    gene = (a.gene or gene_from_filename(src)).upper()
    print(f"\n== quickstart: {gene}\n")

    cfg_path, _ = build(src, gene, a.structure, a.min_reads)

    if a.gff3:
        g = Path(a.gff3).expanduser()
        if not g.exists():
            raise SystemExit(f"{CROSS}{g} not found")
        from .topology import main as tmhmm_main
        print()
        tmhmm_main([str(cfg_path), "--from-file", str(g)])
    else:
        say(WARN, "no --gff3: the topology is empty, so every TM-aware analysis will skip")

    if a.no_run:
        print(f"\n{TICK}config ready. To run:\n"
              f"        python -m mpdms run {cfg_path} --style {a.style}\n")
        return

    from .runner import main as run_main
    print()
    argv2 = [str(cfg_path), "--style", a.style, "--no-cross"]
    if a.only:
        argv2 += ["--only", a.only]
    run_main(argv2)

    from .report_cli import main as report_main
    print()
    report_main([str(cfg_path)])
    summarise(gene, a.style, bool(a.structure))


if __name__ == "__main__":
    sys.exit(main())
