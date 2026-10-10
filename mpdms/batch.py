"""`python -m mpdms batch <folder> [--esm-table t.csv] [--require-structure]`

A folder of DiMSum outputs straight from the pipeline, e.g.

    fitness_singles_aqr1_003.txt  aqr1.gff3  aqr1.cif
    fitness_singles_hxt2.txt      hxt2.gff3  hxt2.cif

becomes one full report per protein. Each fitness table is paired with the gff3 and the
structure whose names match it, allowing for a run or replicate suffix on the fitness file
(aqr1_003 pairs with aqr1.gff3), so the gene keeps its real name and the literature sites
in configs/_literature.yaml still find it.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .quickstart import CROSS, TICK, WARN, gene_from_filename, say

FITNESS = ("*fitness*.txt", "*fitness*.tsv", "*fitness*.csv")
GFF3 = (".gff3", ".gff")
STRUCTURE = (".cif", ".mmcif", ".pdb", ".ent")


def name_candidates(gene: str) -> list[str]:
    """AQR1_003 -> [aqr1_003, aqr1]: the fitness file may carry a run or replicate suffix
    that the gff3 and the structure do not."""
    out = [gene.lower()]
    trimmed = gene.lower()
    while True:
        nxt = re.sub(r"[_-](?:\d+|r\d+|rep\d+|v\d+)$", "", trimmed)
        if nxt == trimmed:
            break
        trimmed = nxt
        out.append(trimmed)
    return out


def find_companion(folder: Path, gene: str, suffixes: tuple[str, ...]) -> Path | None:
    """The gff3 or structure belonging to this protein, by name then by prefix."""
    by_stem: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in suffixes:
            by_stem.setdefault(p.stem.lower(), p)
    for cand in name_candidates(gene):
        if cand in by_stem:
            return by_stem[cand]
    for cand in name_candidates(gene):             # aqr1.deeptmhmm.gff3, hxt2_AF.cif
        hit = [p for s, p in by_stem.items() if s.startswith(cand)]
        if len(hit) == 1:
            return hit[0]
    return None


def discover(folder: Path) -> list[dict]:
    """One entry per fitness table found, with whatever companions it has."""
    seen: dict[Path, None] = {}
    for pat in FITNESS:
        for p in sorted(folder.glob(pat)):
            seen.setdefault(p, None)
    out = []
    for p in seen:
        gene = gene_from_filename(p)
        out.append({"fitness": p, "gene": gene,
                    "ident": gene, "gff3": find_companion(folder, gene, GFF3),
                    "structure": find_companion(folder, gene, STRUCTURE)})
    # the gene is the trimmed name only when that is what the companions are called
    for e in out:
        for cand in name_candidates(e["gene"])[1:]:
            if (e["gff3"] and e["gff3"].stem.lower() == cand) or \
               (e["structure"] and e["structure"].stem.lower() == cand):
                e["gene"] = cand.upper()
                break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m mpdms batch",
        description="Run the full quickstart on every protein in a folder of DiMSum output.")
    ap.add_argument("folder", help="folder holding the fitness tables, gff3s and structures")
    ap.add_argument("--esm-table", help="bulk ESM-1v CSV, imported for each protein in turn")
    ap.add_argument("--require-structure", action="store_true",
                    help="skip proteins with no structure instead of running without it")
    ap.add_argument("--require-gff3", action="store_true", default=True)
    ap.add_argument("--no-require-gff3", dest="require_gff3", action="store_false")
    ap.add_argument("--lower-anchor", default="nonsense")
    ap.add_argument("--missense-only", action="store_true")
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--only", help="comma-separated analysis keys")
    ap.add_argument("--min-reads", type=int, default=0)
    ap.add_argument("--suffix", default="", help="appended to each run id, so a second "
                                                 "setting does not overwrite the first")
    ap.add_argument("--dry-run", action="store_true", help="show the pairing and stop")
    a = ap.parse_args(argv)

    folder = Path(a.folder).expanduser()
    if not folder.is_dir():
        raise SystemExit(f"{CROSS}{folder} is not a folder")
    found = discover(folder)
    if not found:
        raise SystemExit(f"{CROSS}no fitness table in {folder} (looked for {', '.join(FITNESS)})")

    runnable, skipped = [], []
    for e in found:
        missing = [n for n, need in (("gff3", a.require_gff3),
                                     ("structure", a.require_structure)) if need and not e[n]]
        (skipped if missing else runnable).append((e, missing))

    print(f"\n== batch: {len(found)} fitness tables in {folder}\n")
    for e, missing in runnable:
        say(TICK, f"{e['gene']:<10} {e['fitness'].name}"
                  f"  gff3={e['gff3'].name if e['gff3'] else '-'}"
                  f"  structure={e['structure'].name if e['structure'] else '-'}")
    for e, missing in skipped:
        say(WARN, f"{e['gene']:<10} skipped: no {' or '.join(missing)}")
    if a.dry_run or not runnable:
        return 0 if runnable else 1

    from .quickstart import main as quick
    failed = []
    for e, _ in runnable:
        argv_q = [str(e["fitness"]), "--gene", e["gene"],
                  "--id", e["gene"] + a.suffix, "--style", a.style,
                  "--min-reads", str(a.min_reads), "--lower-anchor", a.lower_anchor]
        if e["gff3"]:
            argv_q += ["--gff3", str(e["gff3"])]
        if e["structure"]:
            argv_q += ["--structure", str(e["structure"])]
        if a.esm_table:
            argv_q += ["--esm-table", str(Path(a.esm_table).expanduser())]
        if a.missense_only:
            argv_q += ["--missense-only"]
        if a.only:
            argv_q += ["--only", a.only]
        try:
            quick(argv_q)
        except SystemExit as exc:                 # one bad protein must not lose the other four
            failed.append((e["gene"], str(exc)))
            say(CROSS, f"{e['gene']}: {exc}")
        except Exception as exc:
            failed.append((e["gene"], f"{type(exc).__name__}: {exc}"))
            say(CROSS, f"{e['gene']}: {type(exc).__name__}: {exc}")

    done = [e["gene"] + a.suffix for e, _ in runnable if e["gene"] not in dict(failed)]
    print(f"\n== batch done: {len(done)} of {len(runnable)} proteins")
    for g, why in failed:
        say(CROSS, f"{g}: {why[:120]}")
    if done:
        say(TICK, "compare them: python -m mpdms family "
                  + " ".join(f"configs/{g}.yaml" for g in done))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
