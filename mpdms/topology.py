"""`python -m mpdms tmhmm configs/AQR1.yaml configs/QDR2.yaml`

Define transmembrane regions with TMHMM and write them into the configs, so every
downstream analysis (A04-A08, A13, A14, family) uses the same definition.

TMHMM labels each residue inside (cytoplasmic), membrane, or outside, so unlike the
hydropathy fallback it supplies the *orientation* of each helix directly rather than
inferring it from the positive-inside rule. That matters for A06 and for the family
profiles, where helices of opposite orientation must not be pooled naively.

Sources, tried in this order (or pick one with --source):
  cache     data/external/<GENE>/<ID>.tmhmm written by an earlier run
  file      --from-file: DeepTMHMM .gff3, or TMHMM 2.0 long output
  python    the `tmhmm.py` package (a reimplementation shipping the TMHMM 2.0 model)
  binary    a `tmhmm` executable on PATH (the original DTU distribution)

DeepTMHMM is the current successor and is usually the better choice; run it on the
BioLib server or locally and pass its .gff3 with --from-file.

TMHMM 2.0 and DeepTMHMM are free for academic use under their own licences; this module
only reads their output.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from .config import REPO_ROOT, load_config, protein_sequence

SOLUBLE_MIN = 60   # a non-TM stretch at least this long is a soluble domain, not a loop
LABELS = {"tmhelix": "M", "transmembrane": "M", "membrane": "M",
          "inside": "i", "cytoplasmic": "i", "outside": "o",
          "periplasm": "o", "extracellular": "o", "lumenal": "o", "signal": "S", "beta sheet": "B"}


# ----------------------------------------------------------------- sources
def from_python(seq: str, name: str) -> str:
    import tmhmm
    model = Path(tmhmm.__file__).parent / "TMHMM2.0.model"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = tmhmm.predict(seq, name, str(model), compute_posterior=False)
    ann = r[0] if isinstance(r, tuple) else r      # tuple only when the posterior is requested
    return "".join(ann)


def from_binary(seq: str, name: str, exe: str = "tmhmm") -> str:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        fa = Path(d) / "q.fa"
        fa.write_text(f">{name}\n{seq}\n")
        r = subprocess.run([exe, str(fa)], capture_output=True, text=True, check=True, cwd=d)
    return parse_table(r.stdout, len(seq))


def parse_table(text: str, length: int) -> str:
    """DeepTMHMM .gff3 or TMHMM 2.0 long output -> per-residue i/M/o string."""
    ann = ["i"] * length
    found = False
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t") if "\t" in line else line.split()
        if len(f) < 3:
            continue
        label, nums = None, []
        for tok in f[1:]:
            t = tok.strip().lower()
            if t in LABELS:
                label = LABELS[t]
            elif re.fullmatch(r"\d+", tok.strip()):
                nums.append(int(tok))
        if label is None and len(f) >= 4:
            label = LABELS.get(" ".join(f[1:-2]).strip().lower())
        if label is None or len(nums) < 2:
            continue
        a, b = nums[-2], nums[-1]
        for i in range(max(a, 1), min(b, length) + 1):
            ann[i - 1] = label
        found = True
    if not found:
        raise ValueError("no TMHMM/DeepTMHMM records recognised in that file")
    return "".join(ann)


def annotation(seq: str, name: str, source: str = "auto", from_file: Path | None = None,
               cache: Path | None = None) -> tuple[str, str]:
    """Return (per-residue i/M/o string, which source was used)."""
    if source in ("auto", "cache") and cache and cache.exists():
        a = cache.read_text().strip().splitlines()[-1].strip()
        if len(a) == len(seq):
            return a, "cache"
    if from_file:
        return parse_table(Path(from_file).read_text(), len(seq)), f"file:{Path(from_file).name}"
    if source in ("auto", "python"):
        try:
            return from_python(seq, name), "tmhmm.py"
        except ImportError:
            if source == "python":
                raise
    if source in ("auto", "binary") and shutil.which("tmhmm"):
        return from_binary(seq, name), "tmhmm binary"
    raise RuntimeError(
        "no TMHMM source available. Either install the python package:\n"
        "    pip install 'cython<3' && pip install --no-build-isolation tmhmm.py\n"
        "  (if it fails to build, see scripts/install_tmhmm.sh)\n"
        "or run DeepTMHMM (https://dtu.biolib.com/DeepTMHMM) and pass its .gff3 with --from-file")


# ------------------------------------------------------- annotation -> segments
def segments_from_annotation(ann: str) -> tuple[list[dict], str]:
    """i/M/o string -> the pipeline's topology segment list, with orientations."""
    runs = []
    i = 0
    while i < len(ann):
        j = i
        while j + 1 < len(ann) and ann[j + 1] == ann[i]:
            j += 1
        runs.append((ann[i], i + 1, j + 1))
        i = j + 1
    side = {"i": "cytosolic", "o": "lumenal"}
    n_term = side.get(next((c for c, _, _ in runs if c in side), "i"), "cytosolic")
    segs, ntm, nloop = [], 0, 0
    for k, (c, a, b) in enumerate(runs):
        if c == "M":
            ntm += 1
            prev = next((runs[m][0] for m in range(k - 1, -1, -1) if runs[m][0] in side), None)
            nxt = next((runs[m][0] for m in range(k + 1, len(runs)) if runs[m][0] in side), None)
            if prev in side:
                orient = "in_out" if prev == "i" else "out_in"
            elif nxt in side:
                orient = "in_out" if nxt == "o" else "out_in"
            else:
                orient = "in_out"
            segs.append({"name": f"TM{ntm}", "start": a, "end": b, "type": "TM", "orientation": orient})
        elif c == "S":
            segs.append({"name": "signal", "start": a, "end": b, "type": "signal", "side": n_term})
        else:
            nloop += 1
            first, last = k == 0, k == len(runs) - 1
            name = "Nterm" if first else ("Cterm" if last else f"L{nloop - (1 if not first else 0)}")
            segs.append({"name": name, "start": a, "end": b,
                         "type": "soluble" if (b - a + 1) >= SOLUBLE_MIN else "loop",
                         "side": side.get(c, "cytosolic")})
    return segs, n_term


def write_topology(cfg_path: Path, segs: list[dict], n_term: str, source: str) -> None:
    """Replace the topology block in the config YAML, keeping the rest of the file."""
    import yaml
    from .init_dataset import _Dumper
    txt = Path(cfg_path).read_text()
    block = yaml.dump({"topology": {"n_terminus": n_term,
                                    "source": f"TMHMM ({source})",
                                    "segments": segs}},
                      Dumper=_Dumper, sort_keys=False, width=110)
    m = re.search(r"^topology:\n(?:[ \t].*\n|\n)*", txt, flags=re.M)
    txt = (txt[:m.start()] + block + txt[m.end():]) if m else txt.rstrip() + "\n" + block
    Path(cfg_path).write_text(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mpdms tmhmm")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--source", default="auto", choices=["auto", "cache", "python", "binary"])
    ap.add_argument("--from-file", action="append", default=[],
                    help="DeepTMHMM .gff3 / TMHMM long output; repeat in the same order as the configs")
    ap.add_argument("--dry-run", action="store_true", help="report the prediction without editing the config")
    a = ap.parse_args(argv)
    for k, c in enumerate(a.configs):
        cfg = load_config(Path(c))
        seq = protein_sequence(cfg)
        if not seq:
            print(f"{cfg.id}: no sequence in the config - run `mpdms init` first")
            continue
        gene = str(cfg.get_path("protein.gene") or cfg.id).upper()
        cache = REPO_ROOT / "data" / "external" / gene / f"{cfg.id}.tmhmm"
        ff = Path(a.from_file[k]) if k < len(a.from_file) else None
        try:
            ann, src = annotation(seq, cfg.id, a.source, ff, cache)
        except Exception as e:
            print(f"{cfg.id}: {e}")
            continue
        segs, n_term = segments_from_annotation(ann)
        tms = [s for s in segs if s["type"] == "TM"]
        old = [s for s in (cfg.get_path("topology.segments") or []) if s.get("type") == "TM"]
        print(f"{cfg.id}: {len(tms)} TM helices from {src}; N-terminus {n_term}"
              + (f"  (config previously had {len(old)})" if old else ""))
        for s in tms:
            print(f"    {s['name']:<5} {s['start']:>4}-{s['end']:<4} "
                  f"{s['end'] - s['start'] + 1:>2} aa  {s['orientation']}")
        if old and len(old) != len(tms):
            print(f"    ! TM count differs from the previous annotation ({len(old)} -> {len(tms)}) "
                  "- check before trusting the topology-dependent analyses")
        if a.dry_run:
            continue
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(f"# TMHMM annotation for {cfg.id} ({src})\n{ann}\n")
        write_topology(Path(c), segs, n_term, src)
        print(f"    -> topology written to {c}; annotation cached at "
              f"{cache.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
