"""Batch driver: configs -> canonical tables -> analyses -> stats.json, reports, cross summary."""
from __future__ import annotations

import argparse
import importlib
import json
import time
import traceback
import warnings
from pathlib import Path

from .config import REPO_ROOT, load_config, validate

ANALYSES = [
    ("qc", "qc"),
    ("heatmap", "heatmap_structure"),
    ("a01", "a01_dynamic_range"),
    ("a02", "a02_coverage"),
    ("a03", "a03_reproducibility"),
    ("a04", "a04_positional_topology"),
    ("a05", "a05_substitution_physchem"),
    ("a06", "a06_charge_topology"),
    ("a07", "a07_helical_periodicity"),
    ("a08", "a08_depth_dependence"),
    ("a09", "a09_motifs_breakers"),
    ("a10", "a10_orthogonals"),
    ("a11", "a11_nonsense_profile"),
    ("a12", "a12_synonymous_codons"),
    ("a13", "a13_helix_burial"),
    ("a14", "a14_hydrophobic_facing"),
    ("a15", "a15_esm_functional"),
    ("a16", "a16_assay_validation"),
    ("a18", "a18_helix_kr_vs_pro"),
    ("a19", "a19_literature"),
    ("a20", "a20_variant_panel"),
]
OPT_IN = {"a13", "a14"}  # only run when named in --only


def run_dataset(cfg_path: Path, only: set[str] | None = None, strict: bool = True) -> list[dict]:
    from .io import load_dataset
    cfg = load_config(cfg_path)
    validate(cfg, strict=strict)
    df = load_dataset(cfg)
    warns = validate(cfg, df, strict=strict)
    for w in warns:
        print(f"  [{cfg.id}] warning: {w}")
    outdir = REPO_ROOT / "outputs" / cfg.id
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for key, mod in ANALYSES:
        if (only and key not in only) or (key in OPT_IN and not (only and key in only)):
            continue
        t0 = time.time()
        try:
            m = importlib.import_module(f"mpdms.analyses.{mod}")
            res = m.run(df.copy(), cfg, outdir)
        except Exception as e:  # never crash the batch
            res = {"analysis": key, "dataset_id": cfg.id, "status": "error",
                   "reason": f"{e.__class__.__name__}: {e}", "headline": f"error: {e}",
                   "metrics": {}, "caveats": [], "traceback": traceback.format_exc()}
        res["seconds"] = round(time.time() - t0, 1)
        status = res.get("status")
        print(f"  [{cfg.id}] {key:<8} {status:<8} {res['seconds']:>6}s  {res.get('headline', '')[:110]}")
        if status == "error":
            print("    " + res["traceback"].strip().splitlines()[-1])
        results.append(res)
    # merge with results of earlier runs so `--only` does not drop other analyses
    sp = outdir / "stats.json"
    prev = {r["analysis"]: r for r in json.loads(sp.read_text())} if sp.exists() else {}
    prev.update({r["analysis"]: r for r in results})
    order = [k for k, _ in ANALYSES]
    merged = sorted(prev.values(), key=lambda r: order.index(r["analysis"]) if r["analysis"] in order else 99)
    sp.write_text(json.dumps(merged, indent=2))
    from .report import render_dataset_report
    render_dataset_report(cfg, df, merged)
    return merged


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices in sparse segments
    warnings.filterwarnings("ignore", message=".*Glyph.*")
    warnings.filterwarnings("ignore", message=".*(singular|boundary|positive definite|No artists).*")
    ap = argparse.ArgumentParser(prog="run_all.py")
    ap.add_argument("configs", nargs="*", help="config files (default: configs/*.yaml)")
    ap.add_argument("--only", help="comma-separated analysis keys, e.g. qc,heatmap,a01")
    ap.add_argument("--no-strict", action="store_true", help="downgrade config validation errors to warnings")
    ap.add_argument("--no-cross", action="store_true", help="skip the cross-dataset summary")
    a = ap.parse_args(argv)
    paths = [Path(p) for p in a.configs] or sorted(p for p in (REPO_ROOT / "configs").glob("*.yaml")
                                                   if not p.name.startswith("_"))
    only = set(a.only.split(",")) if a.only else None
    allres = []
    for p in paths:
        print(f"== {p.name}")
        try:
            allres += run_dataset(p, only, strict=not a.no_strict)
        except Exception as e:
            print(f"  FAILED to load {p.name}: {e}")
            traceback.print_exc()
    if allres and not a.no_cross:
        from .cross import build_cross
        build_cross()


if __name__ == "__main__":
    main()
