"""`python -m mpdms helix configs/TPO3.yaml configs/FEN2.yaml configs/QDR2.yaml`

Runs A13 and A14 per protein and pooled across the given proteins (protein as a stratum:
helix-class permutations are within protein, models include a protein term).
Output: outputs/_helix_<IDS>/ and outputs/<ID>/ for each protein.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from .analyses import a13_helix_burial as a13
from .analyses import a14_hydrophobic_facing as a14
from .analyses.base import jsonable
from .config import REPO_ROOT, load_config, validate
from .helixgeom import tm_residue_table, variant_table
from .io import load_dataset


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*Glyph.*")
    warnings.filterwarnings("ignore", module="statsmodels")
    warnings.filterwarnings("ignore", message=".*(singular|boundary|positive definite|No artists).*")
    ap = argparse.ArgumentParser(prog="python -m mpdms helix")
    ap.add_argument("configs", nargs="+")
    a = ap.parse_args(argv)
    loaded = []
    for c in a.configs:
        cfg = load_config(Path(c))
        validate(cfg)
        df = load_dataset(cfg)
        loaded.append((df, cfg))
        print(f"loaded {cfg.id}")
    ids = [cfg.id for _, cfg in loaded]
    pooled_dir = REPO_ROOT / "outputs" / ("_helix_" + "_".join(ids))
    tables, variants, summary = [], [], {}
    for df, cfg in loaded:  # per protein
        t = tm_residue_table(df, cfg)
        tables.append(t)
        variants.append(variant_table(df, t))
        out = REPO_ROOT / "outputs" / cfg.id
        r13 = a13.analyse([t], cfg.display_name, out, cfg)
        a13.sensitivity([(df, cfg)], out)
        r14 = a14.analyse([t], [variants[-1]], cfg.display_name, out, cfg)
        summary[cfg.id] = {"a13": a13.headline(r13), "a14": a14.headline(r14)}
        print(f"  {cfg.id}: {summary[cfg.id]['a13']}\n  {'':{len(cfg.id)}}  {summary[cfg.id]['a14']}")
    if len(loaded) > 1:  # pooled
        title = " + ".join(ids)
        r13 = a13.analyse(tables, title, pooled_dir)
        a13.sensitivity(loaded, pooled_dir)
        r14 = a14.analyse(tables, variants, title, pooled_dir)
        summary["pooled"] = {"a13": a13.headline(r13), "a14": a14.headline(r14)}
        print(f"  pooled: {summary['pooled']['a13']}\n          {summary['pooled']['a14']}")
        (pooled_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
        print(f"pooled outputs -> {pooled_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
