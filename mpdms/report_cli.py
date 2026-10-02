"""`python -m mpdms report configs/AQR1.yaml configs/QDR2.yaml`

Per-protein report: fitness distributions, replicate QC, the DMS heatmap with SSDraw tracks
(mean missense, proline, lysine/arginine) and a solvent-accessibility strip, and the
per-TM-helix comparison of K/R against proline.

Transmembrane regions come from whatever is in the config's `topology:` block, so run
`python -m mpdms tmhmm <configs> --from-file <TMRs.gff3>` first if you want the TMHMM or
DeepTMHMM definition rather than the one `init` guessed.

Writes outputs/<ID>/report/<ID>_report.pdf (figures concatenated, in order) and
reports/<ID>_report.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import warnings
from pathlib import Path

from .analyses.base import jsonable
from .config import REPO_ROOT, load_config, validate as validate_cfg
from .io import load_dataset
from .plotting import git_commit

STEPS = [("qc", "qc"), ("heatmap", "heatmap_structure"), ("a18", "a18_helix_kr_vs_pro")]
FIGURE_ORDER = [
    ("qc_fitness_distributions", "Fitness distributions by variant class (before and after normalisation)"),
    ("qc_replicate_correlation", "Replicate reproducibility"),
    ("qc_read_threshold", "Read-count filter"),
    ("heatmap_ssdraw", "DMS heatmap with SSDraw tracks (mean, proline, Lys/Arg) and RSA strip"),
    ("ssdraw_mean_effect", "SSDraw: mean missense effect"),
    ("ssdraw_proline", "SSDraw: proline"),
    ("ssdraw_lys_arg", "SSDraw: lysine / arginine"),
    ("a18_helix_kr_vs_pro", "Lys/Arg vs proline in each TM helix"),
    ("a18b_rsa_vs_kr_pro", "Solvent accessibility vs Lys/Arg and vs proline effect (TM helices)"),
    ("a19_literature", "Published functional residues against this screen"),
    ("a20_variant_panel", "Replicate fitness of named variants vs synonymous wild type"),
    ("a20b_missense_percentile", "Named variants placed in the missense fitness distribution"),
    ("a21_site_substitutions", "All substitutions at the named sites vs ESM-1v"),
    ("a21b_local_regression", "Named sites with the regression refit on those substitutions only"),
]


def merge_pdfs(paths, out: Path) -> bool:
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf          # older name
        except ImportError:
            return False
    doc = pymupdf.open()
    for p in paths:
        with pymupdf.open(p) as src:
            doc.insert_pdf(src)
    doc.save(out)
    doc.close()
    return True


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|singular|boundary|No artists|invalid value).*")
    ap = argparse.ArgumentParser(prog="python -m mpdms report")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--skip-run", action="store_true",
                    help="reuse figures already in outputs/<ID>/figures instead of recomputing")
    a = ap.parse_args(argv)

    import importlib
    for c in a.configs:
        cfg = load_config(Path(c))
        validate_cfg(cfg)
        df = load_dataset(cfg)
        outdir = REPO_ROOT / "outputs" / cfg.id
        results = {}
        tms = [s for s in (cfg.get_path("topology.segments") or []) if s.get("type") == "TM"]
        print(f"{cfg.id}: {len(df):,} variants, {len(tms)} TM helices "
              f"({cfg.get_path('topology.source', '?')})", flush=True)
        if not a.skip_run:
            for key, mod in STEPS:
                m = importlib.import_module(f"mpdms.analyses.{mod}")
                try:
                    r = m.run(df.copy(), cfg, outdir)
                except Exception as e:
                    r = {"status": "error", "headline": f"{e.__class__.__name__}: {e}", "caveats": []}
                results[key] = r
                print(f"  {key:<8} {r.get('status'):<8} {r.get('headline', '')[:100]}", flush=True)

        rep = outdir / "report"
        rep.mkdir(parents=True, exist_ok=True)
        figs = [(outdir / "figures" / f"{n}.pdf", t) for n, t in FIGURE_ORDER]
        have = [(p, t) for p, t in figs if p.exists()]
        pdf = rep / f"{cfg.id}_report.pdf"
        merged = merge_pdfs([p for p, _ in have], pdf)

        md = [f"# {cfg.display_name} — report", "",
              f"*{dt.date.today().isoformat()} · git `{git_commit()}` · "
              f"{len(df):,} variants · {len(tms)} TM helices from {cfg.get_path('topology.source', '?')}*", ""]
        if results:
            md += ["## Headlines", ""]
            for key, r in results.items():
                md.append(f"- **{key}** — {r.get('headline', '')}")
            md.append("")
            cav = [(k, c) for k, r in results.items() for c in r.get("caveats", [])]
            if cav:
                md += ["## Caveats", ""] + [f"- `{k}` {c}" for k, c in cav] + [""]
        md += ["## Figures", ""]
        for p, t in have:
            md.append(f"### {t}")
            md.append("")
            md.append(f"![{t}]({Path('..') / p.relative_to(REPO_ROOT)})".replace("\\", "/"))
            md.append("")
        missing = [t for p, t in figs if not p.exists()]
        if missing:
            md += ["## Not produced", ""] + [f"- {t}" for t in missing] + [""]
        (REPO_ROOT / "reports").mkdir(exist_ok=True)
        (REPO_ROOT / "reports" / f"{cfg.id}_report.md").write_text("\n".join(md))
        (rep / "summary.json").write_text(json.dumps(jsonable(results), indent=2))
        print(f"  {len(have)}/{len(figs)} figures; "
              + (f"PDF -> {pdf.relative_to(REPO_ROOT)}" if merged else "PDF merge needs `pip install pymupdf`")
              + f"; markdown -> reports/{cfg.id}_report.md", flush=True)


if __name__ == "__main__":
    main()
