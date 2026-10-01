"""`python -m mpdms validate configs/*.yaml`   - is the assay measuring biogenesis, and does it generalise?
`python -m mpdms bench    configs/*.yaml`   - how well do models predict these measurements?

validate produces two kinds of evidence:
  1. the A16 pre-registered battery, per protein and as a protein x test matrix;
  2. leave-one-protein-out transfer: a model using only generalisable biophysical features
     (topology, helix geometry, burial, depth, substitution chemistry) is trained on all
     but one protein and asked to predict the one it has never seen. Transfer that holds
     up across held-out proteins is the evidence that the determinants measured here are
     properties of membrane-protein biogenesis rather than of these particular proteins.
     The within-protein position split is shown alongside as the practical ceiling.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import benchmark as B
from . import plotting as P
from .analyses.a16_assay_validation import KIND_COLOR, battery, figure as battery_figure
from .analyses.base import jsonable
from .config import REPO_ROOT, load_config, validate as validate_cfg
from .features import build, build_external
from .io import load_dataset


def _load(paths):
    out = []
    for c in paths:
        cfg = load_config(Path(c))
        validate_cfg(cfg)
        out.append((load_dataset(cfg), cfg))
        print(f"loaded {cfg.id}", flush=True)
    return out


def battery_matrix_figure(tabs: dict[str, pd.DataFrame]):
    """Protein x test pass/fail matrix - the single slide that summarises assay validity."""
    ids = sorted(tabs)
    tests = tabs[ids[0]][["id", "claim", "kind"]]
    M = np.full((len(tests), len(ids)), np.nan)
    for j, g in enumerate(ids):
        t = tabs[g].set_index("id")
        for i, tid in enumerate(tests.id):
            if tid in t.index and t.loc[tid, "testable"]:
                M[i, j] = 1.0 if t.loc[tid, "passed"] else 0.0
    fig, ax = plt.subplots(figsize=(1.0 * len(ids) + 6.2, 0.33 * len(tests) + 1.4))
    from matplotlib.colors import ListedColormap
    ax.imshow(np.ma.masked_invalid(M), cmap=ListedColormap(["#E8A9A4", "#9ED4BC"]), vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    for i in range(len(tests)):
        for j in range(len(ids)):
            v = M[i, j]
            if not np.isfinite(v):
                ax.plot(j, i, marker="_", ms=7, color=P.MUTED, mew=1.4)
            else:
                ax.plot(j, i, marker="o" if v else "X", ms=6.5 if v else 7,
                        color=P.INK, mfc=P.INK if v else "white", mew=1.4)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(tests)))
    ax.set_yticklabels([f"{r.id}  {r.claim}" for r in tests.itertuples()], fontsize=6.5)
    for i, k in enumerate(tests.kind):
        ax.get_yticklabels()[i].set_color(KIND_COLOR.get(k, P.INK))
    ax.set_xticks(np.arange(-.5, len(ids)), minor=True)
    ax.set_yticks(np.arange(-.5, len(tests)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    n = np.nansum(M), np.isfinite(M).sum()
    ax.set_title(f"Assay validation battery — {int(n[0])}/{int(n[1])} predictions met across {len(ids)} proteins"
                 "\n(filled circle = met, cross = not met, dash = not testable; "
                 "test name colour: grey sanity, blue mechanism, green independent)",
                 loc="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    return fig


def transfer_figure(res: pd.DataFrame):
    """Leave-one-protein-out vs within-protein Spearman, per model."""
    r = res[res.spearman.notna()]
    models = list(r.groupby("model").spearman.mean().sort_values(ascending=False).index)
    splits = [s for s in ("position", "protein") if s in set(r.split)]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.6 * len(splits), 3.2), squeeze=False, sharey=True)
    cols = ["#2F6DB5", "#1B9E77", "#E69F00", "#9467BD", "#6B6B6B", "#D62728"]
    for ax, sp in zip(axes[0], splits):
        d = r[r.split == sp]
        for k, m in enumerate(models):
            v = d[d.model == m]
            if not len(v):
                continue
            ax.scatter(np.full(len(v), k) + np.linspace(-.16, .16, len(v)), v.spearman,
                       s=22, color=cols[k % len(cols)], edgecolor=P.INK, lw=0.3, zorder=3)
            ax.plot([k - .28, k + .28], [v.spearman.mean()] * 2, color=P.INK, lw=1.6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=6.5)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.set_title({"protein": "Leave-one-protein-out (extrapolation)",
                      "position": "Held-out positions, same protein (ceiling)"}[sp], loc="left", fontsize=8.5)
    axes[0, 0].set_ylabel("Spearman ρ (predicted vs measured)")
    fig.suptitle("Do the measured determinants generalise to an unseen membrane protein?",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|singular|boundary|positive definite|No artists).*")
    ap = argparse.ArgumentParser(prog="python -m mpdms validate")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)
    loaded = _load(a.configs)
    out = Path(a.outdir) if a.outdir else REPO_ROOT / "outputs" / "_validation"
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    tabs = {}
    for df, cfg in loaded:
        print(f"  battery: {cfg.id} ...", flush=True)
        t = battery(df, cfg)
        tabs[cfg.id] = t
        P.save(battery_figure(t, cfg.display_name), figdir, f"a16_{cfg.id}", cfg, "A16")
        print(f"    {int(t.passed.sum())}/{int(t.testable.sum())} predictions met", flush=True)
    allb = pd.concat([t.assign(dataset_id=g) for g, t in tabs.items()], ignore_index=True)
    allb.to_csv(tabdir / "validation_battery.csv", index=False)
    if len(tabs) > 1:
        P.save(battery_matrix_figure(tabs), figdir, "validation_matrix", None, "A16")

    print("  building features ...", flush=True)
    feats = build(loaded)
    feats.to_parquet(tabdir / "variant_features.parquet", index=False)
    models = B.default_models(feats)
    res = []
    for sp in ("protein", "position"):
        print(f"  transfer: {sp} split ...", flush=True)
        res.append(B.evaluate(feats, models, split=sp))
    res = pd.concat(res, ignore_index=True)
    res.to_csv(tabdir / "transfer_results.csv", index=False)
    B.summarise(res).to_csv(tabdir / "transfer_summary.csv", index=False)
    P.save(transfer_figure(res), figdir, "transfer_generalisation", None, "LOPO")

    lopo = res[(res.split == "protein") & res.spearman.notna()]
    best = lopo.groupby("model").spearman.mean().sort_values(ascending=False)
    summary = {
        "proteins": sorted(feats.dataset_id.unique()),
        "n_variants": int(len(feats)),
        "battery": {g: {"passed": int(t.passed.sum()), "testable": int(t.testable.sum())} for g, t in tabs.items()},
        "lopo_spearman_by_model": best.round(3).to_dict(),
        "lopo_per_protein": lopo.pivot_table(index="fold", columns="model", values="spearman").round(3).to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    print(f"\nbattery: {int(allb.passed.sum())}/{int(allb.testable.sum())} across {len(tabs)} proteins")
    if len(best):
        print("leave-one-protein-out Spearman (mean over held-out proteins):")
        for m, v in best.items():
            print(f"  {m:<28} {v:+.3f}")
    print(f"outputs -> {out.relative_to(REPO_ROOT)}")


def bench_main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|singular|boundary|No artists).*")
    ap = argparse.ArgumentParser(prog="python -m mpdms bench")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--split", default="protein", choices=["protein", "position", "random", "segment"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--scores", action="append", default=[],
                    help="NAME=path.csv with columns uniprot|gene,pos,mut,score (repeatable)")
    ap.add_argument("--scores-sign", default="auto", help="+1, -1 or auto (orient on the training fold)")
    ap.add_argument("--embeddings", help=".npz of per-variant embeddings keyed 'DATASET:POS:MUT'")
    ap.add_argument("--external", action="append", default=[],
                    help="held-out DMS as CONFIG=CSV, e.g. configs/CFTR.yaml=cftr_dms.csv; the CSV "
                         "needs pos,wt,mut,score (repeatable)")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)

    feats = build(_load(a.configs))
    models = B.default_models(feats)
    sign = a.scores_sign if a.scores_sign == "auto" else float(a.scores_sign)
    for spec in a.scores:
        name, _, path = spec.partition("=")
        feats = B.load_score_csv(Path(path), feats, name)
        models.append(B.ScoreColumn(name, sign=sign))
        print(f"loaded scores {name}: {feats[name].notna().mean():.0%} of variants covered")
    if a.embeddings:
        z = np.load(a.embeddings)
        emb = {}
        for k in z.files:
            ds, pos, mut = k.split(":")
            emb[(ds, int(pos), mut)] = z[k]
        models.append(B.EmbeddingHead(emb))
        print(f"loaded {len(emb):,} embeddings (dim {len(next(iter(emb.values())))})")

    out = Path(a.outdir) if a.outdir else REPO_ROOT / "outputs" / "_benchmark"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    res = B.evaluate(feats, models, split=a.split, n_folds=a.folds)
    res.to_csv(out / "tables" / f"bench_{a.split}.csv", index=False)
    summ = B.summarise(res)
    summ.to_csv(out / "tables" / f"bench_{a.split}_summary.csv", index=False)
    print(f"\n{a.split} split:")
    print(summ[["model", "spearman_mean", "spearman_std"]
               + [c for c in ("auroc_deleterious_mean",) if c in summ]].to_string(index=False))

    for ext in a.external:  # transfer to DMS the models never saw
        cfg_path, _, csv = ext.partition("=")
        if not csv:
            print(f"skip {ext}: expected CONFIG=CSV")
            continue
        name = Path(csv).stem
        try:
            ecfg = load_config(Path(cfg_path))
            e = build_external(csv, ecfg)
        except Exception as exc:
            print(f"skip {ext}: {exc.__class__.__name__}: {exc}")
            continue
        rows = []
        for mk in models:
            m = mk() if callable(mk) and not hasattr(mk, "predict") else mk
            try:
                if hasattr(m, "fit"):
                    m.fit(feats)
                r = B.metrics(e.score_z, m.predict(e))
                r.update(model=m.name, external=name, protein=ecfg.id)
            except Exception as exc:
                r = {"model": getattr(m, "name", "?"), "external": name,
                     "error": f"{exc.__class__.__name__}: {exc}"}
            rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(out / "tables" / f"external_{name}.csv", index=False)
        print(f"\nexternal {name} ({ecfg.id}, n={len(e):,}) - models trained on all yeast data:")
        print(t[[c for c in ("model", "n", "spearman", "auroc_deleterious", "error") if c in t]].to_string(index=False))
    print(f"\noutputs -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
