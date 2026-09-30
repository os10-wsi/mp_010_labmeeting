"""Cross-dataset summary: outputs/_cross/{stats.json, summary.csv, summary.pdf}."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P
from .config import REPO_ROOT

# (column label, analysis, metric key, format, higher_is_better)
METRICS = [
    ("pass variants", "qc", "n_pass", "{:,.0f}", None),
    ("syn→stop AUC", "a01", "auc_stop_below_syn", "{:.2f}", True),
    ("syn SD", "a01", "syn_sd", "{:.2f}", False),
    ("deleterious frac", "a01", "deleterious_frac", "{:.0%}", None),
    ("coverage", "a02", "coverage_missense_postfilter", "{:.0%}", True),
    ("replicate r", "a03", "mean_pearson", "{:.2f}", True),
    ("ICC(2,1)", "a03", "icc_2_1", "{:.2f}", True),
    ("TM−nonTM Δ", "a04", "tm_minus_nontm_mean_of_position_medians", "{:+.2f}", None),
    ("TM vs loop d", "a04", "tm_vs_loop_cohens_d", "{:+.2f}", None),
    ("ΔR² bio−KD (TM)", "a05", "delta_R2_biological_vs_KD", "{:+.3f}", None),
    ("periodic TMDs", "a07", "n_periodic", "{:.0f}", None),
    ("Pro penalty TM", "a09", "P_penalty_TM", "{:+.2f}", None),
    ("orthogonal R² (lin)", "a10", "r2_linear_cv", "{:.2f}", None),
]


def build_cross():
    rows = []
    for p in sorted((REPO_ROOT / "outputs").glob("*/stats.json")):
        if p.parent.name.startswith("_"):
            continue
        rows += json.loads(p.read_text())
    out = REPO_ROOT / "outputs" / "_cross"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(rows, indent=2))
    if not rows:
        return
    by = {(r["dataset_id"], r["analysis"]): r for r in rows}
    datasets = sorted({r["dataset_id"] for r in rows})
    table, flags = [], []
    for d in datasets:
        rec, fl = {"dataset_id": d}, {}
        for label, a, key, _, _ in METRICS:
            r = by.get((d, a), {})
            rec[label] = r.get("metrics", {}).get(key)
            fl[label] = bool(r.get("caveats")) or r.get("status") not in ("ok", None)
        table.append(rec)
        flags.append(fl)
    df = pd.DataFrame(table).set_index("dataset_id")
    fdf = pd.DataFrame(flags, index=df.index)
    df.join(fdf.add_suffix(" [caveat]")).to_csv(out / "summary.csv")

    # slide: table with caveat flags
    fig, ax = plt.subplots(figsize=(1.0 + 0.95 * len(METRICS), 0.9 + 0.32 * len(df)))
    ax.axis("off")
    cells = []
    for d in df.index:
        row = []
        for label, _, _, fmt, _ in METRICS:
            v = df.loc[d, label]
            s = "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)
            row.append(s + (" †" if fdf.loc[d, label] else ""))
        cells.append(row)
    tb = ax.table(cellText=cells, rowLabels=list(df.index), colLabels=[m[0] for m in METRICS], loc="center",
                  cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.5)
    tb.scale(1, 1.35)
    for (i, j), c in tb.get_celld().items():
        c.set_linewidth(0.3)
        c.set_edgecolor("#DDDDDD")
        if i == 0:
            c.set_text_props(fontweight="bold")
            c.set_facecolor("#F2F2F2")
        elif j >= 0 and "†" in c.get_text().get_text():
            c.set_facecolor("#FDECEA")
    ax.set_title("Cross-dataset summary († = caveat fired for that analysis)", loc="left", fontsize=9, fontweight="bold")
    P.stamp(fig, None, "cross")
    fig.savefig(out / "summary.pdf", bbox_inches="tight")
    fig.savefig(out / "summary.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
