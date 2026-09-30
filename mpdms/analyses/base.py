"""Helpers shared by analysis modules (module contract plumbing)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def dirs(outdir: Path) -> tuple[Path, Path]:
    f, t = Path(outdir) / "figures", Path(outdir) / "tables"
    f.mkdir(parents=True, exist_ok=True)
    t.mkdir(parents=True, exist_ok=True)
    return f, t


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else round(v, 6)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, pd.Timestamp):
        return str(x)
    return x


def result(analysis: str, cfg, headline: str, metrics: dict, caveats: list | None = None,
           figures: list | None = None, tables: list | None = None, **extra) -> dict:
    return jsonable({
        "analysis": analysis, "dataset_id": cfg.id, "status": "ok", "headline": headline,
        "metrics": metrics, "caveats": caveats or [],
        "figures": [str(p) for p in (figures or []) if str(p).endswith(".png")],
        "tables": [str(p) for p in (tables or [])], **extra,
    })


def skipped(analysis: str, cfg, reason: str) -> dict:
    return {"analysis": analysis, "dataset_id": cfg.id, "status": "skipped", "reason": reason,
            "headline": f"skipped: {reason}", "metrics": {}, "caveats": []}


def missense(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pass_filter"] & (df["vclass"] == "missense")]


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "p = NA"
    return f"p = {p:.1e}" if p < 1e-3 else f"p = {p:.3f}"
