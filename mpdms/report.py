"""Assemble reports/<dataset_id>.md from the analysis results."""
from __future__ import annotations

import datetime as dt
import os

from jinja2 import Environment, FileSystemLoader

from .config import REPO_ROOT
from .plotting import git_commit


def render_dataset_report(cfg, df, results):
    env = Environment(loader=FileSystemLoader(REPO_ROOT / "templates"), trim_blocks=False, lstrip_blocks=True)
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    rel = []
    for r in results:
        r = dict(r)
        r["figures"] = [os.path.relpath(f, reports) for f in r.get("figures", [])]
        rel.append(r)
    text = env.get_template("dataset_report.md.j2").render(
        cfg=cfg, meta=df.attrs, results=rel, commit=git_commit(), date=dt.date.today().isoformat(),
        n_tm=sum(s.get("type") == "TM" for s in cfg.get_path("topology.segments", []) or []))
    (reports / f"{cfg.id}.md").write_text(text)
