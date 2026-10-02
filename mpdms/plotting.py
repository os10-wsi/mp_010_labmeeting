"""Shared theme, colormaps and figure helpers so every slide looks the same."""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .config import REPO_ROOT  # noqa: E402

# variant classes: missense blue, synonymous green, stop red (checked for deutan/protan separation)
CLASS_COLORS = {"missense": "#2F6DB5", "synonymous": "#1B9E77", "nonsense": "#D62728"}
CLASS_LABELS = {"missense": "Missense", "synonymous": "Synonymous", "nonsense": "Stop"}
CLASS_ORDER = ["missense", "synonymous", "nonsense"]

TM_GREY = "#D9D9D9"
LOOP_TINT = {"cytosolic": "#FFF4E0", "lumenal": "#E8F1FA", "extracellular": "#E8F1FA"}
INK = "#222222"
MUTED = "#6B6B6B"
NA_COLOR = "#B8B8B8"

# heatmap: -1 dark red -> 0 white -> positive light blue (fixed across datasets)
DARK_RED = "#7F0000"
LIGHT_BLUE = "#7FB2DC"
VMIN, VMAX = -1.0, 0.5


def fitness_cmap(vmin: float = VMIN, vmax: float = VMAX):
    neg = LinearSegmentedColormap.from_list(
        "neg", [DARK_RED, "#B2182B", "#E0685B", "#F7C4B5", "#FFFFFF"])
    pos = LinearSegmentedColormap.from_list("pos", ["#FFFFFF", "#D4E6F4", LIGHT_BLUE])
    # stitch so that 0 lands exactly on white regardless of asymmetry
    n = 512
    frac0 = -vmin / (vmax - vmin)
    nneg = int(round(n * frac0))
    colors = np.vstack([neg(np.linspace(0, 1, nneg)), pos(np.linspace(0, 1, n - nneg))])
    cmap = LinearSegmentedColormap.from_list("mpdms_fitness", colors, N=n)
    cmap.set_bad(NA_COLOR)
    cmap.set_under(DARK_RED)
    cmap.set_over(LIGHT_BLUE)
    return cmap


def fitness_norm(vmin: float = VMIN, vmax: float = VMAX):
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def diverging_cmap():
    """Symmetric diverging map for non-fitness quantities (correlations, coefficients)."""
    return LinearSegmentedColormap.from_list("div", ["#B2182B", "#F4A582", "#F7F7F7", "#92C5DE", "#2166AC"])


def set_theme():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,   # embedded TrueType, editable in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
    })


@lru_cache(maxsize=1)
def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT,
                                stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def stamp(fig, cfg=None, analysis: str = ""):
    # The topology source rides on every figure: a panel that shades TM helices should
    # say where those helices came from, so a stale hydropathy guess cannot pass for TMHMM.
    topo = str((cfg.get_path("topology.source", "") if cfg else "") or "").strip()
    txt = f"{analysis}  {cfg.id if cfg else ''}  git:{git_commit()}".strip()
    if topo:
        txt += f"  topology:{topo}"
    fig.text(0.995, 0.002, txt, ha="right", va="bottom", fontsize=5, color=MUTED)


def title(fig, cfg, text: str):
    fig.suptitle(f"{cfg.display_name} — {text}", x=0.01, ha="left", fontsize=10, fontweight="bold")


def save(fig, outdir: Path, name: str, cfg=None, analysis: str = "") -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp(fig, cfg, analysis)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(p)
    plt.close(fig)
    return paths


def shade_topology(ax, cfg, ymin: float = 0.0, ymax: float = 1.0, loops: bool = True, label: bool = False):
    """Grey TM segments; faint tint for loops/soluble domains by side."""
    for s in cfg.get_path("topology.segments", []) or []:
        a, b = s["start"] - 0.5, s["end"] + 0.5
        if s.get("type") in ("TM", "intramembrane"):
            ax.axvspan(a, b, ymin=ymin, ymax=ymax, color=TM_GREY, lw=0, zorder=0)
            if label:
                ax.text((a + b) / 2, 1.0, s["name"], transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=6, color=MUTED)
        elif loops and s.get("side") in LOOP_TINT:
            ax.axvspan(a, b, ymin=ymin, ymax=ymax, color=LOOP_TINT[s["side"]], lw=0, zorder=0)


def topology_track(ax, cfg, xlim):
    """Thin cartoon bar: TM boxes grey, loops as lines coloured by side."""
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for s in cfg.get_path("topology.segments", []) or []:
        a, b = max(s["start"] - 0.5, xlim[0]), min(s["end"] + 0.5, xlim[1])
        if b <= a:
            continue
        if s.get("type") in ("TM", "intramembrane"):
            ax.add_patch(plt.Rectangle((a, 0.1), b - a, 0.8, color="#8C8C8C", lw=0))
            if b - a > 8:
                ax.text((a + b) / 2, 0.5, s["name"], ha="center", va="center", fontsize=5.5, color="white")
        else:
            y = 0.85 if s.get("side") == "cytosolic" else 0.15
            ax.plot([a, b], [y, y], color="#8C8C8C", lw=1.0, solid_capstyle="butt")


def class_legend_handles(classes=CLASS_ORDER):
    from matplotlib.patches import Patch
    return [Patch(color=CLASS_COLORS[c], label=CLASS_LABELS[c]) for c in classes]


set_theme()
