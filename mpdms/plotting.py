"""Shared theme, colormaps and figure helpers so every slide looks the same."""
from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

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


STYLE = "default"          # "default" (slides) or "paper" (journal figure)
_CAPTIONS: dict = {}       # id(fig) -> {"title": str, "panels": [(letter, text)], "stamp": str}


def use_style(name: str) -> None:
    """Switch the global figure style. Call before any figure is built."""
    global STYLE
    if name not in ("default", "paper"):
        raise ValueError(f"unknown style {name!r}")
    STYLE = name
    set_theme()


def paper() -> bool:
    return STYLE == "paper"


# Journal conventions, as applied here: small sans-serif type, bold lower-case panel
# letters outside the axes, no figure title and no provenance stamp on the artwork
# (both move to a caption file so nothing is lost), hairline spines, short outward
# ticks, no grid, and fonts embedded so the PDF stays editable.
PAPER_RC = {
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.titleweight": "normal",
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
    "legend.borderpad": 0.2,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "figure.dpi": 110,
}


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
    if paper():
        plt.rcParams.update(PAPER_RC)


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
    if paper():
        # journal figures carry no provenance on the artwork; it goes in the caption file
        _CAPTIONS.setdefault(id(fig), {}).setdefault("stamp", txt)
        return
    fig.text(0.995, 0.002, txt, ha="right", va="bottom", fontsize=5, color=MUTED)


def title(fig, cfg, text: str):
    if paper():
        _CAPTIONS.setdefault(id(fig), {})["title"] = f"{cfg.display_name} — {text}"
        return
    fig.suptitle(f"{cfg.display_name} — {text}", x=0.01, ha="left", fontsize=10, fontweight="bold")


_PANEL_RE = re.compile(r"^\s*\(([a-zA-Z])\)\s*")


def _relabel_panels(fig) -> list[tuple[str, str]]:
    """Turn '(a) Sites' axes titles into a bold 'a' outside the axes, Nature-style.

    The letter comes from the analysis's own prefix rather than being re-derived, so
    panels keep the lettering their captions and headlines already refer to, and
    colour bars cannot be lettered by mistake. The descriptive half of the title is
    returned for the caption instead of being thrown away.
    """
    panels = []
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        src, setter, m = "", None, None
        for loc in ("left", "center", "right"):   # analyses set titles with loc="left"
            cand = ax.get_title(loc=loc)
            m = _PANEL_RE.match(cand or "")
            if m:
                src = cand
                setter = (lambda txt, _l=loc: ax.set_title(txt, loc=_l))
                break
        if not m:                        # a panel whose label is a text artist (axis off)
            for t in ax.texts:
                m = _PANEL_RE.match(t.get_text() or "")
                if m:
                    src, setter = t.get_text(), t.set_text
                    break
        if not m:
            continue
        letter, rest = m.group(1).lower(), _PANEL_RE.sub("", src)
        setter("")
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -22 / 72, 10 / 72, fig.dpi_scale_trans), ha="left", va="baseline",
            fontsize=8, fontweight="bold", color=INK)
        panels.append((letter, rest.strip()))
    return panels


def write_caption(fig, path: Path, name: str, panels: list[tuple[str, str]]) -> Path:
    """The text the figure no longer carries, as a caption stub to paste into a draft."""
    c = _CAPTIONS.pop(id(fig), {})
    lines = [f"Figure. {c.get('title', name)}", ""]
    lines += [f"({letter}) {text}" for letter, text in panels if text]
    if c.get("stamp"):
        lines += ["", f"[provenance: {c['stamp']}]"]
    path.write_text("\n".join(lines) + "\n")
    return path


def save(fig, outdir: Path, name: str, cfg=None, analysis: str = "") -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp(fig, cfg, analysis)
    panels = _relabel_panels(fig) if paper() else []
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(p)
    if paper():
        write_caption(fig, outdir / f"{name}_caption.txt", name, panels)
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
