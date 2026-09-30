"""SSDraw-style secondary-structure tracks coloured by a per-residue DMS value.

Uses SSDraw's own geometry builders (Chen et al. 2024, Protein Sci; github.com/ncbi/SSDraw)
but draws into a supplied matplotlib Axes with a *fixed* colour normalisation, so the
tracks share the heatmap's scale (-1 dark red, 0 white, >0 light blue) and can be stacked
directly under wrapped heatmap rows.
"""
from __future__ import annotations

import matplotlib.patches as mpatch
import matplotlib.path as mpath
import numpy as np

try:
    from SSDraw import core as _ssd
    HAVE_SSDRAW = True
except Exception:  # pragma: no cover - fallback keeps the pipeline running without SSDraw
    _ssd = None
    HAVE_SSDRAW = False


def _paths_to_patch(coord_sets, ax, z):
    verts, codes = [], []
    for c in coord_sets:
        c = np.asarray(c, float)
        verts.extend(c)
        codes.extend([mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(c) - 1))
    if not verts:
        return None
    patch = mpatch.PathPatch(mpath.Path(np.array(verts), codes), facecolor="none", ec="k", lw=0.45, zorder=z)
    ax.add_patch(patch)
    return patch


def draw_track(ax, ss: str, values, cmap, norm, label: str | None = None):
    """Draw one SSDraw track. `ss` is a per-residue string (H/E/C, '-' = not modelled),
    `values` the per-residue colour values (NaN -> cmap 'bad' colour). x in residue units:
    residue k (0-based) occupies [k, k+1)."""
    n = len(ss)
    vals = np.asarray(values, float)
    ss = "".join(c if c in "HEBC-" else "C" for c in ss)
    ax.set_xlim(0, n)
    ax.set_ylim(1.05, 2.95)
    ax.axis("off")
    if label:
        ax.text(-0.01, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=6.5)
    if not HAVE_SSDRAW or not ss.strip("-"):
        ax.imshow(vals[None, :], extent=[0, n, 1.75, 2.25], cmap=cmap, norm=norm, aspect="auto",
                  interpolation="nearest")
        return
    strand, loop, helix, ssbreak, order, bounds = _ssd.SS_breakdown(ss)
    loop_c, h1, h2, strand_c = [], [], [], []
    ssidx = -(2.0 / _ssd.SPACING)
    for i, kind in enumerate(order):
        prev_ss = order[i - 1] if i else None
        next_ss = order[i + 1] if i < len(order) - 1 else None
        if kind == "L":
            _ssd.build_loop(bounds[i], 0, ssidx, loop_c, len(ss), 1, prev_ss, next_ss, z=0, clr="none")
        elif kind == "H":
            _ssd.build_helix(bounds[i], 0, ssidx, h1, h2, z=i, clr="none", bkg="none")
        elif kind == "E":
            _ssd.build_strand(bounds[i], 0, ssidx, strand_c, next_ss, z=i, clr="none")
    # SSDraw works in units of 1/6 residue; rescale to residue units
    scale = lambda cs: [np.column_stack([np.asarray(c)[:, 0] * 6.0, np.asarray(c)[:, 1]]) for c in cs]  # noqa: E731
    mat = np.tile(vals, (20, 1))
    for k, cs in enumerate([loop_c, h2, strand_c, h1]):
        if not cs:
            continue
        z = 0 if k in (0, 1) else 10
        patch = _paths_to_patch(scale(cs), ax, z)
        im = ax.imshow(mat, extent=[0, n, 0.5, 3], cmap=cmap, norm=norm, interpolation="nearest",
                       aspect="auto", zorder=z)
        im.set_clip_path(patch)
