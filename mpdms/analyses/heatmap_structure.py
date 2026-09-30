"""DMS heatmap (normalised fitness) with SSDraw tracks under every wrapped row:
(i) mean missense effect per position, (ii) the proline substitution at each position.
Also writes B-factor-coloured PDBs + a PyMOL script for structure figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from .. import plotting as P
from ..annot import HEATMAP_ORDER
from ..config import protein_sequence
from ..io import position_matrix
from ..ssdraw_tracks import draw_track
from .base import dirs, result

SLUG = "heatmap"
ROWS = list(HEATMAP_ORDER) + ["*"]


def position_profiles(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["pass_filter"]]
    mis = d[d.vclass == "missense"].groupby("pos")["score_z"]
    prof = pd.DataFrame({"mean_missense": mis.mean(), "median_missense": mis.median(), "n_missense": mis.size()})
    pro = d[(d.mut == "P")].groupby("pos")["score_z"].mean()
    prof["proline"] = pro
    stop = d[d.vclass == "nonsense"].groupby("pos")["score_z"].mean()
    prof["stop"] = stop
    return prof


def structure_ss(cfg, positions) -> tuple[str, pd.DataFrame | None]:
    """Per-position SS string over `positions` from the structure ('-' where absent)."""
    p = cfg.resolve(cfg.get_path("structure.path"))
    if p is None or not p.exists():
        return "-" * len(positions), None
    from ..structure import residue_table
    rt = residue_table(p, cfg.get_path("structure.chain", "A"), int(cfg.get_path("structure.numbering_offset", 0) or 0))
    ssmap = dict(zip(rt["pos"], rt["ss"]))
    return "".join(ssmap.get(i, "-") for i in positions), rt


def heatmap_figure(df, cfg, seq, ss, positions, prof, name, with_tracks=True):
    vmin = cfg.get_path("plotting.heatmap_vmin", P.VMIN)
    vmax = cfg.get_path("plotting.heatmap_vmax", P.VMAX)
    cmap, norm = P.fitness_cmap(vmin, vmax), P.fitness_norm(vmin, vmax)
    mat = position_matrix(df).reindex(index=positions, columns=ROWS)
    per = int(cfg.get_path("plotting.residues_per_row", 100))
    chunks = [positions[i:i + per] for i in range(0, len(positions), per)]
    cell = 0.075
    heat_h = cell * len(ROWS)
    trk = 0.30
    heights = []
    for _ in chunks:
        heights += [0.12, heat_h] + ([trk, trk] if with_tracks else []) + [0.42]
    fig = plt.figure(figsize=(cell * per + 1.4, sum(heights) + 0.5))
    gs = GridSpec(len(heights), 2, figure=fig, height_ratios=heights, width_ratios=[cell * per, 0.12],
                  hspace=0.0, wspace=0.02, left=0.085, right=0.965, top=1 - 0.35 / (sum(heights) + 0.5),
                  bottom=0.02)
    row = 0
    im = None
    for ch in chunks:
        n = len(ch)
        # topology cartoon
        ax_t = fig.add_subplot(gs[row, 0]); row += 1
        P.topology_track(ax_t, cfg, (ch[0] - 0.5, ch[0] - 0.5 + per))
        # heatmap
        ax = fig.add_subplot(gs[row, 0]); row += 1
        m = mat.loc[ch].T.to_numpy(float)
        im = ax.imshow(m, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest",
                       extent=[ch[0] - 0.5, ch[0] - 0.5 + n, len(ROWS) - 0.5, -0.5])
        ax.set_xlim(ch[0] - 0.5, ch[0] - 0.5 + per)
        # WT cells: small dot
        for k, p in enumerate(ch):
            w = seq[p - 1] if seq and 0 < p <= len(seq) else None
            if w in ROWS:
                ax.plot(p, ROWS.index(w), marker="o", ms=1.3, color=P.INK, mew=0)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels(ROWS, fontsize=4.8, fontfamily="monospace")
        ax.tick_params(axis="y", length=0, pad=1)
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if with_tracks:
            for key, lab in (("mean_missense", "Mean"), ("proline", "Pro")):
                ax_s = fig.add_subplot(gs[row, 0]); row += 1
                vals = prof[key].reindex(ch).to_numpy(float)
                sschunk = "".join(ss[positions.index(p)] for p in ch)
                # pad to full row width so all rows share an x-scale
                pad = per - n
                draw_track(ax_s, sschunk + "-" * pad, np.concatenate([vals, np.full(pad, np.nan)]),
                           cmap, norm, label=lab)
        # residue labels
        ax_l = fig.add_subplot(gs[row, 0]); row += 1
        ax_l.set_xlim(ch[0] - 0.5, ch[0] - 0.5 + per)
        ax_l.set_ylim(0, 1)
        ax_l.axis("off")
        for p in ch:
            aa = seq[p - 1] if seq and 0 < p <= len(seq) else ""
            ax_l.text(p, 0.97, aa, ha="center", va="top", fontsize=4.3, fontfamily="monospace")
            if p % 10 == 0:
                ax_l.text(p, 0.62, str(p), ha="center", va="top", fontsize=5.2, color=P.MUTED)
                ax_l.plot([p, p], [0.66, 0.72], color=P.MUTED, lw=0.4)
    cax = fig.add_subplot(gs[1, 1])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_ticks([vmin, -0.5, 0, vmax])
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Normalised fitness", fontsize=6)
    fig.text(0.085, 1 - 0.18 / (sum(heights) + 0.5),
             f"{cfg.display_name} — normalised fitness (syn = 0, stop = −1)"
             + ("; SSDraw tracks: mean missense effect, proline" if with_tracks else ""),
             fontsize=8.5, fontweight="bold", va="center")
    return fig


def ssdraw_only_figure(cfg, ss, positions, prof, key, label):
    vmin = cfg.get_path("plotting.heatmap_vmin", P.VMIN)
    vmax = cfg.get_path("plotting.heatmap_vmax", P.VMAX)
    cmap, norm = P.fitness_cmap(vmin, vmax), P.fitness_norm(vmin, vmax)
    per = int(cfg.get_path("plotting.residues_per_row", 100))
    chunks = [positions[i:i + per] for i in range(0, len(positions), per)]
    fig, axes = plt.subplots(len(chunks) * 2, 1, figsize=(0.075 * per + 1.2, 0.62 * len(chunks) + 0.5),
                             gridspec_kw={"height_ratios": [0.12, 0.4] * len(chunks), "hspace": 0.05})
    axes = np.atleast_1d(axes)
    for k, ch in enumerate(chunks):
        n = len(ch)
        pad = per - n
        at = axes[2 * k]
        P.topology_track(at, cfg, (ch[0] - 0.5, ch[0] - 0.5 + per))
        vals = prof[key].reindex(ch).to_numpy(float)
        sschunk = "".join(ss[positions.index(p)] for p in ch)
        draw_track(axes[2 * k + 1], sschunk + "-" * pad, np.concatenate([vals, np.full(pad, np.nan)]), cmap, norm,
                   label=f"{ch[0]}–{ch[-1]}")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=list(axes), fraction=0.015, pad=0.01, extend="both")
    cb.set_ticks([vmin, -0.5, 0, vmax])
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label("Normalised fitness", fontsize=6)
    fig.suptitle(f"{cfg.display_name} — {label}", x=0.02, ha="left", fontsize=9, fontweight="bold")
    return fig


def write_structure_colouring(cfg, prof, tabdir):
    """PDBs with B-factor = value, plus a PyMOL script using the heatmap colours."""
    p = cfg.resolve(cfg.get_path("structure.path"))
    if p is None or not p.exists():
        return []
    from Bio.PDB import PDBIO, PDBParser
    out = []
    vmin = cfg.get_path("plotting.heatmap_vmin", P.VMIN)
    vmax = cfg.get_path("plotting.heatmap_vmax", P.VMAX)
    cmap, norm = P.fitness_cmap(vmin, vmax), P.fitness_norm(vmin, vmax)
    off = int(cfg.get_path("structure.numbering_offset", 0) or 0)
    for key in ("mean_missense", "proline"):
        s = PDBParser(QUIET=True).get_structure("s", str(p))
        vals = prof[key]
        for res in s.get_residues():
            v = vals.get(res.id[1] + off, np.nan)
            for a in res:
                a.bfactor = float(v) if np.isfinite(v) else 99.0
        f = tabdir / f"structure_{key}.pdb"
        io = PDBIO(); io.set_structure(s); io.save(str(f))
        lines = [f"load {f.name}, {cfg.id}_{key}", f"color grey70, {cfg.id}_{key}"]
        for pos, v in vals.dropna().items():
            r, g, b, _ = cmap(norm(np.clip(v, vmin, vmax)))
            lines.append(f"set_color c{pos}, [{r:.3f},{g:.3f},{b:.3f}]")
            lines.append(f"color c{pos}, {cfg.id}_{key} and resi {pos - off}")
        (tabdir / f"structure_{key}.pml").write_text("\n".join(lines) + "\n")
        out += [f, tabdir / f"structure_{key}.pml"]
    return out


def run(df, cfg, outdir):
    figdir, tabdir = dirs(outdir)
    seq = protein_sequence(cfg)
    lo = int(df["pos"].min())
    hi = int(max(df["pos"].max(), len(seq) if seq else 0))
    positions = list(range(1 if seq else lo, hi + 1))
    prof = position_profiles(df)
    ss, rt = structure_ss(cfg, positions)
    prof.to_csv(tabdir / "position_profiles.csv")
    figs = []
    caveats = []
    if rt is None:
        caveats.append("no structure - SSDraw tracks drawn as flat bars")
    else:
        if seq:
            mism = [(int(r.pos), r.aa) for r in rt.itertuples() if 0 < r.pos <= len(seq) and seq[int(r.pos) - 1] != r.aa]
            if mism:
                caveats.append(f"structure sequence differs from FASTA at {len(mism)} residues, e.g. {mism[:3]}")
    figs += P.save(heatmap_figure(df, cfg, seq, ss, positions, prof, "heatmap", with_tracks=False),
                   figdir, "heatmap", cfg, "heatmap")
    figs += P.save(heatmap_figure(df, cfg, seq, ss, positions, prof, "heatmap_ssdraw", with_tracks=True),
                   figdir, "heatmap_ssdraw", cfg, "heatmap")
    figs += P.save(ssdraw_only_figure(cfg, ss, positions, prof, "mean_missense",
                                      "SSDraw: mean missense effect per position (stops excluded)"),
                   figdir, "ssdraw_mean_effect", cfg, "heatmap")
    figs += P.save(ssdraw_only_figure(cfg, ss, positions, prof, "proline", "SSDraw: proline substitution effect"),
                   figdir, "ssdraw_proline", cfg, "heatmap")
    tabs = [tabdir / "position_profiles.csv"] + write_structure_colouring(cfg, prof, tabdir)
    worst = prof["mean_missense"].nsmallest(5)
    head = ("Most mutation-sensitive positions (mean missense): "
            + ", ".join(f"{seq[p - 1] if seq else ''}{p} ({v:.2f})" for p, v in worst.items()))
    ss_frac = {k: ss.count(k) / max(len(ss), 1) for k in "HEC-"}
    return result("heatmap", cfg, head, {
        "n_positions": len(positions), "coverage_positions": int(prof["n_missense"].gt(0).sum()),
        "helix_frac": ss_frac["H"], "strand_frac": ss_frac["E"],
        "mean_proline_effect": float(prof["proline"].mean()),
    }, caveats, figs, tabs)
