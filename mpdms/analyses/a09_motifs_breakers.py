"""A09 - helix breakers (Pro/Gly) and sequence motifs."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..annot import BIOLOGICAL, load_motifs, motif_effects
from ..config import REPO_ROOT, protein_sequence
from ..stats import bh_fdr, bootstrap_ci
from .base import dirs, missense, result


def breaker_table(mis: pd.DataFrame) -> pd.DataFrame:
    base = mis[~mis.mut.isin(["P", "G"])].groupby("pos").score_z.median().rename("median_other")
    out = []
    for aa in ("P", "G"):
        s = mis[mis.mut == aa].set_index("pos")[["score_z", "seg_type", "wt"]].rename(columns={"score_z": f"score_{aa}"})
        s = s.join(base, how="left")
        s[f"penalty_{aa}"] = s[f"score_{aa}"] - s["median_other"]
        out.append(s)
    t = out[0][["score_P", "penalty_P", "seg_type", "wt", "median_other"]].join(
        out[1][["score_G", "penalty_G"]], how="outer")
    return t


def run(df, cfg, outdir):
    figdir, tabdir = dirs(outdir)
    mis = missense(df)
    seq = protein_sequence(cfg)
    metrics, caveats = {}, []
    br = breaker_table(mis)
    for aa in ("P", "G"):
        for t in ("TM", "loop", "soluble"):
            v = br.loc[br.seg_type == t, f"penalty_{aa}"]
            est, lo, hi = bootstrap_ci(v, n=2000)
            metrics[f"{aa}_penalty_{t}"] = est
            metrics[f"{aa}_penalty_{t}_ci"] = [lo, hi]
    # proline penalty vs hydrophobicity expectation (should be weak)
    pr = br.dropna(subset=["score_P"]).copy()
    pr["hyd_expect"] = BIOLOGICAL["P"] - pr.wt.map(BIOLOGICAL)
    if len(pr) > 10:
        rho = ss.spearmanr(pr.score_P, pr.hyd_expect, nan_policy="omit")[0]
        metrics["rho_proline_vs_hydrophobicity"] = float(rho)
        if abs(rho) > 0.5:
            caveats.append(f"proline effects track hydrophobicity strongly (rho={rho:.2f}) - suspicious for a helix-integrity readout")

    # motifs
    tabs = []
    fig_rows = []
    if seq:
        motifs = load_motifs(REPO_ROOT / "configs" / "_motifs.yaml")
        eff = motif_effects(seq, mis, motifs)
        if len(eff):
            eff = eff.merge(mis[["score_z", "pos", "mut"]], left_on="index", right_index=True)
            rows = []
            for m, g in eff.groupby("motif"):
                ctrl = g[g.effect == "unaffected"].score_z
                for e in ("destroyed", "created"):
                    s = g[g.effect == e].score_z
                    if len(s) >= 3 and len(ctrl) >= 3:
                        u = ss.mannwhitneyu(s, ctrl, alternative="two-sided")
                        rbc = 1 - 2 * u.statistic / (len(s) * len(ctrl))
                        rows.append({"motif": m, "effect": e, "n": len(s), "median": s.median(),
                                     "median_unaffected": ctrl.median(), "rank_biserial": rbc, "p": u.pvalue})
                # matched control for created motifs: unaffected variants at the same positions
                cr = g[g.effect == "created"]
                if len(cr) >= 3:
                    mc = g[(g.effect == "unaffected") & g.pos.isin(cr.pos)].score_z
                    if len(mc) >= 3:
                        u = ss.mannwhitneyu(cr.score_z, mc)
                        rows.append({"motif": m, "effect": "created_vs_same_position", "n": len(cr),
                                     "median": cr.score_z.median(), "median_unaffected": mc.median(),
                                     "rank_biserial": 1 - 2 * u.statistic / (len(cr) * len(mc)), "p": u.pvalue})
            mt = pd.DataFrame(rows)
            if len(mt):
                mt["q"] = bh_fdr(mt.p)
                mt.to_csv(tabdir / "a09_motif_tests.csv", index=False)
                tabs.append(tabdir / "a09_motif_tests.csv")
                glyc = mt[(mt.motif == "N_glyc_sequon") & (mt.effect == "created_vs_same_position")]
                if len(glyc):
                    metrics["created_sequon_q"] = float(glyc.q.iloc[0])
                    metrics["created_sequon_median_shift"] = float(glyc["median"].iloc[0] - glyc.median_unaffected.iloc[0])
                metrics["n_motif_tests_q05"] = int((mt.q < 0.05).sum())
            fig_rows = eff
    assay = cfg.get_path("assay.type", "")
    sequon_hit = metrics.get("created_sequon_q", 1) < 0.05
    if assay in ("glyc_reporter", "surface_expression"):
        caveats.insert(0, "IMPORTANT: for glycosylation-reporter / surface-expression assays a created N-glyc sequon "
                          "can change the readout independently of folding or insertion")
    if sequon_hit:
        caveats.insert(0, f"created N-glycosylation sequons score differently from same-position controls "
                          f"(q={metrics['created_sequon_q']:.2g})")

    fig = plt.figure(figsize=(8.5, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.75)
    ax = fig.add_subplot(gs[0])
    P.shade_topology(ax, cfg)
    med = mis.groupby("pos").score_z.median()
    ax.plot(med.index, med.values, color=P.MUTED, lw=0.8, label="all-missense median", drawstyle="steps-mid")
    ax.plot(br.index, br.score_P, "o", ms=2.2, color="#C0392B", label="→ Pro")
    ax.plot(br.index, br.score_G, "o", ms=2.2, color="#E69F00", label="→ Gly", alpha=0.8)
    ax.axhline(0, color=P.MUTED, lw=0.4)
    ax.set_xlabel("Position")
    ax.set_ylabel(cfg.get_path("plotting.score_label"))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, markerscale=2.5, borderaxespad=0.2)
    ax.set_title(f"(a) Helix breakers — mean Pro penalty in TM {metrics.get('P_penalty_TM', np.nan):.2f}, "
                 f"loop {metrics.get('P_penalty_loop', np.nan):.2f}", loc="left", pad=16)
    ax = fig.add_subplot(gs[1])
    if len(fig_rows):
        cats = []
        for m, g in fig_rows.groupby("motif"):
            for e, col in (("unaffected", "#BBBBBB"), ("destroyed", "#C0392B"), ("created", "#1B4B80")):
                v = g[g.effect == e].score_z.dropna()
                if len(v):
                    cats.append((f"{m}\n{e}", v.to_numpy(), col))
        pos_ = np.arange(len(cats))
        bp = ax.boxplot([c[1] for c in cats], positions=pos_, widths=0.6, showfliers=False, patch_artist=True,
                        medianprops=dict(color=P.INK, lw=1), whiskerprops=dict(lw=0.6, color=P.MUTED),
                        capprops=dict(lw=0.6, color=P.MUTED))
        for patch, c in zip(bp["boxes"], cats):
            patch.set_facecolor(c[2]); patch.set_alpha(0.5); patch.set_linewidth(0.5)
        ax.set_xticks(pos_)
        ax.set_xticklabels([c[0] for c in cats], fontsize=5, rotation=90)
        ax.set_ylabel(cfg.get_path("plotting.score_label"))
        ax.set_title("(b) Motif destroyed / created / unaffected", loc="left")
    else:
        ax.axis("off")
    P.title(fig, cfg, "A09 helix breakers and motifs")
    figs = P.save(fig, figdir, "a09_motifs_breakers", cfg, "A09")
    br.to_csv(tabdir / "a09_breakers.csv")
    tabs.insert(0, tabdir / "a09_breakers.csv")
    head = (f"Proline costs {metrics.get('P_penalty_TM', np.nan):+.2f} beyond the position median inside TMs vs "
            f"{metrics.get('P_penalty_loop', np.nan):+.2f} in loops"
            + ("; CREATED N-GLYC SEQUONS SHIFT THE READOUT" if sequon_hit else ""))
    return result("a09", cfg, head, metrics, caveats, figs, tabs)
