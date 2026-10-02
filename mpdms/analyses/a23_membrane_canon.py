"""A23 - six textbook membrane-protein predictions, each with a declared direction.

Every panel states what should happen BEFORE looking, so a pass is evidence the assay
reads membrane protein biogenesis and a failure is interpretable rather than a shrug.
Three are deliberately predictions that run against the dominant "buried is sensitive"
gradient, which is what makes them worth testing.

 a  Snorkelling     K/R are tolerated near the ends of a TM helix and punishing at its
                    centre, because the long side chain can reach its charge back to the
                    headgroups; a hydrophobic control should NOT show that V. The test
                    is the difference in curvature, not the K/R curve on its own.
 b  Positive-inside K/R cost more on the non-cytosolic flank; D/E mirror it. Four cells,
                    three predicted signs.
 c  Aromatic belt   W/Y matter MORE at the interface than in the hydrocarbon core, i.e.
                    the opposite of the usual depth gradient.
 d  Buried charge   WT-charged TM positions are more sensitive than depth-matched
                    WT-hydrophobic ones, specifically to charge-REMOVING substitutions.
 e  Packing         volume increase is penalised at high-contact positions, and the
                    reference expectation is a threshold rather than a smooth gradient.
 f  Periodicity     ~3.6-residue period in sensitivity along TM helices (A07 territory,
                    summarised here so the canon sits in one figure).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..annot import VOLUME, assign_topology
from ..stats import bh_fdr
from ..structure import membrane_frame, residue_table
from .base import dirs, fmt_p, missense, result, skipped

KR, DE, WY = set("KR"), set("DE"), set("WY")
HYDROPHOBIC = set("AVLIMFC")
CHARGED = set("DEKR")
ZMAX = 18.0          # A from the bilayer centre
CORE_Z = 8.0         # |z| below this is hydrocarbon core; beyond, interface
KR_COLOR, CTRL_COLOR = "#7B5EA7", "#6B6B6B"
MIN_N = 12


# --------------------------------------------------------------------- helpers
def prepared(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    """Missense variants with membrane depth, contacts, segment and flank attached."""
    p = cfg.resolve(cfg.get_path("structure.path"))
    if p is None or not p.exists():
        raise FileNotFoundError("structure.path is null or missing")
    rt = residue_table(p, cfg.get_path("structure.chain", "A"),
                       int(cfg.get_path("structure.numbering_offset", 0) or 0))
    rt, info = membrane_frame(rt, cfg)
    if info.get("method") == "none":
        raise ValueError(f"could not define membrane frame: {info.get('reason', '')}")
    rt["absz"] = rt.zdepth.abs()
    d = missense(assign_topology(df, cfg)).merge(
        rt[["pos", "zdepth", "absz", "radial", "contacts", "rsa"]], on="pos", how="inner")
    d["dvol"] = d.mut.map(VOLUME) - d.wt.map(VOLUME)
    d["cytosolic_flank"] = flank(d, cfg)
    return d, info


def flank(d: pd.DataFrame, cfg) -> pd.Series:
    """True where a TM position lies in the cytoplasmic half of its own helix.

    Orientation comes from TMHMM: in_out means the helix runs N-terminus-inside, so its
    first half is cytosolic. Using the helix's own halves rather than the sign of z
    keeps this independent of how the structure happened to be oriented.
    """
    out = pd.Series(np.nan, index=d.index, dtype="object")
    for s in cfg.get_path("topology.segments", []) or []:
        if s.get("type") != "TM":
            continue
        m = d.pos.between(s["start"], s["end"])
        if not m.any():
            continue
        mid = (s["start"] + s["end"]) / 2
        first_half = d.loc[m, "pos"] < mid
        out.loc[m] = first_half if s.get("orientation") == "in_out" else ~first_half
    return out


def _loess(x, y, grid, frac=0.6):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    f = lowess(y, x, frac=frac, return_sorted=True, delta=0.01 * (np.ptp(x) or 1))
    xs, i = np.unique(f[:, 0], return_index=True)
    return np.interp(grid, xs, f[i, 1])


def curvature(x, y) -> dict:
    """Quadratic coefficient of y ~ a + b|z| + c|z|^2, cluster-robust by position."""
    import statsmodels.formula.api as smf
    t = pd.DataFrame({"y": y, "z": x})
    t["z2"] = t.z ** 2
    m = smf.ols("y ~ z + z2", data=t).fit()
    return {"curvature": float(m.params["z2"]), "se": float(m.bse["z2"]), "p": float(m.pvalues["z2"]),
            "n": int(len(t))}


# ----------------------------------------------------------------- (a) snorkel
def snorkel(d: pd.DataFrame) -> dict:
    """K/R vs hydrophobic-control depth curves; the statistic is the curvature difference."""
    tm = d[(d.seg_type == "TM") & (d.absz <= ZMAX)]
    kr = tm[tm.mut.isin(KR) & tm.wt.isin(HYDROPHOBIC)]
    ct = tm[tm.mut.isin(set("LI")) & tm.wt.isin(HYDROPHOBIC)]
    if len(kr) < MIN_N or len(ct) < MIN_N:
        return {"testable": False, "note": f"too few variants (K/R {len(kr)}, control {len(ct)})"}
    import statsmodels.formula.api as smf
    t = pd.concat([kr.assign(cls="KR"), ct.assign(cls="ctrl")])
    t = t.assign(z=t.absz, z2=t.absz ** 2)
    m = smf.ols("score_z ~ (z + z2) * C(cls, Treatment('ctrl'))", data=t).fit(
        cov_type="cluster", cov_kwds={"groups": t.pos})
    key = [k for k in m.params.index if k.startswith("z2:")]
    delta = float(m.params[key[0]]) if key else np.nan
    return {"testable": True, "kr": curvature(kr.absz, kr.score_z), "control": curvature(ct.absz, ct.score_z),
            "delta_curvature": delta, "p": float(m.pvalues[key[0]]) if key else np.nan,
            "n_kr": len(kr), "n_control": len(ct),
            "passed": bool(key and delta > 0 and m.pvalues[key[0]] < 0.05)}


# ---------------------------------------------------------- (b) positive-inside
def positive_inside(d: pd.DataFrame) -> pd.DataFrame:
    """Cost of introducing K/R and D/E on each flank of the TM helices."""
    tm = d[(d.seg_type == "TM") & d.cytosolic_flank.notna() & d.wt.isin(HYDROPHOBIC)]
    rows = []
    for name, muts, expect in (("K/R", KR, "cytosolic cheaper"), ("D/E", DE, "cytosolic costlier")):
        sub = tm[tm.mut.isin(muts)]
        cy = sub[sub.cytosolic_flank.astype(bool)]
        ex = sub[~sub.cytosolic_flank.astype(bool)]
        if len(cy) < MIN_N or len(ex) < MIN_N:
            continue
        pc = cy.groupby("pos").score_z.median()
        pe = ex.groupby("pos").score_z.median()
        u, pv = ss.mannwhitneyu(pc, pe, alternative="two-sided")
        rows.append({"mutation": name, "expectation": expect,
                     "cytosolic": float(pc.median()), "non_cytosolic": float(pe.median()),
                     "difference": float(pc.median() - pe.median()),
                     "n_pos_cyt": len(pc), "n_pos_non": len(pe), "p": float(pv)})
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = bh_fdr(t.p)
        # predicted signs: K/R less costly inside (difference > 0); D/E the reverse
        t["as_predicted"] = np.where(t.mutation == "K/R", t.difference > 0, t.difference < 0)
    return t


# ------------------------------------------------------------- (c) aromatic belt
def aromatic_belt(d: pd.DataFrame) -> dict:
    """W/Y at the interface vs in the core - predicted to run against the usual gradient."""
    tm = d[(d.seg_type == "TM") & (d.absz <= ZMAX) & d.wt.isin(WY)]
    iface = tm[tm.absz > CORE_Z].groupby("pos").score_z.median()
    core = tm[tm.absz <= CORE_Z].groupby("pos").score_z.median()
    ctrl_t = d[(d.seg_type == "TM") & (d.absz <= ZMAX) & d.wt.isin(HYDROPHOBIC)]
    ci = ctrl_t[ctrl_t.absz > CORE_Z].groupby("pos").score_z.median()
    cc = ctrl_t[ctrl_t.absz <= CORE_Z].groupby("pos").score_z.median()
    if len(iface) < 3 or len(core) < 3:
        return {"testable": False, "note": f"too few W/Y positions (interface {len(iface)}, core {len(core)})"}
    u, pv = ss.mannwhitneyu(iface, core, alternative="two-sided")
    gradient = (float(ci.median() - cc.median()) if len(ci) > 2 and len(cc) > 2 else np.nan)
    return {"testable": True, "interface": float(iface.median()), "core": float(core.median()),
            "difference": float(iface.median() - core.median()),
            "control_gradient": gradient, "n_interface": len(iface), "n_core": len(core), "p": float(pv),
            "passed": bool(iface.median() < core.median() and pv < 0.05)}


# ------------------------------------------------------------ (d) buried charge
def buried_charge(d: pd.DataFrame, n_bins: int = 4) -> dict:
    """WT-charged vs WT-hydrophobic TM positions, matched on depth, split by substitution."""
    tm = d[(d.seg_type == "TM") & (d.absz <= ZMAX)].copy()
    tm["zbin"] = pd.cut(tm.absz, bins=n_bins)
    rows = []
    for removing in (True, False):
        sub = tm[tm.mut.isin(HYDROPHOBIC)] if removing else tm[~tm.mut.isin(CHARGED | HYDROPHOBIC)]
        ch = sub[sub.wt.isin(CHARGED)].groupby(["zbin", "pos"], observed=True).score_z.median()
        hy = sub[sub.wt.isin(HYDROPHOBIC)].groupby(["zbin", "pos"], observed=True).score_z.median()
        if len(ch) < 3 or len(hy) < 3:
            continue
        # depth-matched: difference of medians within each bin, then averaged over bins
        per_bin = []
        for b in tm.zbin.cat.categories:
            a = ch.loc[b] if b in ch.index.get_level_values(0) else pd.Series(dtype=float)
            c = hy.loc[b] if b in hy.index.get_level_values(0) else pd.Series(dtype=float)
            if len(a) >= 2 and len(c) >= 2:
                per_bin.append(float(np.median(a) - np.median(c)))
        if not per_bin:
            continue
        u, pv = ss.mannwhitneyu(ch, hy, alternative="two-sided")
        rows.append({"substitution": "charge-removing" if removing else "other",
                     "charged_wt": float(ch.median()), "hydrophobic_wt": float(hy.median()),
                     "difference_pooled": float(ch.median() - hy.median()),
                     "difference_depth_matched": float(np.mean(per_bin)), "n_bins": len(per_bin),
                     "n_charged_pos": int(ch.index.get_level_values(1).nunique()),
                     "n_hydrophobic_pos": int(hy.index.get_level_values(1).nunique()), "p": float(pv)})
    t = pd.DataFrame(rows)
    if not len(t):
        return {"testable": False, "note": "too few WT-charged TM positions"}
    t["q"] = bh_fdr(t.p)
    rem = t[t.substitution == "charge-removing"]
    oth = t[t.substitution == "other"]
    spec = (float(rem.difference_depth_matched.iloc[0] - oth.difference_depth_matched.iloc[0])
            if len(rem) and len(oth) else np.nan)
    return {"testable": True, "table": t.to_dict("records"), "specificity": spec,
            "passed": bool(len(rem) and rem.difference_depth_matched.iloc[0] < 0 and rem.q.iloc[0] < 0.05)}


# ----------------------------------------------------------------- (e) packing
def packing(d: pd.DataFrame) -> dict:
    """Is a volume increase penalised more where the position is tightly packed?"""
    tm = d[(d.seg_type == "TM") & (d.dvol > 0) & d.contacts.notna()]
    if len(tm) < 4 * MIN_N:
        return {"testable": False, "note": f"too few volume-increasing TM variants ({len(tm)})"}
    import statsmodels.formula.api as smf
    t = tm.assign(dv=tm.dvol / 100.0, ct=tm.contacts.astype(float))
    m = smf.ols("score_z ~ dv * ct", data=t).fit(cov_type="cluster", cov_kwds={"groups": t.pos})
    # threshold vs gradient: compare the interaction model with a split at the contact median
    cut = float(t.ct.median())
    t2 = t.assign(tight=(t.ct > cut).astype(float))
    m2 = smf.ols("score_z ~ dv * tight", data=t2).fit(cov_type="cluster", cov_kwds={"groups": t2.pos})
    return {"testable": True, "slope_interaction": float(m.params.get("dv:ct", np.nan)),
            "p_interaction": float(m.pvalues.get("dv:ct", np.nan)),
            "contact_cut": cut, "slope_loose": float(m2.params.get("dv", np.nan)),
            "slope_tight": float(m2.params.get("dv", np.nan) + m2.params.get("dv:tight", np.nan)),
            "p_threshold": float(m2.pvalues.get("dv:tight", np.nan)), "n": int(len(t)),
            "passed": bool(np.isfinite(m.pvalues.get("dv:ct", np.nan))
                           and m.params.get("dv:ct", 0) < 0 and m.pvalues.get("dv:ct", 1) < 0.05)}


# ------------------------------------------------------------- (f) periodicity
def periodicity(d: pd.DataFrame, lo: float = 3.0, hi: float = 4.5) -> dict:
    """Lomb-Scargle power of per-position sensitivity along each TM helix."""
    from scipy.signal import lombscargle
    tm = d[d.seg_type == "TM"]
    periods, best = [], []
    for h, g in tm.groupby("segment", observed=True):
        s = g.groupby("pos").score_z.median().sort_index()
        if len(s) < 12:
            continue
        x = s.index.to_numpy(float)
        y = s.to_numpy() - s.mean()
        if not np.any(np.abs(y) > 1e-9):
            continue
        per = np.linspace(lo, hi, 200)
        pw = lombscargle(x, y, 2 * np.pi / per, normalize=True)
        periods.append(float(per[int(np.argmax(pw))]))
        best.append(float(pw.max()))
    if not periods:
        return {"testable": False, "note": "no TM helix long enough"}
    return {"testable": True, "median_period": float(np.median(periods)),
            "median_power": float(np.median(best)), "n_helices": len(periods),
            "periods": periods, "passed": bool(3.3 <= np.median(periods) <= 3.9)}


# ------------------------------------------------------------------- figure
def _blank(ax, note: str):
    """An untestable panel: say why, rather than leaving an empty 0-1 axis."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color=P.MUTED, wrap=True)


def _verdict(ax, r: dict):
    """A small corner tag: did the prediction hold?"""
    if not r.get("testable", False):
        txt, col = "not testable", P.MUTED
    elif r.get("passed"):
        txt, col = "as predicted", "#1B9E77"
    else:
        txt, col = "not as predicted", "#D62728"
    ax.text(0.98, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color=col, fontweight="bold")


def figure(cfg, d, res, figdir):
    fig, axg = plt.subplots(2, 3, figsize=(10.4, 6.4), layout="constrained")
    ax = axg[0, 0]
    tm = d[(d.seg_type == "TM") & (d.absz <= ZMAX)]
    grid = np.linspace(0, ZMAX, 80)
    for muts, wts, col, lab in ((KR, HYDROPHOBIC, KR_COLOR, "to Lys/Arg"),
                                (set("LI"), HYDROPHOBIC, CTRL_COLOR, "to Leu/Ile (control)")):
        g = tm[tm.mut.isin(muts) & tm.wt.isin(wts)]
        if len(g) < MIN_N:
            continue
        ax.scatter(g.absz, g.score_z, s=1.5, color=col, alpha=0.18, lw=0, rasterized=True)
        ax.plot(grid, _loess(g.absz.to_numpy(), g.score_z.to_numpy(), grid), color=col, lw=1.5, label=lab)
    ax.axvline(CORE_Z, color=P.MUTED, lw=0.5, ls=(0, (2, 2)))
    s = res["snorkel"]
    if s.get("testable"):
        ax.set_xlabel(f"|depth| from bilayer centre (Å)\nΔcurvature (K/R − control) = "
                      f"{s['delta_curvature']:+.4f}, {fmt_p(s['p'])}", fontsize=6.5)
    else:
        ax.set_xlabel("|depth| from bilayer centre (Å)")
    ax.set_ylabel("Normalised fitness")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.set_title("(a) Snorkelling: K/R rescued near the ends", loc="left", fontsize=8)
    _verdict(ax, s)

    ax = axg[0, 1]
    pit = res["positive_inside_table"]
    if len(pit):
        w, pos = 0.36, np.arange(len(pit))
        ax.bar(pos - w / 2, pit.cytosolic, w, color="#2F6DB5", label="cytosolic flank")
        ax.bar(pos + w / 2, pit.non_cytosolic, w, color="#B8912F", label="non-cytosolic flank")
        for k, r in enumerate(pit.itertuples()):
            ax.text(k, 1.01, f"{fmt_p(r.p)}  sign {'as predicted' if r.as_predicted else 'opposite'}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5,
                    color="#1B9E77" if r.as_predicted else "#D62728")
        ax.set_xticks(pos); ax.set_xticklabels(pit.mutation)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    else:
        _blank(ax, "positive-inside not testable:\ntoo few K/R or D/E variants on one flank")
    ax.axhline(0, color=P.INK, lw=0.5)
    ax.set_ylabel("Median fitness (position medians)")
    ax.set_xlabel("Introduced residue", fontsize=6.5)
    ax.set_title("(b) Positive-inside: K/R cheaper inside, D/E dearer", loc="left", fontsize=8)

    ax = axg[0, 2]
    a = res["aromatic"]
    if a.get("testable"):
        g = d[(d.seg_type == "TM") & (d.absz <= ZMAX) & d.wt.isin(WY)]
        c = d[(d.seg_type == "TM") & (d.absz <= ZMAX) & d.wt.isin(HYDROPHOBIC)]
        P.violin_box(ax, {"W/Y\ncore": g[g.absz <= CORE_Z].score_z.to_numpy(),
                          "W/Y\ninterface": g[g.absz > CORE_Z].score_z.to_numpy(),
                          "hydrophobic\ncore": c[c.absz <= CORE_Z].score_z.to_numpy(),
                          "hydrophobic\ninterface": c[c.absz > CORE_Z].score_z.to_numpy()},
                     colors={"W/Y\ncore": "#B8912F", "W/Y\ninterface": "#E0C56A",
                             "hydrophobic\ncore": "#8899A6", "hydrophobic\ninterface": "#C9D3DD"},
                     width=0.75, show_medians=False)
        ax.set_xlabel(f"W/Y interface − core = {a['difference']:+.2f}, {fmt_p(a['p'])}\n"
                      f"(hydrophobic control gradient {a['control_gradient']:+.2f})", fontsize=6.5)
    else:
        _blank(ax, "aromatic belt not testable:\n" + a.get("note", ""))
    ax.axhline(0, color=P.INK, lw=0.5) if a.get("testable") else None
    ax.set_ylabel("Normalised fitness") if a.get("testable") else None
    ax.set_title("(c) Aromatic belt: W/Y matter at the interface", loc="left", fontsize=8)
    _verdict(ax, a)

    ax = axg[1, 0]
    b = res["buried_charge"]
    if b.get("testable"):
        bt = pd.DataFrame(b["table"])
        pos = np.arange(len(bt)); w = 0.36
        ax.bar(pos - w / 2, bt.charged_wt, w, color="#C2443A", label="WT charged")
        ax.bar(pos + w / 2, bt.hydrophobic_wt, w, color="#8899A6", label="WT hydrophobic")
        for k, r in enumerate(bt.itertuples()):
            ax.text(k, 1.01, f"Δ(depth-matched) {r.difference_depth_matched:+.2f}, {fmt_p(r.p)}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks(pos); ax.set_xticklabels(bt.substitution)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    if b.get("testable"):
        ax.axhline(0, color=P.INK, lw=0.5)
        ax.set_ylabel("Median fitness (position medians)")
        ax.set_xlabel("Substitution class", fontsize=6.5)
    else:
        _blank(ax, "buried charge not testable:\n" + b.get("note", ""))
    ax.set_title("(d) Buried charges are load-bearing", loc="left", fontsize=8)
    _verdict(ax, b)

    ax = axg[1, 1]
    pk = res["packing"]
    tmv = d[(d.seg_type == "TM") & (d.dvol > 0) & d.contacts.notna()]
    if pk.get("testable") and len(tmv):
        cut = pk["contact_cut"]
        for sel, col, lab in ((tmv.contacts <= cut, "#8899A6", f"loose (≤{cut:.0f} contacts)"),
                              (tmv.contacts > cut, "#C2443A", f"tight (>{cut:.0f} contacts)")):
            g = tmv[sel]
            if len(g) < MIN_N:
                continue
            gr = np.linspace(g.dvol.min(), g.dvol.max(), 60)
            ax.scatter(g.dvol, g.score_z, s=1.5, color=col, alpha=0.15, lw=0, rasterized=True)
            ax.plot(gr, _loess(g.dvol.to_numpy(), g.score_z.to_numpy(), gr), color=col, lw=1.5, label=lab)
        ax.set_xlabel(f"Volume increase (Å³)\nslope difference {fmt_p(pk['p_threshold'])}; "
                      f"gradient in contacts {fmt_p(pk['p_interaction'])}", fontsize=6.5)
        ax.legend(frameon=False, fontsize=6, loc="lower left")
    if not pk.get("testable"):
        _blank(ax, "packing not testable:\n" + pk.get("note", ""))
    else:
        ax.set_ylabel("Normalised fitness")
    ax.set_title("(e) Packing: bulk costs more where it is tight", loc="left", fontsize=8)
    _verdict(ax, pk)

    ax = axg[1, 2]
    pr = res["periodicity"]
    if pr.get("testable"):
        ax.hist(pr["periods"], bins=np.linspace(3.0, 4.5, 16), color="#2F6DB5", alpha=0.75, lw=0)
        ax.axvline(3.6, color="#D62728", lw=1.0, ls=(0, (3, 2)))
        ax.text(3.6, 0.98, " α-helix 3.6", transform=ax.get_xaxis_transform(), fontsize=6,
                color="#D62728", va="top")
        ax.set_xlabel(f"Best-fitting period per helix (residues)\nmedian {pr['median_period']:.2f} "
                      f"over {pr['n_helices']} helices", fontsize=6.5)
        ax.set_ylabel("Helices")
    else:
        _blank(ax, "periodicity not testable:\n" + pr.get("note", ""))
    ax.set_title("(f) Helical periodicity ≈ 3.6", loc="left", fontsize=8)
    _verdict(ax, pr)

    P.title(fig, cfg, "A23 canonical membrane-protein predictions")
    return P.save(fig, figdir, "a23_membrane_canon", cfg, "A23")


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    if not [s for s in (cfg.get_path("topology.segments") or []) if s.get("type") == "TM"]:
        return skipped("a23", cfg, "no TM segments in the config - run `mpdms tmhmm` first")
    if cfg.get_path("structure.membrane_normal", "none") == "none":
        return skipped("a23", cfg, "structure.membrane_normal is none - depth undefined")
    try:
        d, info = prepared(df, cfg)
    except (FileNotFoundError, ValueError) as e:
        return skipped("a23", cfg, str(e))
    if not len(d):
        return skipped("a23", cfg, "no missense variants map onto the structure")

    res = {"snorkel": snorkel(d), "positive_inside_table": positive_inside(d),
           "aromatic": aromatic_belt(d), "buried_charge": buried_charge(d),
           "packing": packing(d), "periodicity": periodicity(d)}
    pit = res["positive_inside_table"]
    res["positive_inside"] = {"testable": bool(len(pit)),
                              "n_as_predicted": int(pit.as_predicted.sum()) if len(pit) else 0,
                              "n_tested": int(len(pit)),
                              "passed": bool(len(pit) and pit.as_predicted.all()
                                             and (pit.q < 0.05).any())}
    figs = figure(cfg, d, res, figdir)

    rows = [{"prediction": k, "testable": v.get("testable", False), "passed": v.get("passed", False),
             "note": v.get("note", "")} for k, v in res.items() if k != "positive_inside_table"]
    tp = tabdir / "a23_predictions.csv"
    pd.DataFrame(rows).to_csv(tp, index=False)
    pp = tabdir / "a23_positive_inside.csv"
    pit.to_csv(pp, index=False)

    tested = [r for r in rows if r["testable"]]
    met = [r["prediction"] for r in tested if r["passed"]]
    caveats = ["each panel states its direction in advance; a failure is a result about this "
               "protein or this assay, not a bug",
               f"depth comes from the membrane frame ({info.get('method')}) on a predicted "
               "structure, so |z| is approximate and the core/interface split at "
               f"{CORE_Z:g} Å is a convention",
               "flank membership uses each helix's own halves and TMHMM orientation, so it does "
               "not depend on how the structure happens to be oriented"]
    notest = [r["prediction"] for r in rows if not r["testable"]]
    if notest:
        caveats.append("not testable here: " + ", ".join(notest))
    head = (f"{len(met)}/{len(tested)} canonical predictions met"
            + (f" ({', '.join(met)})" if met else "")
            + (f"; failed: {', '.join(r['prediction'] for r in tested if not r['passed'])}"
               if len(met) < len(tested) else ""))
    return result("a23", cfg, head,
                  {k: (v if not isinstance(v, pd.DataFrame) else v.to_dict("records"))
                   for k, v in res.items()},
                  caveats, figs, [tp, pp])
