"""A24 - what the structure says about where mutations are tolerated.

Four structural variables the model already provides are used together, because in a
membrane protein no one of them separates the classes that matter: relative solvent
accessibility, contact number, depth along the membrane normal, and RADIAL distance
from the transmembrane bundle axis. The radial term is what makes "accessible" useful:
in a monomer embedded in a bilayer an accessible TM position faces either lipid or the
substrate cavity, never water, and those two have opposite predictions.

 a  Helices ranked by burial, not by sequence. Buried helices should be less tolerant,
    and the correlation should survive controlling for depth.
 b  Three-way facing classes crossed with substitution type:
      lipid-facing    accessible and peripheral - tolerates hydrophobic swaps, not charge
      protein-facing  buried                    - tolerates neither bulk nor charge
      cavity-lining   accessible and central    - the substrate pathway
 c  The cavity-lining prediction that matters: abundance-tolerant but ESM-constrained,
    i.e. function rather than folding. Needs A15 to have run.
 d  Contact order. Positions whose contacts are sequence-distant hold the fold together;
    tested with burial in the model so it is not just burial again.
 e  Depth x burial: are the two gradients independent or multiplicative?
 f  Helix-helix interface matrix: mean effect at each pair's shared interface.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as ss

from .. import plotting as P
from ..annot import VOLUME, assign_topology
from ..stats import bh_fdr
from ..structure import contact_order, contact_pairs, membrane_frame, residue_table
from .base import dirs, fmt_p, missense, result, skipped

SLAB = 15.0          # |z| within which a position counts as in the membrane
RSA_OPEN = 0.25      # above this, the side chain is exposed (to lipid or to the cavity)
CHARGED = set("DEKR")
HYDROPHOBIC = set("AVLIMFC")
FACE_COLORS = {"lipid-facing": "#B8912F", "protein-facing": "#2F6DB5", "cavity-lining": "#1B9E77"}
FACE_ORDER = ["protein-facing", "lipid-facing", "cavity-lining"]
MIN_N = 10


def prepared(df: pd.DataFrame, cfg):
    p = cfg.resolve(cfg.get_path("structure.path"))
    if p is None or not p.exists():
        raise FileNotFoundError("structure.path is null or missing")
    chain = cfg.get_path("structure.chain", "A")
    off = int(cfg.get_path("structure.numbering_offset", 0) or 0)
    rt = residue_table(p, chain, off)
    rt, info = membrane_frame(rt, cfg)
    if info.get("method") == "none":
        raise ValueError(f"could not define membrane frame: {info.get('reason', '')}")
    pairs = contact_pairs(p, chain, off)
    rt["absz"] = rt.zdepth.abs()
    rt["contact_order"] = contact_order(pairs, rt.pos).to_numpy()
    rt["in_slab"] = rt.absz <= SLAB
    # the radial split is defined on TM-slab residues only, so loops cannot move it
    med_radial = float(rt.loc[rt.in_slab, "radial"].median())
    rt["face"] = np.where(
        ~rt.in_slab, "non-membrane",
        np.where(rt.rsa < RSA_OPEN, "protein-facing",
                 np.where(rt.radial > med_radial, "lipid-facing", "cavity-lining")))
    d = missense(assign_topology(df, cfg)).merge(
        rt[["pos", "zdepth", "absz", "radial", "rsa", "contacts", "contact_order", "in_slab", "face"]],
        on="pos", how="inner")
    d["dvol"] = d.mut.map(VOLUME) - d.wt.map(VOLUME)
    d["sub_class"] = np.where(d.mut.isin(CHARGED), "to charged",
                              np.where(d.dvol > 30, "to bulkier",
                                       np.where(d.mut.isin(HYDROPHOBIC), "to hydrophobic", "other")))
    return d, rt, pairs, info, med_radial


# ------------------------------------------------------------- (a) helix burial
def helix_burial(d: pd.DataFrame, rt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tm = d[(d.seg_type == "TM") & d.in_slab]
    rows = []
    for h, g in tm.groupby("segment", observed=True):
        r = rt[rt.pos.isin(g.pos.unique())]
        rows.append({"helix": h, "burial": float(r.contacts.mean()), "mean_rsa": float(r.rsa.mean()),
                     "mean_absz": float(g.absz.mean()), "n_pos": int(g.pos.nunique()),
                     "median_effect": float(g.score_z.median())})
    t = pd.DataFrame(rows).sort_values("burial", ascending=False).reset_index(drop=True)
    if len(t) < 4:
        return t, {"testable": False, "note": f"only {len(t)} helices"}
    rho, pv = ss.spearmanr(t.burial, t.median_effect)
    m = smf.ols("median_effect ~ burial + mean_absz", data=t).fit()
    return t, {"testable": True, "spearman": float(rho), "p": float(pv), "n_helices": len(t),
               "beta_burial_given_depth": float(m.params["burial"]),
               "p_burial_given_depth": float(m.pvalues["burial"]),
               "passed": bool(rho < 0 and pv < 0.05)}


# ------------------------------------------------------------------ (b) facing
def facing_table(d: pd.DataFrame) -> pd.DataFrame:
    sub = d[d.in_slab & d.face.isin(FACE_ORDER) & d.sub_class.isin(["to charged", "to bulkier", "to hydrophobic"])]
    rows = []
    for (face, cls), g in sub.groupby(["face", "sub_class"], observed=True):
        pos = g.groupby("pos").score_z.median()
        if len(pos) < 3:
            continue
        rows.append({"face": face, "sub_class": cls, "n_variants": len(g), "n_positions": len(pos),
                     "median": float(pos.median())})
    return pd.DataFrame(rows)


def facing_tests(d: pd.DataFrame) -> pd.DataFrame:
    """Within each substitution class, is one facing class more tolerant than another?"""
    sub = d[d.in_slab & d.face.isin(FACE_ORDER)]
    rows = []
    for cls, g in sub.groupby("sub_class", observed=True):
        if cls not in ("to charged", "to bulkier", "to hydrophobic"):
            continue
        for a, b in (("lipid-facing", "protein-facing"), ("cavity-lining", "protein-facing"),
                     ("cavity-lining", "lipid-facing")):
            xa = g[g.face == a].groupby("pos").score_z.median()
            xb = g[g.face == b].groupby("pos").score_z.median()
            if len(xa) < 3 or len(xb) < 3:
                continue
            rows.append({"sub_class": cls, "a": a, "b": b, "n_a": len(xa), "n_b": len(xb),
                         "difference": float(xa.median() - xb.median()),
                         "p": float(ss.mannwhitneyu(xa, xb, alternative="two-sided")[1])})
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = bh_fdr(t.p)
    return t


# --------------------------------------------------------- (c) cavity vs ESM
def cavity_vs_esm(d: pd.DataFrame, outdir: Path) -> dict:
    vp = Path(outdir) / "tables" / "a15_variants.csv"
    if not vp.exists():
        return {"testable": False, "note": "a15_variants.csv not found - run a15 first"}
    v = pd.read_csv(vp)[["pos", "mut", "z"]]
    m = d[d.in_slab & d.face.isin(FACE_ORDER)].merge(v, on=["pos", "mut"], how="inner")
    if not len(m):
        return {"testable": False, "note": "no overlap with the A15 variant table"}
    rows = []
    for face, g in m.groupby("face", observed=True):
        pos = g.groupby("pos").agg(abundance=("score_z", "median"), esm_z=("z", "median"))
        if len(pos) < 3:
            continue
        rows.append({"face": face, "n_positions": len(pos),
                     "median_abundance": float(pos.abundance.median()),
                     "median_esm_z": float(pos.esm_z.median())})
    t = pd.DataFrame(rows)
    if len(t) < 2:
        return {"testable": False, "note": "too few positions per facing class"}
    cav = t[t.face == "cavity-lining"]
    oth = t[t.face != "cavity-lining"]
    return {"testable": True, "table": t.to_dict("records"),
            "passed": bool(len(cav) and len(oth)
                           and cav.median_abundance.iloc[0] > oth.median_abundance.min()
                           and cav.median_esm_z.iloc[0] < oth.median_esm_z.max())}


# -------------------------------------------------------- (d) contact order
def contact_order_test(d: pd.DataFrame) -> dict:
    tm = d[d.in_slab & d.contact_order.notna() & d.contacts.notna()]
    pos = tm.groupby("pos").agg(y=("score_z", "median"), co=("contact_order", "first"),
                                ct=("contacts", "first"), absz=("absz", "first")).dropna()
    if len(pos) < 20:
        return {"testable": False, "note": f"only {len(pos)} positions with contact order"}
    rho, pv = ss.spearmanr(pos.co, pos.y)
    m = smf.ols("y ~ co + ct + absz", data=pos).fit()
    return {"testable": True, "spearman": float(rho), "p": float(pv), "n_positions": len(pos),
            "beta_contact_order": float(m.params["co"]), "p_adjusted": float(m.pvalues["co"]),
            "passed": bool(m.params["co"] < 0 and m.pvalues["co"] < 0.05)}


# ------------------------------------------------------- (e) depth x burial
def depth_burial(d: pd.DataFrame) -> dict:
    tm = d[d.in_slab & d.contacts.notna()]
    pos = tm.groupby("pos").agg(y=("score_z", "median"), ct=("contacts", "first"),
                                absz=("absz", "first")).dropna()
    if len(pos) < 20:
        return {"testable": False, "note": f"only {len(pos)} positions"}
    m = smf.ols("y ~ ct * absz", data=pos).fit()
    return {"testable": True, "beta_contacts": float(m.params["ct"]), "beta_absz": float(m.params["absz"]),
            "beta_interaction": float(m.params["ct:absz"]), "p_interaction": float(m.pvalues["ct:absz"]),
            "n_positions": len(pos),
            "passed": bool(m.pvalues["ct:absz"] < 0.05)}


# ------------------------------------------------- (f) helix-helix interfaces
def interface_matrix(d: pd.DataFrame, pairs: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    tms = [s for s in (cfg.get_path("topology.segments") or []) if s.get("type") == "TM"]
    names = [s["name"] for s in tms]
    seg_of = {}
    for s in tms:
        for p_ in range(s["start"], s["end"] + 1):
            seg_of[p_] = s["name"]
    pr = pairs.assign(hi=pairs.pos_i.map(seg_of), hj=pairs.pos_j.map(seg_of)).dropna(subset=["hi", "hj"])
    pr = pr[pr.hi != pr.hj]
    eff = d[d.in_slab].groupby("pos").score_z.median()
    rows = []
    for (a, b), g in pr.groupby(["hi", "hj"], observed=True):
        res = set(g.pos_i) | set(g.pos_j)
        vals = eff.reindex(sorted(res)).dropna()
        if len(vals) < 4:
            continue
        lo, hi = sorted([a, b], key=lambda n: names.index(n))
        rows.append({"helix_a": lo, "helix_b": hi, "n_contacts": len(g),
                     "n_interface_pos": len(vals), "median_effect": float(vals.median())})
    t = pd.DataFrame(rows)
    mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for r in t.itertuples():
        mat.loc[r.helix_a, r.helix_b] = r.median_effect
        mat.loc[r.helix_b, r.helix_a] = r.median_effect
    return t, mat


# ------------------------------------------------------------------- figure
def _tag(ax, r: dict):
    if not r.get("testable", False):
        t, c = "not testable", P.MUTED
    elif r.get("passed"):
        t, c = "as predicted", "#1B9E77"
    else:
        t, c = "not as predicted", "#D62728"
    # top-right: the lower corners carry legends and fit annotations in these panels
    ax.text(0.98, 0.97, t, transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color=c, fontweight="bold")


def figure(cfg, d, rt, res, figdir):
    fig, axg = plt.subplots(2, 3, figsize=(10.6, 6.6), layout="constrained")

    ax = axg[0, 0]
    ht, hs = res["helix_table"], res["helix_burial"]
    if len(ht):
        ax.scatter(ht.burial, ht.median_effect, s=26, color="#2F6DB5", lw=0.4, edgecolor="white", zorder=3)
        for r in ht.itertuples():
            ax.annotate(r.helix, (r.burial, r.median_effect), xytext=(3, 3),
                        textcoords="offset points", fontsize=5.5, color=P.MUTED)
        if hs.get("testable"):
            b = np.polyfit(ht.burial, ht.median_effect, 1)
            g = np.linspace(ht.burial.min(), ht.burial.max(), 20)
            ax.plot(g, np.polyval(b, g), color=P.FIT_RED, lw=1.2)
            ax.set_xlabel(f"Mean contacts per residue (burial)\nSpearman ρ = {hs['spearman']:+.2f}, "
                          f"{fmt_p(hs['p'])}; given depth {fmt_p(hs['p_burial_given_depth'])}", fontsize=6.5)
    ax.set_ylabel("Median missense effect")
    ax.set_title("(a) Buried helices are less tolerant", loc="left", fontsize=8)
    _tag(ax, hs)

    ax = axg[0, 1]
    ft = res["facing_table"]
    if len(ft):
        classes = ["to hydrophobic", "to bulkier", "to charged"]
        faces = [f for f in FACE_ORDER if f in set(ft.face)]
        w = 0.8 / max(len(faces), 1)
        for k, f in enumerate(faces):
            sub = ft[ft.face == f].set_index("sub_class").reindex(classes)
            ax.bar(np.arange(len(classes)) + (k - (len(faces) - 1) / 2) * w, sub["median"].to_numpy(),
                   w, color=FACE_COLORS[f], label=f)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([c.replace("to ", "") for c in classes])
        ax.legend(frameon=False, fontsize=5.5, loc="lower left")
        ft_sig = res["facing_tests"]
        nsig = int((ft_sig.q < 0.05).sum()) if len(ft_sig) else 0
        ax.set_xlabel(f"Substitution\n{nsig}/{len(ft_sig)} facing contrasts significant at q<0.05", fontsize=6.5)
    ax.axhline(0, color=P.INK, lw=0.5)
    ax.set_ylabel("Median missense effect")
    ax.set_title("(b) Facing class × substitution type", loc="left", fontsize=8)

    ax = axg[0, 2]
    cv = res["cavity"]
    if cv.get("testable"):
        ct = pd.DataFrame(cv["table"])
        for r in ct.itertuples():
            ax.scatter([r.median_abundance], [r.median_esm_z], s=60, color=FACE_COLORS.get(r.face, P.INK),
                       lw=0.5, edgecolor="white", zorder=3, label=f"{r.face} (n = {r.n_positions})")
        ax.axhline(-1.0, color="#D62728", lw=0.6, ls=(0, (3, 3)))
        ax.axvline(-0.5, color=P.MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.legend(frameon=False, fontsize=5.5, loc="upper left")
        ax.set_xlabel("Median abundance", fontsize=6.5)
        ax.set_ylabel("Median ESM-1v residual z", fontsize=6.5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.5, 0.5, "cavity vs ESM not testable:\n" + cv.get("note", ""), transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=P.MUTED)
    ax.set_title("(c) Cavity lining: tolerated but constrained", loc="left", fontsize=8)
    _tag(ax, cv)

    ax = axg[1, 0]
    co = res["contact_order"]
    tm = d[d.in_slab & d.contact_order.notna()]
    if co.get("testable") and len(tm):
        pos = tm.groupby("pos").agg(y=("score_z", "median"), co=("contact_order", "first"),
                                    ct=("contacts", "first")).dropna()
        sc = ax.scatter(pos.co, pos.y, s=6, c=pos.ct, cmap="viridis", lw=0, alpha=0.8)
        fig.colorbar(sc, ax=ax, fraction=0.04).set_label("contacts", fontsize=6)
        b = np.polyfit(pos.co, pos.y, 1)
        g = np.linspace(pos.co.min(), pos.co.max(), 20)
        ax.plot(g, np.polyval(b, g), color=P.FIT_RED, lw=1.2)
        ax.set_xlabel(f"Mean sequence separation of contacts\nρ = {co['spearman']:+.2f}; "
                      f"with contacts and depth in the model {fmt_p(co['p_adjusted'])}", fontsize=6.5)
    ax.set_ylabel("Median missense effect")
    ax.set_title("(d) Long-range contacts hold the fold", loc="left", fontsize=8)
    _tag(ax, co)

    ax = axg[1, 1]
    db = res["depth_burial"]
    if db.get("testable"):
        pos = d[d.in_slab].groupby("pos").agg(y=("score_z", "median"), ct=("contacts", "first"),
                                              absz=("absz", "first")).dropna()
        cut = pos.ct.median()
        for sel, col, lab in ((pos.ct <= cut, "#8899A6", f"loose (≤{cut:.0f})"),
                              (pos.ct > cut, "#C2443A", f"tight (>{cut:.0f})")):
            g = pos[sel]
            if len(g) < MIN_N:
                continue
            ax.scatter(g.absz, g.y, s=6, color=col, lw=0, alpha=0.6)
            b = np.polyfit(g.absz, g.y, 1)
            gg = np.linspace(g.absz.min(), g.absz.max(), 20)
            ax.plot(gg, np.polyval(b, gg), color=col, lw=1.4, label=lab)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
        ax.set_xlabel(f"|depth| from bilayer centre (Å)\ncontacts × depth interaction "
                      f"{fmt_p(db['p_interaction'])}", fontsize=6.5)
    ax.set_ylabel("Median missense effect")
    ax.set_title("(e) Depth and burial, together", loc="left", fontsize=8)
    _tag(ax, db)

    ax = axg[1, 2]
    mat = res["interface_mat"]
    if mat is not None and mat.notna().to_numpy().any():
        vals = mat.to_numpy(float)
        im = ax.imshow(np.ma.masked_invalid(vals), cmap=P.fitness_cmap(), norm=P.fitness_norm(),
                       interpolation="nearest")
        ax.set_xticks(range(len(mat))); ax.set_xticklabels(mat.columns, fontsize=5, rotation=90)
        ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index, fontsize=5)
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.045).set_label("median effect at the interface", fontsize=6)
        ax.set_xlabel(f"{int(mat.notna().to_numpy().sum() / 2)} helix pairs in contact", fontsize=6.5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "no helix-helix interface with enough positions", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=P.MUTED)
    ax.set_title("(f) Helix–helix interfaces", loc="left", fontsize=8)

    P.title(fig, cfg, "A24 structural determinants of tolerance")
    return P.save(fig, figdir, "a24_structural_tolerance", cfg, "A24")


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    if not [s for s in (cfg.get_path("topology.segments") or []) if s.get("type") == "TM"]:
        return skipped("a24", cfg, "no TM segments - run `mpdms tmhmm` first")
    if cfg.get_path("structure.membrane_normal", "none") == "none":
        return skipped("a24", cfg, "structure.membrane_normal is none - depth and radius undefined")
    try:
        d, rt, pairs, info, med_radial = prepared(df, cfg)
    except (FileNotFoundError, ValueError) as e:
        return skipped("a24", cfg, str(e))
    if not len(d):
        return skipped("a24", cfg, "no missense variants map onto the structure")

    ht, hs = helix_burial(d, rt)
    itab, imat = interface_matrix(d, pairs, cfg)
    res = {"helix_table": ht, "helix_burial": hs, "facing_table": facing_table(d),
           "facing_tests": facing_tests(d), "cavity": cavity_vs_esm(d, outdir),
           "contact_order": contact_order_test(d), "depth_burial": depth_burial(d),
           "interface_table": itab, "interface_mat": imat}
    figs = figure(cfg, d, rt, res, figdir)

    t1 = tabdir / "a24_helix_burial.csv"; ht.to_csv(t1, index=False)
    t2 = tabdir / "a24_facing.csv"
    res["facing_table"].merge(res["facing_tests"], on="sub_class", how="outer").to_csv(t2, index=False)
    t3 = tabdir / "a24_interfaces.csv"; itab.to_csv(t3, index=False)
    t4 = tabdir / "a24_positions.csv"
    (rt[["pos", "aa", "zdepth", "absz", "radial", "rsa", "contacts", "contact_order", "in_slab", "face"]]
     .to_csv(t4, index=False))

    counts = d[d.in_slab].face.value_counts().to_dict()
    caveats = [f"facing classes come from a monomer: accessible TM positions face lipid or the "
               f"cavity, split at the median radius of {med_radial:.1f} Å, with RSA < {RSA_OPEN} "
               "counted as protein-facing",
               f"membrane frame: {info.get('method')}; the {SLAB:g} Å slab and the RSA cut are "
               "conventions, so check that a result near either boundary survives moving it",
               "tests use position medians, not variants, since variants at one position are not "
               "independent"]
    keys = ["helix_burial", "cavity", "contact_order", "depth_burial"]
    tested = [k for k in keys if res[k].get("testable")]
    met = [k for k in tested if res[k].get("passed")]
    ftests = res["facing_tests"]
    head = (f"{len(met)}/{len(tested)} structural predictions met"
            + (f" ({', '.join(met)})" if met else "")
            + (f"; facing: {int((ftests.q < 0.05).sum())}/{len(ftests)} contrasts significant"
               if len(ftests) else "")
            + (f"; TM positions {counts}" if counts else ""))
    return result("a24", cfg, head,
                  {"helix_burial": hs, "cavity": res["cavity"], "contact_order": res["contact_order"],
                   "depth_burial": res["depth_burial"], "median_radial_A": med_radial,
                   "face_counts": counts, "membrane_frame": info.get("method"),
                   "facing": res["facing_table"].to_dict("records"),
                   "facing_tests": ftests.to_dict("records"),
                   "interfaces": itab.to_dict("records")},
                  caveats, figs, [t1, t2, t3, t4])
