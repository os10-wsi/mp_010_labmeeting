"""`python -m mpdms arch configs/AQR1_fitness.yaml configs/HXT1.yaml configs/HXT3.yaml`

Ten structural questions about a membrane protein, each asked within one protein.

Everything here is a property of where a residue sits in its own fold and of what a
substitution does to it chemically. Nothing is pooled across proteins: these assays
differ in dynamic range, and one of them (HXT1) was normalised against a wild-type
anchor rather than synonymous variants, so absolute effect sizes are not comparable
between panels even though ranks within a panel are.

    s01  helix-helix interface map: which packing interfaces are load-bearing
    s02  small residues at interfaces: is there room to make them bigger?
    s03  proline, both directions: introducing one, and removing a native kink
    s04  contact order: holding your own helix against holding the fold together
    s05  packing strain: is too big worse than too small?
    s06  salt bridges: do partners matter, and is reversal worse than a swap?
    s07  pseudo-symmetry: does the fold treat its two bundles the same way?
    s08  how far constraint travels through space, in angstroms
    s09  which burial measure actually predicts tolerance
    s10  shells outward from the translocation pathway
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .analyses.base import missense
from .config import REPO_ROOT, load_config
from .environs import effect_color, grid, panel, protein_frame
from .io import load_dataset
from .structure import contact_pairs, load_structure, membrane_frame, residue_table

OUT = REPO_ROOT / "outputs" / "_arch"
SALT_BRIDGE = 4.0        # A between a carboxyl oxygen and a basic nitrogen
HBOND = 3.5              # A between heavy-atom donor and acceptor
MIN_GROUP = 15           # variants, before a group gets a number


def protein_bits(cfg) -> dict:
    """Everything the ten analyses share, loaded once per protein."""
    df = load_dataset(cfg)
    var, pos = protein_frame(cfg)
    sp = cfg.resolve(cfg.get_path("structure.path"))
    if sp is None or not sp.exists():
        return {}
    off = int(cfg.get_path("structure.numbering_offset", 0) or 0)
    rt = residue_table(sp, cfg.get_path("structure.chain", "A"), off)
    rt, frame = membrane_frame(rt, cfg)
    helix = {}
    for s in (cfg.get_path("topology.segments", []) or []):
        if s.get("type") == "TM":
            for p in range(int(s["start"]), int(s["end"]) + 1):
                helix[p] = s["name"]
    pairs = contact_pairs(sp, cfg.get_path("structure.chain", "A"), off)
    return {"cfg": cfg, "df": df, "var": var, "pos": pos, "rt": rt, "pairs": pairs,
            "helix": helix, "path": sp, "offset": off, "frame": frame,
            "effect": dict(zip(pos.pos, pos.effect))}


def tm_order(bits: dict) -> list[str]:
    """TM helix names in sequence order."""
    segs = [s for s in (bits["cfg"].get_path("topology.segments", []) or [])
            if s.get("type") == "TM"]
    return [s["name"] for s in sorted(segs, key=lambda s: s["start"])]


def interface_pairs(bits: dict) -> pd.DataFrame:
    """Contacts between residues in two DIFFERENT TM helices."""
    p = bits["pairs"].copy()
    if not len(p):
        return p
    p["helix_i"] = p.pos_i.map(bits["helix"])
    p["helix_j"] = p.pos_j.map(bits["helix"])
    p = p.dropna(subset=["helix_i", "helix_j"])
    return p[p.helix_i != p.helix_j].reset_index(drop=True)


# ------------------------------------------------- s01 helix-helix interface map
def s01_interfaces(bits: dict, names: list[str], outdir: Path) -> dict:
    """Mean effect at the residues forming each helix-helix packing interface.

    A transporter is held together by a handful of these. The matrix says which, and
    the diagonal is each helix on its own, so an interface can be read against the
    helices that make it.
    """
    rows = []
    fig, pairs_ax, axes = grid(names, w=3.3, h=3.1, extra=0.3)
    im = None
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        order = tm_order(b)
        ip = interface_pairs(b)
        eff = b["effect"]
        m = np.full((len(order), len(order)), np.nan)
        idx = {h: k for k, h in enumerate(order)}
        for (hi, hj), sub in ip.groupby(["helix_i", "helix_j"]):
            if hi not in idx or hj not in idx:
                continue
            res = set(sub.pos_i) | set(sub.pos_j)
            v = [eff[p] for p in res if p in eff]
            if len(v) < 4:
                continue
            a, c = idx[hi], idx[hj]
            m[a, c] = m[c, a] = float(np.mean(v))
            rows.append({"protein": g, "helix_a": hi, "helix_b": hj,
                         "n_contacts": int(len(sub)), "n_residues": int(len(res)),
                         "n_scored": int(len(v)), "mean_effect": float(np.mean(v))})
        for h, k in idx.items():                      # diagonal: the helix itself
            v = [e for p, e in eff.items() if b["helix"].get(p) == h]
            if len(v) >= 4:
                m[k, k] = float(np.mean(v))
        im = ax.imshow(m, cmap=P.fitness_cmap(), norm=P.fitness_norm())
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=6)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6)
        d = [r for r in rows if r["protein"] == g]
        worst = min(d, key=lambda r: r["mean_effect"]) if d else None
        panel(ax, i, g, (f"{len(d)} interfaces, worst {worst['helix_a']}–"
                         f"{worst['helix_b']} {worst['mean_effect']:.2f}") if worst else "",
              size=7.5)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01).set_label(
            "Mean effect at the interface", fontsize=8, fontweight="bold")
    fig.suptitle("Off-diagonal: residues in contact between two different TM helices. "
                 "Diagonal: the whole helix.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "s01_interface_map", None, "S01")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "s01_interfaces.csv", index=False)
    head = "; ".join(f"{g.replace('_fitness','')} {len(d)} interfaces, worst "
                     f"{d.loc[d.mean_effect.idxmin(), 'helix_a']}–"
                     f"{d.loc[d.mean_effect.idxmin(), 'helix_b']} "
                     f"{d.mean_effect.min():.2f}" for g, d in t.groupby("protein"))
    return {"status": "ok", "headline": head}


# --------------------------------------- s02 small residues at packing interfaces
SMALL = set("GAS")


def s02_small_residues(bits: dict, names: list[str], outdir: Path) -> dict:
    """Is a small residue at an interface intolerant of being made bigger?

    Glycine and alanine at a helix-helix interface are what lets two helices approach.
    If that is why they are there, enlarging them should cost at an interface and cost
    much less anywhere else, and the comparison is the slope of effect on the change in
    side-chain volume.
    """
    rows = []
    fig, pairs_ax, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        ip = interface_pairs(b)
        at_iface = set(ip.pos_i) | set(ip.pos_j)
        v = b["var"].dropna(subset=["score_z", "delta_volume"])
        v = v[v.wt.isin(list(SMALL)) & (v.is_tm > 0.5)]
        for lab, sub, col in (("at an interface", v[v.pos.isin(at_iface)], P.DARK_RED),
                              ("elsewhere in a helix",
                               v[~v.pos.isin(at_iface)], "#2166AC")):
            if len(sub) < MIN_GROUP * 2:
                continue
            bn = pd.qcut(sub.delta_volume, 5, duplicates="drop")
            m = sub.groupby(bn, observed=True).agg(x=("delta_volume", "mean"),
                                                   y=("score_z", "mean"),
                                                   e=("score_z", "sem")).dropna()
            ax.errorbar(m.x, m.y, yerr=m.e, fmt="-o", ms=4, lw=1.3, color=col, capsize=2,
                        label=f"{lab} (n = {len(sub):,})", mec=P.INK, mew=0.4)
            r = ss.linregress(sub.delta_volume, sub.score_z)
            rows.append({"protein": g, "group": lab, "slope_per_A3": float(r.slope),
                         "se": float(r.stderr), "p": float(r.pvalue), "n": int(len(sub)),
                         "n_positions": int(sub.pos.nunique())})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (1, 2)))
        d = [r for r in rows if r["protein"] == g]
        panel(ax, i, g, "  ".join(f"{r['group'].split()[0]} {r['slope_per_A3']:+.4f}"
                                  for r in d), size=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
        if i % 2 == 0:
            ax.set_ylabel("mean effect", fontsize=7.5)
        ax.set_xlabel("Δ side-chain volume (Å³)", fontsize=7.5)
    fig.suptitle("Wild-type glycine, alanine or serine inside a TM helix. The claim is a "
                 "steeper fall to the right at interfaces: no room to grow.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "s02_small_residue_packing", None, "S02")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "s02_small_residues.csv", index=False)
    head = "; ".join(f"{g.replace('_fitness','')} "
                     + " vs ".join(f"{r.slope_per_A3:+.4f}" for r in d.itertuples())
                     for g, d in t.groupby("protein"))
    return {"status": "ok", "headline": f"slope per Å³, interface vs elsewhere: {head}"}


# ------------------------------------------------------- s03 proline, both ways
def s03_proline(bits: dict, names: list[str], outdir: Path) -> dict:
    """Introducing a proline into a helix, and removing one that is already there.

    Both directions, because they ask different things. A proline put into the middle
    of a transmembrane helix breaks its backbone hydrogen bonding and should be close
    to null. A proline already in a transmembrane helix usually marks a kink the fold
    is built around, and removing it should cost more than removing an ordinary
    residue there.

    Introducing is measured against the position's own other substitutions, so a
    proline at an intolerant position is not credited with that position's intolerance.
    """
    rows, native = [], []
    fig, pairs_ax, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        v = b["var"].dropna(subset=["score_z"])
        tm = v[v.is_tm > 0.5]
        if len(tm) < 50:
            continue
        other = tm[tm.mut != "P"].groupby("pos").score_z.mean()
        pro = tm[tm.mut == "P"].set_index("pos").score_z
        both = pd.concat([pro.rename("pro"), other.rename("rest")], axis=1).dropna()
        both = both.join(b["pos"].set_index("pos")[["abs_rel", "rsa"]], how="left")
        both["excess"] = both.pro - both.rest
        if len(both) >= 10:
            bn = pd.cut(both.abs_rel, [0, 0.25, 0.5, 0.75, 1.0], include_lowest=True)
            m = both.groupby(bn, observed=True).agg(x=("abs_rel", "mean"),
                                                    y=("excess", "mean"),
                                                    e=("excess", "sem"),
                                                    n=("excess", "size")).dropna()
            ax.errorbar(m.x, m.y, yerr=m.e, fmt="-o", ms=5, lw=1.6, color=P.DARK_RED,
                        capsize=2, mec=P.INK, mew=0.4)
            rows.append({"protein": g, "n_positions": int(len(both)),
                         "excess_mid_helix": float(both.loc[both.abs_rel < 0.5,
                                                            "excess"].mean()),
                         "excess_helix_end": float(both.loc[both.abs_rel >= 0.5,
                                                            "excess"].mean()),
                         "p": float(ss.mannwhitneyu(both.loc[both.abs_rel < 0.5, "excess"],
                                                    both.loc[both.abs_rel >= 0.5,
                                                             "excess"]).pvalue)
                         if both.abs_rel.lt(0.5).sum() > 3
                         and both.abs_rel.ge(0.5).sum() > 3 else np.nan})
        ax.axhline(0, color=P.MUTED, lw=0.8)
        ax.set_xlabel("position in the helix  (0 = middle, 1 = either end)", fontsize=7)
        if i % 2 == 0:
            ax.set_ylabel("proline minus the position's\nother substitutions",
                          fontsize=7.5)

        # the other direction: positions whose wild type IS a proline
        pt = b["pos"]
        tmp = pt[(pt.is_tm > 0.5)]
        wt_at = dict(zip(b["rt"].pos, b["rt"].aa))
        tmp = tmp.assign(wt=tmp.pos.map(wt_at))
        natP = tmp[tmp.wt == "P"]
        rest = tmp[tmp.wt != "P"]
        if len(natP) >= 2 and len(rest) >= 10:
            native.append({"protein": g, "n_native_proline": int(len(natP)),
                           "mean_effect_native_P": float(natP.effect.mean()),
                           "mean_effect_other_TM": float(rest.effect.mean()),
                           "difference": float(natP.effect.mean() - rest.effect.mean()),
                           "p": float(ss.mannwhitneyu(natP.effect, rest.effect).pvalue),
                           "positions": ",".join(str(int(p)) for p in sorted(natP.pos))})
        n = native[-1] if native and native[-1]["protein"] == g else None
        panel(ax, i, g, (f"{n['n_native_proline']} native TM prolines, "
                         f"{n['difference']:+.2f} vs the rest") if n else "", size=7.5)
    fig.suptitle("Above zero: proline is better tolerated than the average substitution "
                 "at the same position. Below: worse.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "s03_proline", None, "S03")
    plt.close(fig)
    t, nt = pd.DataFrame(rows), pd.DataFrame(native)
    t.to_csv(outdir / "s03_proline_introduced.csv", index=False)
    nt.to_csv(outdir / "s03_proline_native.csv", index=False)
    head = "; ".join(f"{r.protein.replace('_fitness','')} mid {r.excess_mid_helix:+.3f} "
                     f"vs end {r.excess_helix_end:+.3f}" for r in t.itertuples())
    if len(nt):
        head += " | native P: " + "; ".join(
            f"{r.protein.replace('_fitness','')} {r.n_native_proline} sites, "
            f"{r.difference:+.2f}" for r in nt.itertuples())
    return {"status": "ok", "headline": head, "native": nt.to_dict("records")}


# ----------------------------------------------- s04 local against long-range contacts
def s04_contact_order(bits: dict, names: list[str], outdir: Path) -> dict:
    """Does holding the fold together matter more than holding your own helix?

    Contact number counts both. Splitting it says which kind of packing a position is
    paying for: contacts inside its own TM helix, or contacts to a different one. The
    two are fitted together, so each coefficient is the effect of one at a fixed value
    of the other.
    """
    import statsmodels.formula.api as smf
    rows = []
    fig, pairs_ax, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        p = b["pairs"]
        if not len(p):
            continue
        hel = b["helix"]
        same, other = {}, {}
        for a, c in zip(p.pos_i, p.pos_j):
            ha, hc = hel.get(a), hel.get(c)
            d = same if (ha is not None and ha == hc) else other
            d[a] = d.get(a, 0) + 1
            d[c] = d.get(c, 0) + 1
        t = b["pos"].copy()
        t["n_local"] = t.pos.map(same).fillna(0)
        t["n_distant"] = t.pos.map(other).fillna(0)
        t = t[(t.is_tm > 0.5)].dropna(subset=["effect"])
        if len(t) < 30:
            continue
        for lab, col, c in (("contacts in its own helix", "n_local", "#2166AC"),
                            ("contacts to another helix", "n_distant", P.DARK_RED)):
            bn = pd.qcut(t[col], 5, duplicates="drop")
            m = t.groupby(bn, observed=True).agg(x=(col, "mean"), y=("effect", "mean"),
                                                 e=("effect", "sem")).dropna()
            ax.errorbar(m.x, m.y, yerr=m.e, fmt="-o", ms=4, lw=1.3, color=c, capsize=2,
                        label=lab, mec=P.INK, mew=0.4)
        f = smf.ols("effect ~ n_local + n_distant", data=t).fit()
        rows.append({"protein": g, "n_positions": int(len(t)),
                     "beta_local": float(f.params["n_local"]),
                     "se_local": float(f.bse["n_local"]),
                     "p_local": float(f.pvalues["n_local"]),
                     "beta_distant": float(f.params["n_distant"]),
                     "se_distant": float(f.bse["n_distant"]),
                     "p_distant": float(f.pvalues["n_distant"]),
                     "corr_local_distant": float(t.n_local.corr(t.n_distant))})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        r = rows[-1]
        panel(ax, i, g, f"own {r['beta_local']:+.4f}, other {r['beta_distant']:+.4f}",
              size=7.5)
        ax.set_xlabel("number of contacts", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect at the position", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("Fitted together, so each coefficient holds the other kind of contact "
                 "fixed.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "s04_contact_order", None, "S04")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "s04_contact_order.csv", index=False)
    head = "; ".join(f"{r.protein.replace('_fitness','')} own {r.beta_local:+.4f} "
                     f"(p {r.p_local:.2g}), other {r.beta_distant:+.4f} "
                     f"(p {r.p_distant:.2g})" for r in t.itertuples())
    return {"status": "ok", "headline": head}


# --------------------------------------------------------- s05 packing strain
def s05_packing_strain(bits: dict, names: list[str], outdir: Path) -> dict:
    """In a buried core, is making a residue bigger worse than making it smaller?

    A cavity costs the protein some packing energy; a clash costs it the geometry of
    the whole helix pair. The two slopes are fitted on either side of zero and the
    asymmetry between them is the result.
    """
    rows = []
    fig, pairs_ax, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        v = bits[g]["var"].dropna(subset=["score_z", "delta_volume", "rsa"])
        core = v[(v.is_tm > 0.5) & (v.rsa < 0.15)]
        if len(core) < 100:
            continue
        bn = pd.qcut(core.delta_volume, 8, duplicates="drop")
        m = core.groupby(bn, observed=True).agg(x=("delta_volume", "mean"),
                                                y=("score_z", "mean"),
                                                e=("score_z", "sem")).dropna()
        ax.fill_between(m.x, m.y - m.e, m.y + m.e, color=P.MUTED, alpha=0.2, lw=0)
        ax.plot(m.x, m.y, "-o", ms=4, lw=1.4, color=P.INK, mec=P.INK)
        out = {"protein": g, "n": int(len(core))}
        for side, sub, col in (("smaller", core[core.delta_volume < 0], "#2166AC"),
                               ("bigger", core[core.delta_volume > 0], P.DARK_RED)):
            if len(sub) < MIN_GROUP * 2:
                continue
            r = ss.linregress(sub.delta_volume, sub.score_z)
            xs = np.linspace(sub.delta_volume.min(), sub.delta_volume.max(), 10)
            ax.plot(xs, r.intercept + r.slope * xs, color=col, lw=1.8,
                    ls=(0, (4, 3)), label=f"{side}: {r.slope:+.4f} per Å³")
            out[f"slope_{side}"] = float(r.slope)
            out[f"se_{side}"] = float(r.stderr)
            out[f"n_{side}"] = int(len(sub))
        if "slope_bigger" in out and "slope_smaller" in out:
            # the asymmetry: how much steeper the cost of growing is than of shrinking
            out["asymmetry"] = float(abs(out["slope_bigger"]) - abs(out["slope_smaller"]))
            out["z_asymmetry"] = float(
                (abs(out["slope_bigger"]) - abs(out["slope_smaller"]))
                / np.hypot(out["se_bigger"], out["se_smaller"]))
        rows.append(out)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.INK, lw=0.8, ls=(0, (1, 2)))
        panel(ax, i, g, f"asymmetry z = {out.get('z_asymmetry', float('nan')):+.1f}",
              size=7.5)
        ax.set_xlabel("Δ side-chain volume (Å³)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("Buried transmembrane positions only (RSA < 0.15). A steeper line to "
                 "the right than the left means a clash costs more than a cavity.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "s05_packing_strain", None, "S05")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "s05_packing_strain.csv", index=False)
    head = "; ".join(f"{r.protein.replace('_fitness','')} bigger "
                     f"{getattr(r, 'slope_bigger', float('nan')):+.4f} vs smaller "
                     f"{getattr(r, 'slope_smaller', float('nan')):+.4f} "
                     f"(z {getattr(r, 'z_asymmetry', float('nan')):+.1f})"
                     for r in t.itertuples())
    return {"status": "ok", "headline": head}


# ------------------------------------------------------------ s06 salt bridges
ACIDIC = {"D": ("OD1", "OD2"), "E": ("OE1", "OE2")}
BASIC = {"K": ("NZ",), "R": ("NE", "NH1", "NH2"), "H": ("ND1", "NE2")}


def salt_bridges(path: Path, chain: str = "A", offset: int = 0,
                 cutoff: float = SALT_BRIDGE) -> pd.DataFrame:
    """Acidic-basic side-chain pairs close enough to be a salt bridge.

    Charged-group atoms only, not centroids: a lysine whose backbone is near a
    glutamate is not a salt bridge, and the distinction is the whole point of looking.
    """
    from Bio.PDB.Polypeptide import index_to_one, three_to_index
    _, _, ch = load_structure(Path(path), chain)
    acid, base = [], []
    for res in ch:
        if res.id[0] != " ":
            continue
        try:
            aa = index_to_one(three_to_index(res.get_resname()))
        except (KeyError, ValueError):
            continue
        pos = res.id[1] + offset
        for names, store in ((ACIDIC.get(aa), acid), (BASIC.get(aa), base)):
            if not names:
                continue
            for n in names:
                if n in res:
                    store.append((pos, aa, np.asarray(res[n].coord, float)))
    rows = []
    for pa, aaa, ca in acid:
        for pb, aab, cb in base:
            if pa == pb:
                continue
            d = float(np.linalg.norm(ca - cb))
            if d <= cutoff:
                rows.append({"pos_acid": pa, "aa_acid": aaa, "pos_base": pb,
                             "aa_base": aab, "distance": d})
    out = pd.DataFrame(rows)
    if len(out):           # one row per residue pair, at its closest approach
        out = (out.sort_values("distance")
               .drop_duplicates(subset=["pos_acid", "pos_base"]).reset_index(drop=True))
    return out


def s06_salt_bridges(bits: dict, names: list[str], outdir: Path) -> dict:
    """Do salt-bridge partners matter, and is reversing a charge worse than keeping it?

    Two comparisons. Charged residues that have a partner against charged residues that
    do not, which asks whether the pairing is doing the work rather than the charge.
    Then, at the paired positions only, substitutions that keep the charge (K to R)
    against substitutions that reverse or remove it, which asks the same question from
    the mutation's side.
    """
    rows, subs, found = [], [], {}
    fig, pairs_ax, _ = grid(names, w=3.1, h=2.9, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        sb = salt_bridges(b["path"], b["cfg"].get_path("structure.chain", "A"),
                          b["offset"])
        found[g] = sb
        partnered = set(sb.pos_acid) | set(sb.pos_base) if len(sb) else set()
        wt_at = dict(zip(b["rt"].pos, b["rt"].aa))
        pos = b["pos"].assign(wt=b["pos"].pos.map(wt_at)).dropna(subset=["effect"])
        charged = pos[pos.wt.isin(list("DEKRH"))]
        groups = {"in a salt bridge": charged[charged.pos.isin(partnered)].effect,
                  "charged, unpaired": charged[~charged.pos.isin(partnered)].effect,
                  "everything else": pos[~pos.wt.isin(list("DEKRH"))].effect}
        use = {k: v.to_numpy() for k, v in groups.items() if len(v) >= 5}
        if use:
            P.violin_box(ax, use, colors={k: effect_color(float(np.median(v)))
                                          for k, v in use.items()})
            ax.tick_params(axis="x", labelsize=6)
        if len(use) >= 2 and "in a salt bridge" in use and "charged, unpaired" in use:
            a, c = use["in a salt bridge"], use["charged, unpaired"]
            rows.append({"protein": g, "n_bridges": int(len(sb)),
                         "n_partnered": len(partnered),
                         "mean_partnered": float(np.mean(a)),
                         "mean_unpaired": float(np.mean(c)),
                         "difference": float(np.mean(a) - np.mean(c)),
                         "p": float(ss.mannwhitneyu(a, c).pvalue)})
        # and from the substitution's side, at the partnered positions
        v = b["var"].dropna(subset=["score_z"])
        v = v[v.pos.isin(partnered) & v.wt.isin(list("DEKRH"))]
        if len(v) >= MIN_GROUP * 2:
            keep = ((v.wt.isin(list("KRH")) & v.mut.isin(list("KRH")))
                    | (v.wt.isin(list("DE")) & v.mut.isin(list("DE"))))
            rev = ((v.wt.isin(list("KRH")) & v.mut.isin(list("DE")))
                   | (v.wt.isin(list("DE")) & v.mut.isin(list("KRH"))))
            if keep.sum() >= 5 and rev.sum() >= 5:
                subs.append({"protein": g, "n_keep": int(keep.sum()),
                             "n_reverse": int(rev.sum()),
                             "mean_keep": float(v.loc[keep, "score_z"].mean()),
                             "mean_reverse": float(v.loc[rev, "score_z"].mean()),
                             "difference": float(v.loc[rev, "score_z"].mean()
                                                 - v.loc[keep, "score_z"].mean()),
                             "p": float(ss.mannwhitneyu(v.loc[rev, "score_z"],
                                                        v.loc[keep, "score_z"]).pvalue)})
        r = rows[-1] if rows and rows[-1]["protein"] == g else None
        panel(ax, i, g, (f"{r['n_bridges']} bridges over {r['n_partnered']} residues, "
                         f"{r['difference']:+.2f}") if r else f"{len(sb)} bridges",
              size=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect at the position", fontsize=7.5)
    fig.suptitle(f"A salt bridge is a charged side-chain oxygen within "
                 f"{SALT_BRIDGE:.0f} Å of a charged nitrogen in the AlphaFold model.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "s06_salt_bridges", None, "S06")
    plt.close(fig)
    t, st = pd.DataFrame(rows), pd.DataFrame(subs)
    t.to_csv(outdir / "s06_salt_bridge_positions.csv", index=False)
    st.to_csv(outdir / "s06_charge_substitutions.csv", index=False)
    pd.concat([d.assign(protein=g) for g, d in found.items() if len(d)],
              ignore_index=True).to_csv(outdir / "s06_bridges.csv", index=False) \
        if any(len(d) for d in found.values()) else None
    head = "; ".join(f"{r.protein.replace('_fitness','')} paired−unpaired "
                     f"{r.difference:+.2f} (p {r.p:.2g}, {r.n_bridges} bridges)"
                     for r in t.itertuples())
    if len(st):
        head += " | reversal−conservative: " + "; ".join(
            f"{r.protein.replace('_fitness','')} {r.difference:+.2f}"
            for r in st.itertuples())
    return {"status": "ok", "headline": head}


# --------------------------------------------------------- s07 pseudo-symmetry
def kabsch(a: np.ndarray, b: np.ndarray):
    """Rotation and translation that best put `a` onto `b`, with the RMSD."""
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    moved = (a - ca) @ r.T + cb
    return r, ca, cb, float(np.sqrt(((moved - b) ** 2).sum(1).mean()))


def superpose_bundles(bits: dict, n_first: int = 6) -> dict:
    """Superpose the first half of the TM helices onto the second half.

    The two bundles of an MFS transporter are an inverted repeat, and the textbook
    pairing is TM1 with TM7, TM2 with TM8 and so on. Rather than assume it, every
    one-to-one pairing of the six against the six is tried, each scored by the RMSD of
    a Kabsch fit on the helix centroids, and the best is kept. The canonical pairing's
    own RMSD is reported next to it, so the figure can say whether the textbook answer
    is what the structure gives.

    Residues are then matched inside each paired helix by their rank along the helix,
    which is what makes two helices of different length comparable at all; the CA RMSD
    of that residue-level correspondence is what to judge the whole thing by.
    """
    from itertools import permutations
    order = tm_order(bits)
    if len(order) < 2 * n_first:
        return {}
    first, second = order[:n_first], order[n_first:2 * n_first]
    rt = bits["rt"].set_index("pos")
    members = {h: sorted(p for p, hh in bits["helix"].items() if hh == h) for h in order}
    members = {h: [p for p in v if p in rt.index] for h, v in members.items()}
    if any(len(members[h]) < 5 for h in first + second):
        return {}
    cen = {h: rt.loc[members[h], ["x", "y", "z"]].to_numpy().mean(0) for h in order}
    best = None
    for perm in permutations(range(n_first)):
        a = np.array([cen[first[k]] for k in range(n_first)])
        b = np.array([cen[second[perm[k]]] for k in range(n_first)])
        _, _, _, rms = kabsch(a, b)
        if best is None or rms < best[0]:
            best = (rms, perm)
    canon = tuple(range(n_first))
    a = np.array([cen[h] for h in first])
    rms_canon = kabsch(a, np.array([cen[second[k]] for k in canon]))[3]
    rms, perm = best
    pairing = [(first[k], second[perm[k]]) for k in range(n_first)]

    # residue-level correspondence inside each paired helix, by rank along the helix
    pa, pb = [], []
    for ha, hb in pairing:
        ia, ib = members[ha], members[hb]
        n = min(len(ia), len(ib))
        qa = np.linspace(0, len(ia) - 1, n).round().astype(int)
        qb = np.linspace(0, len(ib) - 1, n).round().astype(int)
        pa += [ia[k] for k in qa]
        pb += [ib[k] for k in qb]
    ca = rt.loc[pa, ["x", "y", "z"]].to_numpy()
    cb = rt.loc[pb, ["x", "y", "z"]].to_numpy()
    _, _, _, rms_res = kabsch(ca, cb)
    return {"pairing": pairing, "rmsd_centroid": rms, "rmsd_centroid_canonical": rms_canon,
            "rmsd_residue": rms_res, "canonical": perm == canon,
            "pos_a": pa, "pos_b": pb}


def s07_pseudo_symmetry(bits: dict, names: list[str], outdir: Path) -> dict:
    """Does the fold treat its two bundles the same way?

    The only internal replicate this data has: two halves of one protein, built on the
    same fold, measured in the same experiment. Where they agree, constraint follows
    the architecture. Where they do not, something other than the fold is at work, and
    that is usually function.
    """
    rows, pairs_out = [], []
    fig, pairs_ax, _ = grid(names, w=3.1, h=3.0, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        sup = superpose_bundles(b)
        if not sup:
            panel(ax, i, g, "fewer than twelve TM helices", size=7.5)
            continue
        eff = b["effect"]
        d = pd.DataFrame({"pos_a": sup["pos_a"], "pos_b": sup["pos_b"]})
        d["effect_a"] = d.pos_a.map(eff)
        d["effect_b"] = d.pos_b.map(eff)
        d = d.dropna()
        if len(d) < 20:
            panel(ax, i, g, "too few positions measured in both halves", size=7.5)
            continue
        lim = [min(d.effect_a.min(), d.effect_b.min()) - 0.05,
               max(d.effect_a.max(), d.effect_b.max()) + 0.05]
        ax.plot(lim, lim, ls=(0, (5, 4)), color=P.INK, lw=0.9)
        ax.scatter(d.effect_a, d.effect_b, s=14, c=[effect_color((x + y) / 2)
                                                    for x, y in zip(d.effect_a, d.effect_b)],
                   edgecolor=P.INK, linewidth=0.3)
        rho, p = ss.spearmanr(d.effect_a, d.effect_b)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel("first bundle (TM1–6)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("second bundle (TM7–12)", fontsize=7.5)
        panel(ax, i, g, f"ρ = {rho:+.2f} (p = {p:.2g}), CA RMSD "
                        f"{sup['rmsd_residue']:.1f} Å", size=7.5)
        rows.append({"protein": g, "n_pairs": int(len(d)), "spearman": float(rho),
                     "p": float(p), "pearson": float(ss.pearsonr(d.effect_a, d.effect_b)[0]),
                     "rmsd_residue": sup["rmsd_residue"],
                     "rmsd_centroid": sup["rmsd_centroid"],
                     "rmsd_centroid_canonical": sup["rmsd_centroid_canonical"],
                     "canonical_pairing": bool(sup["canonical"]),
                     "pairing": " ".join(f"{a}:{c}" for a, c in sup["pairing"])})
        pairs_out.append(d.assign(protein=g))
    fig.suptitle("Structurally equivalent positions in the two halves of the fold, "
                 "paired by a superposition searched over all helix correspondences.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "s07_pseudo_symmetry", None, "S07")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "s07_pseudo_symmetry.csv", index=False)
    if pairs_out:
        pd.concat(pairs_out, ignore_index=True).to_csv(
            outdir / "s07_paired_positions.csv", index=False)
    head = "; ".join(f"{r.protein.replace('_fitness','')} ρ {r.spearman:+.2f}, RMSD "
                     f"{r.rmsd_residue:.1f} Å, "
                     + ("canonical pairing" if r.canonical_pairing
                        else "NON-canonical pairing") for r in t.itertuples())
    return {"status": "ok", "headline": head}


# ------------------------------------- s08 how far constraint travels through space
DIST_BINS = np.array([0, 6, 8, 10, 12, 15, 18, 22, 26, 30, 40])
N_PERM_SPATIAL = 2000


def correlogram(xyz: np.ndarray, v: np.ndarray, bins: np.ndarray = DIST_BINS):
    """Correlation of the effect between pairs of residues, by how far apart they are.

    The centred product of two positions' effects, averaged within a distance bin and
    divided by the variance: 1 if positions that far apart always agree, 0 if knowing
    one says nothing about the other. Reading where it crosses zero gives a length
    scale in angstroms rather than a yes or no.
    """
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    c = v - v.mean()
    prod = np.outer(c, c) / (v.var() or 1.0)
    iu = np.triu_indices(len(v), 1)
    dd, pp = d[iu], prod[iu]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (dd >= lo) & (dd < hi)
        if m.sum() >= 20:
            out.append({"lo": float(lo), "hi": float(hi), "mid": float((lo + hi) / 2),
                        # NOT named "corr": that shadows DataFrame.corr, and attribute
                        # access then hands back the method with no error at all
                        "r": float(pp[m].mean()), "n_pairs": int(m.sum())})
    return pd.DataFrame(out)


def morans_i(xyz: np.ndarray, v: np.ndarray, cutoff: float = 10.0) -> float:
    """Moran's I with a simple neighbourhood: 1 within the cutoff, 0 beyond."""
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    w = ((d <= cutoff) & (d > 0)).astype(float)
    s = w.sum()
    if s == 0:
        return np.nan
    c = v - v.mean()
    denom = (c ** 2).sum()
    return float(len(v) / s * (w * np.outer(c, c)).sum() / denom) if denom else np.nan


def s08_spatial(bits: dict, names: list[str], outdir: Path) -> dict:
    """Is constraint a property of regions or of individual residues?

    If sensitive positions come in patches, the correlation between two positions'
    effects should fall away with the distance between them and reach zero at some
    characteristic size. If sensitivity is a property of each residue on its own,
    the curve is flat at zero from the start.

    The null shuffles the effects among the same positions, which keeps the structure
    and the distribution of effects and destroys only their arrangement in space - and
    it is done separately within transmembrane and non-transmembrane positions, since
    TM residues are both closer together and less tolerant, and that alone would
    manufacture a correlation.
    """
    rows, curves = [], []
    rng = np.random.default_rng(8)
    fig, pairs_ax, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        t = b["pos"].dropna(subset=["effect"]).merge(
            b["rt"][["pos", "x", "y", "z"]], on="pos", how="inner")
        if len(t) < 50:
            continue
        xyz = t[["x", "y", "z"]].to_numpy(float)
        v = t.effect.to_numpy(float)
        strata = (t.is_tm > 0.5).to_numpy()
        obs = correlogram(xyz, v)
        null = []
        for _ in range(200):          # the band: a correlogram is the expensive part
            s = v.copy()
            for m in (strata, ~strata):
                if m.sum() > 1:
                    s[m] = rng.permutation(s[m])
            null.append(correlogram(xyz, s).set_index("mid")["r"])
        nb = pd.DataFrame(null)
        ax.fill_between(nb.columns, nb.quantile(0.025), nb.quantile(0.975),
                        color=P.MUTED, alpha=0.25, lw=0, label="null, 95%")
        ax.plot(obs.mid, obs["r"], "-o", ms=4, lw=1.6, color=P.DARK_RED, mec=P.INK,
                mew=0.4, label="observed")
        ax.axhline(0, color=P.INK, lw=0.7)
        # the length scale: where the observed curve first falls inside the null band
        inside = obs["r"].to_numpy() <= nb.quantile(0.975).reindex(obs.mid).to_numpy()
        scale = float(obs.mid[np.argmax(inside)]) if inside.any() else np.nan
        mi = morans_i(xyz, v)
        mi_null = np.array([morans_i(xyz, rng.permutation(v))
                            for _ in range(N_PERM_SPATIAL // 20)])
        rows.append({"protein": g, "n_positions": int(len(t)), "morans_i": mi,
                     "morans_i_null_mean": float(mi_null.mean()),
                     "p_morans": float(((mi_null >= mi).sum() + 1) / (len(mi_null) + 1)),
                     "length_scale_A": scale,
                     "corr_at_6A": float(obs["r"].iloc[0]) if len(obs) else np.nan})
        curves.append(obs.assign(protein=g))
        panel(ax, i, g, f"Moran's I = {mi:+.3f} (p = {rows[-1]['p_morans']:.3f}), "
                        f"reaches the null at ~{scale:.0f} Å", size=7.5)
        ax.set_xlabel("distance between two residues (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("correlation of their effects", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("The null shuffles effects among positions within transmembrane and "
                 "non-transmembrane separately, so the topology cannot manufacture a "
                 "patch.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "s08_spatial_scale", None, "S08")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "s08_spatial.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(outdir / "s08_correlogram.csv",
                                                    index=False)
    head = "; ".join(f"{r.protein.replace('_fitness','')} I {r.morans_i:+.3f} "
                     f"(p {r.p_morans:.3f}), ~{r.length_scale_A:.0f} Å"
                     for r in t.itertuples())
    return {"status": "ok", "headline": head}


# ---------------------------------------- s09 which burial measure predicts tolerance
def weighted_contacts(xyz: np.ndarray) -> np.ndarray:
    """Weighted contact number: every other residue counts, by one over its distance
    squared. A smoother burial measure than a cutoff count, with no threshold to pick."""
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return (1.0 / d ** 2).sum(1)


MEASURES = [("rsa", "solvent accessibility"), ("contacts", "contact number"),
            ("wcn", "weighted contact number"), ("abs_z", "depth |z|"),
            ("radial", "distance from the axis")]


def s09_burial_measures(bits: dict, names: list[str], outdir: Path) -> dict:
    """Five ways of saying "buried", raced against each other.

    They are all measuring something close to the same thing, so the univariate
    correlations will all look respectable and tell us little. The joint fit is the
    point: with every measure in the model at once, which coefficients survive.
    """
    import statsmodels.formula.api as smf
    rows, joint = [], []
    fig, pairs_ax, _ = grid(names, w=3.2, h=2.9, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        t = b["pos"].merge(b["rt"][["pos", "x", "y", "z"]], on="pos", how="inner")
        t = t[t.is_tm > 0.5].dropna(subset=["effect"])
        if len(t) < 40:
            continue
        t = t.assign(wcn=weighted_contacts(t[["x", "y", "z"]].to_numpy(float)))
        use = [k for k, _ in MEASURES if k in t and t[k].notna().sum() > 30]
        for k in use:
            rho, p = ss.spearmanr(t[k], t.effect, nan_policy="omit")
            rows.append({"protein": g, "measure": k, "spearman": float(rho),
                         "p": float(p), "n": int(t[k].notna().sum())})
        d = t.dropna(subset=use + ["effect"]).copy()
        for k in use:                       # standardised, so coefficients compare
            d[k] = (d[k] - d[k].mean()) / (d[k].std() or 1.0)
        if len(d) > 40:
            f = smf.ols("effect ~ " + " + ".join(use), data=d).fit()
            for k in use:
                joint.append({"protein": g, "measure": k, "beta": float(f.params[k]),
                              "se": float(f.bse[k]), "p": float(f.pvalues[k]),
                              "r2": float(f.rsquared), "n": int(len(d))})
        s = pd.DataFrame([r for r in rows if r["protein"] == g]).set_index("measure")
        j = pd.DataFrame([r for r in joint if r["protein"] == g]).set_index("measure") \
            if joint else pd.DataFrame()
        y = np.arange(len(use))[::-1]
        ax.barh(y - 0.19, [abs(s.loc[k, "spearman"]) for k in use], height=0.34,
                color=P.MUTED, edgecolor=P.INK, linewidth=0.4, label="|ρ| on its own")
        if len(j):
            ax.barh(y + 0.19, [abs(j.loc[k, "beta"]) for k in use], height=0.34,
                    color=P.DARK_RED, edgecolor=P.INK, linewidth=0.4,
                    label="|β| with the others")
            for k, yy in zip(use, y):
                if j.loc[k, "p"] < 0.05:
                    ax.text(abs(j.loc[k, "beta"]) + 0.01, yy + 0.19, "*", fontsize=8,
                            va="center")
        ax.set_yticks(y)
        ax.set_yticklabels([dict(MEASURES)[k] for k in use], fontsize=7)
        r2 = j.r2.iloc[0] if len(j) else float("nan")
        panel(ax, i, g, f"joint R² = {r2:.2f}", size=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.suptitle("Transmembrane positions only. Grey: each measure alone. Red: the same "
                 "measure standardised and fitted with the other four; * is p < 0.05.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "s09_burial_measures", None, "S09")
    plt.close(fig)
    u, j = pd.DataFrame(rows), pd.DataFrame(joint)
    u.to_csv(outdir / "s09_univariate.csv", index=False)
    j.to_csv(outdir / "s09_joint.csv", index=False)
    head = "; ".join(f"{g.replace('_fitness','')} survives: "
                     + (", ".join(d[d.p < 0.05].measure) or "none")
                     for g, d in j.groupby("protein")) if len(j) else ""
    return {"status": "ok", "headline": head}


# -------------------------------------- s10 shells outward from the central pathway
MAX_SHELL = 4


def s10_shells(bits: dict, names: list[str], outdir: Path) -> dict:
    """Tolerance by distance, in contacts, from the residues lining the pathway.

    Shell 1 is the cavity-lining residues themselves. Shell 2 is the transmembrane
    residues in contact with those and not already counted, and so outward. How fast
    sensitivity decays says which mechanism is at work: a sharp drop after the first
    shell points at contact with whatever moves through the protein, a gradual one at
    the stability of the fold.
    """
    rows = []
    fig, pairs_ax, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        pos = b["pos"].dropna(subset=["effect"])
        lining = set(pos.loc[pos.environ == "cavity-lining", "pos"])
        if len(lining) < 5:
            panel(ax, i, g, "no cavity-lining positions", size=7.5)
            continue
        adj = {}
        for a, c in zip(b["pairs"].pos_i, b["pairs"].pos_j):
            adj.setdefault(a, set()).add(c)
            adj.setdefault(c, set()).add(a)
        tm = set(pos.loc[pos.is_tm > 0.5, "pos"])
        shell, seen, cur = {}, set(lining), set(lining)
        for p in cur:
            shell[p] = 1
        for k in range(2, MAX_SHELL + 1):
            nxt = {q for p in cur for q in adj.get(p, ()) if q in tm and q not in seen}
            for q in nxt:
                shell[q] = k
            seen |= nxt
            cur = nxt
            if not cur:
                break
        t = pos.assign(shell=pos.pos.map(shell))
        groups = {}
        for k in range(1, MAX_SHELL + 1):
            v = t.loc[t.shell == k, "effect"].dropna()
            if len(v) >= 5:
                groups[str(k)] = v.to_numpy()
        far = t.loc[t.shell.isna() & (t.is_tm > 0.5), "effect"].dropna()
        if len(far) >= 5:
            groups["beyond"] = far.to_numpy()
        if not groups:
            continue
        xs = np.arange(len(groups))
        mu = [float(np.mean(v)) for v in groups.values()]
        se = [float(np.std(v, ddof=1) / np.sqrt(len(v))) for v in groups.values()]
        ax.bar(xs, mu, yerr=se, color=[effect_color(m) for m in mu], edgecolor=P.INK,
               linewidth=0.5, capsize=3)
        ax.set_xticks(xs); ax.set_xticklabels(list(groups), fontsize=7)
        ax.set_xlabel("shell from the cavity lining (contacts)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect at the position", fontsize=7.5)
        for k, (lab, v) in enumerate(groups.items()):
            rows.append({"protein": g, "shell": lab, "n": int(len(v)),
                         "mean_effect": mu[k], "se": se[k]})
        drop = (mu[1] - mu[0]) if len(mu) > 1 else np.nan
        panel(ax, i, g, f"{len(lining)} lining, shell 1→2 {drop:+.3f}", size=7.5)
    fig.suptitle("Shell 1 lines the translocation pathway; each shell outward is one "
                 "more structural contact away from it.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "s10_cavity_shells", None, "S10")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "s10_shells.csv", index=False)
    head = "; ".join(f"{g.replace('_fitness','')} " + " ".join(
        f"{r.shell}:{r.mean_effect:.2f}" for r in d.itertuples())
        for g, d in t.groupby("protein"))
    return {"status": "ok", "headline": head}


ANALYSES = {"s01": "helix-helix interface map",
            "s02": "small residues at interfaces",
            "s03": "proline, introduced and removed",
            "s04": "local against long-range contacts",
            "s05": "packing strain and its asymmetry",
            "s06": "salt bridges and charge substitutions",
            "s07": "pseudo-symmetry of the two bundles",
            "s08": "how far constraint travels in space",
            "s09": "which burial measure predicts tolerance",
            "s10": "shells outward from the pathway"}
RUN = {"s01": s01_interfaces, "s02": s02_small_residues, "s03": s03_proline,
       "s04": s04_contact_order, "s05": s05_packing_strain, "s06": s06_salt_bridges,
       "s07": s07_pseudo_symmetry, "s08": s08_spatial, "s09": s09_burial_measures,
       "s10": s10_shells}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "Degrees of freedom|All-NaN|divide).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms arch",
        description="Ten structural questions about a membrane protein, each asked "
                    "within one protein. Panels follow the order of the configs.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)
    keys = [k.strip() for k in (a.only.split(",") if a.only else ANALYSES)]
    bad = [k for k in keys if k not in ANALYSES]
    if bad:
        raise SystemExit(f"unknown analysis {bad}; available: {', '.join(ANALYSES)}")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bits, names = {}, []
    for c in a.configs:
        cfg = load_config(Path(c))
        b = protein_bits(cfg)
        if not b or not len(b.get("pos", [])):
            print(f"  {cfg.id:<16} skipped: needs a structure, a topology and a dataset",
                  flush=True)
            continue
        bits[cfg.id], _ = b, names.append(cfg.id)
        ip = interface_pairs(b)
        print(f"  {cfg.id:<16} {len(b['pos']):>4} scored positions  "
              f"{len(tm_order(b)):>2} TM helices  {len(b['pairs']):>5} contacts "
              f"({len(ip):,} between helices)"
              + ("   WILD-TYPE ANCHOR: scores not on the syn=0/stop=-1 scale"
                 if b["df"].attrs.get("normalization_fallback") else ""), flush=True)
    if not names:
        raise SystemExit("no protein could be loaded")
    rows = int(np.ceil(len(names) / a.cols))
    print("\n  panels: " + " | ".join(
        ", ".join(n.replace("_fitness", "") for n in names[r * a.cols:(r + 1) * a.cols])
        for r in range(rows)), flush=True)

    out = {}
    print()
    for k in keys:
        try:
            r = RUN[k](bits, names, outdir)
        except Exception as e:
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        out[k] = r
        print(f"  {k}  {r.get('status'):<8} {r.get('headline') or r.get('reason', '')}",
              flush=True)
    (outdir / "arch.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
