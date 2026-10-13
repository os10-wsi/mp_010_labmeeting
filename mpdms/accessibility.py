"""`python -m mpdms access configs/AQR1_fitness.yaml configs/HXT1.yaml configs/HXT3.yaml`

Ten questions about solvent accessibility, asked inside one protein.

Relative solvent accessibility is the single strongest structural predictor of
mutational tolerance in a soluble protein, and in a membrane protein it is a broken
measurement. It is computed by rolling a *water* probe over an isolated monomer, so a
residue against acyl chains and a residue in the cytosol both score "exposed" while
being in opposite chemical worlds, and in a transporter a residue lining the
water-filled cavity scores exposed as well. That is not a reason to drop RSA. It is the
reason these ten analyses exist: most of them are about what RSA is confusing with what.

    r01  the dose-response, fitted separately inside and outside the membrane
    r02  the accessible surface split into the part facing water and the part facing lipid
    r03  where the knee is, and whether the inherited RSA < 0.15 cut belongs there
    r04  your own burial against your neighbourhood's
    r05  how much of the variance RSA explains that nothing else does
    r06  burial changes the spread of effects at a position, not only the mean
    r07  which amino acids care about being buried
    r08  per helix: RSA cannot predict where RSA does not vary
    r09  the fraction of substitutions that reach stop-codon level, not the mean
    r10  burial that moves: RSA in two conformations, if a second one is available

Nothing is pooled across proteins. Every figure is one panel per protein in the order
the configs are given, and every model fitted to variants has its standard errors
clustered by position.
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .annot import MAX_ASA
from .architecture import protein_bits
from .burial import binned, cluster_ols, empty_panel, short_name, zscore
from .config import REPO_ROOT, load_config
from .environs import effect_color, grid, panel

OUT = REPO_ROOT / "outputs" / "_access"
RSA_BINS = np.array([0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 1.01])
DEAD = -0.8              # score_z at or below which a substitution is stop-like
DEFAULT_CUT = 0.15       # the RSA "buried" cut the rest of the pipeline inherits
MIN_GROUP = 15           # variants, before a group gets a number
MIN_POSITIONS = 8
N_BOOT = 400
PRED = {"rsa": "solvent accessibility", "contacts": "contact number",
        "abs_z": "depth |z|", "radial": "distance from the axis"}


# --------------------------------------------------------------------- loading
def atom_sasa(path: Path, chain: str = "A", offset: int = 0) -> pd.DataFrame:
    """Per-residue accessible area, split by where on the residue it is exposed.

    Shrake-Rupley at the atom level, then each exposed atom's area is assigned to lipid
    or to water by its own depth in the membrane frame. Rolling a water probe over a
    membrane protein and calling the answer "exposure" is the problem; this is the
    cheapest honest repair, and it needs no new structure - only the frame we already
    have.
    """
    from Bio.PDB.Polypeptide import index_to_one, three_to_index
    from Bio.PDB.SASA import ShrakeRupley
    from .structure import load_structure
    _, _, ch = load_structure(path, chain)
    ShrakeRupley().compute(ch, level="A")
    rows = []
    for res in ch:
        if res.id[0] != " " or "CA" not in res:
            continue
        try:
            aa = index_to_one(three_to_index(res.get_resname()))
        except (KeyError, ValueError):
            aa = "X"
        for a in res:
            if a.element == "H":
                continue
            rows.append({"pos": res.id[1] + offset, "aa": aa,
                         "sasa": float(getattr(a, "sasa", 0.0) or 0.0),
                         "x": float(a.coord[0]), "y": float(a.coord[1]),
                         "z": float(a.coord[2])})
    return pd.DataFrame(rows)


def split_accessibility(path: Path, cfg, chain: str = "A", offset: int = 0) -> pd.DataFrame:
    """asa_lipid and asa_water per residue, from the per-atom areas and the membrane frame.

    An exposed atom inside the bilayer slab is against acyl chains; one outside it is
    against water. The slab half-thickness is taken from the protein's own TM residues
    rather than assumed, so a thin bundle is not forced into a 30 Å membrane.
    """
    from .structure import membrane_frame, residue_table
    at = atom_sasa(path, chain, offset)
    if not len(at):
        return pd.DataFrame(columns=["pos", "asa_lipid", "asa_water", "rsa_lipid",
                                     "rsa_water"])
    rt = residue_table(path, chain, offset)
    rt, _ = membrane_frame(rt, cfg)
    # the same frame applied to atoms: project each atom onto the membrane normal
    ca = rt.dropna(subset=["zdepth"])
    if len(ca) < 10:
        return pd.DataFrame(columns=["pos", "asa_lipid", "asa_water", "rsa_lipid",
                                     "rsa_water"])
    fit = np.linalg.lstsq(
        np.c_[ca[["x", "y", "z"]].to_numpy(float), np.ones(len(ca))],
        ca.zdepth.to_numpy(float), rcond=None)[0]
    at["zdepth"] = at[["x", "y", "z"]].to_numpy(float) @ fit[:3] + fit[3]
    tm = rt[rt.pos.isin(_tm_positions(cfg))].dropna(subset=["zdepth"])
    half = float(tm.zdepth.abs().quantile(0.9)) if len(tm) else 15.0
    at["in_lipid"] = at.zdepth.abs() <= half
    g = at.groupby("pos").apply(
        lambda d: pd.Series({"aa": d.aa.iloc[0],
                             "asa_lipid": float(d.loc[d.in_lipid, "sasa"].sum()),
                             "asa_water": float(d.loc[~d.in_lipid, "sasa"].sum())}),
        include_groups=False).reset_index()
    mx = g.aa.map(MAX_ASA)
    g["rsa_lipid"] = (g.asa_lipid / mx).clip(upper=1.0)
    g["rsa_water"] = (g.asa_water / mx).clip(upper=1.0)
    g.attrs["half_thickness"] = half
    return g.drop(columns="aa")


def _tm_positions(cfg) -> set[int]:
    out = set()
    for s in (cfg.get_path("topology.segments", []) or []):
        if s.get("type") == "TM":
            out |= set(range(int(s["start"]), int(s["end"]) + 1))
    return out


def neighbour_rsa(rt: pd.DataFrame, cutoff: float = 8.0, exclude: int = 4) -> pd.Series:
    """Mean RSA of the residues packed around each one, skipping the sequence-local ones.

    The same construction as the local hydrophobicity in b09, so "is it you or your
    surroundings" is asked the same way for burial as it is for chemistry.
    """
    from scipy.spatial import cKDTree
    xyz = rt[["x", "y", "z"]].to_numpy(float)
    rsa = rt.rsa.to_numpy(float)
    pos = rt.pos.to_numpy(int)
    out = np.full(len(rt), np.nan)
    for i, nb in enumerate(cKDTree(xyz).query_ball_point(xyz, cutoff)):
        keep = [j for j in nb if abs(pos[j] - pos[i]) > exclude and np.isfinite(rsa[j])]
        if keep:
            out[i] = float(np.mean(rsa[keep]))
    return pd.Series(out, index=rt.index)


def bits_for(cfg, alt: Path | None = None) -> dict:
    """Everything the ten analyses share, loaded once per protein."""
    b = protein_bits(cfg)
    if not b:
        return {}
    rt = b["rt"].copy()
    rt["rsa_env"] = neighbour_rsa(rt)
    sp, off = b["path"], b["offset"]
    sa = split_accessibility(sp, cfg, cfg.get_path("structure.chain", "A"), off)
    extra = rt[["pos", "aa", "rsa_env"]].rename(columns={"aa": "wt_struct"})
    if len(sa):
        extra = extra.merge(sa, on="pos", how="left")
    var, pos = b["var"].copy(), b["pos"].copy()
    out = {}
    for name, t in (("var", var), ("pos", pos)):
        t = t.drop(columns=[c for c in extra.columns if c != "pos" and c in t.columns])
        out[name] = t.merge(extra, on="pos", how="left")
    var, pos = out["var"], out["pos"]
    wt = var.groupby("pos").wt.first()
    pos["wt"] = pos.pos.map(wt)
    pos["wt"] = pos.wt.where(pos.wt.notna(), pos.wt_struct)
    # the shape of each position's distribution, which the mean alone cannot carry
    # NOT named "skew" or "std": those shadow the DataFrame methods of the same name,
    # and attribute access then hands back a bound method with no error at all
    sh = var.dropna(subset=["score_z"]).groupby("pos").score_z.agg(
        sd="std", skewness=lambda s: float(ss.skew(s)) if len(s) > 2 else np.nan,
        frac_dead=lambda s: float((s <= DEAD).mean()), n_scored="size")
    pos = pos.merge(sh.reset_index(), on="pos", how="left")
    b.update(var=var, pos=pos, rt=rt, half_thickness=sa.attrs.get("half_thickness"))
    if alt is not None and Path(alt).exists():
        b["alt_rsa"] = _alt_rsa(Path(alt), cfg, off)
    return b


def _alt_rsa(path: Path, cfg, offset: int) -> pd.DataFrame:
    from .structure import residue_table
    rt = residue_table(path, cfg.get_path("structure.chain", "A"), offset)
    return rt[["pos", "aa", "rsa"]].rename(columns={"rsa": "rsa_alt", "aa": "aa_alt"})


# ------------------------------------------------------- r01 the dose-response
def r01_dose_response(bits: dict, names: list[str], outdir: Path) -> dict:
    """Mean effect against RSA, inside and outside the membrane separately.

    In a soluble protein this is the strongest structural predictor there is. Here the
    loops should behave like a soluble protein and the TM segments should not, because
    inside the membrane "exposed" means exposed to lipid, which is not a permissive
    place to be. The ratio of the two slopes is the number to quote.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["effect", "rsa"])
        for lab, sub, col in (("in a TM helix", p[p.is_tm > 0.5], P.DARK_RED),
                              ("outside the membrane", p[p.is_tm <= 0.5], "#2166AC")):
            if len(sub) < MIN_POSITIONS * 2:
                continue
            ax.scatter(sub.rsa, sub.effect, s=10, color=col, alpha=0.3, lw=0)
            binned(ax, sub.rsa, sub.effect, RSA_BINS, col, lab, ms=4.2)
            sl = ss.linregress(sub.rsa, sub.effect)
            rows.append({"protein": g, "band": lab, "n": int(len(sub)),
                         "slope": float(sl.slope), "se": float(sl.stderr),
                         "p": float(sl.pvalue),
                         "spearman": float(ss.spearmanr(sub.rsa, sub.effect)[0]),
                         "rsa_iqr": float(sub.rsa.quantile(0.75)
                                          - sub.rsa.quantile(0.25))})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        d = {r["band"]: r for r in rows if r["protein"] == g}
        ratio = (d["in a TM helix"]["slope"] / d["outside the membrane"]["slope"]
                 if len(d) == 2 and d["outside the membrane"]["slope"] else np.nan)
        panel(ax, i, g, (f"TM/loop slope ratio {ratio:.2f}" if np.isfinite(ratio)
                         else ""), size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("relative solvent accessibility", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect at the position", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.suptitle("A ratio below one means RSA buys less inside the membrane than outside "
                 "it, which is what a water probe rolled over a lipid-facing residue "
                 "should do.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "r01_dose_response", None, "R01")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "r01_slopes.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} {r.band.split(' ')[0]} slope "
                     f"{r.slope:+.3f} (ρ {r.spearman:+.2f})" for r in per.itertuples())
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# ---------------------------------------- r02 water-facing against lipid-facing area
def r02_water_vs_lipid(bits: dict, names: list[str], outdir: Path) -> dict:
    """One predictor becomes two, with opposite meanings.

    Every exposed atom is assigned to water or to lipid by its own depth, so a residue
    gets an area facing each. Being exposed to water should be protective; being exposed
    to lipid should not be, or should be much less so. If the two coefficients come out
    the same, the membrane adds nothing to the burial story and RSA was fine as it was.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["effect", "rsa_water", "rsa_lipid"])
        if len(p) < MIN_POSITIONS * 3:
            empty_panel(ax, i, g, "no per-atom areas: needs a membrane frame")
            continue
        for lab, col, x in (("facing water", "#2166AC", p.rsa_water),
                            ("facing lipid", P.DARK_RED, p.rsa_lipid)):
            binned(ax, x, p.effect, RSA_BINS, col, lab, ms=4.2)
        d = p.copy()
        d["w"] = zscore(d.rsa_water)
        d["l"] = zscore(d.rsa_lipid)
        import statsmodels.formula.api as smf
        m = smf.ols("effect ~ w + l", data=d[["effect", "w", "l"]]).fit(cov_type="HC3")
        t = m.t_test("w - l = 0")
        rows.append({"protein": g, "n": int(m.nobs), "r2": float(m.rsquared),
                     "water": float(m.params["w"]), "water_se": float(m.bse["w"]),
                     "water_p": float(m.pvalues["w"]),
                     "lipid": float(m.params["l"]), "lipid_se": float(m.bse["l"]),
                     "lipid_p": float(m.pvalues["l"]),
                     "difference": float(m.params["w"] - m.params["l"]),
                     "difference_se": float(np.ravel(t.sd)[0]),
                     "difference_p": float(np.ravel(t.pvalue)[0]),
                     "half_thickness": bits[g].get("half_thickness")})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        r = rows[-1]
        panel(ax, i, g, f"water {r['water']:+.3f} vs lipid {r['lipid']:+.3f}, "
                        f"difference p {r['difference_p']:.2g}", size=7.5)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("accessible area / max ASA, for that part of the surface",
                          fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect at the position", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.suptitle("Each residue's accessible surface is split by where the exposed atoms "
                 "sit: inside the bilayer slab counts as lipid, outside it as water. "
                 "Coefficients are per SD, from both terms in one model.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "r02_water_vs_lipid", None, "R02")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "r02_water_vs_lipid.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} water {r.water:+.3f} vs lipid "
                     f"{r.lipid:+.3f} (Δ p {r.difference_p:.2g})"
                     for r in per.itertuples()) if len(per) else ""
    return {"status": "ok", "headline": head, "models": per.to_dict("records")}


# ---------------------------------------------------- r03 where is the knee?
def segmented(x, y, grid_q=np.linspace(0.06, 0.85, 80)) -> dict:
    """Least-squares breakpoint: the RSA at which the slope changes, found by search.

    A hinge at an unknown place is not a linear model, so the breakpoint is found by
    fitting the hinge at every candidate and keeping the one with the smallest residual
    sum of squares. Candidates are quantiles of the observed RSA, so the search cannot
    propose a breakpoint with no data either side of it.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 30:
        return {}
    best = None
    for q in grid_q:
        c = float(np.quantile(x, q))
        if (x < c).sum() < 10 or (x >= c).sum() < 10:
            continue
        A = np.c_[np.ones(len(x)), np.minimum(x, c), np.maximum(x - c, 0)]
        beta, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        rss = float(np.sum((y - A @ beta) ** 2))
        if best is None or rss < best["rss"]:
            best = {"breakpoint": c, "rss": rss, "below": float(beta[1]),
                    "above": float(beta[2]), "intercept": float(beta[0])}
    if best is None:
        return {}
    lin = np.polyfit(x, y, 1)
    best["rss_linear"] = float(np.sum((y - np.polyval(lin, x)) ** 2))
    best["linear_slope"] = float(lin[0])
    best["n"] = int(len(x))
    return best


def r03_breakpoint(bits: dict, names: list[str], outdir: Path,
                   n_boot: int = N_BOOT) -> dict:
    """Does tolerance change at RSA 0.15, or somewhere else?

    The 0.15 cut this project uses everywhere is inherited from soluble-protein work and
    has never been checked against these data. Either the fitted breakpoint lands near
    it and the rest of the pipeline is safe, or it does not and several figures are
    cutting in the wrong place.
    """
    rows = []
    rng = np.random.default_rng(23)
    fig, pairs, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["effect", "rsa"])
        fit = segmented(p.rsa, p.effect)
        if not fit:
            empty_panel(ax, i, g, "too few scored positions to fit a breakpoint")
            continue
        boot = []
        for _ in range(n_boot):
            s = p.sample(len(p), replace=True, random_state=int(rng.integers(1 << 31)))
            f = segmented(s.rsa, s.effect)
            if f:
                boot.append(f["breakpoint"])
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) > 20 else (np.nan,) * 2)
        ax.scatter(p.rsa, p.effect, s=10, color="#D9D9D9", lw=0)
        binned(ax, p.rsa, p.effect, RSA_BINS, P.INK, "")
        c = fit["breakpoint"]
        xs = np.linspace(float(p.rsa.min()), float(p.rsa.max()), 100)
        ax.plot(xs, fit["intercept"] + fit["below"] * np.minimum(xs, c)
                + fit["above"] * np.maximum(xs - c, 0), color=P.DARK_RED, lw=1.6)
        if np.isfinite(lo):
            ax.axvspan(lo, hi, color=P.DARK_RED, alpha=0.12, lw=0)
        ax.axvline(c, color=P.DARK_RED, lw=1.0, ls=(0, (4, 3)))
        ax.axvline(DEFAULT_CUT, color=P.INK, lw=1.0, ls=(0, (1, 2)))
        ax.axhline(0, color=P.MUTED, lw=0.6)
        rows.append({"protein": g, **fit, "boot_lo": float(lo), "boot_hi": float(hi),
                     "default_cut": DEFAULT_CUT,
                     "covers_default_cut": bool(np.isfinite(lo)
                                                and lo <= DEFAULT_CUT <= hi)})
        panel(ax, i, g, f"knee at RSA {c:.2f} [{lo:.2f}, {hi:.2f}]", size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("relative solvent accessibility", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect at the position", fontsize=7.5)
    fig.suptitle("Dashed red = the fitted breakpoint with its bootstrap interval; dotted "
                 "black = the RSA < 0.15 cut this pipeline uses elsewhere.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "r03_breakpoint", None, "R03")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "r03_breakpoint.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} knee {r.breakpoint:.2f} "
                     f"[{r.boot_lo:.2f}, {r.boot_hi:.2f}]"
                     + (" (covers the 0.15 cut)" if r.covers_default_cut
                        else " (excludes the 0.15 cut)")
                     for r in per.itertuples()) if len(per) else ""
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# ------------------------------------------- r04 your burial or your neighbourhood's
def r04_own_vs_neighbourhood(bits: dict, names: list[str], outdir: Path) -> dict:
    """A residue can be solvent-exposed in a buried pocket, or buried on an open face.

    Own RSA and the mean RSA of everything within 8 Å go into one model. Whichever
    coefficient survives says whether burial acts on the side chain itself or on the
    region it sits in - the same question b09 asks about chemistry, asked about space.
    """
    rows = []
    fig, pairs, axes = grid(names, w=3.0, h=2.8, extra=0.3)
    im = None
    eb = np.array([0.0, 0.12, 0.20, 0.28, 0.40, 1.01])
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["effect", "rsa", "rsa_env"]).copy()
        if len(p) < MIN_POSITIONS * 3:
            empty_panel(ax, i, g, "too few scored positions")
            continue
        m = np.full((len(eb) - 1, len(RSA_BINS) - 1), np.nan)
        for a in range(len(eb) - 1):
            for b_ in range(len(RSA_BINS) - 1):
                s = p[(p.rsa_env >= eb[a]) & (p.rsa_env < eb[a + 1])
                      & (p.rsa >= RSA_BINS[b_]) & (p.rsa < RSA_BINS[b_ + 1])].effect
                if len(s) >= 4:
                    m[a, b_] = float(s.mean())
        from .burial import empty_safe_cmap
        im = ax.imshow(m, origin="lower", aspect="auto", cmap=empty_safe_cmap(),
                       norm=P.fitness_norm(),
                       extent=(0, len(RSA_BINS) - 1, 0, len(eb) - 1))
        ax.set_xticks(np.arange(len(RSA_BINS) - 1) + 0.5)
        ax.set_xticklabels([f"{RSA_BINS[b_]:.2f}" for b_ in range(len(RSA_BINS) - 1)],
                           fontsize=5.5)
        ax.set_yticks(np.arange(len(eb) - 1) + 0.5)
        ax.set_yticklabels([f"{eb[a]:.2f}" for a in range(len(eb) - 1)], fontsize=6)
        p["own"] = zscore(p.rsa); p["env"] = zscore(p.rsa_env)
        import statsmodels.formula.api as smf
        fit = smf.ols("effect ~ own + env", data=p[["effect", "own", "env"]]).fit(
            cov_type="HC3")
        rows.append({"protein": g, "n": int(fit.nobs), "r2": float(fit.rsquared),
                     "own": float(fit.params["own"]), "own_se": float(fit.bse["own"]),
                     "own_p": float(fit.pvalues["own"]),
                     "env": float(fit.params["env"]), "env_se": float(fit.bse["env"]),
                     "env_p": float(fit.pvalues["env"]),
                     "correlation": float(p.rsa.corr(p.rsa_env))})
        r = rows[-1]
        panel(ax, i, g, f"own {r['own']:+.3f}, neighbourhood {r['env']:+.3f}", size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("own RSA", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean RSA within 8 Å", fontsize=7.5)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01).set_label(
            "Mean effect", fontsize=8, fontweight="bold")
    fig.suptitle("If only your own burial mattered the colour would change left to right "
                 "and not bottom to top. Coefficients are per SD, both in one model.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "r04_own_vs_neighbourhood", None, "R04")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "r04_own_vs_neighbourhood.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} own {r.own:+.3f} (p {r.own_p:.2g}), "
                     f"neighbourhood {r.env:+.3f} (p {r.env_p:.2g})"
                     for r in per.itertuples()) if len(per) else ""
    return {"status": "ok", "headline": head, "models": per.to_dict("records")}


# ---------------------------------------- r05 unique and shared variance
def lmg_shares(d: pd.DataFrame, y: str, xs: list[str]) -> dict:
    """Shapley decomposition of R²: each predictor's average marginal contribution.

    With correlated predictors the order they enter a model changes what each appears to
    explain, and no single ordering is the right one. Averaging the increment over every
    ordering is the Shapley value - the LMG measure - and it has the property that makes
    the figure readable: the shares are non-negative and add up to exactly the full
    model's R².
    """
    def r2(cols):
        if not cols:
            return 0.0
        A = np.c_[np.ones(len(d)), d[list(cols)].to_numpy(float)]
        beta = np.linalg.lstsq(A, d[y].to_numpy(float), rcond=None)[0]
        resid = d[y].to_numpy(float) - A @ beta
        tot = float(np.sum((d[y] - d[y].mean()) ** 2))
        return 1.0 - float(np.sum(resid ** 2)) / tot if tot else 0.0

    cache, share = {}, {x: 0.0 for x in xs}
    def cached(cols):
        k = tuple(sorted(cols))
        if k not in cache:
            cache[k] = r2(k)
        return cache[k]
    orders = list(itertools.permutations(xs))
    for order in orders:
        seen = []
        for x in order:
            share[x] += cached(seen + [x]) - cached(seen)
            seen.append(x)
    share = {k: v / len(orders) for k, v in share.items()}
    # the semipartial: what each predictor adds last, which is its unique contribution
    full = cached(xs)
    unique = {x: full - cached([c for c in xs if c != x]) for x in xs}
    return {"share": share, "unique": unique, "r2": full}


def readable_on(color) -> str:
    """Black or white, whichever the eye can read on that fill.

    Picking by position in the stack rather than by the colour itself puts dark text on
    a dark block as soon as the ordering changes, which is how the label went missing
    the first time this figure was drawn.
    """
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else P.INK


def r05_variance(bits: dict, names: list[str], outdir: Path) -> dict:
    """How much of the variance does RSA explain that nothing else could?

    s09 asks which burial measure predicts best. This asks a different question: of the
    variance the four measures explain between them, how much belongs to each when the
    overlap is split fairly? The shares add to the full model's R², and the tick on each
    bar is what that predictor adds when it enters last - its unique part. A tall bar
    with a tiny tick is a predictor doing no work the others could not do.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.1, h=2.7, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["effect"] + list(PRED)).copy()
        if len(p) < MIN_POSITIONS * 4:
            empty_panel(ax, i, g, "too few positions with all four measures")
            continue
        for c in PRED:
            p[c] = zscore(p[c])
        out = lmg_shares(p, "effect", list(PRED))
        order = sorted(PRED, key=lambda c: -out["share"][c])
        bottom = 0.0
        cols = P.series_colors(len(PRED))
        for k, c in enumerate(order):
            v = out["share"][c]
            ax.bar(0, v, 0.55, bottom=bottom, color=cols[k], edgecolor=P.INK, lw=0.5,
                   label=PRED[c])
            ax.plot([0.33, 0.33], [bottom, bottom + out["unique"][c]], color=P.INK,
                    lw=3.0, solid_capstyle="butt")
            if v > 0.012:
                ax.text(0, bottom + v / 2, f"{v:.3f}", ha="center", va="center",
                        fontsize=6, color=readable_on(cols[k]))
            bottom += v
            rows.append({"protein": g, "predictor": c, "lmg_share": float(v),
                         "unique": float(out["unique"][c]),
                         "share_of_model": float(v / out["r2"]) if out["r2"] else np.nan})
        ax.set_xlim(-0.6, 1.5); ax.set_xticks([])
        panel(ax, i, g, f"total R² = {out['r2']:.3f}", size=8)
        if i % 2 == 0:
            ax.set_ylabel("variance in mean effect explained", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.suptitle("Stacked: each predictor's Shapley (LMG) share, which add to the full "
                 "model's R². The black tick inside each block is that predictor's unique "
                 "contribution - what it adds entering last.", fontsize=7, x=0.01,
                 ha="left")
    P.save(fig, outdir, "r05_variance_shares", None, "R05")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "r05_variance_shares.csv", index=False)
    d = per[per.predictor == "rsa"]
    head = "; ".join(f"{short_name(r.protein)} RSA share {r.lmg_share:.3f} "
                     f"({100 * r.share_of_model:.0f}% of the model), unique {r.unique:.3f}"
                     for r in d.itertuples()) if len(d) else ""
    return {"status": "ok", "headline": head, "shares": per.to_dict("records")}


# ------------------------------------- r06 burial changes the shape, not just the mean
def r06_shape(bits: dict, names: list[str], outdir: Path) -> dict:
    """A buried position is uniformly intolerant; an exposed one is mostly fine.

    Two positions can have the same mean and be nothing alike: one where every
    substitution costs a little, and one where most cost nothing and three are lethal.
    The spread and the skew of a position's own substitutions separate them, and the
    claim is that burial predicts the shape as well as the average.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.1, h=2.7, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["rsa", "sd"])
        p = p[p.n_scored >= 5]
        if len(p) < MIN_POSITIONS * 2:
            empty_panel(ax, i, g, "too few positions with 5+ variants")
            continue
        binned(ax, p.rsa, p.sd, RSA_BINS, P.DARK_RED, "SD of the position's effects")
        ax2 = ax.twinx()
        sk = p.dropna(subset=["skewness"])
        binned(ax2, sk.rsa, sk.skewness, RSA_BINS, "#2166AC", "skew")
        ax2.axhline(0, color="#2166AC", lw=0.5, ls=(0, (1, 2)))
        ax2.tick_params(axis="y", labelsize=6, colors="#2166AC", direction="out")
        ax2.set_ylabel("skew", fontsize=7, color="#2166AC")
        # the twin inherits the shared style's hidden right spine, which leaves its
        # ticks floating next to nothing; give it back its own axis line in its colour
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("#2166AC")
        P.no_square(ax2)
        for lab, col in (("sd", "sd"), ("skew", "skewness")):
            s = p.dropna(subset=[col])
            if len(s) > 20:
                sl = ss.linregress(s.rsa, s[col])
                rows.append({"protein": g, "statistic": lab, "n": int(len(s)),
                             "slope": float(sl.slope), "se": float(sl.stderr),
                             "p": float(sl.pvalue),
                             "spearman": float(ss.spearmanr(s.rsa, s[col])[0])})
        d = {r["statistic"]: r for r in rows if r["protein"] == g}
        panel(ax, i, g, (f"SD ρ {d['sd']['spearman']:+.2f}, skew ρ "
                         f"{d['skew']['spearman']:+.2f}" if len(d) == 2 else ""), size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("relative solvent accessibility", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("SD across the position's substitutions", fontsize=7.5)
        if i == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6, loc="upper left")
    fig.suptitle("Red, left axis: how much the identity of the substitution matters at "
                 "that position. Blue, right axis: skew - negative means a tolerant "
                 "position with a few disasters.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "r06_shape_of_the_distribution", None, "R06")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "r06_shape.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} {r.statistic} ρ {r.spearman:+.2f} "
                     f"(p {r.p:.2g})" for r in per.itertuples()) if len(per) else ""
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# ---------------------------------------- r07 which residues care about being buried
def r07_by_mutant(bits: dict, names: list[str], outdir: Path,
                  min_n: int = 40) -> dict:
    """For each of the twenty residues, how much its cost depends on burial.

    Proline and the charged residues should be ruinous when buried and harmless when
    exposed - a steep positive slope. Alanine and serine should be nearly flat. If that
    ordering does not come out, something upstream is wrong, which makes this the best
    sanity check in the module as well as the most readable figure in it.
    """
    rows = []
    for g in names:
        v = bits[g]["var"].dropna(subset=["score_z", "rsa"])
        for mut, sub in v.groupby("mut"):
            if len(sub) < min_n or sub.pos.nunique() < 6:
                continue
            m = cluster_ols("score_z ~ rsa", sub[["score_z", "rsa"]], sub.pos)
            if m is None:
                continue
            rows.append({"protein": g, "mut": mut, "n": int(m.nobs),
                         "slope": float(m.params["rsa"]), "se": float(m.bse["rsa"]),
                         "p": float(m.pvalues["rsa"]),
                         "mean": float(sub.score_z.mean())})
    if not rows:
        return {"status": "skipped", "reason": "no mutant residue has enough variants"}
    # one row order for every panel, from the mean slope across proteins. Sorting each
    # panel by its own slopes would make the residues move between panels, and the
    # comparison the figure is for is between proteins at the same residue. Only the
    # order is pooled; every number in a panel is still that protein's alone.
    allr = pd.DataFrame(rows)
    order = list(allr.groupby("mut").slope.mean().sort_values().index)

    fig, pairs, _ = grid(names, w=3.2, h=2.9, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        by = {r["mut"]: r for r in rows if r["protein"] == g}
        per = [by[m] for m in order if m in by]
        if not per:
            empty_panel(ax, i, g, "no mutant residue has enough variants")
            continue
        y = np.arange(len(per))
        ax.barh(y, [r["slope"] for r in per],
                xerr=[1.96 * r["se"] for r in per],
                color=[effect_color(r["mean"]) for r in per], edgecolor=P.INK,
                linewidth=0.4, error_kw=dict(lw=0.6, capsize=1.4))
        ax.set_yticks(y)
        ax.set_yticklabels([r["mut"] for r in per], fontsize=6)
        ax.axvline(0, color=P.MUTED, lw=0.7)
        top = max(per, key=lambda r: r["slope"])
        panel(ax, i, g, f"steepest: {top['mut']} ({top['slope']:+.2f})", size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("slope of effect on RSA, for that mutant residue", fontsize=7.5)
    fig.suptitle("Rows are in the same order in every panel, by the mean slope across "
                 "proteins. Bars coloured by that residue's mean effect in this protein. "
                 "A steep positive slope means the residue is tolerated when exposed and "
                 "not when buried.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "r07_slope_by_mutant", None, "R07")
    plt.close(fig)
    per = allr
    per.to_csv(outdir / "r07_slope_by_mutant.csv", index=False)
    head = "; ".join(
        f"{short_name(g)} steepest {d.sort_values('slope').mut.iloc[-1]}, flattest "
        f"{d.loc[d.slope.abs().idxmin(), 'mut']}" for g, d in per.groupby("protein")
    ) if len(per) else ""
    return {"status": "ok", "headline": head, "by_mutant": per.to_dict("records")}


# ------------------------------------------- r08 per helix: no range, no prediction
def r08_by_helix(bits: dict, names: list[str], outdir: Path) -> dict:
    """RSA cannot predict anything in a helix where RSA does not vary.

    A helix buried in the middle of the bundle has every residue at the same low
    accessibility, so the slope fitted inside it is noise with a wide interval - and
    that is a fact about the geometry, not a failure of the idea. Plotting the slope
    against the helix's own RSA range says which helices the question can even be asked
    in.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["effect", "rsa"])
        p = p[(p.is_tm > 0.5) & p.helix.notna()]
        per = []
        for h, sub in p.groupby("helix"):
            if len(sub) < MIN_POSITIONS or sub.rsa.std() == 0:
                continue
            sl = ss.linregress(sub.rsa, sub.effect)
            per.append({"protein": g, "helix": h, "n": int(len(sub)),
                        "slope": float(sl.slope), "se": float(sl.stderr),
                        "p": float(sl.pvalue),
                        "rsa_range": float(sub.rsa.quantile(0.9)
                                           - sub.rsa.quantile(0.1)),
                        "rsa_median": float(sub.rsa.median()),
                        "mean_effect": float(sub.effect.mean())})
        if not per:
            empty_panel(ax, i, g, "no TM helix with enough scored positions")
            continue
        rows += per
        d = pd.DataFrame(per)
        ax.errorbar(d.rsa_range, d.slope, yerr=d.se, fmt="o", ms=5.5, lw=0,
                    elinewidth=0.8, capsize=2,
                    color=P.DARK_RED, mec=P.INK, mew=0.5)
        P.label_points(ax, d.rsa_range.to_numpy(), d.slope.to_numpy(),
                       d.helix.tolist(), fontsize=6)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        if len(d) >= 4:
            rho, pv = ss.spearmanr(d.rsa_range, d.se)
            panel(ax, i, g, f"range vs error ρ = {rho:+.2f} (p = {pv:.2g})", size=8)
        else:
            panel(ax, i, g, size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("RSA range within the helix (10th–90th percentile)",
                          fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("slope of effect on RSA, within the helix", fontsize=7.5)
    fig.suptitle("A helix with no RSA range (left) cannot have a slope estimated in it; "
                 "the error bars should shrink to the right, which is what the quoted "
                 "correlation tests.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "r08_by_helix", None, "R08")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "r08_by_helix.csv", index=False)
    head = "; ".join(f"{short_name(g)} {len(d)} helices, slope "
                     f"{d.slope.min():+.2f} to {d.slope.max():+.2f}"
                     for g, d in per.groupby("protein")) if len(per) else ""
    return {"status": "ok", "headline": head, "by_helix": per.to_dict("records")}


# ------------------------------------------- r09 the fraction dead, not the mean
def r09_fraction_dead(bits: dict, names: list[str], outdir: Path) -> dict:
    """Burial may not shift everything down a little; it may make a few things lethal.

    A mean of -0.5 can be twenty substitutions at -0.5 or ten at 0 and ten at -1, and
    those are different claims about what burial does. Plotting the fraction of a
    position's substitutions at stop-codon level alongside the rescaled mean separates
    them: if the fraction is the steeper curve, burial converts substitutions into null
    alleles rather than degrading them all.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.1, h=2.7, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["rsa", "frac_dead", "effect"])
        p = p[p.n_scored >= 5]
        if len(p) < MIN_POSITIONS * 2:
            empty_panel(ax, i, g, "too few positions with 5+ variants")
            continue
        binned(ax, p.rsa, p.frac_dead, RSA_BINS, P.DARK_RED,
               f"fraction at or below {DEAD:g}")
        # the mean on the same axis, rescaled so the two curves can be compared by shape
        scaled = p.effect / float(DEAD)
        binned(ax, p.rsa, scaled, RSA_BINS, P.MUTED, "mean effect, rescaled")
        for lab, y in (("fraction_dead", p.frac_dead), ("mean_rescaled", scaled)):
            sl = ss.linregress(p.rsa, y)
            rows.append({"protein": g, "measure": lab, "n": int(len(p)),
                         "slope": float(sl.slope), "se": float(sl.stderr),
                         "p": float(sl.pvalue)})
        d = {r["measure"]: r for r in rows if r["protein"] == g}
        ratio = (d["fraction_dead"]["slope"] / d["mean_rescaled"]["slope"]
                 if d.get("mean_rescaled", {}).get("slope") else np.nan)
        panel(ax, i, g, (f"fraction/mean slope ratio {ratio:.2f}"
                         if np.isfinite(ratio) else ""), size=8)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("relative solvent accessibility", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel(f"fraction of substitutions ≤ {DEAD:g}", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.suptitle(f"The mean is divided by {DEAD:g} so that both curves run from 0 "
                 "(tolerant) to 1 (stop-like) and only their shapes are being compared.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "r09_fraction_dead", None, "R09")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "r09_fraction_dead.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} {r.measure} slope {r.slope:+.3f}"
                     for r in per.itertuples()) if len(per) else ""
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# ----------------------------------------------- r10 burial that moves between states
def r10_alternative_state(bits: dict, names: list[str], outdir: Path) -> dict:
    """A transporter is not one structure, and AlphaFold gives one.

    Where a second conformation is available, the positions whose accessibility changes
    between the two are the moving parts: the gates and the pathway that opens and
    closes. The claim is that they are less tolerant than positions equally buried in
    both states - that it costs more to sit on a hinge than in a wall.
    """
    have = [g for g in names if "alt_rsa" in bits[g]]
    if not have:
        return {"status": "skipped",
                "reason": "no protein has a second structure; set structure.alt_path in "
                          "the config or pass --alt GENE=FILE"}
    rows, frames = [], []
    fig, pairs, _ = grid(have, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["effect", "rsa"]).merge(
            bits[g]["alt_rsa"], on="pos", how="inner")
        p = p.dropna(subset=["rsa_alt"])
        if "wt" in p and "aa_alt" in p:
            agree = float((p.wt == p.aa_alt).mean())
            p = p[p.wt == p.aa_alt]
        else:
            agree = np.nan
        if len(p) < MIN_POSITIONS * 3:
            empty_panel(ax, i, g, "too few positions shared with the second structure")
            continue
        p = p.copy()
        p["d_rsa"] = (p.rsa_alt - p.rsa).abs()
        p["mean_rsa"] = (p.rsa_alt + p.rsa) / 2
        if p.d_rsa.std() < 1e-6:
            # the same coordinates twice: the model would be rank-deficient and would
            # report a coefficient of zero with a p-value of nan, which reads as a result
            empty_panel(ax, i, g, "the second structure has identical accessibility —\n"
                                  "is it the same file?")
            continue
        ax.scatter(p.d_rsa, p.effect, s=12, c=[effect_color(e) for e in p.effect],
                   edgecolor=P.INK, linewidth=0.25)
        binned(ax, p.d_rsa, p.effect, np.array([0, 0.05, 0.10, 0.20, 0.35, 1.01]),
               P.INK, "")
        import statsmodels.formula.api as smf
        p["d"] = zscore(p.d_rsa); p["m"] = zscore(p.mean_rsa)
        fit = smf.ols("effect ~ d + m", data=p[["effect", "d", "m"]]).fit(cov_type="HC3")
        rows.append({"protein": g, "n": int(fit.nobs), "wt_agreement": agree,
                     "n_moving": int((p.d_rsa > 0.1).sum()),
                     "median_d_rsa": float(p.d_rsa.median()),
                     "r2": float(fit.rsquared),
                     "delta_rsa": float(fit.params["d"]),
                     "delta_rsa_se": float(fit.bse["d"]),
                     "delta_rsa_p": float(fit.pvalues["d"]),
                     "mean_rsa": float(fit.params["m"]),
                     "mean_rsa_p": float(fit.pvalues["m"])})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        r = rows[-1]
        panel(ax, i, g, f"Δ RSA {r['delta_rsa']:+.3f} (p {r['delta_rsa_p']:.2g}), "
                        f"mean RSA held fixed", size=7.5)
        frames.append(p.assign(protein=g))
        if i // 2 == (len(have) - 1) // 2:
            ax.set_xlabel("|RSA(state 2) − RSA(state 1)|", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect at the position", fontsize=7.5)
    fig.suptitle("Mean RSA across the two states is in the model, so the Δ coefficient is "
                 "about moving rather than about being exposed on average.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "r10_alternative_state", None, "R10")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "r10_alternative_state.csv", index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(outdir / "r10_positions.csv",
                                                    index=False)
    head = "; ".join(f"{short_name(r.protein)} ΔRSA {r.delta_rsa:+.3f} "
                     f"(p {r.delta_rsa_p:.2g}), {r.n_moving} positions move by > 0.1"
                     for r in per.itertuples()) if len(per) else ""
    return {"status": "ok",
            "headline": head or "no protein had a second structure that differs from "
                                "the first",
            "models": per.to_dict("records")}


ANALYSES = {"r01": "the dose-response, TM against loop",
            "r02": "water-facing against lipid-facing accessible area",
            "r03": "where the knee is",
            "r04": "own burial against the neighbourhood's",
            "r05": "unique and shared variance",
            "r06": "the shape of the distribution, not only its mean",
            "r07": "which residues care about being buried",
            "r08": "per helix: no range, no prediction",
            "r09": "the fraction at stop-codon level",
            "r10": "burial that moves between two conformations"}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "divide by zero|Degrees of freedom|All-NaN|"
                                              "tight_layout|DataFrameGroupBy.apply).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms access",
        description="What solvent accessibility does and does not say about tolerance in "
                    "a membrane protein. One panel per protein, in the order given.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
    ap.add_argument("--alt", action="append", default=[], metavar="GENE=FILE",
                    help="a second conformation for r10, as a PDB or mmCIF file whose "
                         "numbering matches the config")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help="bootstrap replicates for the r03 breakpoint interval")
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)
    keys = [k.strip() for k in (a.only.split(",") if a.only else ANALYSES)]
    bad = [k for k in keys if k not in ANALYSES]
    if bad:
        raise SystemExit(f"unknown analysis {bad}; available: {', '.join(ANALYSES)}")
    alt = {}
    for spec in a.alt:
        if "=" not in spec:
            raise SystemExit(f"--alt needs GENE=FILE, got {spec!r}")
        k, v = spec.split("=", 1)
        alt[k] = Path(v)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bits, names = {}, []
    for c in a.configs:
        cfg = load_config(Path(c))
        second = alt.get(cfg.id) or alt.get(cfg.get_path("protein.gene") or "")
        if second is None:
            cp = cfg.get_path("structure.alt_path")
            second = cfg.resolve(cp) if cp else None
        b = bits_for(cfg, second)
        if not b or b["pos"].rsa.isna().all():
            print(f"  {cfg.id:<16} skipped: needs a structure with accessibility",
                  flush=True)
            continue
        bits[cfg.id] = b
        names.append(cfg.id)
        p = b["pos"]
        print(f"  {cfg.id:<16} {len(p):>4} positions  RSA {p.rsa.min():.2f}–"
              f"{p.rsa.max():.2f} (median {p.rsa.median():.2f})"
              + (f"  bilayer ±{b['half_thickness']:.0f} Å"
                 if b.get("half_thickness") else "")
              + ("  second structure loaded" if "alt_rsa" in b else ""), flush=True)
    if not names:
        raise SystemExit("no protein has a structure to compute accessibility from")
    rows = int(np.ceil(len(names) / a.cols))
    print("\n  panels: " + " | ".join(
        ", ".join(short_name(n) for n in names[r * a.cols:(r + 1) * a.cols])
        for r in range(rows)), flush=True)
    pd.concat([b["pos"].assign(protein=g) for g, b in bits.items()],
              ignore_index=True).to_csv(outdir / "positions.csv", index=False)

    calls = {"r01": lambda: r01_dose_response(bits, names, outdir),
             "r02": lambda: r02_water_vs_lipid(bits, names, outdir),
             "r03": lambda: r03_breakpoint(bits, names, outdir, a.boot),
             "r04": lambda: r04_own_vs_neighbourhood(bits, names, outdir),
             "r05": lambda: r05_variance(bits, names, outdir),
             "r06": lambda: r06_shape(bits, names, outdir),
             "r07": lambda: r07_by_mutant(bits, names, outdir),
             "r08": lambda: r08_by_helix(bits, names, outdir),
             "r09": lambda: r09_fraction_dead(bits, names, outdir),
             "r10": lambda: r10_alternative_state(bits, names, outdir)}

    out = {}
    print()
    for k in keys:
        try:
            r = calls[k]()
        except Exception as e:
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        out[k] = r
        print(f"  {k}  {r.get('status'):<8} {r.get('headline') or r.get('reason', '')}",
              flush=True)
    (outdir / "access.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
