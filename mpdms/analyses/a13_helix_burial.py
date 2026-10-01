"""A13 - TM helices: position dependence, surface vs core helices, helix-by-helix effects.

Fitness = per-position mean normalised fitness of missense variants, split three ways:
rest (all missense except Pro/Gly), Pro, Gly. Burial = relative SASA on the AlphaFold
model (no membrane), so lipid-facing counts as exposed. Helix class = helix median RSA
above/below the protein's median (median split) or top/bottom third (tertile split).
The helix is the unit for class statistics (positions within a helix are not independent).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as ss

from .. import plotting as P
from ..helixgeom import CLASS_COLORS, SUB_COLORS, SUBS, tm_residue_table
from ..stats import bh_fdr
from .a08_depth_dependence import _lowess_curve
from .base import dirs, fmt_p, result, skipped

COLS = {"rest": "fit_rest", "P": "fit_P", "G": "fit_G"}
SPLITS = {"class_median": "median split", "class_tertile": "tertile split (middle third dropped)"}
N_PERM = 5000


# ------------------------------------------------------------------ statistics
def _hid(t):
    return t.dataset_id + ":" + t.helix


def _rank(x):
    return ss.rankdata(x)


def _perm_within(groups, n, rng):
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    keys = rng.random((n, len(groups))) + g_sorted[None, :] * 2.0
    shuffled = order[np.argsort(keys, axis=1)]          # positions of each group's members, shuffled
    out = np.empty_like(shuffled)
    out[:, order] = shuffled                             # slot k (original order) receives a member of its group
    return out


def position_stats(t: pd.DataFrame, col: str, rng, n_perm: int = 2000) -> dict:
    d = t[t.confident & t[col].notna()].copy()
    if len(d) < 10:
        return {}
    d["hid"] = _hid(d)
    rho = ss.spearmanr(d.abs_rel, d[col])[0]
    # permutation: shuffle fitness within each helix (keeps helix means, breaks position)
    gcode = pd.factorize(d.hid)[0]
    rx = _rank(d.abs_rel.to_numpy()); rx = (rx - rx.mean()) / rx.std()
    ry = _rank(d[col].to_numpy()); ry = (ry - ry.mean()) / ry.std()
    perms = _perm_within(gcode, n_perm, rng)
    null = (ry[perms] * rx[None, :]).mean(axis=1)
    p_perm = (1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (1 + n_perm)
    out = {"n_positions": len(d), "n_helices": d.hid.nunique(), "spearman_absrel": rho, "p_perm_within_helix": p_perm}
    try:
        form = f"{col} ~ abs_rel + I(abs_rel**2)" + (" + C(dataset_id)" if d.dataset_id.nunique() > 1 else "")
        m = smf.ols(form, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.hid})
        out.update(coef_absrel=m.params["abs_rel"], p_absrel=m.pvalues["abs_rel"],
                   coef_absrel2=m.params["I(abs_rel ** 2)"], p_absrel2=m.pvalues["I(abs_rel ** 2)"])
        out["centre_minus_end"] = -(m.params["abs_rel"] + m.params["I(abs_rel ** 2)"])
    except Exception:
        pass
    # cytosolic vs lumenal half, per helix then bootstrap over helices
    cyt = d[d.rel_pos < 0].groupby("hid")[col].mean()
    lum = d[d.rel_pos > 0].groupby("hid")[col].mean()
    hh = (cyt - lum).dropna().to_numpy()
    if len(hh) >= 3:
        boots = hh[rng.integers(0, len(hh), (2000, len(hh)))].mean(axis=1)
        out.update(cyto_minus_lumen=float(hh.mean()), cyto_minus_lumen_ci=[float(np.quantile(boots, .025)),
                                                                           float(np.quantile(boots, .975))],
                   p_cyto_vs_lumen=float(ss.wilcoxon(hh).pvalue) if len(hh) >= 6 else np.nan)
    return out


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if sp == 0:
        return np.nan
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9)))


def class_stats(t: pd.DataFrame, col: str, split: str, rng, n_perm: int = N_PERM, mixed: bool = True) -> dict:
    d = t[t.confident & t[col].notna() & t[split].isin(["surface", "core"])].copy()
    if d.empty:
        return {}
    d["hid"] = _hid(d)
    h = d.groupby("hid").agg(fit=(col, "mean"), cls=(split, "first"), ds=("dataset_id", "first"),
                             helix_rsa=("helix_rsa", "first"))
    s_, c_ = h[h.cls == "surface"].fit, h[h.cls == "core"].fit
    out = {"n_surface_helices": len(s_), "n_core_helices": len(c_),
           "surface_mean": float(s_.mean()) if len(s_) else np.nan,
           "core_mean": float(c_.mean()) if len(c_) else np.nan}
    if len(s_) >= 2 and len(c_) >= 2:
        obs = s_.mean() - c_.mean()
        # shuffle surface/core labels across helices within each protein
        lab = (h.cls == "surface").to_numpy()
        perms = _perm_within(pd.factorize(h.ds)[0], n_perm, rng)
        L = lab[perms]
        f = h.fit.to_numpy()[None, :]
        null = (f * L).sum(1) / L.sum(1) - (f * ~L).sum(1) / (~L).sum(1)
        g = hedges_g(s_, c_)
        sa, ca = s_.to_numpy(), c_.to_numpy()
        gb = [hedges_g(sa[rng.integers(0, len(sa), len(sa))], ca[rng.integers(0, len(ca), len(ca))]) for _ in range(1000)]
        out.update(surface_minus_core=float(obs), p_perm=float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))),
                   hedges_g=g, hedges_g_ci=[float(np.nanquantile(gb, .025)), float(np.nanquantile(gb, .975))])
    if len(h) >= 4:
        out["spearman_helixRSA_vs_fitness"] = float(ss.spearmanr(h.helix_rsa, h.fit)[0])
        out["p_spearman_helixRSA"] = float(ss.spearmanr(h.helix_rsa, h.fit)[1])
    if not mixed:
        return out
    try:  # residue-level mixed model, random intercept per helix
        form = f"{col} ~ C({split}, Treatment('core'))" + (" + C(dataset_id)" if d.dataset_id.nunique() > 1 else "")
        m = smf.mixedlm(form, d, groups=d["hid"]).fit(reml=True, method="lbfgs")
        k = [x for x in m.params.index if "surface" in x][0]
        out.update(mixedlm_surface_coef=float(m.params[k]), mixedlm_p=float(m.pvalues[k]))
    except Exception as e:
        out["mixedlm_error"] = str(e)[:80]
    return out


def per_helix_stats(t: pd.DataFrame, col: str) -> pd.DataFrame:
    d = t[t.confident & t[col].notna()]
    rows = []
    for (ds, hx), g in d.groupby(["dataset_id", "helix"]):
        others = d[(d.dataset_id == ds) & (d.helix != hx)][col]
        p = ss.mannwhitneyu(g[col], others).pvalue if len(g) >= 3 and len(others) >= 3 else np.nan
        rows.append({"dataset_id": ds, "helix": hx, "start": g.helix_start.iloc[0], "end": g.helix_end.iloc[0],
                     "length": int(g.helix_len.iloc[0]), "helix_rsa": g.helix_rsa.iloc[0],
                     "partner_helices": g.partner_helices.iloc[0],
                     "class_median": g.class_median.iloc[0], "class_tertile": g.class_tertile.iloc[0],
                     "frac_buried_res": float((g.rsa < 0.10).mean()), "n_positions": len(g),
                     "median": g[col].median(), "mean": g[col].mean(),
                     "frac_deleterious": float((g[col] < -0.3).mean()),
                     "median_minus_other_helices": g[col].median() - others.median(), "p_vs_other_helices": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_vs_other_helices"] = bh_fdr(out.p_vs_other_helices)
        # helices that deviate from their own class (e.g. intolerant surface helix)
        for cls in ("surface", "core"):
            m = out.class_median == cls
            if m.sum() >= 3:
                z = (out.loc[m, "median"] - out.loc[m, "median"].mean()) / out.loc[m, "median"].std()
                out.loc[m, "z_within_class"] = z
    return out


# ------------------------------------------------------------------ figures
def _band(ax, d, col, color, label, rng, grid=np.linspace(-1, 1, 81), n_boot: int = 100):
    d = d[d[col].notna()]
    if len(d) < 10:
        return
    ax.scatter(d.rel_pos, d[col], s=5, color=color, alpha=0.3, lw=0)
    x, y = d.rel_pos.to_numpy(), d[col].to_numpy()
    c = _lowess_curve(x, y, grid, frac=0.5)
    idx = [np.flatnonzero(m) for m in (pd.factorize(_hid(d))[0][None, :] == np.arange(_hid(d).nunique())[:, None])]
    boots = []
    for _ in range(n_boot):  # resample helices
        pick = np.concatenate([idx[k] for k in rng.integers(0, len(idx), len(idx))])
        boots.append(_lowess_curve(x[pick], y[pick], grid, frac=0.5))
    boots = np.array(boots)
    ax.fill_between(grid, np.nanquantile(boots, .025, 0), np.nanquantile(boots, .975, 0), color=color, alpha=0.2, lw=0)
    ax.plot(grid, c, color=color, lw=1.6, label=label)


def fig_position(t, title, rng):
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 7.2), sharex=True, sharey="col")
    d = t[t.confident]
    for j, (sub, col) in enumerate(COLS.items()):
        _band(axes[0, j], d, col, SUB_COLORS[sub], "all TM helices", rng)
        axes[0, j].set_title(SUBS[sub], fontsize=8.5)
        for i, split in enumerate(SPLITS, 1):
            for cls in ("core", "surface"):
                _band(axes[i, j], d[d[split] == cls], col, CLASS_COLORS[cls], f"{cls} helices", rng)
    for i, lab in enumerate(["All TM helices", "Surface vs core\n(median split)", "Surface vs core\n(tertile split)"]):
        axes[i, 0].set_ylabel(f"{lab}\nmean normalised fitness", fontsize=7)
        axes[i, 0].legend(loc="lower left", fontsize=6)
    for ax in axes.ravel():
        ax.axhline(0, color=P.MUTED, lw=0.4)
        ax.axvline(0, color=P.MUTED, lw=0.4, ls=(0, (2, 2)))
    for ax in axes[-1]:
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["cyto\nend", "", "centre", "", "lumen\nend"])
        ax.set_xlabel("Position along TM helix")
    fig.suptitle(f"{title} — fitness vs position along TM helices (LOESS ± helix bootstrap)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    return fig


def fig_classes(t, stats_tab, title, rng):
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 7.6))
    d = t[t.confident]
    for j, (sub, col) in enumerate(COLS.items()):
        for i, split in enumerate(SPLITS):
            ax = axes[i, j]
            data, cols, labs = [], [], []
            for cls in ("core", "surface"):
                v = d.loc[d[split] == cls, col].dropna()
                data.append(v.to_numpy()); cols.append(CLASS_COLORS[cls]); labs.append(f"{cls}\n(n={len(v)})")
            if any(len(x) for x in data):
                parts = ax.violinplot([x if len(x) else [np.nan] for x in data], showextrema=False, widths=0.8)
                for b, c in zip(parts["bodies"], cols):
                    b.set_facecolor(c); b.set_alpha(0.35); b.set_edgecolor("none")
                for k, x in enumerate(data):
                    ax.scatter(k + 1 + rng.uniform(-0.15, 0.15, len(x)), x, s=4, color=cols[k], lw=0, alpha=0.6)
                    if len(x):
                        ax.hlines(np.median(x), k + 0.7, k + 1.3, color=P.INK, lw=1.2)
                # helix means as larger dots
                hm = d[d[split].isin(["core", "surface"])].groupby(_hid(d)).agg(f=(col, "mean"), c=(split, "first"))
                for k, cls in enumerate(("core", "surface")):
                    v = hm[hm.c == cls].f
                    ax.scatter(np.full(len(v), k + 1.38), v, s=16, marker="D", facecolor="white",
                               edgecolor=CLASS_COLORS[cls], lw=0.9, zorder=3)
            ax.set_xticks([1, 2]); ax.set_xticklabels(labs, fontsize=6.5)
            ax.axhline(0, color=P.MUTED, lw=0.4)
            r = stats_tab[(stats_tab.subtype == sub) & (stats_tab.split == split)]
            if len(r) and np.isfinite(r.iloc[0].get("p_perm", np.nan)):
                r = r.iloc[0]
                ax.set_title(f"{SUBS[sub]} · Δ={r.surface_minus_core:+.2f}, g={r.hedges_g:+.2f}\nhelix-perm {fmt_p(r.p_perm)}",
                             fontsize=7)
            else:
                ax.set_title(SUBS[sub], fontsize=7)
        # continuous: helix exposure vs helix mean
        ax = axes[2, j]
        hm = d.groupby(_hid(d)).agg(f=(col, "mean"), rsa=("helix_rsa", "first"), c=("class_median", "first"))
        for cls in ("core", "surface"):
            v = hm[hm.c == cls]
            ax.scatter(v.rsa, v.f, s=22, color=CLASS_COLORS[cls], edgecolor=P.INK, lw=0.3, label=cls)
        if hm.f.notna().sum() >= 4:
            rho, p = ss.spearmanr(hm.rsa, hm.f, nan_policy="omit")
            ax.set_title(f"{SUBS[sub]} · helix RSA vs mean: ρ={rho:+.2f} ({fmt_p(p)})", fontsize=7)
        ax.set_xlabel("Helix median RSA (AlphaFold, no membrane)")
        ax.axhline(0, color=P.MUTED, lw=0.4)
    axes[0, 0].set_ylabel("Median split\nposition mean fitness", fontsize=7)
    axes[1, 0].set_ylabel("Tertile split\nposition mean fitness", fontsize=7)
    axes[2, 0].set_ylabel("Helix mean fitness", fontsize=7)
    axes[2, 0].legend(loc="lower right", fontsize=6)
    fig.suptitle(f"{title} — surface (lipid-exposed) vs core TM helices (open diamonds = helix means)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    return fig


def fig_per_helix(t, ph, title):
    d = t[t.confident]
    keys = list(d.groupby(["dataset_id", "helix"], sort=False).groups)
    order = sorted(keys, key=lambda k: (k[0], d[(d.dataset_id == k[0]) & (d.helix == k[1])].helix_start.iloc[0]))
    n = len(order)
    ncol = min(6, n)
    nrow = int(np.ceil(n / ncol)) + 1
    H = 1.45 * nrow + 1.1
    fig = plt.figure(figsize=(1.75 * ncol + 0.6, H))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.75, wspace=0.25, top=1 - 0.75 / H, bottom=0.9 / H)
    for k, (ds, hx) in enumerate(order):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        g = d[(d.dataset_id == ds) & (d.helix == hx)].sort_values("rel_pos")
        cls = g.class_median.iloc[0]
        ax.set_facecolor("#F7F3FA" if cls == "core" else "#F0F7FB")
        ax.plot(g.rel_pos, g.fit_rest, "-o", ms=2.2, lw=0.9, color=SUB_COLORS["rest"])
        ax.plot(g.rel_pos, g.fit_P, "s", ms=2.6, color=SUB_COLORS["P"])
        ax.plot(g.rel_pos, g.fit_G, "^", ms=2.6, color=SUB_COLORS["G"])
        ax.axhline(0, color=P.MUTED, lw=0.4)
        ax.set_ylim(-1.6, 0.6)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["cyto", "0", "lum"], fontsize=5.5)
        ax.tick_params(labelsize=5.5)
        if k % ncol:
            ax.set_yticklabels([])
        lab = f"{ds}:" if t.dataset_id.nunique() > 1 else ""
        ax.set_title(f"{lab}{hx} {g.helix_start.iloc[0]}–{g.helix_end.iloc[0]}\n{cls}, RSA {g.helix_rsa.iloc[0]:.2f}",
                     fontsize=6.2)
    # ranking panel
    ax = fig.add_subplot(gs[nrow - 1, :])
    r = ph.sort_values("median").reset_index(drop=True)
    for i, row in r.iterrows():
        g = d[(d.dataset_id == row.dataset_id) & (d.helix == row.helix)]
        for sub, col, off in (("rest", "fit_rest", 0), ("P", "fit_P", -0.22), ("G", "fit_G", 0.22)):
            v = g[col].dropna()
            if len(v):
                ax.plot([i + off] * 2, np.quantile(v, [0.25, 0.75]), color=SUB_COLORS[sub], lw=1.2)
                ax.plot(i + off, v.median(), "o", ms=3.5, color=SUB_COLORS[sub],
                        mec=CLASS_COLORS[row.class_median] if sub == "rest" else SUB_COLORS[sub], mew=1.2)
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels([(f"{a}:" if t.dataset_id.nunique() > 1 else "") + b + ("*" if q < 0.05 else "")
                        for a, b, q in zip(r.dataset_id, r.helix, r.q_vs_other_helices.fillna(1))], rotation=90, fontsize=5.5)
    ax.axhline(0, color=P.MUTED, lw=0.4)
    ax.set_ylabel("Median (IQR)", fontsize=7)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=SUB_COLORS[s], marker="o", lw=1.2, label=SUBS[s]) for s in SUBS]
              + [Line2D([], [], color="white", marker="o", mec=CLASS_COLORS[c], mew=1.5, label=f"{c} helix") for c in ("surface", "core")],
              loc="lower right", bbox_to_anchor=(1.0, 1.12), ncol=5, fontsize=6)
    ax.set_title("Helices ranked by median (missense excl. Pro/Gly); * = differs from the protein's other TM helices (q<0.05)",
                 loc="left", fontsize=7, pad=22)
    fig.suptitle(f"{title} — each TM helix independently (shading: blue = surface, purple = core)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    return fig


# ------------------------------------------------------------------ driver
def analyse(tables: list[pd.DataFrame], title: str, outdir, cfg=None) -> dict:
    figdir, tabdir = dirs(outdir)
    rng = np.random.default_rng(13)
    t = pd.concat(tables, ignore_index=True)
    pos_rows, cls_rows = [], []
    for sub, col in COLS.items():
        pos_rows.append({"subtype": sub, **position_stats(t, col, rng)})
        for split in SPLITS:
            cls_rows.append({"subtype": sub, "split": split, **class_stats(t, col, split, rng)})
    pos_tab, cls_tab = pd.DataFrame(pos_rows), pd.DataFrame(cls_rows)
    ph = per_helix_stats(t, "fit_rest")
    for sub in ("P", "G"):
        x = per_helix_stats(t, COLS[sub])[["dataset_id", "helix", "median", "p_vs_other_helices", "q_vs_other_helices"]]
        ph = ph.merge(x.rename(columns={"median": f"median_{sub}", "p_vs_other_helices": f"p_{sub}",
                                        "q_vs_other_helices": f"q_{sub}"}), on=["dataset_id", "helix"], how="left")
    figs = []
    figs += P.save(fig_position(t, title, rng), figdir, "a13a_helix_position", cfg, "A13")
    figs += P.save(fig_classes(t, cls_tab, title, rng), figdir, "a13b_surface_vs_core", cfg, "A13")
    figs += P.save(fig_per_helix(t, ph, title), figdir, "a13c_per_helix", cfg, "A13")
    tabs = []
    for name, tab in (("a13_position_stats", pos_tab), ("a13_class_stats", cls_tab), ("a13_per_helix", ph),
                      ("a13_residues", t)):
        p = tabdir / f"{name}.csv"
        tab.to_csv(p, index=False)
        tabs.append(p)
    return {"pos": pos_tab, "cls": cls_tab, "ph": ph, "t": t, "figs": figs, "tabs": tabs}


def sensitivity(dfs_cfgs, outdir) -> pd.DataFrame:
    """Headline class/position statistics under UniProt vs DSSP helix boundaries."""
    rng = np.random.default_rng(14)
    rows = []
    for bnd in ("uniprot", "dssp"):
        t = pd.concat([tm_residue_table(df, cfg, boundaries=bnd) for df, cfg in dfs_cfgs], ignore_index=True)
        for sub, col in COLS.items():
            ps = position_stats(t, col, rng, n_perm=1000)
            for split in SPLITS:
                cs = class_stats(t, col, split, rng, n_perm=1000, mixed=False)
                rows.append({"boundaries": bnd, "subtype": sub, "split": split,
                             "spearman_absrel": ps.get("spearman_absrel"), "p_perm_position": ps.get("p_perm_within_helix"),
                             "surface_minus_core": cs.get("surface_minus_core"), "hedges_g": cs.get("hedges_g"),
                             "p_perm_class": cs.get("p_perm")})
    out = pd.DataFrame(rows)
    out.to_csv(dirs(outdir)[1] / "a13_sensitivity_boundaries.csv", index=False)
    return out


def headline(res) -> str:
    c = res["cls"]
    r = c[(c.subtype == "rest") & (c.split == "class_median")]
    pr = res["pos"][res["pos"].subtype == "rest"]
    bits = []
    if len(r) and np.isfinite(r.iloc[0].get("surface_minus_core", np.nan)):
        r = r.iloc[0]
        bits.append(f"surface − core helices = {r.surface_minus_core:+.2f} (g = {r.hedges_g:+.2f}, helix-permutation {fmt_p(r.p_perm)})")
    if len(pr) and "spearman_absrel" in pr:
        p = pr.iloc[0]
        bits.append(f"|distance from helix centre| vs fitness ρ = {p.spearman_absrel:+.2f} ({fmt_p(p.p_perm_within_helix)})")
    return "; ".join(bits) or "insufficient TM data"


def run(df, cfg, outdir):
    p = cfg.resolve(cfg.get_path("structure.path"))
    if p is None or not p.exists():
        return skipped("a13", cfg, "no structure (needed for RSA)")
    if not any(s.get("type") == "TM" for s in cfg.get_path("topology.segments", []) or []):
        return skipped("a13", cfg, "no TM segments")
    t = tm_residue_table(df, cfg)
    res = analyse([t], cfg.display_name, outdir, cfg)
    sens = sensitivity([(df, cfg)], outdir)
    caveats = ["AlphaFold monomer: helices at an oligomer interface will look 'surface'",
               f"{res['ph'].shape[0]} TM helices - helix-level tests have limited power; read effect sizes and CIs"]
    flip = sens.groupby(["subtype", "split"]).surface_minus_core.apply(lambda x: np.sign(x).nunique() > 1)
    if flip.any():
        caveats.append("surface−core sign changes between UniProt and DSSP helix boundaries for: "
                       + ", ".join(f"{a}/{b}" for (a, b), v in flip.items() if v))
    r = res["cls"]
    metrics = {f"{row.subtype}_{row.split}_{k}": row[k] for _, row in r.iterrows()
               for k in ("surface_minus_core", "hedges_g", "p_perm", "mixedlm_p") if k in row and pd.notna(row[k])}
    metrics.update({f"{row.subtype}_spearman_absrel": row.get("spearman_absrel") for _, row in res["pos"].iterrows()})
    return result("a13", cfg, headline(res), metrics, caveats, res["figs"], res["tabs"])
