"""A06 - charge introduction and the positive-inside rule."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import sem as ss_sem

from .. import plotting as P
from ..annot import segments_df
from .base import dirs, missense, result, skipped

FLANK = 10
ACID, BASE = set("DE"), set("KR")
BREAKERS = {"proline": ("P", "#E07B39"), "glycine": ("G", "#7B5EA7")}  # distinct from the D/E red
THIRDS = ["cyto third", "central third", "lumenal third"]


def annotate(d: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    """Map each variant to (TM, x) where x = 0 at the cytosolic end, 1 at the lumenal end;
    flanks get x in [-0.5, 0) (cytosolic) or (1, 1.5] (lumenal)."""
    rows = []
    tms = segs[segs.type == "TM"]
    for s in tms.itertuples():
        L = s.end - s.start + 1
        in_out = (s.orientation or "in_out") == "in_out"
        lo, hi = s.start - FLANK, s.end + FLANK
        sub = d[d.pos.between(lo, hi)]
        for idx, r in sub.iterrows():
            if s.start <= r.pos <= s.end:
                f = (r.pos - s.start) / max(L - 1, 1)
                x = f if in_out else 1 - f
                region = THIRDS[min(int(x * 3), 2)]
            else:
                before = r.pos < s.start
                dist = (s.start - r.pos) if before else (r.pos - s.end)
                cyto = before == in_out
                x = -dist / (2 * FLANK) if cyto else 1 + dist / (2 * FLANK)
                region = "cyto flank" if cyto else "lumenal flank"
            rows.append({"idx": idx, "tm": s.name, "orientation": s.orientation, "x": x, "region": region})
    a = pd.DataFrame(rows)
    if a.empty:
        return a
    return a.merge(d, left_on="idx", right_index=True)


def run(df, cfg, outdir):
    segs = segments_df(cfg)
    if segs.empty or not (segs.type == "TM").any():
        return skipped("a06", cfg, "no TM segments")
    figdir, tabdir = dirs(outdir)
    d = missense(df)
    d = d[~d.wt.isin(list(ACID | BASE)) & d.mut.isin(list(ACID | BASE))].copy()
    d["charge"] = np.where(d.mut.isin(list(ACID)), "acidic", "basic")
    br = missense(df)
    br = br[~br.wt.isin(["P", "G"]) & br.mut.isin(["P", "G"])].copy()
    br["breaker"] = np.where(br.mut == "P", "proline", "glycine")
    br = annotate(br, segs) if len(br) else br

    a = annotate(d, segs)
    if a.empty or a.charge.nunique() < 2:
        return skipped("a06", cfg, "too few charge-introducing variants in/near TMs")

    rows, metrics, caveats = [], {}, []
    for tm, g in [("pooled", a)] + list(a.groupby("tm")):
        for reg in ["cyto flank"] + THIRDS + ["lumenal flank"]:
            s = g[g.region == reg]
            ac, ba = s[s.charge == "acidic"].score_z.to_numpy(), s[s.charge == "basic"].score_z.to_numpy()
            if len(ac) >= 2 and len(ba) >= 2:
                r = np.random.default_rng(0)
                boots = [np.mean(r.choice(ac, len(ac))) - np.mean(r.choice(ba, len(ba))) for _ in range(2000)]
                diff, lo, hi = np.mean(ac) - np.mean(ba), *np.quantile(boots, [0.025, 0.975])
            else:
                diff = lo = hi = np.nan
            rows.append({"tm": tm, "region": reg, "n_acidic": len(ac), "n_basic": len(ba),
                         "mean_acidic": np.mean(ac) if len(ac) else np.nan, "mean_basic": np.mean(ba) if len(ba) else np.nan,
                         "acid_minus_basic": diff, "ci_lo": lo, "ci_hi": hi})
    # helix breakers go in their own table: mixing them into the charge rows would give
    # the pooled lookups duplicate region labels and silently return Series
    brows = []
    for name in BREAKERS:
        g = br[br.breaker == name] if len(br) else br
        for reg in ["cyto flank"] + THIRDS + ["lumenal flank"]:
            v = g[g.region == reg].score_z.to_numpy() if len(g) else np.array([])
            brows.append({"introduced": name, "region": reg, "n": len(v),
                          "mean": float(np.mean(v)) if len(v) else np.nan,
                          "sem": float(ss_sem(v)) if len(v) > 1 else np.nan})
    brtab = pd.DataFrame(brows)
    tab = pd.DataFrame(rows)
    pooled = tab[tab.tm == "pooled"].set_index("region")
    for reg in THIRDS:
        key = reg.split()[0]
        metrics[f"acid_minus_basic_{key}"] = pooled.loc[reg, "acid_minus_basic"]
        metrics[f"acid_minus_basic_{key}_ci"] = [pooled.loc[reg, "ci_lo"], pooled.loc[reg, "ci_hi"]]

    # two-way model: charge x third (TM interior only)
    inner = a[a.region.isin(THIRDS)].copy()
    tests = []
    for tm, g in [("pooled", inner)] + list(inner.groupby("tm")):
        if g.charge.nunique() == 2 and g.region.nunique() >= 2 and len(g) >= 12:
            try:
                m = smf.ols("score_z ~ C(charge) * C(region)", data=g).fit(
                    cov_type="cluster", cov_kwds={"groups": g["pos"]}) if g.pos.nunique() > 5 else \
                    smf.ols("score_z ~ C(charge) * C(region)", data=g).fit()
                inter = [k for k in m.params.index if ":" in k]
                wald = m.wald_test(" = 0, ".join(inter) + " = 0", scalar=True) if inter else None
                p = float(wald.pvalue) if wald is not None else np.nan
            except Exception:
                p = np.nan
            tests.append({"tm": tm, "interaction_p": p, "n": len(g)})
    tests = pd.DataFrame(tests)
    metrics["pooled_interaction_p"] = float(tests.loc[tests.tm == "pooled", "interaction_p"].iloc[0]) if len(tests) else np.nan

    # direction check per TM: positive-inside predicts acid-basic more negative at the cytosolic end
    recheck = []
    for tm, g in tab[tab.tm != "pooled"].groupby("tm"):
        g = g.set_index("region")
        c = g.loc["cyto third", "acid_minus_basic"] if "cyto third" in g.index else np.nan
        l_ = g.loc["lumenal third", "acid_minus_basic"] if "lumenal third" in g.index else np.nan
        if np.isfinite(c) and np.isfinite(l_) and c - l_ > 0.15:
            s = segs.set_index("name").loc[tm]
            recheck.append(f"{tm} ({s.start}-{s.end})")
    consistent = np.isfinite(metrics.get("acid_minus_basic_cyto", np.nan)) and \
        metrics["acid_minus_basic_cyto"] <= metrics.get("acid_minus_basic_lumenal", np.inf)
    metrics["positive_inside_consistent"] = bool(consistent)
    metrics["tms_inverted"] = recheck
    if recheck:
        caveats.append("asymmetry INVERTED relative to annotated topology in " + ", ".join(recheck)
                       + " - real finding or orientation error; re-check these boundaries/n_terminus")

    # figure: faceted by orientation
    orients = [o for o in ("in_out", "out_in") if o in set(a.orientation)]
    fig, axes = plt.subplots(1, len(orients), figsize=(3.4 * len(orients), 2.5), squeeze=False, sharey=True)
    bins = np.concatenate([np.linspace(-0.5, 0, 3), np.linspace(0, 1, 7)[1:], np.linspace(1, 1.5, 3)[1:]])
    for ax, o in zip(axes[0], orients):
        g = a[a.orientation == o]
        ax.axvspan(0, 1, color=P.TM_GREY, lw=0, zorder=0)
        for ch, col in (("acidic", "#C0392B"), ("basic", "#1B4B80")):
            s = g[g.charge == ch]
            b = pd.cut(s.x, bins, include_lowest=True)
            m = s.groupby(b, observed=False).score_z.agg(["mean", "sem", "size"])
            xc = [iv.mid for iv in m.index]
            ax.errorbar(xc, m["mean"], yerr=m["sem"], fmt="-o", ms=3, lw=1, color=col, capsize=0,
                        label=f"{'D/E' if ch == 'acidic' else 'K/R'} introduced")
        for name, (aa, col) in BREAKERS.items():
            s_ = br[br.orientation == o] if len(br) else br
            s_ = s_[s_.breaker == name] if len(s_) else s_
            if len(s_) < 6:
                continue
            b = pd.cut(s_.x, bins, include_lowest=True)
            m = s_.groupby(b, observed=False).score_z.agg(["mean", "sem", "size"])
            ax.errorbar([iv.mid for iv in m.index], m["mean"], yerr=m["sem"], fmt="--s", ms=2.5,
                        lw=0.9, color=col, capsize=0, label=f"{aa} introduced")
        ax.axhline(0, color=P.MUTED, lw=0.5)
        ax.set_xticks([-0.25, 0, 0.5, 1, 1.25])
        ax.set_xticklabels(["cyto\nflank", "cyto\nend", "centre", "lumen\nend", "lumen\nflank"])
        ax.set_title(f"TMs oriented {o.replace('_', '→')}", loc="left")
        ax.text(0.99, 0.02, f"{g.tm.nunique()} helices", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color=P.MUTED)
    axes[0, 0].set_ylabel(cfg.get_path("plotting.score_label"))
    # one legend under the panels: inside the axes it sat on the cytosolic-flank points
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=len(lb), frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02))
    P.title(fig, cfg, "A06 charge and helix-breaker introduction across TMDs")
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    figs = P.save(fig, figdir, "a06_charge_topology", cfg, "A06")
    t1, t2 = tabdir / "a06_charge_by_region.csv", tabdir / "a06_interaction_tests.csv"
    tab.to_csv(t1, index=False)
    tests.to_csv(t2, index=False)
    t3 = tabdir / "a06_breaker_by_region.csv"
    brtab.to_csv(t3, index=False)
    ip = metrics["pooled_interaction_p"]
    verdict = ("no significant charge × depth interaction" if not (np.isfinite(ip) and ip < 0.05)
               else "consistent with positive-inside" if consistent else "OPPOSITE to positive-inside")
    metrics["verdict"] = verdict
    head = (f"Acidic−basic score difference is {metrics['acid_minus_basic_cyto']:+.2f} at the cytosolic third vs "
            f"{metrics['acid_minus_basic_lumenal']:+.2f} at the lumenal third "
            f"({verdict}; interaction p = {metrics['pooled_interaction_p']:.2g})")
    # the breaker lines carry a prediction too: proline should be worst mid-membrane
    bp = brtab[brtab.introduced == "proline"].set_index("region")["mean"]
    if {"central third", "cyto flank", "lumenal flank"} <= set(bp.index) and bp.notna().all():
        edge = float(np.nanmean([bp["cyto flank"], bp["lumenal flank"]]))
        metrics["proline_centre_minus_flank"] = float(bp["central third"] - edge)
        head += f"; proline centre − flank = {metrics['proline_centre_minus_flank']:+.2f}"
    metrics["breaker_profile"] = brtab.to_dict("records")
    return result("a06", cfg, head, metrics, caveats, figs, [t1, t2, t3])
