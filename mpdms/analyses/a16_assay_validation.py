"""A16 - does this assay measure membrane-protein biogenesis? A pre-registered battery.

Each test states a direction that a genuine biogenesis/stability readout must satisfy,
derived from independent biophysics (not from these data). The battery is scored
pass/fail with an effect size and CI, so the claim "I am measuring biogenesis" becomes a
number (tests passed / tests testable) rather than an impression.

The tests are deliberately of three kinds:
  * sanity      - the assay separates its own controls and replicates (K01-K02)
  * mechanism   - it reproduces known membrane-insertion physics (K03-K09)
  * independent - it agrees with data the assay never saw (K10-K12)

K12 is the strongest internal control available for an MFS transporter: the N- and
C-terminal 6-TM bundles are structurally equivalent by pseudo-twofold symmetry, so a real
structural readout must give correlated profiles across the two halves, while an artefact
of library construction or sequencing has no reason to.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..annot import segments_df
from ..io import replicate_cols
from ..stats import auc_below, circular_shift_test
from .base import dirs, fmt_p, result

POLAR = set("DEKRNQHST")
HYDROPHOBIC = set("AVLIFMWC")
N_BOOT = 1000


def _t(tid, claim, value, lo=np.nan, hi=np.nan, p=np.nan, passed=None, note="", kind="mechanism"):
    return {"id": tid, "kind": kind, "claim": claim, "value": value, "ci_lo": lo, "ci_hi": hi,
            "p": p, "testable": passed is not None, "passed": bool(passed) if passed is not None else False,
            "note": note}


def _diff(a, b, seed=0):
    """mean(a) - mean(b) with a bootstrap CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
          for _ in range(N_BOOT)]
    return float(a.mean() - b.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


# --------------------------------------------------------------------- tests
def k01_dynamic_range(df, cfg, mis, pos):
    d = df[df.pass_filter]
    syn = d.loc[d.vclass == "synonymous", "score_z"]
    stop = d.loc[d.vclass == "nonsense", "score_z"]
    if len(syn) < 10 or len(stop) < 10:
        return _t("K01", "stop variants separate from synonymous", np.nan, kind="sanity",
                  note="too few controls")
    auc = auc_below(stop, syn)
    return _t("K01", "stop variants separate from synonymous (AUC ≥ 0.90)", auc,
              passed=auc >= 0.90, kind="sanity", note=f"n_stop={len(stop)}, n_syn={len(syn)}")


def k02_reproducibility(df, cfg, mis, pos):
    reps = replicate_cols(df)
    if len(reps) < 2:
        return _t("K02", "replicates agree", np.nan, kind="sanity", note="<2 replicates")
    d = df[df.pass_filter]
    rs = [ss.pearsonr(*[d[c] for c in (a, b)][:2])[0] for a in reps for b in reps if a < b
          and d[[a, b]].notna().all(axis=1).sum() > 10]
    r = float(np.mean(rs))
    return _t("K02", "replicates agree (mean Pearson r ≥ 0.6)", r, passed=r >= 0.6, kind="sanity",
              note=f"{len(reps)} replicates")


def k03_tm_sensitivity(df, cfg, mis, pos):
    prof = mis.groupby("pos").score_z.median()
    full = prof.reindex(range(int(df.pos.min()), int(df.pos.max()) + 1))
    tm = full.index.isin(pos.loc[pos.is_tm == 1, "pos"])
    ok = full.notna().to_numpy()
    if tm.sum() < 10 or (~tm).sum() < 10:
        return _t("K03", "TM positions more sensitive than non-TM", np.nan, note="no topology")
    r = circular_shift_test(full.to_numpy()[ok], tm[ok], n=5000, alternative="less")
    return _t("K03", "TM positions more sensitive than non-TM (circular-shift test)",
              r["observed"], p=r["p"], passed=(r["observed"] < 0 and r["p"] < 0.05),
              note="shift test preserves positional autocorrelation")


def k04_biological_scale(df, cfg, mis, pos):
    """von Heijne's biological (insertion) scale should explain TM substitutions better than
    Kyte-Doolittle: the assay reads out Sec61-mediated insertion, not bulk transfer."""
    import statsmodels.formula.api as smf
    d = mis[mis.seg_type == "TM"].dropna(subset=["delta_biological", "delta_kd", "score_z"])
    if len(d) < 100:
        return _t("K04", "biological hydrophobicity scale beats Kyte-Doolittle inside TMDs", np.nan,
                  note="too few TM variants")
    r2 = {}
    for scale in ("biological", "kd"):
        m = smf.ols(f"score_z ~ delta_{scale} + delta_volume + delta_charge + delta_helix", data=d).fit()
        r2[scale] = m.rsquared_adj
    v = r2["biological"] - r2["kd"]
    return _t("K04", "biological hydrophobicity scale beats Kyte-Doolittle inside TMDs (ΔadjR² > 0)",
              v, passed=v > 0, note=f"adjR² bio {r2['biological']:.3f} vs KD {r2['kd']:.3f}")


def k05_polar_in_tm(df, cfg, mis, pos):
    sub = mis[mis.wt.isin(list(HYDROPHOBIC)) & mis.mut.isin(list(POLAR))]
    v, lo, hi = _diff(sub.loc[sub.seg_type == "TM", "score_z"],
                      sub.loc[sub.seg_type.isin(["loop", "soluble"]), "score_z"], seed=5)
    return _t("K05", "hydrophobic→polar is worse inside TMDs than in loops", v, lo, hi,
              passed=None if not np.isfinite(v) else (v < 0 and hi < 0))


def k06_proline(df, cfg, mis, pos):
    pro = mis[mis.mut == "P"]
    v, lo, hi = _diff(pro.loc[pro.seg_type == "TM", "score_z"],
                      pro.loc[pro.seg_type.isin(["loop", "soluble"]), "score_z"], seed=6)
    return _t("K06", "proline is worse inside TM helices than in loops", v, lo, hi,
              passed=None if not np.isfinite(v) else (v < 0 and hi < 0))


def k07_charge_depth(df, cfg, mis, pos):
    """Charge burial costs most at the bilayer centre: compare the central third of each TMD
    with the terminal thirds."""
    sub = mis[(mis.seg_type == "TM") & ~mis.wt.isin(list("DEKR")) & mis.mut.isin(list("DEKR"))]
    sub = sub.dropna(subset=["abs_rel"])
    v, lo, hi = _diff(sub.loc[sub.abs_rel <= 1 / 3, "score_z"], sub.loc[sub.abs_rel >= 2 / 3, "score_z"], seed=7)
    return _t("K07", "introduced charge is worse at the TMD centre than at its ends", v, lo, hi,
              passed=None if not np.isfinite(v) else (v < 0 and hi < 0))


def k08_positive_inside(df, cfg, mis, pos):
    """von Heijne's positive-inside rule: K/R are better tolerated towards the cytosolic end
    of a TMD than D/E, and the asymmetry reverses towards the lumenal end."""
    sub = mis[(mis.seg_type == "TM") & ~mis.wt.isin(list("DEKR"))].dropna(subset=["rel_pos"])
    acid, base = sub[sub.mut.isin(list("DE"))], sub[sub.mut.isin(list("KR"))]
    cyto = _diff(acid.loc[acid.rel_pos < -1 / 3, "score_z"], base.loc[base.rel_pos < -1 / 3, "score_z"], seed=8)[0]
    lum = _diff(acid.loc[acid.rel_pos > 1 / 3, "score_z"], base.loc[base.rel_pos > 1 / 3, "score_z"], seed=9)[0]
    v = cyto - lum
    return _t("K08", "positive-inside: K/R favoured over D/E at the cytosolic end vs the lumenal end",
              v, passed=None if not np.isfinite(v) else v < 0,
              note=f"acid−base cytosolic {cyto:+.2f}, lumenal {lum:+.2f}")


def k09_burial(df, cfg, mis, pos):
    d = mis[(mis.seg_type == "TM") & mis.plddt.ge(70)].dropna(subset=["rsa"])
    v, lo, hi = _diff(d.loc[d.rsa < 0.10, "score_z"], d.loc[d.rsa > 0.25, "score_z"], seed=10)
    return _t("K09", "buried TM residues are more sensitive than lipid-facing ones", v, lo, hi,
              passed=None if not np.isfinite(v) else (v < 0 and hi < 0), kind="independent",
              note="RSA from the AlphaFold monomer")


def k10_evolution(df, cfg, mis, pos):
    d = mis.dropna(subset=["esm1v", "score_z"])
    if len(d) < 200:
        return _t("K10", "evolutionary constraint (ESM-1v) correlates with measured sensitivity",
                  np.nan, kind="independent", note="no ESM-1v scores - run `mpdms esm`")
    site = d.groupby("pos").agg(s=("score_z", "mean"), e=("esm1v", "mean"))
    rho, p = ss.spearmanr(site.s, site.e)
    return _t("K10", "evolutionary constraint (ESM-1v) correlates with measured sensitivity (ρ > 0)",
              rho, p=p, passed=(rho > 0 and p < 0.05), kind="independent",
              note=f"{len(site)} positions, variant-level ρ={ss.spearmanr(d.score_z, d.esm1v)[0]:.2f}")


def k11_nonsense_profile(df, cfg, mis, pos):
    """A biogenesis readout must register truncation anywhere in the protein. If stops are
    deleterious in only part of the sequence, the assay is reporting on a fragment (a
    terminal tag, a partial transcript) rather than on the full-length protein."""
    from .a11_nonsense_profile import changepoint
    d = df[df.pass_filter & (df.vclass == "nonsense")].dropna(subset=["score_z"])
    if len(d) < 20:
        return _t("K11", "truncation is deleterious throughout the protein", np.nan, kind="independent",
                  note="too few nonsense variants")
    lo, hi = int(df.pos.min()), int(df.pos.max())
    cut = lo + 0.9 * (hi - lo)            # the last 10% may be a dispensable tail
    body = d[d.pos <= cut]
    edges = np.arange(lo, cut + 20, 20)
    med = body.groupby(pd.cut(body.pos, edges, include_lowest=True), observed=True).score_z.median().dropna()
    frac = float((med < -0.5).mean()) if len(med) else np.nan
    cp = changepoint(d.pos.to_numpy(), d.score_z.to_numpy())
    note = f"{len(med)} windows of 20 residues" + (f"; tolerance changepoint at {cp}" if cp else "")
    return _t("K11", "truncation is deleterious throughout the protein (≥80% of windows)", frac,
              passed=None if not np.isfinite(frac) else frac >= 0.8, kind="independent", note=note)


def _bundle_profiles(mis, segs, n_grid=12):
    """Interpolate each TM helix's sensitivity profile onto a common relative-position grid."""
    prof = mis.groupby("pos").score_z.mean()
    grid = np.linspace(-1, 1, n_grid)
    out = {}
    for s in segs.itertuples():
        p = np.arange(s.start, s.end + 1)
        y = prof.reindex(p).to_numpy(float)
        in_out = (s.orientation or "in_out") == "in_out"
        f = (p - s.start) / max(len(p) - 1, 1)
        rel = (2 * f - 1) if in_out else (1 - 2 * f)
        o = np.argsort(rel)
        ok = np.isfinite(y[o])
        if ok.sum() >= 6:
            out[s.name] = np.interp(grid, rel[o][ok], y[o][ok])
    return out


def k12_pseudosymmetry(df, cfg, mis, pos):
    """MFS transporters are built from two structurally equivalent 6-TM bundles. A readout of
    protein structure must give correlated profiles across the pseudo-twofold; a technical
    artefact need not."""
    segs = segments_df(cfg)
    tms = segs[segs.type == "TM"].sort_values("start").reset_index(drop=True)
    n = len(tms)
    if n < 8 or n % 2:
        return _t("K12", "the two structural bundles give correlated profiles", np.nan,
                  kind="independent", note=f"{n} TM helices - not an even two-bundle fold")
    prof = _bundle_profiles(mis, tms)
    half = n // 2
    pairs = [(tms.name[i], tms.name[i + half]) for i in range(half)
             if tms.name[i] in prof and tms.name[i + half] in prof]
    if len(pairs) < 4:
        return _t("K12", "the two structural bundles give correlated profiles", np.nan,
                  kind="independent", note="too few helices with coverage")
    a = np.concatenate([prof[x] for x, _ in pairs])
    b = np.concatenate([prof[y] for _, y in pairs])
    rho = ss.spearmanr(a, b)[0]
    # null: re-pair the two bundles at random
    rng = np.random.default_rng(12)
    nulls = []
    ys = [y for _, y in pairs]
    for _ in range(2000):
        perm = rng.permutation(len(ys))
        if np.all(perm == np.arange(len(ys))):
            continue
        nulls.append(ss.spearmanr(a, np.concatenate([prof[ys[k]] for k in perm]))[0])
    p = (1 + np.sum(np.array(nulls) >= rho)) / (1 + len(nulls))
    return _t("K12", "the two pseudo-symmetric TM bundles give correlated profiles (ρ > 0)", rho,
              p=p, passed=(rho > 0 and p < 0.05), kind="independent",
              note=f"{len(pairs)} helix pairs, null = random re-pairing")


TESTS = [k01_dynamic_range, k02_reproducibility, k03_tm_sensitivity, k04_biological_scale,
         k05_polar_in_tm, k06_proline, k07_charge_depth, k08_positive_inside, k09_burial,
         k10_evolution, k11_nonsense_profile, k12_pseudosymmetry]


def battery(df, cfg) -> pd.DataFrame:
    from ..features import position_table, variant_features
    mis = variant_features(df, cfg)
    pos = position_table(df, cfg)
    rows = []
    for fn in TESTS:
        try:
            rows.append(fn(df, cfg, mis, pos))
        except Exception as e:  # a failing test must not kill the battery
            rows.append(_t(fn.__name__[:3].upper(), fn.__doc__ or fn.__name__, np.nan,
                           note=f"error: {e.__class__.__name__}: {e}"))
    return pd.DataFrame(rows)


KIND_COLOR = {"sanity": "#6B6B6B", "mechanism": "#2F6DB5", "independent": "#1B9E77"}


def figure(tab: pd.DataFrame, title: str):
    t = tab.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.4, 0.34 * len(t) + 1.6))
    for i, r in t.iterrows():
        col = KIND_COLOR.get(r.kind, P.MUTED)
        if not r.testable:
            ax.text(0.01, i, "not testable: " + r.note, transform=ax.get_yaxis_transform(),
                    va="center", fontsize=6, color=P.MUTED)
            continue
        v = r.value
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.4, solid_capstyle="butt")
        ax.plot(v, i, "o", ms=5, color=col if r.passed else "white", mec=col, mew=1.4, zorder=3)
        lab = f"{v:+.2f}" if abs(v) < 10 else f"{v:.3g}"
        if np.isfinite(r.p):
            lab += f"  {fmt_p(r.p)}"
        ax.text(1.02, i, ("met    " if r.passed else "NOT MET  ") + lab,
                transform=ax.get_yaxis_transform(), va="center", fontsize=6.5,
                color=col if r.passed else P.CLASS_COLORS["nonsense"])
    ax.axvline(0, color=P.INK, lw=0.6)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{r.id}  {r.claim}" for _, r in t.iterrows()], fontsize=6.5)
    ax.set_xlabel("Effect size (units of normalised fitness, or ρ / AUC where noted)")
    n_pass, n_test = int(tab.passed.sum()), int(tab.testable.sum())
    ax.set_title(f"{title} — assay validation battery: {n_pass}/{n_test} pre-registered predictions met",
                 loc="left", fontsize=9.5, fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=c, marker="o", lw=1.4, label=k) for k, c in KIND_COLOR.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=6.5, ncol=3)
    ax.set_ylim(-0.6, len(t) - 0.4)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    return fig


def run(df, cfg, outdir):
    figdir, tabdir = dirs(outdir)
    tab = battery(df, cfg)
    figs = P.save(figure(tab, cfg.display_name), figdir, "a16_assay_validation", cfg, "A16")
    t = tabdir / "a16_validation_battery.csv"
    tab.to_csv(t, index=False)
    n_pass, n_test = int(tab.passed.sum()), int(tab.testable.sum())
    failed = tab[tab.testable & ~tab.passed]
    caveats = [f"{r.id} failed: {r.claim} (value {r.value:+.3g})" for _, r in failed.iterrows()]
    caveats += [f"{r.id} not testable: {r.note}" for _, r in tab[~tab.testable].iterrows()]
    caveats.append("passing the battery shows the readout behaves like membrane-protein biogenesis; "
                   "it does not by itself separate insertion from post-insertional degradation")
    head = (f"{n_pass}/{n_test} pre-registered predictions met"
            + (f"; failed: {', '.join(failed.id)}" if len(failed) else " (all testable predictions met)"))
    return result("a16", cfg, head, {"n_passed": n_pass, "n_testable": n_test,
                                     "fraction_passed": n_pass / n_test if n_test else np.nan,
                                     **{r.id: r.value for _, r in tab.iterrows()}},
                  caveats, figs, [t])
