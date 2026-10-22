"""`python -m mpdms stability configs/AQR1_fitness.yaml --ddg AQR1_fitness=ThermoMPNN_P53943.csv`

Does a predicted change in folding stability explain the measured fitness effect, and
does it explain it equally well everywhere in the protein?

ThermoMPNN predicts ddG for a point substitution in a folded protein. It was trained on
soluble proteins, where "buried" means buried in water-excluded packing, so a membrane
protein asks it something it has not seen: the lipid-facing surface of a transporter is
as water-excluded as a soluble core while being chemically its opposite, and the
cavity is a water-filled channel in the middle of the membrane. A single correlation
over the whole protein averages those worlds together and reports a number that belongs
to none of them.

So every analysis here is the same correlation asked separately in each environment:

    t01  the whole protein, one point per variant - the number people usually quote
    t02  the same correlation split by burial class, with the differences tested
    t03  the same correlation against relative solvent accessibility, as a curve

Burial classes are the pipeline's own, the ones `burial` and `environ` use, so a
position called core here is the same residue called core there:

    core      inside a TM helix bundle, packed against other helices
    cavity    inside the membrane, facing the central translocation pathway
    lipid     inside the membrane, facing acyl chains
    aqueous   outside the membrane, in water

Every interval is a bootstrap over POSITIONS, not over variants. Nineteen substitutions
at one residue are not nineteen independent tests of a prediction method: they share
that residue's structural context, and resampling variants would give intervals several
times too narrow.
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
from .burial import BCLASSES, BCOLORS, BLABEL, bits_for, empty_panel, short_name
from .config import REPO_ROOT, load_config
from .environs import grid, panel

OUT = REPO_ROOT / "outputs" / "_stability"
MIN_PAIRS = 40            # variants in a group, before it gets a correlation
MIN_POSITIONS = 8         # positions in a group, before a bootstrap means anything
N_BOOT = 2000
MIN_WT_AGREEMENT = 0.95   # wild-type letters matching, before an offset is believed
RSA_BINS = np.array([0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.01])
BURIED, EXPOSED = 0.15, 0.35      # the cuts the rest of the pipeline inherits
DDG_LABEL = "ThermoMPNN ΔΔG (kcal/mol, + = destabilising)"


# --------------------------------------------------------------------- loading
DDG_COLUMNS = {"ddg_pred": "ddg", "ddg": "ddg", "ddg_predicted": "ddg",
               "predicted_ddg": "ddg", "position": "position", "pos": "position",
               "wildtype": "wt", "wt": "wt", "wt_aa": "wt",
               "mutation": "mut", "mut": "mut", "mutant": "mut"}


def read_ddg(path: Path) -> pd.DataFrame:
    """A ThermoMPNN inference table, as position / wt / mut / ddg.

    Column names are matched case-insensitively against the handful of spellings these
    files come with, rather than assumed, because a renamed column in one protein's CSV
    would otherwise fail somewhere far from the cause.
    """
    d = pd.read_csv(path)
    ren = {c: DDG_COLUMNS[c.strip().lower()] for c in d.columns
           if c.strip().lower() in DDG_COLUMNS}
    d = d.rename(columns=ren)
    missing = {"position", "wt", "mut", "ddg"} - set(d.columns)
    if missing:
        raise SystemExit(f"{path}: no column for {sorted(missing)}; "
                         f"found {list(d.columns)}")
    d = d[["position", "wt", "mut", "ddg"]].dropna()
    d["position"] = d.position.astype(int)
    return d.reset_index(drop=True)


def find_offset(var: pd.DataFrame, ddg: pd.DataFrame,
                search=range(-200, 201)) -> tuple[int, float]:
    """The shift from the prediction's numbering to the assay's, by wild-type agreement.

    ThermoMPNN numbers from the structure it was given, which is usually 0-indexed and
    may start part-way into the protein. The only honest way to line the two up is the
    wild-type sequence itself: the right offset is the one where the predicted
    wild-type letter matches the assay's at (almost) every shared position.
    """
    a = var.drop_duplicates("pos").set_index("pos").wt
    b = ddg.drop_duplicates("position").set_index("position").wt
    best, best_frac, best_match = 0, 0.0, -1
    for off in search:
        idx = a.index.intersection(b.index + off)
        if len(idx) < 20:
            continue
        same = a.loc[idx].to_numpy() == b.reindex(idx - off).to_numpy()
        # ranked by how many residues MATCH, not by the fraction that do: a short
        # overlap that happens to agree everywhere would otherwise outrank the real
        # alignment, which is a residue or two short of perfect in most real files
        if (int(same.sum()), len(idx)) > (best_match, 0):
            best, best_frac, best_match = off, float(same.mean()), int(same.sum())
    return best, best_frac


def attach_ddg(b: dict, ddg: pd.DataFrame, offset: int | None = None,
               min_agreement: float = MIN_WT_AGREEMENT) -> tuple[pd.DataFrame, dict]:
    """Merge predictions onto the variant table, refusing a numbering we cannot trust.

    A wrong offset does not fail loudly: it produces a full table of variants whose
    predictions belong to other residues, and a correlation near zero that looks like a
    result. So poor wild-type agreement stops the protein here rather than warning.
    """
    var = b["var"]
    off, frac = find_offset(var, ddg) if offset is None else (offset, np.nan)
    if offset is None and frac < min_agreement:
        raise SystemExit(
            f"best offset {off:+d} matches only {frac:.1%} of wild-type residues "
            f"(need {min_agreement:.0%}). The structure and the assay are probably not "
            f"the same construct - pass --offset to override if you know they are.")
    d = ddg.assign(pos=ddg.position + off).rename(columns={"wt": "wt_ddg"})
    m = var.merge(d[["pos", "wt_ddg", "mut", "ddg"]], on=["pos", "mut"], how="inner")
    bad = int((m.wt != m.wt_ddg).sum())
    m = m[m.wt == m.wt_ddg].drop(columns="wt_ddg")
    m = m.dropna(subset=["score_z", "ddg"])
    return m, {"offset": int(off), "wt_agreement": float(frac), "n_merged": int(len(m)),
               "n_wt_mismatch": bad}


# ------------------------------------------------------------- the correlation
def corr_with_ci(d: pd.DataFrame, method: str = "spearman", n_boot: int = N_BOOT,
                 seed: int = 0) -> dict:
    """Correlation of ddG with fitness, with an interval bootstrapped over positions.

    Resampling positions rather than variants is the whole point: the question is how
    well the predictor does across the protein's residues, and a residue contributes one
    piece of independent evidence however many substitutions were measured at it.
    """
    d = d.dropna(subset=["ddg", "score_z"])
    if len(d) < MIN_PAIRS or d.pos.nunique() < MIN_POSITIONS:
        return {"n": int(len(d)), "n_positions": int(d.pos.nunique()), "r": np.nan}
    f = ss.spearmanr if method == "spearman" else ss.pearsonr
    obs = float(f(d.ddg, d.score_z)[0])
    groups = [g for _, g in d.groupby("pos")[["ddg", "score_z"]]]
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[k] for k in pick])
        if s.ddg.nunique() > 2 and s.score_z.nunique() > 2:
            draws[i] = f(s.ddg, s.score_z)[0]
    draws = draws[np.isfinite(draws)]
    return {"n": int(len(d)), "n_positions": int(d.pos.nunique()), "r": obs,
            "lo": float(np.quantile(draws, 0.025)) if len(draws) else np.nan,
            "hi": float(np.quantile(draws, 0.975)) if len(draws) else np.nan,
            "se": float(np.std(draws, ddof=1)) if len(draws) > 1 else np.nan,
            "p": float(f(d.ddg, d.score_z)[1]), "method": method,
            "draws": draws}


def difference_in_corr(a: pd.DataFrame, b: pd.DataFrame, method: str = "spearman",
                       n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Is the predictor better in one group than the other?

    Two correlations and two intervals do not answer that: overlapping intervals are not
    a test, and non-overlapping ones overstate the evidence. The difference is
    bootstrapped directly, resampling positions inside each group, so the two are
    compared on the same draws.
    """
    ra = corr_with_ci(a, method, n_boot, seed)
    rb = corr_with_ci(b, method, n_boot, seed + 1)
    if not np.isfinite(ra.get("r", np.nan)) or not np.isfinite(rb.get("r", np.nan)):
        return {"difference": np.nan, "n_a": ra.get("n", 0), "n_b": rb.get("n", 0)}
    n = min(len(ra["draws"]), len(rb["draws"]))
    diff = ra["draws"][:n] - rb["draws"][:n]
    # a two-sided bootstrap p: how often the difference lands on the other side of
    # zero, with the add-one correction, so a difference no draw crosses is reported as
    # the smallest p the number of draws can support rather than as exactly zero
    k = int((diff <= 0).sum())
    p = min(1.0, 2 * (min(k, len(diff) - k) + 1) / (len(diff) + 1))
    return {"difference": float(ra["r"] - rb["r"]),
            "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975)),
            "p": float(p), "n_boot": int(len(diff)),
            "r_a": ra["r"], "r_b": rb["r"], "n_a": ra["n"], "n_b": rb["n"],
            "n_positions_a": ra["n_positions"], "n_positions_b": rb["n_positions"]}


def _suptitle(fig, text: str) -> None:
    """A left-aligned caption wrapped to the figure's own width.

    Under constrained layout an unwrapped suptitle widens the canvas to fit itself, so
    a one-protein figure comes out three times wider than its single panel. Wrapping
    against the current figure width keeps the panel the size it was asked for.
    """
    import textwrap
    fig.suptitle(textwrap.fill(" ".join(text.split()),
                               width=max(40, int(14 * fig.get_figwidth()))),
                 fontsize=7, x=0.01, ha="left")


# --------------------------------------------------- t01 the whole-protein number
def _density(ax, x, y, max_kde: int = 2000, seed: int = 0):
    """Scatter coloured by point density, with the kernel fitted on a subsample.

    A Gaussian KDE evaluated at every one of ten thousand points is quadratic and slow
    for a picture that would look the same either way, so the kernel is estimated from
    at most `max_kde` points and then evaluated everywhere.
    """
    xy = np.vstack([x, y])
    try:
        rng = np.random.default_rng(seed)
        fit = xy[:, rng.choice(xy.shape[1], min(max_kde, xy.shape[1]), replace=False)]
        z = ss.gaussian_kde(fit)(xy)
    except Exception:
        z = np.zeros(xy.shape[1])
    o = z.argsort()
    return ax.scatter(xy[0][o], xy[1][o], c=z[o], s=7, cmap="viridis", lw=0,
                      rasterized=True)


def t01_overall(data: dict, names: list[str], outdir: Path,
                cols: int = 2) -> dict:
    """One correlation per protein, over every missense variant with a prediction."""
    rows = []
    fig, pairs, _ = grid(names, w=3.2, h=2.9, cols=cols, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        d = data[g]["var"]
        if len(d) < MIN_PAIRS:
            empty_panel(ax, i, g, "too few variants with a prediction")
            continue
        sp = corr_with_ci(d, "spearman", seed=1)
        pe = corr_with_ci(d, "pearson", seed=2)
        _density(ax, d.ddg.to_numpy(float), d.score_z.to_numpy(float))
        ax.axhline(0, color=P.INK, lw=0.8, ls=(0, (1, 2)))
        ax.axhline(-1, color="#B2182B", lw=0.8, ls=(0, (1, 2)))
        ax.axvline(0, color=P.INK, lw=0.8, ls=(0, (1, 2)))
        rows.append({"protein": g, "spearman": sp["r"], "spearman_lo": sp["lo"],
                     "spearman_hi": sp["hi"], "pearson": pe["r"],
                     "pearson_lo": pe["lo"], "pearson_hi": pe["hi"],
                     "n": sp["n"], "n_positions": sp["n_positions"],
                     **{k: data[g]["merge"][k] for k in ("offset", "wt_agreement")}})
        panel(ax, i, g, f"ρ = {sp['r']:+.2f} [{sp['lo']:+.2f}, {sp['hi']:+.2f}]  "
                        f"r = {pe['r']:+.2f}  n = {sp['n']:,}", size=7)
        if i % 2 == 0:
            ax.set_ylabel("Normalised fitness", fontsize=7.5)
    fig.supxlabel(DDG_LABEL, fontsize=8)
    _suptitle(fig, "Every missense variant with a prediction. Dotted lines mark the "
                 "wild-type and nonsense anchors of the fitness scale and zero predicted "
                 "ΔΔG. Intervals are bootstrapped over positions.")
    P.save(fig, outdir, "t01_ddg_vs_fitness", None, "T01")
    plt.close(fig)
    if not rows:
        return {"status": "skipped", "reason": "no protein has enough merged variants"}
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "t01_overall.csv", index=False)
    return {"status": "ok", "by_protein": t.to_dict("records"),
            "headline": "; ".join(f"{short_name(r.protein)} ρ {r.spearman:+.2f} "
                                  f"[{r.spearman_lo:+.2f}, {r.spearman_hi:+.2f}] "
                                  f"n {r.n:,}" for r in t.itertuples())}


# ------------------------------------------------- t02 the same, split by environment
def t02_by_class(data: dict, names: list[str], outdir: Path,
                 method: str = "spearman", cols: int = 2) -> dict:
    """The correlation asked separately in each burial class, and the differences tested.

    The claim worth testing is that a stability predictor trained on soluble proteins
    works where a membrane protein most resembles one - the packed helix-helix core -
    and degrades on the lipid-facing surface, where burial is real but the surrounding
    chemistry is the opposite of what the training set contained. Each difference is
    core minus that class, bootstrapped over positions.
    """
    rows, diffs = [], []
    fig, pairs, _ = grid(names, w=3.2, h=2.7, cols=cols, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        d = data[g]["var"]
        got = []
        for c in BCLASSES:
            sub = d[d.bclass == c]
            r = corr_with_ci(sub, method, seed=hash((g, c)) % 2**31)
            if not np.isfinite(r.get("r", np.nan)):
                continue
            got.append((c, r))
            rows.append({"protein": g, "bclass": c, "method": method,
                         "r": r["r"], "lo": r["lo"], "hi": r["hi"], "se": r["se"],
                         "p": r["p"], "n": r["n"], "n_positions": r["n_positions"]})
        if not got:
            empty_panel(ax, i, g, "no class has enough variants")
            continue
        y = np.arange(len(got))[::-1]
        for k, (c, r) in enumerate(got):
            ax.errorbar(r["r"], y[k], xerr=[[r["r"] - r["lo"]], [r["hi"] - r["r"]]],
                        fmt="o", ms=6, lw=0, elinewidth=1.4, capsize=3,
                        color=BCOLORS[c], mec=P.INK, mew=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{BLABEL[c]}\n{r['n']:,} variants, {r['n_positions']} "
                            f"positions" for c, r in got], fontsize=6)
        ax.axvline(0, color=P.INK, lw=0.7, ls=(0, (3, 3)))
        core = d[d.bclass == "core"]
        best = ""
        for c in ("lipid", "cavity", "aqueous"):
            if not len(core):
                break
            dd = difference_in_corr(core, d[d.bclass == c], method,
                                    seed=hash((g, c, "d")) % 2**31)
            if not np.isfinite(dd.get("difference", np.nan)):
                continue
            diffs.append({"protein": g, "contrast": f"core − {c}", "method": method, **dd})
            if c == "lipid":
                best = (f"core − lipid {dd['difference']:+.2f} "
                        f"[{dd['lo']:+.2f}, {dd['hi']:+.2f}] p {dd['p']:.2g}")
        panel(ax, i, g, best, size=7)
    fig.supxlabel(f"{'Spearman ρ' if method == 'spearman' else 'Pearson r'} between "
                  "predicted ΔΔG and fitness", fontsize=8)
    _suptitle(fig, "The same correlation asked inside each environment. A predictor "
                 "trained on soluble proteins is expected to do best in the packed "
                 "core; the lipid-facing surface is buried but chemically unlike "
                 "anything in its training set. Intervals bootstrap positions.")
    P.save(fig, outdir, "t02_correlation_by_class", None, "T02")
    plt.close(fig)
    if not rows:
        return {"status": "skipped", "reason": "no class has enough variants"}
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "t02_correlation_by_class.csv", index=False)
    dt = pd.DataFrame(diffs)
    if len(dt):
        dt.to_csv(outdir / "t02_class_differences.csv", index=False)
    head = "; ".join(
        f"{short_name(r.protein)} " + " ".join(
            f"{c}{t[(t.protein == r.protein) & (t.bclass == c)].r.iloc[0]:+.2f}"
            for c in BCLASSES if ((t.protein == r.protein) & (t.bclass == c)).any())
        for r in t.drop_duplicates("protein").itertuples())
    return {"status": "ok", "headline": head, "by_class": t.to_dict("records"),
            "differences": dt.to_dict("records")}


# ------------------------------------------------------- t03 against accessibility
def t03_by_rsa(data: dict, names: list[str], outdir: Path,
               method: str = "spearman", cols: int = 2) -> dict:
    """The correlation as a function of relative solvent accessibility.

    The class split is a decision about where the boundaries are; this is the same
    question without one. If the predictor is reading packing, the curve should be
    strongest at low RSA and decay as residues become exposed - and in a membrane
    protein it should NOT, because much of the exposed surface is exposed to lipid.
    Comparing this curve with t02 is how you tell those apart.
    """
    rows, cuts = [], []
    fig, pairs, _ = grid(names, w=3.2, h=2.7, cols=cols, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        d = data[g]["var"].dropna(subset=["rsa"])
        pts = []
        for lo, hi in zip(RSA_BINS[:-1], RSA_BINS[1:]):
            sub = d[(d.rsa >= lo) & (d.rsa < hi)]
            r = corr_with_ci(sub, method, n_boot=max(N_BOOT // 4, 200),
                             seed=hash((g, lo)) % 2**31)
            if not np.isfinite(r.get("r", np.nan)):
                continue
            pts.append({"mid": float((lo + hi) / 2), **r})
            rows.append({"protein": g, "rsa_lo": float(lo), "rsa_hi": float(hi),
                         "method": method, "r": r["r"], "lo": r["lo"], "hi": r["hi"],
                         "n": r["n"], "n_positions": r["n_positions"]})
        if not pts:
            empty_panel(ax, i, g, "no accessibility bin has enough variants")
            continue
        m = pd.DataFrame(pts)
        ax.fill_between(m["mid"], m["lo"], m["hi"], color=P.DARK_RED, alpha=0.18, lw=0)
        ax.plot(m["mid"], m["r"], "-o", ms=4.5, lw=1.5, color=P.DARK_RED, mec=P.INK,
                mew=0.4)
        ax.axhline(0, color=P.INK, lw=0.7)
        ax.axvline(BURIED, color=P.MUTED, lw=0.7, ls=(0, (3, 3)))
        dd = difference_in_corr(d[d.rsa < BURIED], d[d.rsa >= EXPOSED], method,
                                seed=hash((g, "rsa")) % 2**31)
        if np.isfinite(dd.get("difference", np.nan)):
            cuts.append({"protein": g, "contrast": f"RSA < {BURIED} − RSA ≥ {EXPOSED}",
                         "method": method, **dd})
            panel(ax, i, g, f"buried − exposed {dd['difference']:+.2f} "
                            f"[{dd['lo']:+.2f}, {dd['hi']:+.2f}] p {dd['p']:.2g}", size=7)
        else:
            panel(ax, i, g, "", size=7)
        if i % 2 == 0:
            ax.set_ylabel(f"{'ρ' if method == 'spearman' else 'r'} with fitness",
                          fontsize=7.5)
    fig.supxlabel("relative solvent accessibility", fontsize=8)
    _suptitle(fig, "The dashed line is the RSA < 0.15 'buried' cut the rest of the "
                 "pipeline inherits. RSA is computed with a water probe on a monomer, "
                 "so in a membrane protein a lipid-facing residue reads as exposed: "
                 "read this against t02, not instead of it.")
    P.save(fig, outdir, "t03_correlation_by_rsa", None, "T03")
    plt.close(fig)
    if not rows:
        return {"status": "skipped", "reason": "no accessibility bin has enough variants"}
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "t03_correlation_by_rsa.csv", index=False)
    ct = pd.DataFrame(cuts)
    if len(ct):
        ct.to_csv(outdir / "t03_buried_vs_exposed.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} buried − exposed {r.difference:+.2f} "
                     f"[{r.lo:+.2f}, {r.hi:+.2f}] p {r.p:.2g}"
                     for r in ct.itertuples()) if len(ct) else "no contrast"
    return {"status": "ok", "headline": head, "by_bin": t.to_dict("records"),
            "contrasts": ct.to_dict("records")}


ANALYSES = {"t01": "predicted ΔΔG against fitness, whole protein",
            "t02": "the same correlation split by burial class",
            "t03": "the same correlation against solvent accessibility"}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "divide by zero|Degrees of freedom|All-NaN|"
                                              "tight_layout).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms stability",
        description="A predicted stability change against measured fitness, asked "
                    "separately in each environment of a membrane protein.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--ddg", action="append", default=[], metavar="ID=PATH",
                    help="ThermoMPNN inference CSV for one protein, by config id; "
                         "repeatable. Without it the path is read from the config key "
                         "predictions.thermompnn")
    ap.add_argument("--offset", type=int, default=None,
                    help="force the numbering offset instead of detecting it from "
                         "wild-type agreement (applies to every protein)")
    ap.add_argument("--min-wt-agreement", type=float, default=MIN_WT_AGREEMENT)
    ap.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)
    globals()["N_BOOT"] = int(a.boot)
    keys = [k.strip() for k in (a.only.split(",") if a.only else ANALYSES)]
    bad = [k for k in keys if k not in ANALYSES]
    if bad:
        raise SystemExit(f"unknown analysis {bad}; available: {', '.join(ANALYSES)}")
    given = {}
    for s in a.ddg:
        if "=" not in s:
            raise SystemExit(f"--ddg wants ID=PATH, got {s!r}")
        k, v = s.split("=", 1)
        given[k] = Path(v).expanduser()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data, names = {}, []
    for c in a.configs:
        cfg = load_config(Path(c))
        path = given.get(cfg.id) or cfg.resolve(cfg.get_path("predictions.thermompnn"))
        if path is None:
            print(f"  {cfg.id:<16} skipped: no prediction file (--ddg {cfg.id}=... or "
                  f"predictions.thermompnn in the config)", flush=True)
            continue
        if not Path(path).exists():
            raise SystemExit(f"{cfg.id}: no such prediction file: {path}")
        b = bits_for(cfg)
        if not b or b["pos"].bclass.isna().all():
            print(f"  {cfg.id:<16} skipped: needs a structure and a topology", flush=True)
            continue
        var, info = attach_ddg(b, read_ddg(Path(path)), a.offset, a.min_wt_agreement)
        if len(var) < MIN_PAIRS:
            print(f"  {cfg.id:<16} skipped: only {len(var)} variants merged", flush=True)
            continue
        data[cfg.id], _ = {"var": var, "merge": info}, names.append(cfg.id)
        counts = var.bclass.value_counts()
        print(f"  {cfg.id:<16} {len(var):>5} variants  offset {info['offset']:+d} "
              f"(wt {info['wt_agreement']:.1%})  "
              + "  ".join(f"{k} {n}" for k, n in counts.items()), flush=True)
    if not names:
        raise SystemExit("no protein had both a prediction file and a classified "
                         "structure")
    rows = int(np.ceil(len(names) / a.cols))
    print("\n  panels: " + " | ".join(
        ", ".join(short_name(n) for n in names[r * a.cols:(r + 1) * a.cols])
        for r in range(rows)), flush=True)
    pd.concat([d["var"].assign(protein=g) for g, d in data.items()],
              ignore_index=True).to_csv(outdir / "variants_with_ddg.csv", index=False)

    calls = {"t01": lambda: t01_overall(data, names, outdir, a.cols),
             "t02": lambda: t02_by_class(data, names, outdir, a.method, a.cols),
             "t03": lambda: t03_by_rsa(data, names, outdir, a.method)}
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
    (outdir / "stability.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
