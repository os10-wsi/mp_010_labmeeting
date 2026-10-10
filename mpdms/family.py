"""`python -m mpdms family configs/AQR1.yaml configs/QDR2.yaml`

Paired analysis for proteins of one family (e.g. two MFS/DHA1 transporters).

Part A - aligned positions. Align the sequences, keep columns where both proteins have a
residue, and ask how far the mutational effect at a position in one protein predicts the
effect at the structurally equivalent position in the other. Restricted to TM helices,
this asks whether homologous positions carry homologous constraints.

  The null matters here. Two MFS transporters both have sensitive TM cores and tolerant
  loops, so a positive correlation across all positions is guaranteed by topology alone.
  The null used is a permutation *within* the same class (TM with TM, loop with loop),
  which preserves that structure and destroys only the residue-level correspondence. A
  correlation that survives it is evidence of position-specific, not regional, agreement.

Part B - helix profiles. For TM segments that are entirely alpha-helical in the model,
plot the effect of proline, glycine and lysine/arginine at each position along the helix
from N to C.

  Helices in a polytopic protein alternate orientation, so a given N-terminal end is
  cytosolic in half of them and lumenal in the other half. Pooling strictly N to C will
  therefore cancel any membrane-sided effect (the positive-inside signal in particular).
  Both views are produced: N to C as asked, and the same data split by orientation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .analyses.a08_depth_dependence import _lowess_curve
from .analyses.base import jsonable
from .annot import AA20, segments_df
from .config import REPO_ROOT, load_config, protein_sequence, validate as validate_cfg
from .io import load_dataset
from .structure import residue_table

CLASSES = {"P": ("→ Proline", "#D62728"), "G": ("→ Glycine", "#E69F00"),
           "KR": ("→ Lys / Arg", "#7B5EA7")}
HELICAL_FRAC = 0.7   # fraction of a TMHMM segment that must be alpha-helical in the model
MIN_VARIANTS = 3


# ------------------------------------------------------------------ alignment
def align_sequences(seqs: dict[str, str]) -> pd.DataFrame:
    """One row per alignment column; columns `<id>_pos` (1-based, NaN at a gap) and `<id>_aa`."""
    ids = list(seqs)
    if len(ids) < 2:
        raise RuntimeError("need at least two sequences")
    if shutil.which("mafft"):
        rows = _mafft(seqs)
        rows.attrs.setdefault("method", "MAFFT")
    elif len(ids) == 2:
        rows = _pairwise(seqs)
        rows.attrs.setdefault("method", "pairwise")
    else:
        rows = center_star(seqs)
    return rows


def _to_table(ids, aligned: list[str]) -> pd.DataFrame:
    out = {}
    for i, g in enumerate(ids):
        pos, n = [], 0
        for c in aligned[i]:
            if c == "-":
                pos.append(np.nan)
            else:
                n += 1
                pos.append(n)
        out[f"{g}_pos"] = pos
        out[f"{g}_aa"] = list(aligned[i])
    t = pd.DataFrame(out)
    t.insert(0, "col", np.arange(1, len(t) + 1))
    return t


def _pairwise(seqs: dict[str, str]) -> pd.DataFrame:
    from Bio import Align
    from Bio.Align import substitution_matrices
    ids = list(seqs)
    al = Align.PairwiseAligner(mode="global")
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score = -11, -1
    for attr in ("end_insertion_score", "end_deletion_score",      # Biopython >= 1.85
                 "target_end_gap_score", "query_end_gap_score"):   # older
        try:
            setattr(al, attr, 0.0)
        except (AttributeError, ValueError):
            pass
    a = al.align(seqs[ids[0]], seqs[ids[1]])[0]
    return _to_table(ids, [str(a[0]), str(a[1])])


def center_star(seqs: dict[str, str]) -> pd.DataFrame:
    """Reference-anchored alignment for more than two sequences, without MAFFT.

    The most central sequence becomes the reference and every other is aligned to it
    pairwise, so the columns are the reference's own residues. Insertions relative to the
    reference are dropped, which a real multiple alignment would keep; for mapping each
    protein's positions onto one shared frame, which is all that is needed here, that
    loses only the parts no reference residue corresponds to. Use MAFFT when it is
    available - `align_sequences` prefers it.
    """
    from Bio import Align
    from Bio.Align import substitution_matrices
    ids = list(seqs)
    al = Align.PairwiseAligner(mode="global")
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score = -11, -1
    for attr in ("end_insertion_score", "end_deletion_score",
                 "target_end_gap_score", "query_end_gap_score"):
        try:
            setattr(al, attr, 0.0)
        except (AttributeError, ValueError):
            pass
    score = {g: 0.0 for g in ids}
    pairs = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            aln = al.align(seqs[a], seqs[b])[0]
            pairs[(a, b)] = aln
            sa, sb = str(aln[0]), str(aln[1])
            ident = sum(x == y for x, y in zip(sa, sb) if x != "-")
            frac = ident / max(1, min(len(seqs[a]), len(seqs[b])))
            score[a] += frac
            score[b] += frac
    ref = max(ids, key=lambda g: score[g])
    L = len(seqs[ref])
    out = {f"{ref}_pos": list(range(1, L + 1)), f"{ref}_aa": list(seqs[ref])}
    for g in ids:
        if g == ref:
            continue
        aln = pairs.get((ref, g)) or pairs.get((g, ref))
        sr, sg = (str(aln[0]), str(aln[1])) if (ref, g) in pairs else (str(aln[1]), str(aln[0]))
        pos, aa = [np.nan] * L, ["-"] * L
        ir = ig = 0
        for cr, cg in zip(sr, sg):
            if cr != "-":
                if cg != "-":
                    pos[ir], aa[ir] = ig + 1, cg
                ir += 1
            if cg != "-":
                ig += 1
        out[f"{g}_pos"], out[f"{g}_aa"] = pos, aa
    t = pd.DataFrame(out)
    t.insert(0, "col", np.arange(1, len(t) + 1))
    t.attrs["reference"] = ref
    t.attrs["method"] = "center-star (no MAFFT)"
    return t


def _mafft(seqs: dict[str, str]) -> pd.DataFrame:
    if not shutil.which("mafft"):
        raise RuntimeError("more than two proteins needs MAFFT (conda install -c bioconda mafft)")
    import tempfile
    ids = list(seqs)
    with tempfile.TemporaryDirectory() as d:
        fa = Path(d) / "in.fa"
        fa.write_text("".join(f">{g}\n{s}\n" for g, s in seqs.items()))
        out = subprocess.run(["mafft", "--auto", "--quiet", str(fa)], capture_output=True, text=True, check=True)
    cur, al = None, {g: [] for g in ids}
    for line in out.stdout.splitlines():
        if line.startswith(">"):
            cur = line[1:].strip()
        elif cur:
            al[cur].append(line.strip())
    return _to_table(ids, ["".join(al[g]) for g in ids])


# -------------------------------------------------------------- per-protein
def protein_table(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """One row per position: topology, secondary structure, and mean effect by mutant class."""
    d = df[df.pass_filter & (df.vclass == "missense")]
    lo, hi = int(df.pos.min()), int(df.pos.max())
    t = pd.DataFrame({"pos": np.arange(lo, hi + 1)})
    segs = segments_df(cfg)
    seg_type, helix, orient = {}, {}, {}
    for s in segs.itertuples():
        for p in range(s.start, s.end + 1):
            seg_type[p] = s.type
            if s.type == "TM":
                helix[p] = s.name
                orient[p] = s.orientation or "in_out"
    t["seg_type"] = t.pos.map(seg_type).fillna("unannotated")
    t["helix"] = t.pos.map(helix)
    t["orientation"] = t.pos.map(orient)
    sp = cfg.resolve(cfg.get_path("structure.path"))
    if sp and sp.exists():
        rt = residue_table(sp, cfg.get_path("structure.chain", "A"),
                           int(cfg.get_path("structure.numbering_offset", 0) or 0))
        t = t.merge(rt[["pos", "aa", "ss", "plddt", "rsa"]], on="pos", how="left")
    else:
        for c in ("aa", "ss", "plddt", "rsa"):
            t[c] = np.nan
    t["mean_all"] = t.pos.map(d.groupby("pos").score_z.mean())
    t["n_variants"] = t.pos.map(d.groupby("pos").score_z.size()).fillna(0)
    for k in ("P", "G"):
        g = d[d.mut == k].groupby("pos").score_z.mean()
        t[f"mean_{k}"] = t.pos.map(g)
    t["mean_KR"] = t.pos.map(d[d.mut.isin(["K", "R"])].groupby("pos").score_z.mean())
    t["mean_rest"] = t.pos.map(d[~d.mut.isin(["P", "G", "K", "R"])].groupby("pos").score_z.mean())
    # the 19-vector of substitution effects, for profile correlation
    wide = d.pivot_table(index="pos", columns="mut", values="score_z", aggfunc="mean").reindex(columns=list(AA20))
    t = t.merge(wide.add_prefix("sub_"), on="pos", how="left")
    # helix-level: fraction alpha-helical, and position along the helix N->C
    hs = {}
    for name, g in t[t.helix.notna()].groupby("helix"):
        frac = float((g.ss == "H").mean()) if g.ss.notna().any() else np.nan
        hs[name] = frac
    t["helix_helical_frac"] = t.helix.map(hs)
    t["helix_len"] = t.helix.map(t[t.helix.notna()].groupby("helix").size())
    idx = t[t.helix.notna()].groupby("helix").cumcount() + 1
    t.loc[t.helix.notna(), "idx_from_N"] = idx
    t["idx_from_C"] = t.helix_len - t.idx_from_N + 1
    t["frac_NC"] = (t.idx_from_N - 1) / (t.helix_len - 1)
    return t


# ---------------------------------------------------- A: aligned positions
def aligned_table(aln: pd.DataFrame, tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    t = aln.copy()
    for g, pt in tabs.items():
        cols = ["pos", "seg_type", "helix", "ss", "plddt", "mean_all", "mean_P", "mean_G", "mean_KR",
                "n_variants"] + [f"sub_{a}" for a in AA20]
        t = t.merge(pt[cols].add_prefix(f"{g}_"), left_on=f"{g}_pos", right_on=f"{g}_pos", how="left")
    a, b = list(tabs)
    t["both_residues"] = t[f"{a}_aa"].ne("-") & t[f"{b}_aa"].ne("-")
    t["identical"] = t.both_residues & t[f"{a}_aa"].eq(t[f"{b}_aa"])
    t["both_tm"] = t[f"{a}_seg_type"].eq("TM") & t[f"{b}_seg_type"].eq("TM")
    t["both_helical"] = t.both_tm & t[f"{a}_ss"].eq("H") & t[f"{b}_ss"].eq("H")
    t["both_loop"] = t[f"{a}_seg_type"].isin(["loop", "soluble"]) & t[f"{b}_seg_type"].isin(["loop", "soluble"])
    return t


def _within_class_null(t, a, b, mask, col="mean_all", n=2000, seed=0):
    """Permute the pairing within the same topology class, preserving regional structure."""
    d = t[mask].dropna(subset=[f"{a}_{col}", f"{b}_{col}"])
    if len(d) < 15:
        return np.nan, np.nan, np.array([]), 0
    x, y = d[f"{a}_{col}"].to_numpy(), d[f"{b}_{col}"].to_numpy()
    obs = ss.spearmanr(x, y)[0]
    cls = np.where(d.both_tm, "TM", np.where(d.both_loop, "loop", "other"))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n):
        yp = y.copy()
        for c in np.unique(cls):
            m = cls == c
            yp[m] = rng.permutation(y[m])
        null.append(ss.spearmanr(x, yp)[0])
    null = np.array(null)
    p = (1 + np.sum(null >= obs)) / (1 + len(null))
    return float(obs), float(p), null, len(d)


def substitution_profile_correlation(t, a, b, mask, seed=0):
    """At each aligned position, correlate the 19-vector of substitution effects."""
    cols = [f"sub_{x}" for x in AA20]
    d = t[mask]
    A = d[[f"{a}_{c}" for c in cols]].to_numpy(float)
    B = d[[f"{b}_{c}" for c in cols]].to_numpy(float)
    rhos = []
    for i in range(len(A)):
        ok = np.isfinite(A[i]) & np.isfinite(B[i])
        if ok.sum() >= 8:
            rhos.append(ss.spearmanr(A[i][ok], B[i][ok])[0])
    rhos = np.array([r for r in rhos if np.isfinite(r)])
    rng = np.random.default_rng(seed)
    null = []
    idx = np.arange(len(A))
    for _ in range(200):
        perm = rng.permutation(idx)
        for i, j in zip(idx, perm):
            ok = np.isfinite(A[i]) & np.isfinite(B[j])
            if ok.sum() >= 8:
                r = ss.spearmanr(A[i][ok], B[j][ok])[0]
                if np.isfinite(r):
                    null.append(r)
    return rhos, np.array(null)


def part_a(t, a, b, outdir, title):
    figdir, tabdir = outdir / "figures", outdir / "tables"
    rows = []
    subsets = {"all aligned": t.both_residues, "both TM": t.both_tm,
               "both TM & α-helical": t.both_helical, "both loop/soluble": t.both_loop}
    for name, m in subsets.items():
        obs, p, null, n = _within_class_null(t, a, b, m)
        rows.append({"subset": name, "n_positions": n, "spearman": obs,
                     "p_vs_within_class_null": p,
                     "null_mean": float(np.mean(null)) if len(null) else np.nan,
                     "null_p95": float(np.quantile(null, 0.95)) if len(null) else np.nan})
    stats = pd.DataFrame(rows)
    rhos, null = substitution_profile_correlation(t, a, b, t.both_tm)

    fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.5), width_ratios=[1.15, 1, 1], layout="constrained")
    ax = axs[0]
    d = t[t.both_residues].dropna(subset=[f"{a}_mean_all", f"{b}_mean_all"])
    ax.scatter(d.loc[~d.both_tm, f"{a}_mean_all"], d.loc[~d.both_tm, f"{b}_mean_all"],
               s=7, color="#BBBBBB", lw=0, label="other aligned positions")
    dt = d[d.both_tm]
    ax.scatter(dt[f"{a}_mean_all"], dt[f"{b}_mean_all"], s=9, color="#2F6DB5", lw=0, label="both TM")
    lim = [min(d[f"{a}_mean_all"].min(), d[f"{b}_mean_all"].min()) - .05,
           max(d[f"{a}_mean_all"].max(), d[f"{b}_mean_all"].max()) + .05]
    ax.plot(lim, lim, color=P.MUTED, lw=0.6, ls=(0, (2, 2)))
    r_tm = stats.loc[stats.subset == "both TM"].iloc[0]
    ax.set_xlabel(f"{a} mean missense effect")
    ax.set_ylabel(f"{b} mean missense effect")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title(f"(a) Aligned positions · both TM ρ = {r_tm.spearman:.2f}", loc="left", fontsize=8.5)

    ax = axs[1]
    y = np.arange(len(stats))
    ax.barh(y, stats.spearman, color=["#BBBBBB", "#2F6DB5", "#1B4B80", "#E69F00"], height=0.6)
    for i, r in stats.iterrows():
        if np.isfinite(r.null_p95):
            ax.plot([r.null_p95] * 2, [i - .33, i + .33], color=P.CLASS_COLORS["nonsense"], lw=1.4)
        if np.isfinite(r.spearman):
            ax.text(0.02, i, f"  n={r.n_positions}, p={r.p_vs_within_class_null:.3g}", va="center", fontsize=6,
                    color=P.INK)
    ax.set_yticks(y)
    ax.set_yticklabels(stats.subset, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ between proteins")
    ax.set_title("(b) vs within-class permutation (red = null 95th pct)", loc="left", fontsize=8.5)

    ax = axs[2]
    bins = np.linspace(-1, 1, 33)
    if len(null):
        ax.hist(null, bins=bins, color="#BBBBBB", density=True, label=f"shuffled pairing (n={len(null)})")
    if len(rhos):
        ax.hist(rhos, bins=bins, color="#2F6DB5", alpha=0.75, density=True,
                label=f"aligned TM positions (n={len(rhos)})")
        ax.axvline(np.median(rhos), color="#1B4B80", lw=1.4)
    u = ss.mannwhitneyu(rhos, null).pvalue if len(rhos) > 5 and len(null) > 5 else np.nan
    ax.set_xlabel("ρ between the two proteins' 19-substitution profiles")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title((f"(c) Substitution profiles · median ρ = {np.median(rhos):.2f} "
                  f"vs {np.median(null):+.2f} shuffled") if len(rhos) and len(null) else "(c)",
                 loc="left", fontsize=8.5)
    fig.suptitle(f"{title} — do aligned positions carry the same constraints?", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    P.save(fig, figdir, "a17a_aligned_correlation", None, "family")
    stats["sub_profile_median_rho"] = np.median(rhos) if len(rhos) else np.nan
    stats["sub_profile_p"] = u
    stats.to_csv(tabdir / "a17a_aligned_correlation.csv", index=False)
    t.to_csv(tabdir / "a17a_alignment.csv", index=False)
    return stats, rhos, u


# ------------------------------------------------- B: helix N->C profiles
def helix_frame(tabs: dict[str, pd.DataFrame], helical_frac: float = HELICAL_FRAC) -> pd.DataFrame:
    rows = []
    for g, t in tabs.items():
        d = t[t.helix.notna()].copy()
        keep = d.helix_helical_frac.isna() | (d.helix_helical_frac >= helical_frac)
        d = d[keep]
        d["protein"] = g
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def part_b(hf: pd.DataFrame, outdir, title, n_boot: int = 300):
    figdir, tabdir = outdir / "figures", outdir / "tables"
    proteins = sorted(hf.protein.unique())
    rng = np.random.default_rng(17)
    grid = np.linspace(0, 1, 60)

    def band(ax, d, col, color, label):
        d = d.dropna(subset=[col, "frac_NC"])
        if len(d) < 8:
            return
        ax.scatter(d.frac_NC, d[col], s=5, color=color, alpha=0.3, lw=0)
        ax.plot(grid, _lowess_curve(d.frac_NC.to_numpy(), d[col].to_numpy(), grid, frac=0.55),
                color=color, lw=1.7, label=label)
        hx = d.helix.unique()
        idx = {h: d.index[d.helix == h].to_numpy() for h in hx}
        bs = []
        for _ in range(n_boot):
            pick = np.concatenate([idx[h] for h in rng.choice(hx, len(hx))])
            s = d.loc[pick]
            bs.append(_lowess_curve(s.frac_NC.to_numpy(), s[col].to_numpy(), grid, frac=0.55))
        bs = np.array(bs)
        ax.fill_between(grid, np.nanquantile(bs, .025, 0), np.nanquantile(bs, .975, 0),
                        color=color, alpha=0.2, lw=0)

    # figure 1: pooled N->C, then the same split by helix orientation
    nrow = 1 + len(proteins)
    fig, axes = plt.subplots(nrow, 3, figsize=(10.4, 2.5 * nrow), squeeze=False, sharex=True, sharey=True)
    for j, (k, (lab, col)) in enumerate(CLASSES.items()):
        ax = axes[0, j]
        for g, ls in zip(proteins, ["-", (0, (4, 1.6))]):
            d = hf[hf.protein == g]
            band(ax, d, f"mean_{k}", col, g)
            if ax.lines:
                ax.lines[-1].set_linestyle(ls)
        ax.set_title(lab, fontsize=9, color=col)
        ax.axhline(0, color=P.MUTED, lw=0.5)
        if j == 0:
            ax.set_ylabel("Both proteins\nmean normalised fitness", fontsize=7.5)
            ax.legend(loc="lower left", fontsize=6.5, frameon=True, framealpha=0.85, edgecolor="none")
    for i, g in enumerate(proteins, 1):
        for j, (k, (lab, col)) in enumerate(CLASSES.items()):
            ax = axes[i, j]
            d = hf[hf.protein == g]
            for o, shade in (("in_out", col), ("out_in", "#8C8C8C")):
                sub = d[d.orientation == o]
                if len(sub):
                    band(ax, sub, f"mean_{k}", shade,
                         {"in_out": "N-term cytosolic", "out_in": "N-term lumenal"}[o])
            ax.axhline(0, color=P.MUTED, lw=0.5)
            if j == 0:
                ax.set_ylabel(f"{g}\nby helix orientation", fontsize=7.5)
                ax.legend(loc="lower left", fontsize=6, frameon=True, framealpha=0.85, edgecolor="none")
    for ax in axes[-1]:
        ax.set_xlabel("Position along TM helix (0 = N-terminal end, 1 = C-terminal end)")
    fig.suptitle(f"{title} — proline, glycine and K/R along TM helices, N to C",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    P.save(fig, figdir, "a17b_helix_profiles", None, "family")

    # figure 2: per-helix small multiples
    keys = sorted(hf.groupby(["protein", "helix"]).groups, key=lambda x: (x[0], int("".join(filter(str.isdigit, x[1])) or 0)))
    ncol = min(6, max(1, len(keys)))
    nr = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nr, ncol, figsize=(1.9 * ncol + 0.5, 1.6 * nr + 0.8), squeeze=False,
                             sharey=True, sharex=True)
    for ax in axes.ravel():
        ax.axis("off")
    for n, (g, h) in enumerate(keys):
        ax = axes[n // ncol, n % ncol]
        ax.axis("on")
        d = hf[(hf.protein == g) & (hf.helix == h)].sort_values("idx_from_N")
        for k, (lab, col) in CLASSES.items():
            ax.plot(d.idx_from_N, d[f"mean_{k}"], "-o", ms=2.2, lw=0.9, color=col)
        ax.plot(d.idx_from_N, d.mean_rest, lw=0.8, color="#BBBBBB", zorder=0)
        ax.axhline(0, color=P.MUTED, lw=0.4)
        o = d.orientation.iloc[0] if len(d) else ""
        ax.set_title(f"{g} {h}  ({'N cyto' if o == 'in_out' else 'N lumen'})", fontsize=6.3)
        ax.tick_params(labelsize=5.5)
        if n % ncol == 0:
            ax.set_ylabel("fitness", fontsize=6)
        if n // ncol == nr - 1:
            ax.set_xlabel("residue from helix N-term", fontsize=6)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=c, marker="o", lw=1, label=l) for l, c in CLASSES.values()]
               + [Line2D([], [], color="#BBBBBB", lw=1, label="other missense")],
               loc="lower center", ncol=4, fontsize=7)
    fig.suptitle(f"{title} — each entirely α-helical TM domain, N to C", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    P.save(fig, figdir, "a17c_helix_small_multiples", None, "family")

    # statistics
    import statsmodels.formula.api as smf
    rows = []
    for g in proteins:
        for k in CLASSES:
            d = hf[(hf.protein == g)].dropna(subset=[f"mean_{k}", "frac_NC"]).rename(columns={f"mean_{k}": "y"})
            if len(d) < 25 or d.helix.nunique() < 3:
                continue
            m = smf.ols("y ~ frac_NC + I(frac_NC**2)", data=d).fit(cov_type="cluster",
                                                                   cov_kwds={"groups": d.helix})
            per_h = d.groupby("helix").apply(
                lambda s: s.loc[s.frac_NC < .5, "y"].mean() - s.loc[s.frac_NC >= .5, "y"].mean(),
                include_groups=False).dropna()
            nb = rng.choice(per_h.to_numpy(), (2000, len(per_h))).mean(1) if len(per_h) >= 3 else np.array([np.nan])
            ends = d[(d.idx_from_N <= 4) | (d.idx_from_C <= 4)].y
            mid = d[(d.idx_from_N > 4) & (d.idx_from_C > 4)].y
            rows.append({"protein": g, "class": k, "n_positions": len(d), "n_helices": d.helix.nunique(),
                         "mean": d.y.mean(), "coef_frac": m.params["frac_NC"], "p_frac": m.pvalues["frac_NC"],
                         "coef_frac2": m.params["I(frac_NC ** 2)"], "p_frac2": m.pvalues["I(frac_NC ** 2)"],
                         "Nhalf_minus_Chalf": float(per_h.mean()) if len(per_h) else np.nan,
                         "Nhalf_minus_Chalf_lo": float(np.nanquantile(nb, .025)),
                         "Nhalf_minus_Chalf_hi": float(np.nanquantile(nb, .975)),
                         "ends_minus_middle": ends.mean() - mid.mean(),
                         "p_ends_vs_middle": ss.mannwhitneyu(ends, mid).pvalue if min(len(ends), len(mid)) > 5 else np.nan})
    stats = pd.DataFrame(rows)
    stats.to_csv(tabdir / "a17b_helix_profile_stats.csv", index=False)
    hf.to_csv(tabdir / "a17b_helix_positions.csv", index=False)
    return stats


# --------------------------------------------------------------------- CLI
def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|singular|boundary|No artists|invalid value).*")
    ap = argparse.ArgumentParser(prog="python -m mpdms family")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--helical-frac", type=float, default=HELICAL_FRAC,
                    help="minimum fraction of a TM segment that must be α-helical (default 0.8)")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)

    loaded = []
    for c in a.configs:
        cfg = load_config(Path(c))
        validate_cfg(cfg)
        loaded.append((load_dataset(cfg), cfg))
        print(f"loaded {cfg.id}", flush=True)
    ids = [cfg.id for _, cfg in loaded]
    out = Path(a.outdir) if a.outdir else REPO_ROOT / "outputs" / ("_family_" + "_".join(ids))
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    title = " vs ".join(ids)

    tabs = {cfg.id: protein_table(df, cfg) for df, cfg in loaded}
    for g, t in tabs.items():
        src = str(load_config(Path(a.configs[ids.index(g)])).get_path("topology.source", "?"))
        d = t[t.helix.notna()]
        hel = d.groupby("helix").agg(frac=("helix_helical_frac", "first"), n=("pos", "size"))
        kept = hel[hel.frac.isna() | (hel.frac >= a.helical_frac)]
        print(f"  {g}: {len(hel)} TM segments ({src}); {len(kept)} kept at ≥{a.helical_frac:.0%} α-helical")
        print("      " + "  ".join(f"{h}:{r.frac:.0%}" if np.isfinite(r.frac) else f"{h}:?"
                                   for h, r in hel.iterrows()), flush=True)
        if len(kept) < len(hel):
            print(f"      dropped {', '.join(hel.index.difference(kept.index))} "
                  "(lower --helical-frac to keep them)", flush=True)

    seqs = {cfg.id: protein_sequence(cfg) for _, cfg in loaded}
    missing = [g for g, s in seqs.items() if not s]
    summary = {}
    if missing:
        print(f"  no sequence for {missing} - skipping the alignment half")
    else:
        print("  aligning ...", flush=True)
        aln = align_sequences(seqs)
        at = aligned_table(aln, tabs)
        ident = float(at.identical.sum() / max(at.both_residues.sum(), 1))
        print(f"  {int(at.both_residues.sum())} aligned columns, {ident:.0%} identity", flush=True)
        if len(ids) == 2:
            st, rhos, u = part_a(at, ids[0], ids[1], out, title)
            summary["identity"] = ident
            summary["aligned_correlation"] = st.to_dict("records")
            print("\n  aligned-position agreement (Spearman, vs within-class permutation):")
            for _, r in st.iterrows():
                if np.isfinite(r.spearman):
                    print(f"    {r.subset:<22} ρ={r.spearman:+.2f}  n={r.n_positions:<5} "
                          f"p={r.p_vs_within_class_null:.3g}  (null 95th pct {r.null_p95:+.2f})")
            if len(rhos):
                print(f"    substitution profiles at aligned TM positions: median ρ={np.median(rhos):+.2f} "
                      f"vs shuffled {np.median(u) if np.isscalar(u) else ''} p={u:.3g}")
        else:
            print("  more than two proteins: pairwise correlation figures are made for pairs only")

    print("\n  helix profiles ...", flush=True)
    hf = helix_frame(tabs, a.helical_frac)
    if hf.empty:
        print("  no TM helices passed the α-helical filter")
    else:
        st = part_b(hf, out, title)
        summary["helix_profiles"] = st.to_dict("records")
        print("\n  position dependence along the helix (N→C):")
        for _, r in st.iterrows():
            print(f"    {r.protein:<8} {r['class']:<3} n={r.n_positions:<4} mean={r['mean']:+.2f}  "
                  f"N-half − C-half {r.Nhalf_minus_Chalf:+.2f} "
                  f"[{r.Nhalf_minus_Chalf_lo:+.2f},{r.Nhalf_minus_Chalf_hi:+.2f}]  "
                  f"ends − middle {r.ends_minus_middle:+.2f} (p={r.p_ends_vs_middle:.3g})")
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    print(f"\noutputs -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
