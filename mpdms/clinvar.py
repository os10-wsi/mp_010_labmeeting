"""`python -m mpdms clinvar --family hexose=... --human SLC2A1=P11166 --table variant_summary.txt.gz`

Does the yeast abundance signal predict which human variants are pathogenic?

The family posterior from `mpdms bayes` is a per-alignment-column estimate of how much a
position matters, learnt from yeast transporters. A human member of the same Pfam family
is aligned into the same frame, every ClinVar missense variant is mapped to its column,
and pathogenic and benign variants are compared by the constraint the yeast data assigns
to the position they sit at.

Three things this cannot do, which the output states rather than hides:

  * It is a POSITION score, not a variant score. Two substitutions at one residue get the
    same prediction, so the ceiling is set by how often pathogenic and benign variants
    occupy different positions rather than different amino acids.
  * ClinVar is ascertained, not sampled. Pathogenic variants cluster where people looked,
    and benign ones are enriched in the tolerant parts of well-sequenced genes, which
    inflates any position-based separation.
  * Yeast and human members of this family are around 30% identical, so the alignment
    itself is a source of error; the per-column protein count is reported alongside.
"""
from __future__ import annotations

import argparse
import gzip
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .bayes import column_frame, pool_columns, position_estimates
from .config import REPO_ROOT, load_config, protein_sequence
from .family import align_sequences
from .io import load_dataset

OUT = REPO_ROOT / "outputs" / "_clinvar"
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
       "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
       "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
PROT_RE = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
PATHOGENIC = ("pathogenic",)
BENIGN = ("benign",)


def classify_significance(s: str) -> str | None:
    """ClinVar's free text to pathogenic / benign / None.

    Anything conflicting, uncertain or otherwise hedged returns None: a variant of
    uncertain significance is not a negative, and counting it as one is the commonest way
    to make a predictor look good on ClinVar.
    """
    t = str(s).strip().lower()
    if not t or "conflict" in t or "uncertain" in t or "not provided" in t:
        return None
    has_p = "pathogenic" in t
    has_b = "benign" in t
    if has_p and has_b:
        return None
    if has_p:
        return "pathogenic"
    if has_b:
        return "benign"
    return None


def parse_protein_change(name: str) -> tuple[str, int, str] | None:
    m = PROT_RE.search(str(name))
    if not m:
        return None
    wt, pos, mut = AA3.get(m.group(1)), int(m.group(2)), AA3.get(m.group(3))
    if not wt or not mut or wt == mut:
        return None
    return wt, pos, mut


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_clinvar(path: Path, genes: set[str]) -> pd.DataFrame:
    """Missense ClinVar records for these genes, as (gene, wt, pos, mut, label).

    Reads the standard variant_summary.txt(.gz) by column name, and also accepts any table
    carrying a gene, a protein change and a clinical significance under recognisable names.
    """
    path = Path(path)
    with _open(path) as fh:
        header = fh.readline().lstrip("#").rstrip("\n").split("\t")
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in header}

    def pick(*names):
        return next((cols[n] for n in names if n in cols), None)

    gene_c = pick("genesymbol", "gene", "symbol")
    name_c = pick("name", "proteinchange", "hgvsp", "variant")
    sig_c = pick("clinicalsignificance", "germlineclassification", "significance",
                 "clinicalsignificancelastevaluated")
    if not (gene_c and name_c and sig_c):
        raise SystemExit(
            f"{path.name}: need a gene column, a protein-change column and a clinical "
            f"significance column.\n        found: {header[:12]}")
    want = {g.upper() for g in genes}
    keep = []
    for chunk in pd.read_csv(path, sep="\t", chunksize=200_000, low_memory=False,
                             usecols=lambda c: c in (gene_c, name_c, sig_c),
                             compression="gzip" if str(path).endswith(".gz") else None):
        chunk = chunk[chunk[gene_c].astype(str).str.upper().isin(want)]
        if len(chunk):
            keep.append(chunk)
    if not keep:
        raise SystemExit(f"{path.name}: no rows for {', '.join(sorted(want))}")
    # select by name, never by position: usecols does not promise the order asked for
    t = pd.concat(keep, ignore_index=True)[[gene_c, name_c, sig_c]]
    t.columns = ["gene", "name", "significance"]
    t["label"] = t.significance.map(classify_significance)
    t["change"] = t.name.map(parse_protein_change)
    # one mask for both, applied once: filtering the frame and the parsed changes
    # separately leaves them on different indices
    t = t[t.label.notna() & t.change.notna()]
    out = pd.DataFrame({"gene": t.gene.astype(str).str.upper(),
                        "wt": [c[0] for c in t.change],
                        "pos": [c[1] for c in t.change],
                        "mut": [c[2] for c in t.change],
                        "label": t.label.to_numpy()}).drop_duplicates()
    if not len(out):
        raise SystemExit(f"{path.name}: no missense variant with a clear benign or "
                         "pathogenic classification for these genes")
    return out


def fetch_sequence(acc: str) -> str:
    """UniProt FASTA, cached under data/external so a rerun needs no network."""
    cache = REPO_ROOT / "data" / "external" / "_uniprot" / f"{acc}.fasta"
    if cache.exists():
        return "".join(cache.read_text().split("\n")[1:]).strip()
    import urllib.request
    with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta",
                                timeout=60) as r:
        txt = r.read().decode()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(txt)
    return "".join(txt.split("\n")[1:]).strip()


def family_with_queries(members: list[str], queries: dict[str, str], min_proteins: int):
    """Pool the measured family, and map each query protein's residues onto the same columns.

    The query sequences take part in the alignment so that their numbering is in the same
    frame, but they contribute no data: only the measured proteins are pooled.
    """
    cfgs = {}
    for c in members:
        cfg = load_config(Path(c))
        cfgs[cfg.id] = cfg
    seqs = {i: (protein_sequence(c) or "") for i, c in cfgs.items()}
    bad = [i for i, s in seqs.items() if not s]
    if bad:
        raise SystemExit(f"no sequence for {', '.join(bad)}")
    aln = align_sequences({**seqs, **queries})
    est = {i: position_estimates(load_dataset(c)) for i, c in cfgs.items()}
    long = column_frame(aln, est)
    cols = pool_columns(long, min_proteins)
    maps = {}
    for g in queries:
        c = f"{g}_pos"
        if c not in aln.columns:
            continue
        m = aln[["col", c]].dropna()
        maps[g] = dict(zip(m[c].astype(int), m["col"].astype(int)))
    return cols, maps, aln, long


def column_conservation(aln: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """Per column, how conserved the measured proteins' residues are.

    This is the control the whole comparison turns on. A conserved column is both where the
    yeast data says the position matters and where human pathogenic variants sit, so a
    constraint score that merely tracks conservation has rediscovered conservation. One
    minus the normalised Shannon entropy of the aligned residues: 1 when every member
    carries the same amino acid, 0 when they are maximally mixed.
    """
    rows = []
    for r in aln.itertuples(index=False):
        aas = [getattr(r, f"{m}_aa") for m in members if hasattr(r, f"{m}_aa")]
        aas = [a for a in aas if isinstance(a, str) and a not in ("-", "")]
        if len(aas) < 2:
            rows.append({"col": int(r.col), "conservation": np.nan, "n_aa": len(aas)})
            continue
        v = pd.Series(aas).value_counts(normalize=True)
        h = float(-(v * np.log(v)).sum())
        rows.append({"col": int(r.col), "n_aa": len(aas),
                     "conservation": 1.0 - h / np.log(len(aas))})
    return pd.DataFrame(rows)


def conservation_control(scored: pd.DataFrame) -> dict:
    """Does the yeast constraint add anything once conservation is accounted for?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    d = scored.dropna(subset=["mu", "conservation"])
    y = (d.label == "pathogenic").to_numpy()
    if y.sum() < 5 or (~y).sum() < 5 or len(d) < 30:
        return {"reason": "too few variants with both a pooled column and a conservation value"}
    mu = -d.mu.to_numpy()
    cons = d.conservation.to_numpy()
    out = {"n": int(len(d)), "auroc_abundance": float(roc_auc_score(y, mu)),
           "auroc_conservation": float(roc_auc_score(y, cons)),
           "corr_abundance_conservation": float(ss.spearmanr(mu, cons)[0])}
    X = np.column_stack([(mu - mu.mean()) / (mu.std() or 1),
                         (cons - cons.mean()) / (cons.std() or 1)])
    m = LogisticRegression(max_iter=2000).fit(X, y)
    out["auroc_both"] = float(roc_auc_score(y, m.decision_function(X)))
    out["coef_abundance"] = float(m.coef_[0][0])
    out["coef_conservation"] = float(m.coef_[0][1])
    # the decisive number: abundance scored after conservation has had its say
    resid = mu - np.polyval(np.polyfit(cons, mu, 1), cons)
    out["auroc_abundance_given_conservation"] = float(roc_auc_score(y, resid))
    return out


def evaluate(scored: pd.DataFrame, score: str = "mu") -> dict:
    """AUROC of the yeast constraint for pathogenic against benign, with a bootstrap CI."""
    from sklearn.metrics import roc_auc_score
    d = scored.dropna(subset=[score])
    lab = (d.label == "pathogenic").to_numpy()
    if lab.sum() < 5 or (~lab).sum() < 5:
        return {"n": int(len(d)), "n_pathogenic": int(lab.sum()), "n_benign": int((~lab).sum()),
                "reason": "need at least 5 of each class"}
    # more negative mu = the yeast family says the position matters = predicted pathogenic
    x = -d[score].to_numpy()
    auc = float(roc_auc_score(lab, x))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(d), len(d))
        if 0 < lab[i].sum() < len(i):
            boots.append(roc_auc_score(lab[i], x[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    u = ss.mannwhitneyu(d.loc[lab, score], d.loc[~lab, score], alternative="less")
    return {"n": int(len(d)), "n_pathogenic": int(lab.sum()), "n_benign": int((~lab).sum()),
            "auroc": auc, "auroc_lo": float(lo), "auroc_hi": float(hi),
            "p_mannwhitney": float(u.pvalue),
            "median_pathogenic": float(d.loc[lab, score].median()),
            "median_benign": float(d.loc[~lab, score].median()),
            "n_positions": int(d.pos.nunique()),
            "variants_per_position": float(len(d) / max(d.pos.nunique(), 1))}


def figure(scored: pd.DataFrame, res: dict, by_gene: pd.DataFrame, outdir: Path, name: str):
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1.1]})
    d = scored.dropna(subset=["mu"])
    ax = axes[0]
    groups = {"benign": d.loc[d.label == "benign", "mu"].to_numpy(),
              "pathogenic": d.loc[d.label == "pathogenic", "mu"].to_numpy()}
    # benign and pathogenic are a status pair, not two arbitrary identities, so they keep
    # a semantic green/red rather than two samples of a sequential ramp
    P.violin_box(ax, groups, colors={"benign": "#1B9E77", "pathogenic": "#C0392B"})
    # violin_box already prints each median under its violin, so the counts go above;
    # putting them in the tick labels lands them on top of the medians
    for k, lab in enumerate(("benign", "pathogenic")):
        ax.text(k, 1.01, f"n = {len(groups[lab]):,}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5, color=P.MUTED, clip_on=False)
    ax.set_ylabel("Yeast family constraint at that column (μ)")
    ax.set_title("(a) Where the variants sit", loc="left", fontsize=9)
    ax = axes[1]
    if "auroc" in res:
        fpr, tpr, _ = roc_curve((d.label == "pathogenic"), -d.mu)
        ax.plot(fpr, tpr, lw=2.0, color=P.SERIES[0])
        ax.text(0.97, 0.05, f"AUROC = {res['auroc']:.2f}\n"
                            f"[{res['auroc_lo']:.2f}, {res['auroc_hi']:.2f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5)
    ax.plot([0, 1], [0, 1], color=P.MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(b) Pathogenic vs benign", loc="left", fontsize=9)
    ax = axes[2]
    if len(by_gene):
        g = by_gene.dropna(subset=["auroc"]).sort_values("auroc")
        y = np.arange(len(g))
        cm = P.sequential_cmap()
        ax.barh(y, g.auroc, color=[cm(0.15 + 0.7 * v) for v in g.auroc], height=0.6)
        ax.errorbar(g.auroc, y, xerr=[g.auroc - g.auroc_lo, g.auroc_hi - g.auroc],
                    fmt="none", ecolor=P.INK, elinewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.gene}  ({int(r.n_pathogenic)}P/{int(r.n_benign)}B)"
                            for r in g.itertuples()], fontsize=6.5)
        ax.axvline(0.5, color=P.MUTED, lw=0.9, ls=(0, (3, 3)))
        ax.set_xlim(0, 1)
        # one gene must not become a bar the height of the panel
        ax.set_ylim(-0.8, max(len(g) - 0.2, 1.4))
        ax.set_xlabel("AUROC")
    ax.set_title("(c) Per gene", loc="left", fontsize=9)
    P.save(fig, outdir, f"c01_{name}_clinvar", None, "C01")
    plt.close(fig)


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms clinvar",
        description="Test the family's abundance signal against ClinVar classifications.")
    ap.add_argument("--family", action="append", required=True, metavar="NAME=cfg1,cfg2,...",
                    help="the measured proteins to pool, as in `mpdms bayes`; repeatable, "
                         "so a narrow and a broad family can be compared on the same variants")
    ap.add_argument("--human", action="append", required=True, metavar="GENE=ACCESSION",
                    help="a human member of the same family, e.g. SLC2A1=P11166 (repeatable)")
    ap.add_argument("--table", required=True,
                    help="ClinVar variant_summary.txt(.gz), or any table with gene, "
                         "protein change and clinical significance columns")
    ap.add_argument("--min-proteins", type=int, default=3)
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)

    fams = {}
    for spec in a.family:
        name, _, members = spec.partition("=")
        members = [m.strip() for m in members.split(",") if m.strip()]
        if not name or not members:
            raise SystemExit(f"--family {spec}: expected NAME=cfg1.yaml,cfg2.yaml,...")
        bad = [m for m in members if not Path(m).is_file()]
        if bad:
            raise SystemExit(f"--family {name}: not a config file: {', '.join(bad)}")
        fams[name] = members
    queries = {}
    for h in a.human:
        g, _, acc = h.partition("=")
        if not acc:
            raise SystemExit(f"--human {h}: expected GENE=ACCESSION, e.g. SLC2A1=P11166")
        queries[g.strip().upper()] = acc.strip()

    print(f"  fetching {len(queries)} human sequence(s) ...", flush=True)
    seqs = {}
    for g, acc in queries.items():
        try:
            seqs[g] = fetch_sequence(acc)
        except Exception as e:
            raise SystemExit(f"could not fetch {acc} for {g}: {type(e).__name__}: {e}\n"
                             f"        put the FASTA at data/external/_uniprot/{acc}.fasta "
                             "and rerun")
        print(f"    {g:<10} {acc}  {len(seqs[g])} residues", flush=True)

    print(f"  reading {Path(a.table).name} ...", flush=True)
    cv = read_clinvar(Path(a.table), set(seqs))
    print(f"    {len(cv):,} classified missense variants: "
          f"{(cv.label == 'pathogenic').sum():,} pathogenic, "
          f"{(cv.label == 'benign').sum():,} benign", flush=True)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, members in fams.items():
        summary[name] = run_one(name, members, seqs, cv, a.min_proteins, outdir)
    if len(summary) > 1:
        print("\n  == families side by side")
        for n, r in summary.items():
            o = r.get("overall", {})
            if "auroc" in o:
                print(f"    {n:<12} AUROC {o['auroc']:.3f} "
                      f"[{o['auroc_lo']:.3f}, {o['auroc_hi']:.3f}]   "
                      f"{o['n_pathogenic']}P/{o['n_benign']}B over "
                      f"{r['n_columns']:,} pooled columns")
            else:
                print(f"    {n:<12} not evaluated: {o.get('reason')}")
    (outdir / "clinvar.json").write_text(json.dumps(summary, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


def run_one(name, members, seqs, cv, min_proteins, outdir):
    import json
    print(f"\n  == {name}: aligning and pooling {len(members)} measured proteins",
          flush=True)
    cols, maps, aln, long = family_with_queries(members, seqs, min_proteins)
    print(f"    {len(cols):,} pooled columns from {long.protein.nunique()} proteins "
          f"({aln.attrs.get('method', '?')})", flush=True)

    by_col = cols.set_index("col")
    rows = []
    for r in cv.itertuples(index=False):
        col = maps.get(r.gene, {}).get(int(r.pos))
        seq = seqs.get(r.gene, "")
        wt_ok = bool(seq) and 0 < r.pos <= len(seq) and seq[r.pos - 1] == r.wt
        rec = {"gene": r.gene, "pos": r.pos, "wt": r.wt, "mut": r.mut, "label": r.label,
               "col": col, "wt_matches_uniprot": wt_ok}
        if col is not None and col in by_col.index:
            c = by_col.loc[col]
            rec.update(mu=float(c.mu), tau=float(c.tau), J=int(c.J),
                       seg_type=c.get("seg_type"))
        rows.append(rec)
    scored = pd.DataFrame(rows)
    cons = column_conservation(aln, sorted(long.protein.unique()))
    scored = scored.merge(cons, on="col", how="left")
    mism = int((~scored.wt_matches_uniprot).sum())
    if mism:
        print(f"    WARNING: {mism:,} variants whose wild-type residue does not match the "
              "UniProt sequence (different transcript or isoform); they are dropped")
        scored = scored[scored.wt_matches_uniprot]
    mapped = scored.mu.notna()
    print(f"    {int(mapped.sum()):,} of {len(scored):,} variants land on a pooled column "
          f"({mapped.mean():.0%})", flush=True)

    scored.to_csv(outdir / f"{name}_scored_variants.csv", index=False)
    res = evaluate(scored)
    res["conservation_control"] = conservation_control(scored)
    per = []
    for g, gg in scored.groupby("gene"):
        r = evaluate(gg)
        r["gene"] = g
        per.append(r)
    by_gene = pd.DataFrame(per)
    by_gene.to_csv(outdir / f"{name}_by_gene.csv", index=False)
    if "auroc" in res:
        figure(scored, res, by_gene, outdir, name)

    print()
    if "auroc" in res:
        print(f"  AUROC {res['auroc']:.3f}  [{res['auroc_lo']:.3f}, {res['auroc_hi']:.3f}]"
              f"   {res['n_pathogenic']:,} pathogenic vs {res['n_benign']:,} benign"
              f"   p = {res['p_mannwhitney']:.2g}")
        gap = res["median_benign"] - res["median_pathogenic"]
        print(f"  median constraint: pathogenic {res['median_pathogenic']:+.3f}, "
              f"benign {res['median_benign']:+.3f}   (gap {gap:+.3f})")
        tau = float(cols.tau.median())
        spread = float(cols.mu.std(ddof=1))
        print(f"  pooled columns: median τ {tau:.3f}, SD of μ {spread:.3f}, "
              f"ratio τ/SD(μ) {tau / spread:.2f}")
        if tau / spread > 0.6:
            print("    the family disagrees about as much as the columns differ, so μ is "
                  "shrunk hard toward the family average and the contrast above is "
                  "flattened; a tighter family will separate better")
        res["median_tau"] = tau
        res["sd_mu"] = spread
        res["gap"] = gap
        print(f"  {res['n_positions']:,} distinct positions carry these "
              f"{res['n']:,} variants ({res['variants_per_position']:.1f} per position): "
              "this is a position score, so that ratio is the ceiling on what it can "
              "resolve")
        cc = res.get("conservation_control", {})
        if "auroc_abundance" in cc:
            print(f"  conservation control (n = {cc['n']:,}):")
            print(f"    yeast abundance alone        AUROC {cc['auroc_abundance']:.3f}")
            print(f"    column conservation alone    AUROC {cc['auroc_conservation']:.3f}")
            print(f"    both together                AUROC {cc['auroc_both']:.3f}")
            print(f"    abundance after conservation AUROC "
                  f"{cc['auroc_abundance_given_conservation']:.3f}"
                  "   <- what abundance adds that conservation does not")
            print(f"    they correlate at rho = {cc['corr_abundance_conservation']:+.2f}; "
                  f"standardised coefficients: abundance {cc['coef_abundance']:+.2f}, "
                  f"conservation {cc['coef_conservation']:+.2f}")
            if cc["auroc_abundance_given_conservation"] < 0.55:
                print("    NOTE: once conservation is removed the abundance signal is close "
                      "to chance, so this result is mostly conservation rediscovered")
        else:
            print(f"  conservation control: {cc.get('reason', 'not computed')}")
        for r in by_gene.itertuples():
            if np.isfinite(getattr(r, "auroc", np.nan)):
                print(f"    {r.gene:<10} AUROC {r.auroc:.3f}  "
                      f"({int(r.n_pathogenic)}P / {int(r.n_benign)}B)")
            else:
                print(f"    {r.gene:<10} not evaluated: {getattr(r, 'reason', '')}")
    else:
        print(f"  not evaluated: {res.get('reason')} "
              f"({res.get('n_pathogenic', 0)} pathogenic, {res.get('n_benign', 0)} benign)")
    return {"overall": res, "by_gene": by_gene.to_dict("records"),
            "alignment": aln.attrs.get("method"), "n_columns": int(len(cols)),
            "n_proteins": int(long.protein.nunique())}


if __name__ == "__main__":
    raise SystemExit(main())
