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
    """ClinVar's free text to pathogenic / benign / uncertain / None.

    "uncertain" is returned for a variant ClinVar itself calls uncertain, and it is never
    a label in any comparison: a variant of uncertain significance is not a negative, and
    counting it as one is the commonest way to make a predictor look good on ClinVar.
    Everything that reaches the evaluation drops it. It is kept so it can be SCORED, which
    is a different question - what the model says about variants nobody has classified -
    and that question has no right answer to be flattered by.

    Conflicting interpretations stay None. A variant two submitters disagree about is not
    the same thing as one nobody has called, and pooling them would hide a real signal:
    the model may well side with one submitter.
    """
    t = str(s).strip().lower()
    if not t or "conflict" in t or "not provided" in t:
        return None
    if "uncertain" in t:
        return "uncertain"
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


def read_clinvar(path: Path, genes: set[str], include_uncertain: bool = False) -> pd.DataFrame:
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
    keep_labels = {"pathogenic", "benign"} | ({"uncertain"} if include_uncertain else set())
    t = t[t.label.isin(keep_labels) & t.change.notna()]
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


def read_alphamissense(path: Path, uniprots: set[str]) -> pd.DataFrame:
    """AlphaMissense substitution scores for these UniProt accessions.

    The released table is 216 million rows, so it is streamed and filtered by accession
    rather than read: the same lesson the ESM import taught, and the difference between
    seconds and minutes. Columns are found by NAME (uniprot_id, protein_variant,
    am_pathogenicity) because the published order is not something to rely on, and any
    leading comment lines are skipped.
    """
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    header, first, n_skip = None, None, 0
    with opener(path, "rt") as fh:
        for line in fh:
            n_skip += 1
            if line.startswith("#"):
                continue
            if header is None:
                header = line.rstrip("\n").split("\t")
                continue
            first = line.rstrip("\n").split("\t")
            break
    if header is None:
        raise SystemExit(f"{path.name}: no header row")
    # The released file names three columns and writes four: am_class is in every data
    # row and missing from the header. pandas resolves that by making the first column
    # an index, which shifts every name one to the left - uniprot_id then holds "M1A"
    # and am_pathogenicity holds "benign" - and the accession filter quietly matches
    # nothing. Naming the surplus columns before reading is what stops that.
    extra = (len(first) - len(header)) if first else 0
    if extra > 0:
        header = header + (["am_class"] if extra == 1 and "am_class" not in header
                           else [f"column_{len(header) + i + 1}" for i in range(extra)])
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in header}
    uni = cols.get("uniprotid") or cols.get("uniprot")
    var = cols.get("proteinvariant") or cols.get("variant")
    sc = cols.get("ampathogenicity") or cols.get("pathogenicity")
    if not (uni and var and sc):
        raise SystemExit(f"{path.name}: need uniprot_id, protein_variant and "
                         f"am_pathogenicity columns; found {header[:8]}")
    want = {u.upper() for u in uniprots}
    keep = []
    for chunk in pd.read_csv(path, sep="\t", comment="#", chunksize=1_000_000,
                             header=None, names=header, skiprows=n_skip - 1,
                             usecols=lambda c: c in (uni, var, sc), low_memory=False):
        hit = chunk[chunk[uni].astype(str).str.upper().isin(want)]
        if len(hit):
            keep.append(hit)
    if not keep:
        raise SystemExit(f"{path.name}: no rows for {', '.join(sorted(want))}")
    t = pd.concat(keep, ignore_index=True)[[uni, var, sc]]
    t.columns = ["uniprot", "variant", "am_pathogenicity"]
    v = t.variant.astype(str).str.extract(r"^([A-Z])(\d+)([A-Z])$")
    t = t[v[0].notna()]
    return pd.DataFrame({"uniprot": t.uniprot.str.upper(),
                         "wt": v[0][v[0].notna()], "pos": v[1][v[0].notna()].astype(int),
                         "mut": v[2][v[0].notna()],
                         "am_pathogenicity": t.am_pathogenicity.to_numpy()})


# how each predictor points. `sign` multiplies the raw value so that, after it, HIGHER
# always means "more likely pathogenic" - which is what the AUROC needs and the one
# thing that is silently catastrophic to get wrong.
def read_mochi(path: Path, min_folds: int = 8) -> pd.DataFrame:
    """A MoCHI weights table: one energy per (alignment column, mutant residue).

    The model is fitted on a masked alignment of the yeast members, so a row is not a
    substitution of a named wild-type residue but "this residue at this column", and
    the id carries no wild-type letter - it reads like `-3A`, a gap where the wild type
    would be. That is why the merge below is on (column, mutant) and never checks a
    wild-type identity: there is none to check, and a wt column invented here would be
    a false reassurance.

    Energies are taken in kcal/mol. Each row is the mean over the folds that converged,
    and the ones that converged in only a few are kept out: with ten folds a row at n=8
    can carry a 95% interval several kcal/mol wide, which is not a measurement of
    anything. The count dropped is reported rather than absorbed.
    """
    w = pd.read_csv(path, sep="\t")
    need = {"id", "Pos", "mean_kcal/mol"}
    missing = need - set(w.columns)
    if missing:
        raise SystemExit(f"{path}: MoCHI weights lack {sorted(missing)}")
    w = w[w.id.astype(str).str.upper() != "WT"].copy()
    w["Pos"] = pd.to_numeric(w.Pos, errors="coerce")
    w = w.dropna(subset=["Pos", "mean_kcal/mol"])
    w["col"] = w.Pos.astype(int)
    # the mutant residue is the last character of the id; anything else is not a
    # single-residue weight and has no place in a per-variant predictor
    w["mut"] = w.id.astype(str).str.strip().str[-1].str.upper()
    w = w[w.mut.isin(list("ACDEFGHIKLMNPQRSTVWY"))]
    n_all = len(w)
    if "n" in w:
        w = w[pd.to_numeric(w.n, errors="coerce").fillna(0) >= min_folds]
    out = w.rename(columns={"mean_kcal/mol": "mochi_ddg"})[["col", "mut", "mochi_ddg"]]
    out = out.groupby(["col", "mut"], as_index=False).mochi_ddg.mean()
    out.attrs["n_dropped_low_folds"] = int(n_all - len(w))
    out.attrs["n_weights"] = int(len(out))
    out.attrs["max_col"] = int(out.col.max()) if len(out) else 0
    return out


def read_alignment(path: Path) -> dict[str, str]:
    """A FASTA alignment as {header: aligned sequence}, every sequence the same length."""
    seqs, name = {}, None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
            seqs[name] = []
        elif name and line:
            seqs[name].append(line)
    out = {k: "".join(v) for k, v in seqs.items()}
    lens = {len(v) for v in out.values()}
    if len(lens) > 1:
        raise SystemExit(f"{path}: not an alignment - sequence lengths {sorted(lens)}")
    return out


def residue_to_column(aligned: str) -> dict[int, int]:
    """Residue number (1-based, ungapped) -> column number (1-based) in the alignment."""
    out, k = {}, 0
    for i, c in enumerate(aligned, start=1):
        if c != "-":
            k += 1
            out[k] = i
    return out


def mochi_variant_scores(weights: pd.DataFrame, aln: dict[str, str],
                         acc: dict[str, str]) -> pd.DataFrame:
    """The energy of each human variant, by the column its residue aligns to.

    The human protein is matched to its sequence in the alignment by UniProt accession,
    which the `--human GENE=ACCESSION` argument already supplies, rather than by a
    header substring that could match two entries.
    """
    width = len(next(iter(aln.values()))) if aln else 0
    if weights.attrs.get("max_col", 0) > width:
        raise SystemExit(
            f"the MoCHI weights run to column {weights.attrs['max_col']} but the "
            f"alignment is {width} columns wide: these are different column frames, "
            "so the mapping would be silently wrong. Supply the alignment the model "
            "was fitted on.")
    rows = []
    for gene, a in acc.items():
        hit = [v for k, v in aln.items() if a.upper() in k.upper()]
        if len(hit) != 1:
            continue
        r2c = residue_to_column(hit[0])
        m = pd.DataFrame({"pos": list(r2c), "col": [r2c[p] for p in r2c]})
        m["gene"] = gene
        rows.append(m)
    if not rows:
        return pd.DataFrame(columns=["gene", "pos", "mut", "mochi_ddg"])
    pos_map = pd.concat(rows, ignore_index=True)
    out = pos_map.merge(weights, on="col", how="inner")
    return out[["gene", "pos", "mut", "col", "mochi_ddg"]]


PREDICTORS = [("am_pathogenicity", "AlphaMissense", +1, "higher = pathogenic"),
              ("mochi_ddg", "MoCHI mask ΔΔG", +1, "higher = destabilising"),
              ("esm1v", "ESM-1v", -1, "lower = deleterious"),
              ("mu", "Family model μ", -1, "lower = constrained"),
              ("conservation", "Column conservation", +1, "higher = conserved")]
VIOLIN_ORDER = ["am_pathogenicity", "esm1v", "mochi_ddg", "mu"]


def predictor_table(scored: pd.DataFrame, am: pd.DataFrame | None,
                    esm: dict[str, pd.DataFrame], acc: dict[str, str],
                    mochi: pd.DataFrame | None = None) -> pd.DataFrame:
    """The known variants with every available predictor attached, one column each."""
    d = scored[scored.label.isin(["benign", "pathogenic"])].copy()
    if mochi is not None and len(mochi):
        d = d.merge(mochi.drop(columns=[c for c in ("col",) if c in mochi]),
                    on=["gene", "pos", "mut"], how="left")
    if am is not None and len(am):
        d["uniprot"] = d.gene.map({g: a.upper() for g, a in acc.items()})
        d = d.merge(am, on=["uniprot", "pos", "wt", "mut"], how="left")
    for g, e in (esm or {}).items():
        m = e.rename(columns={"esm1v": "_e"})[["pos", "mut", "_e"]]
        m = m.assign(gene=g)
        d = d.merge(m, on=["gene", "pos", "mut"], how="left")
        d["esm1v"] = d["_e"] if "esm1v" not in d else d.esm1v.fillna(d["_e"])
        d = d.drop(columns="_e")
    return d


def score_predictors(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """AUROC per predictor, on its own variants and on the shared ones."""
    both = d.dropna(subset=[k for k in keys if k in d])
    rows = []
    for k, lab, sign, note in PREDICTORS:
        if k not in d or d[k].isna().all():
            continue
        for scope, sub in (("own", d.dropna(subset=[k])), ("shared", both)):
            # evaluate() was written for mu and takes LOWER = pathogenic, so each
            # predictor is flipped into that convention here rather than special-cased
            # inside it. -sign puts every predictor the same way round: AlphaMissense
            # (high = pathogenic) is negated, ESM-1v and mu (low = pathogenic) are not.
            r = evaluate(sub.assign(_s=-sign * sub[k]), score="_s")
            rows.append({"predictor": lab, "key": k, "scope": scope, "direction": note,
                         **{x: r.get(x) for x in ("n", "n_pathogenic", "n_benign",
                                                  "auroc", "auroc_lo", "auroc_hi",
                                                  "p_mannwhitney")}})
    return pd.DataFrame(rows)


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
           "n_conservation_levels": int(pd.Series(cons).nunique()),
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


def place_uncertain(scored: pd.DataFrame) -> pd.DataFrame:
    """Where each variant of uncertain significance falls against the two known sets.

    Not a classification. For each VUS, `beats_benign` is the fraction of benign variants
    it is more constrained than and `beats_pathogenic` the fraction of pathogenic ones -
    the same quantity `rank_variants` computes, so a VUS can be read on the same axis as
    the variants whose answer is known. `sits_with` says only which reference set it
    resembles, and "beyond" marks a VUS more constrained than every pathogenic variant.

    The honest reading of a high `beats_benign` is "this position behaves, in yeast, like
    the positions where SLC2A10 variants are known to cause disease". It is a hypothesis
    about a position, not a call on a patient's variant.
    """
    d = scored.dropna(subset=["mu"])
    vus = d[d.label == "uncertain"].copy()
    ben = d.loc[d.label == "benign", "mu"].to_numpy()
    pat = d.loc[d.label == "pathogenic", "mu"].to_numpy()
    if not len(vus) or not len(ben) or not len(pat):
        return pd.DataFrame()
    vus["beats_benign"] = [(np.sum(ben > m) + 0.5 * np.sum(ben == m)) / len(ben)
                           for m in vus.mu]
    vus["beats_pathogenic"] = [(np.sum(pat > m) + 0.5 * np.sum(pat == m)) / len(pat)
                               for m in vus.mu]
    # the gap between the two reference ranges is its own answer: a VUS more constrained
    # than every benign variant but less than every pathogenic one resembles neither set,
    # and calling it pathogenic-like on the first half of that sentence would overstate it
    vus["sits_with"] = np.select(
        [vus.mu < pat.min(),
         (vus.beats_benign >= 1.0) & (vus.beats_pathogenic > 0.0),
         (vus.beats_pathogenic <= 0.0) & (vus.beats_benign < 1.0)],
        ["beyond every pathogenic", "with the pathogenic set", "with the benign set"],
        default="between the two")
    cols = ["gene", "pos", "wt", "mut", "mu", "tau", "J", "seg_type", "conservation",
            "beats_benign", "beats_pathogenic", "sits_with", "col"]
    return vus[[c for c in cols if c in vus.columns]].sort_values("mu").reset_index(drop=True)


def collapse_to_positions(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per position, because mu is a property of the position, not the variant.

    Three substitutions at one arginine get the same score and count three times in a
    variant-level AUROC, so a gene whose pathogenic variants cluster at a few residues is
    scored on fewer independent observations than the variant count suggests. Collapsing
    first is the honest unit. A position carrying both a pathogenic and a benign call is
    dropped rather than assigned one of them: the score cannot separate what it cannot
    see apart.
    """
    d = scored.dropna(subset=["mu"])
    if not len(d):
        return d
    g = d.groupby(["gene", "pos"])
    keep = g.label.nunique() == 1
    out = (g.first().assign(n_variants=g.size(), label=g.label.first())
           .loc[keep[keep].index].reset_index())
    out.attrs["n_mixed_positions"] = int((~keep).sum())
    return out


def rank_variants(scored: pd.DataFrame, score: str = "mu") -> pd.DataFrame:
    """Every scored variant, ordered by constraint, with how well each one is called.

    The AUROC is an average over all pathogenic-benign pairs, so it hides which variants
    the model gets right. This takes it apart. For a pathogenic variant, `pairs_beaten`
    is the fraction of benign variants it is more constrained than; for a benign variant,
    the fraction of pathogenic variants it is less constrained than. Averaging that
    column over either class returns the AUROC exactly, so a variant with a low value is
    literally one of the ones costing the score.

    The threshold is Youden's J on these same variants, which is the best this score can
    do here rather than a threshold anyone could have set in advance - so `called` is a
    description of the figure, not a prediction, and `pairs_beaten` is the number to rank
    by.
    """
    d = scored.dropna(subset=[score]).copy()
    path = (d.label == "pathogenic").to_numpy()
    if path.sum() == 0 or (~path).sum() == 0:
        return pd.DataFrame()
    x = -d[score].to_numpy()                      # higher = more constrained
    pos, neg = x[path], x[~path]
    beaten = np.where(
        path,
        [(np.sum(neg < v) + 0.5 * np.sum(neg == v)) / len(neg) for v in x],
        [(np.sum(pos > v) + 0.5 * np.sum(pos == v)) / len(pos) for v in x])
    d["pairs_beaten"] = beaten
    d["constraint_rank"] = ss.rankdata(-x, method="min").astype(int)   # 1 = most constrained
    d["percentile"] = 100 * (1 - ss.rankdata(-x) / len(x))
    # Youden's J over the thresholds the data actually offers
    cuts = np.unique(x)
    j = [(np.mean(pos >= c) - np.mean(neg >= c), c) for c in cuts]
    best = max(j)[1]
    d["called"] = np.where(x >= best, "pathogenic", "benign")
    d["outcome"] = np.select(
        [path & (d.called == "pathogenic"), path & (d.called == "benign"),
         ~path & (d.called == "benign")],
        ["true positive", "missed (false negative)", "true negative"],
        default="false alarm (false positive)")
    d.attrs["threshold_mu"] = float(-best)
    d.attrs["youden_j"] = float(max(j)[0])
    cols = ["gene", "pos", "wt", "mut", "label", "outcome", "pairs_beaten",
            "constraint_rank", "percentile", "mu", "tau", "J", "seg_type",
            "conservation", "col"]
    out = d[[c for c in cols if c in d.columns]].sort_values("constraint_rank")
    out.attrs = dict(d.attrs)
    return out.reset_index(drop=True)


def figure_variants(ranked: pd.DataFrame, outdir: Path, name: str, gene: str = ""):
    """Every variant as its own point, so the misses can be named rather than averaged."""
    if not len(ranked):
        return
    d = ranked.sort_values("mu")
    y = np.arange(len(d))[::-1]
    col = {"benign": "#1B9E77", "pathogenic": "#C0392B"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, max(3.4, 0.17 * len(d) + 1.1)),
                             layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    thr = d.attrs.get("threshold_mu")
    if thr is not None:
        ax.axvline(thr, color=P.INK, lw=1.0, ls=(0, (4, 3)))
        ax.text(thr, len(d) + 0.4, f"  best threshold μ = {thr:+.3f}", fontsize=6,
                va="bottom", color=P.INK)
    ax.hlines(y, d.mu.min() - 0.01, d.mu, color="#DDDDDD", lw=0.7, zorder=0)
    miss = d.outcome.str.startswith(("missed", "false alarm"))
    ax.scatter(d.mu, y, s=30, c=[col[v] for v in d.label], zorder=4,
               edgecolor=np.where(miss, P.INK, "none"),
               linewidth=np.where(miss, 1.1, 0.0))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.wt}{int(r.pos)}{r.mut}" for r in d.itertuples()], fontsize=6)
    for t, m in zip(ax.get_yticklabels(), miss):
        if m:
            t.set_fontweight("bold")
    ax.set_xlabel("Yeast family constraint at that column (μ)")
    ax.set_ylim(-1, len(d))
    ax.set_title(f"(a) every {gene or 'query'} variant, ordered by constraint "
                 "(ringed and bold = on the wrong side)", loc="left", fontsize=8.5)
    hand = [plt.Line2D([], [], marker="o", ls="", color=col[k], label=k)
            for k in ("pathogenic", "benign")]
    ax.legend(handles=hand, frameon=False, fontsize=6.5, loc="lower right")

    ax = axes[1]
    for k, lab in enumerate(("pathogenic", "benign")):
        g = d[d.label == lab].sort_values("pairs_beaten")
        ax.scatter(g.pairs_beaten, np.arange(len(g)) + (0 if k == 0 else 0),
                   s=26, color=col[lab], label=lab, alpha=0.85, edgecolor=P.INK, lw=0.3)
    ax.axvline(0.5, color=P.MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("fraction of the other class it is correctly ordered against")
    ax.set_ylabel("variants of that class, sorted")
    ax.set_title("(b) what each variant contributes to the AUROC", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    P.save(fig, outdir, f"c02_{name}_variants", None, "C02")
    plt.close(fig)


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


def _vus_panel(ax, known: pd.DataFrame, vus: pd.DataFrame, gene: str | None):
    """The VUS placed beside the two reference sets, rather than classified.

    Three strips of the same quantity, so the comparison is direct. Shaded full-range
    bands were the obvious alternative and the wrong one: the pathogenic range runs from
    the most constrained column in the gene up past the three variants the model misses,
    so it covers the benign range entirely and the picture says nothing. Points and
    medians say what the bands could not.
    """
    col = {"benign": "#1B9E77", "uncertain": P.INK, "pathogenic": "#C0392B"}
    sets = [("benign", known.loc[known.label == "benign", "mu"].to_numpy()),
            ("uncertain", vus.mu.to_numpy()),
            ("pathogenic", known.loc[known.label == "pathogenic", "mu"].to_numpy())]
    rng = np.random.default_rng(0)
    floor = float(sets[0][1].min())
    for k, (lab, v) in enumerate(sets):
        below = v < floor
        ax.scatter(k + rng.uniform(-0.19, 0.19, len(v)), v, s=22, color=col[lab],
                   alpha=0.85, lw=np.where(below & (lab == "uncertain"), 0.9, 0.0),
                   edgecolor=P.INK, zorder=4)
        ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color=P.INK, lw=1.6, zorder=5)
        ax.text(k, 1.01, f"n = {len(v)}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6.5, color=P.MUTED, clip_on=False)
    ax.axhline(floor, color=P.INK, lw=0.8, ls=(0, (4, 3)))
    n_below = int((vus.mu < floor).sum())
    ax.set_xticks(range(3))
    ax.set_xticklabels(["benign", "uncertain", "pathogenic"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Yeast family constraint at that column (μ)")
    ax.set_title(f"({'c' if gene else 'd'}) where the {len(vus)} VUS fall — {n_below} "
                 "below every benign variant", loc="left", fontsize=9)


def figure(scored: pd.DataFrame, res: dict, by_gene: pd.DataFrame, outdir: Path, name: str):
    from sklearn.metrics import roc_curve
    # with one query gene the per-gene panel is the same number as panel (b) drawn as a
    # bar, so it is dropped rather than repeated; the gene's name moves into the titles
    one = by_gene.dropna(subset=["auroc"]) if "auroc" in by_gene else by_gene
    single = str(one.gene.iloc[0]) if len(one) == 1 else None
    vus = scored[(scored.label == "uncertain") & scored.mu.notna()]
    n = (2 if single else 3) + (1 if len(vus) else 0)
    ratios = [1, 1] + ([] if single else [1.1]) + ([1] if len(vus) else [])
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + (0.3 if single else 0.6), 3.2),
                             layout="constrained", gridspec_kw={"width_ratios": ratios})
    # the known variants carry the comparison; a VUS is scored but never counted in it
    d = scored[scored.label.isin(["benign", "pathogenic"])].dropna(subset=["mu"])
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
    ax.set_title(f"(a) Where the variants sit"
                 + (f" in {single}" if single else ""), loc="left", fontsize=9)
    ax = axes[1]
    if "auroc" in res:
        fpr, tpr, _ = roc_curve((d.label == "pathogenic"), -d.mu)
        ax.plot(fpr, tpr, lw=2.0, color=P.SERIES[0])
        ax.text(0.97, 0.05, f"AUROC = {res['auroc']:.2f}\n"
                            f"[{res['auroc_lo']:.2f}, {res['auroc_hi']:.2f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5)
    ax.plot([0, 1], [0, 1], color=P.MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(b) Pathogenic vs benign"
                 + (f" in {single}" if single else ""), loc="left", fontsize=9)
    if len(vus):
        _vus_panel(axes[2 if single else 3], d, vus, single)
    if single:
        P.save(fig, outdir, f"c01_{name}_clinvar", None, "C01")
        plt.close(fig)
        return
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


BENIGN_GREY = "#9A9A9A"
PATHOGENIC_BLUE = "#2A5DA8"
# one colour per predictor, by key. Conservation is the dashed grey reference, and a
# predictor added later without an entry falls back to it rather than crashing.
ROC_COLOR = {"am_pathogenicity": "#C0392B", "esm1v": "#1B9E77",
             "mochi_ddg": "#8E44AD", "mu": PATHOGENIC_BLUE, "conservation": "#6B6B6B"}


def figure_predictors(d: pd.DataFrame, res: pd.DataFrame, outdir: Path, name: str,
                      gene: str = ""):
    """Each predictor's two distributions, then all of them on one ROC.

    The violins show each predictor in its own units, because they have no common
    scale and rescaling them to share one would invent a comparison. The ROC is where
    they are actually compared, and it is drawn on the variants that ALL of them score:
    an AUROC computed on a predictor's own larger set is a different question, so those
    are reported in the table beside it rather than overlaid here.
    """
    from sklearn.metrics import roc_curve
    keys = [k for k in VIOLIN_ORDER if k in d and d[k].notna().any()]
    if not keys:
        return
    roc_keys = keys + [k for k in ("conservation",) if k in d and d[k].notna().any()]
    shared = d.dropna(subset=roc_keys)
    n = len(keys) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n + 0.4, 3.3), layout="constrained")
    look = {k: (lab, sign, note) for k, lab, sign, note in PREDICTORS}
    for ax, k in zip(axes, keys):
        lab, _, note = look[k]
        sub = d.dropna(subset=[k])
        groups = {"benign": sub.loc[sub.label == "benign", k].to_numpy(),
                  "pathogenic": sub.loc[sub.label == "pathogenic", k].to_numpy()}
        P.violin_box(ax, groups, colors={"benign": BENIGN_GREY,
                                         "pathogenic": PATHOGENIC_BLUE})
        for j, g in enumerate(("benign", "pathogenic")):
            ax.text(j, 1.01, f"n = {len(groups[g]):,}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.5, color=P.MUTED, clip_on=False)
        ax.set_ylabel(f"{lab}  ({note})", fontsize=7.5)
        r = res[(res.key == k) & (res.scope == "own")]
        extra = f"AUROC {r.auroc.iloc[0]:.2f}" if len(r) and pd.notna(r.auroc.iloc[0]) else ""
        ax.set_title(f"({chr(97 + list(axes).index(ax))}) {lab}  {extra}", loc="left",
                     fontsize=9)
    ax = axes[-1]
    # keyed by predictor, not zipped against a positional list: a fifth predictor used
    # to run off the end of that list and leave conservation with no colour at all
    cols = {k: ROC_COLOR.get(k, P.MUTED) for k in roc_keys}
    y = (shared.label == "pathogenic").to_numpy()
    for k in roc_keys:
        lab, sign, _ = look[k]
        if len(shared) < 10 or y.sum() == 0 or (~y).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y, sign * shared[k].to_numpy())
        r = res[(res.key == k) & (res.scope == "shared")]
        a = r.auroc.iloc[0] if len(r) else np.nan
        ax.plot(fpr, tpr, lw=2.0 if k != "conservation" else 1.3, color=cols[k],
                ls=(0, (4, 3)) if k == "conservation" else "-",
                label=f"{lab}  {a:.2f}")
    ax.plot([0, 1], [0, 1], color=P.INK, lw=0.8, ls=(0, (2, 3)))
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", title=None)
    ax.set_title(f"({chr(97 + n - 1)}) the same {len(shared):,} variants", loc="left",
                 fontsize=9)
    fig.suptitle(f"{gene or 'query'}: benign in grey, pathogenic in blue. Every predictor "
                 "in its own units; the ROC compares them on the variants all of them "
                 "score.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, f"c03_{name}_predictors", None, "C03")
    plt.close(fig)


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms clinvar",
        description="Test the family's abundance signal against ClinVar classifications.")
    ap.add_argument("--family", action="append", default=[], metavar="NAME=cfg1,cfg2,...",
                    help="the measured proteins to pool, as in `mpdms bayes`; repeatable, "
                         "so a narrow and a broad family can be compared on the same variants")
    ap.add_argument("--human", action="append", required=True, metavar="GENE=ACCESSION",
                    help="a human member of the same family, e.g. SLC2A1=P11166 (repeatable)")
    ap.add_argument("--table", required=True,
                    help="ClinVar variant_summary.txt(.gz), or any table with gene, "
                         "protein change and clinical significance columns")
    ap.add_argument("--min-proteins", type=int, default=3)
    ap.add_argument("--alphamissense", metavar="FILE",
                    help="AlphaMissense_aa_substitutions.tsv.gz (or any table with "
                         "uniprot_id, protein_variant and am_pathogenicity). Streamed "
                         "and filtered by accession, so the full release is fine")
    ap.add_argument("--esm", action="append", default=[], metavar="GENE=FILE",
                    help="per-variant ESM-1v scores for a query protein, as a csv with "
                         "pos, mut and esm1v/llr/score (repeatable)")
    ap.add_argument("--mochi", metavar="FILE",
                    help="a MoCHI weights table from a masked-alignment energy model: "
                         "id, Pos, mean_kcal/mol, one row per (alignment column, "
                         "mutant residue). Scored per variant, unlike the family model")
    ap.add_argument("--mochi-aln", metavar="FILE",
                    help="the alignment the MoCHI model was fitted on, as FASTA, "
                         "including the human query sequence. Pos in the weights is a "
                         "column of THIS alignment, not of the family alignment this "
                         "command builds")
    ap.add_argument("--mochi-min-folds", type=int, default=8,
                    help="drop weights fitted in fewer than this many folds (default 8)")
    ap.add_argument("--include-vus", action="store_true",
                    help="also score the variants ClinVar calls uncertain, and add a panel "
                         "showing where they fall. They never enter the AUROC, the "
                         "conservation control or any per-variant ranking")
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
        if len(members) < a.min_proteins:
            raise SystemExit(
                f"--family {name} has {len(members)} proteins but --min-proteins is "
                f"{a.min_proteins}, so no alignment column can ever reach the "
                f"threshold and nothing would be pooled. Pass --min-proteins "
                f"{len(members)} to use every column these proteins share, or add "
                "more family members.")
        fams[name] = members

    # Every input path is checked before any work starts. The expensive steps - fetching
    # the sequences, streaming ClinVar, streaming AlphaMissense - run in that order, so
    # a path typed wrong in the last argument used to surface as a traceback minutes in,
    # after the earlier reads had already been paid for. All of them are named at once.
    wanted = [("--table", a.table), ("--alphamissense", a.alphamissense),
              ("--mochi", a.mochi), ("--mochi-aln", a.mochi_aln)]
    wanted += [(f"--esm {spec}", spec.partition("=")[2]) for spec in a.esm]
    missing = [(flag, v) for flag, v in wanted if v and not Path(v).is_file()]
    if missing:
        raise SystemExit("file not found:\n" + "\n".join(
            f"        {flag}  {v}" for flag, v in missing))

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
    cv = read_clinvar(Path(a.table), set(seqs), include_uncertain=a.include_vus)
    print(f"    {int(cv.label.isin(['pathogenic', 'benign']).sum()):,} classified missense "
          f"variants: {(cv.label == 'pathogenic').sum():,} pathogenic, "
          f"{(cv.label == 'benign').sum():,} benign"
          + (f"; plus {(cv.label == 'uncertain').sum():,} of uncertain significance"
             if a.include_vus else ""), flush=True)

    am = None
    if a.alphamissense:
        print(f"  reading {Path(a.alphamissense).name} ...", flush=True)
        am = read_alphamissense(Path(a.alphamissense), set(queries.values()))
        print(f"    {len(am):,} AlphaMissense substitutions for "
              f"{am.uniprot.nunique()} accession(s)", flush=True)
    esm = {}
    for spec in a.esm:
        g, _, f = spec.partition("=")
        if not f:
            raise SystemExit(f"--esm {spec}: expected GENE=FILE")
        from .analyses.a15_esm_functional import read_esm
        e = read_esm(Path(f))
        if e is None:
            raise SystemExit(f"--esm {f}: need pos and mut (or a mutation column like "
                             "M1A) plus a score column (esm1v/llr/score/mean)")
        gene = g.strip().upper()
        # The table is merged on position, so a file for the wrong protein would line up
        # silently and score nonsense. Its wild-type residues are the check, and they
        # are worth making because an id column can say one thing while the numbers are
        # another: these files are routinely named and labelled by hand.
        if "wt" in e and gene in seqs:
            q = seqs[gene]
            inb = e[(e.pos >= 1) & (e.pos <= len(q))]
            agree = float((inb.wt.to_numpy() ==
                           np.array([q[p - 1] for p in inb.pos])).mean()) if len(inb) else 0.0
            print(f"    {len(e):,} ESM-1v scores for {gene} "
                  f"({e.attrs.get('source_column', '?')} column); wild-type residues "
                  f"agree with the sequence at {agree:.1%} of {len(inb):,} rows",
                  flush=True)
            if agree < 0.9:
                raise SystemExit(
                    f"--esm {f}: the wild-type residues match {gene} at only {agree:.1%}. "
                    "That table is for a different protein, a different isoform, or a "
                    "different numbering. Scoring it would merge on position and look "
                    "fine, so it is refused.")
            if e.pos.max() > len(q) or e.pos.nunique() < len(q):
                print(f"    note: covers {e.pos.nunique()} of {len(q)} positions "
                      f"(max {e.pos.max()})", flush=True)
        else:
            print(f"    {len(e):,} ESM-1v scores for {gene} (no wt column to check "
                  "against the sequence)", flush=True)
        esm[gene] = e

    mochi_scores = None
    if a.mochi:
        if not a.mochi_aln:
            raise SystemExit("--mochi needs --mochi-aln: Pos in the weights is a column "
                             "of the alignment the model was fitted on, and without it "
                             "there is nothing to map a human residue through")
        w = read_mochi(Path(a.mochi), a.mochi_min_folds)
        aln = read_alignment(Path(a.mochi_aln))
        width = len(next(iter(aln.values()))) if aln else 0
        print(f"\n  MoCHI: {w.attrs['n_weights']:,} weights over "
              f"{w.col.nunique()} columns (max {w.attrs['max_col']}); alignment is "
              f"{width} columns over {len(aln)} sequences; dropped "
              f"{w.attrs['n_dropped_low_folds']} weights fitted in fewer than "
              f"{a.mochi_min_folds} folds", flush=True)
        mochi_scores = mochi_variant_scores(w, aln, queries)
        if not len(mochi_scores):
            print("    none of the query accessions appear in the alignment headers - "
                  "no MoCHI score will be attached", flush=True)
        else:
            for gene, d in mochi_scores.groupby("gene"):
                print(f"    {gene}: {len(d):,} scored substitutions over "
                      f"{d.pos.nunique()} residues", flush=True)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    if not fams:
        summary["predictors"] = run_predictors_only(
            cv, outdir, am=am, esm=esm, acc=queries, mochi=mochi_scores)
        print(f"\n  -> {outdir}/")
        return 0
    for name, members in fams.items():
        summary[name] = run_one(name, members, seqs, cv, a.min_proteins, outdir,
                                am=am, esm=esm, acc=queries, mochi=mochi_scores)
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


def run_predictors_only(cv, outdir, am=None, esm=None, acc=None, mochi=None):
    """The predictor comparison without a family model.

    With no `--family` there is no alignment, no pooling and no mu, so none of the
    machinery that exists to turn yeast measurements into a per-column constraint runs.
    What is left is the comparison between predictors that score a human variant
    directly - AlphaMissense, ESM-1v, and a MoCHI energy model - on the ClinVar labels.

    That also removes the position-score ceiling this module usually warns about: every
    predictor here is per variant, so two substitutions at one residue can differ and
    the comparison is not bounded by how often pathogenic and benign variants occupy
    different positions.
    """
    import json
    known = cv[cv.label.isin(["benign", "pathogenic"])].copy()
    if not len(known):
        raise SystemExit("no classified missense variants to score")
    pred = predictor_table(known, am, esm or {}, acc or {}, mochi)
    keys = [k for k, *_ in PREDICTORS if k in pred and pred[k].notna().any()]
    if not keys:
        raise SystemExit("no predictor scored any variant: pass at least one of "
                         "--alphamissense, --esm or --mochi, or add --family")
    pr = score_predictors(pred, keys)
    pred.to_csv(outdir / "c03_predictors.csv", index=False)
    pr.to_csv(outdir / "c03_predictor_auroc.csv", index=False)
    gene = str(known.gene.iloc[0]) if known.gene.nunique() == 1 else ""
    figure_predictors(pred, pr, outdir, "predictors", gene)
    print(f"\n  {len(known):,} classified variants, no family model "
          f"({int((known.label == 'pathogenic').sum())} pathogenic, "
          f"{int((known.label == 'benign').sum())} benign)")
    print("  predictors, each on its own variants and on the shared set:")
    for k in keys:
        rr = pr[pr.key == k].set_index("scope")
        lab = next(l for kk, l, *_ in PREDICTORS if kk == k)
        bits = []
        for sc in ("own", "shared"):
            if sc in rr.index and pd.notna(rr.loc[sc, "auroc"]):
                bits.append(f"{sc} {rr.loc[sc, 'auroc']:.3f} "
                            f"[{rr.loc[sc, 'auroc_lo']:.3f}, {rr.loc[sc, 'auroc_hi']:.3f}] "
                            f"n={int(rr.loc[sc, 'n'])}")
        print(f"    {lab:<22} " + "   ".join(bits))
    print("    compare the shared column: the own column is a different set of "
          "variants for each predictor")
    res = {"mode": "predictors only", "n_known": int(len(known)),
           "predictors": pr.to_dict("records")}
    (outdir / "c03_predictors.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def run_one(name, members, seqs, cv, min_proteins, outdir, am=None, esm=None,
            acc=None, mochi=None):
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
    # one split, before any number is computed: a variant of uncertain significance is
    # scored and plotted, and never takes part in a comparison that has a right answer
    known = scored[scored.label.isin(["benign", "pathogenic"])]
    uncertain = place_uncertain(scored)
    if len(uncertain):
        uncertain.to_csv(outdir / f"{name}_uncertain.csv", index=False)
    res = evaluate(known)
    res["conservation_control"] = conservation_control(known)
    per = []
    for g, gg in known.groupby("gene"):
        r = evaluate(gg)
        r["gene"] = g
        per.append(r)
    by_gene = pd.DataFrame(per)
    by_gene.to_csv(outdir / f"{name}_by_gene.csv", index=False)
    at_pos = collapse_to_positions(known)
    if len(at_pos):
        res_pos = evaluate(at_pos)
        res["by_position"] = res_pos
        at_pos.to_csv(outdir / f"{name}_scored_positions.csv", index=False)
    ranked = rank_variants(known)
    if len(ranked):
        ranked.to_csv(outdir / f"{name}_variant_ranking.csv", index=False)
        res["threshold_mu"] = ranked.attrs.get("threshold_mu")
        res["youden_j"] = ranked.attrs.get("youden_j")
        res["outcomes"] = {k: int(v) for k, v in ranked.outcome.value_counts().items()}
    if "auroc" in res:
        figure(scored, res, by_gene, outdir, name)
        one = by_gene.dropna(subset=["auroc"]) if "auroc" in by_gene else by_gene
        figure_variants(ranked, outdir, name,
                        str(one.gene.iloc[0]) if len(one) == 1 else "")

    print()
    if "auroc" in res:
        print(f"  AUROC {res['auroc']:.3f}  [{res['auroc_lo']:.3f}, {res['auroc_hi']:.3f}]"
              f"   {res['n_pathogenic']:,} pathogenic vs {res['n_benign']:,} benign"
              f"   p = {res['p_mannwhitney']:.2g}")
        rp = res.get("by_position", {})
        if "auroc" in rp:
            print(f"  by position: AUROC {rp['auroc']:.3f}  "
                  f"[{rp['auroc_lo']:.3f}, {rp['auroc_hi']:.3f}]"
                  f"   {rp['n_pathogenic']} pathogenic vs {rp['n_benign']} benign positions"
                  f"   p = {rp['p_mannwhitney']:.2g}")
            print("    <- the honest unit: mu is a property of the position, so several "
                  "substitutions at one residue are one observation, not several")
        elif rp:
            print(f"  by position: not evaluated: {rp.get('reason')}")
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
            lv = cc.get("n_conservation_levels", 0)
            if lv and lv < 8:
                print(f"    CAUTION: conservation takes only {lv} distinct values over "
                      "these variants - it is one minus the entropy of a handful of "
                      "aligned residues, so it is a coarse baseline and beating it is a "
                      "weaker claim than beating a deep-alignment conservation score")
            if cc["auroc_abundance_given_conservation"] < 0.55:
                print("    NOTE: once conservation is removed the abundance signal is close "
                      "to chance, so this result is mostly conservation rediscovered")
        else:
            print(f"  conservation control: {cc.get('reason', 'not computed')}")
        if len(ranked):
            print(f"\n  per variant (best threshold μ = {ranked.attrs['threshold_mu']:+.3f}, "
                  f"Youden J = {ranked.attrs['youden_j']:.2f}):")
            for k, v in ranked.outcome.value_counts().items():
                print(f"    {k:<28} {v}")
            bad = ranked[ranked.outcome.str.startswith(("missed", "false alarm"))]
            if len(bad):
                print("    on the wrong side, worst first:")
                for r in bad.sort_values("pairs_beaten").itertuples():
                    print(f"      {r.wt}{int(r.pos)}{r.mut:<4} {r.label:<11} μ {r.mu:+.3f}"
                          f"  beats {r.pairs_beaten:.0%} of the other class"
                          + (f"  [{r.seg_type}]" if isinstance(getattr(r, "seg_type", None), str)
                             else ""))
            print(f"    full table: {name}_variant_ranking.csv")
        pred = predictor_table(known, am, esm or {}, acc or {}, mochi)
        keys = [k for k, *_ in PREDICTORS if k in pred and pred[k].notna().any()]
        if len(keys) > 1:
            pr = score_predictors(pred, keys)
            pred.to_csv(outdir / f"{name}_predictors.csv", index=False)
            pr.to_csv(outdir / f"{name}_predictor_auroc.csv", index=False)
            one = by_gene.dropna(subset=["auroc"]) if "auroc" in by_gene else by_gene
            figure_predictors(pred, pr, outdir, name,
                              str(one.gene.iloc[0]) if len(one) == 1 else "")
            res["predictors"] = pr.to_dict("records")
            print("\n  predictors, each on its own variants and on the shared set:")
            for k in keys:
                rr = pr[pr.key == k].set_index("scope")
                lab = next(l for kk, l, *_ in PREDICTORS if kk == k)
                bits = []
                for sc in ("own", "shared"):
                    if sc in rr.index and pd.notna(rr.loc[sc, "auroc"]):
                        bits.append(f"{sc} {rr.loc[sc, 'auroc']:.3f} "
                                    f"[{rr.loc[sc, 'auroc_lo']:.3f}, "
                                    f"{rr.loc[sc, 'auroc_hi']:.3f}] "
                                    f"n={int(rr.loc[sc, 'n'])}")
                print(f"    {lab:<22} " + "   ".join(bits))
            print("    compare the shared column: the own column is a different set of "
                  "variants for each predictor")
        if len(uncertain):
            ben_min = float(known.loc[known.label == "benign", "mu"].min())
            below = int((uncertain.mu < ben_min).sum())
            res["uncertain"] = {
                "n": int(len(uncertain)),
                "n_below_every_benign": below,
                "n_with_pathogenic_set": int((uncertain.sits_with ==
                                              "with the pathogenic set").sum()),
                "n_beyond_every_pathogenic": int((uncertain.sits_with ==
                                                  "beyond every pathogenic").sum()),
                "median_mu": float(uncertain.mu.median())}
            print(f"\n  {len(uncertain):,} variants of uncertain significance, scored but "
                  "never counted in anything above:")
            for k, v in uncertain.sits_with.value_counts().items():
                print(f"    {k:<26} {v}")
            print(f"    {below} sit below every benign variant (μ < {ben_min:+.3f}); "
                  f"median μ {uncertain.mu.median():+.3f} against "
                  f"{known.loc[known.label == 'benign', 'mu'].median():+.3f} benign and "
                  f"{known.loc[known.label == 'pathogenic', 'mu'].median():+.3f} pathogenic")
            if below:
                print("    most constrained, worth a second look:")
                for r in uncertain.head(min(below, 10)).itertuples():
                    print(f"      {r.wt}{int(r.pos)}{r.mut:<4} μ {r.mu:+.3f}"
                          f"  more constrained than {r.beats_benign:.0%} of benign and "
                          f"{r.beats_pathogenic:.0%} of pathogenic"
                          + (f"  [{r.seg_type}]"
                             if isinstance(getattr(r, "seg_type", None), str) else ""))
            print("    these are hypotheses about positions, not calls on variants")
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
