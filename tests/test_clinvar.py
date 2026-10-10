"""ClinVar parsing and the pathogenic-vs-benign test. The filtering decides the result."""
import gzip

import numpy as np
import pandas as pd
import pytest

from mpdms.clinvar import (classify_significance, evaluate, parse_protein_change,
                           read_clinvar)


@pytest.mark.parametrize("text,expect", [
    ("Pathogenic", "pathogenic"),
    ("Likely pathogenic", "pathogenic"),
    ("Pathogenic/Likely pathogenic", "pathogenic"),
    ("Benign", "benign"),
    ("Likely benign", "benign"),
    ("Benign/Likely benign", "benign"),
    ("Uncertain significance", None),
    ("Conflicting interpretations of pathogenicity", None),
    ("Conflicting classifications of pathogenicity", None),
    ("not provided", None),
    ("", None),
    ("drug response", None),
])
def test_clinical_significance_is_read_conservatively(text, expect):
    assert classify_significance(text) == expect


def test_a_variant_called_both_ways_is_not_counted_as_either():
    """Counting an uncertain or conflicting call as benign is how a predictor is flattered."""
    assert classify_significance("Pathogenic; Benign") is None
    assert classify_significance("Benign, Pathogenic") is None


@pytest.mark.parametrize("name,expect", [
    ("NM_006516.4(SLC2A1):c.277C>T (p.Arg93Trp)", ("R", 93, "W")),
    ("p.Gly1Ala", ("G", 1, "A")),
    ("NM_1(X):c.1A>T (p.Met1Leu)", ("M", 1, "L")),
])
def test_protein_changes_are_parsed(name, expect):
    assert parse_protein_change(name) == expect


@pytest.mark.parametrize("name", [
    "NM_1(X):c.1A>T (p.Arg93=)",          # synonymous
    "NM_1(X):c.1A>T (p.Arg93Ter)",        # nonsense: Ter is not an amino acid here
    "NM_1(X):c.1A>T (p.Arg93Arg)",        # no change
    "NM_1(X):c.277C>T",                   # no protein consequence
    "",
])
def test_non_missense_changes_are_rejected(name):
    assert parse_protein_change(name) is None


def _table(tmp_path, rows, gz=True, sig_col="ClinicalSignificance"):
    t = pd.DataFrame(rows).rename(columns={"ClinicalSignificance": sig_col})
    p = tmp_path / ("cv.txt.gz" if gz else "cv.txt")
    t.to_csv(p, sep="\t", index=False, compression="gzip" if gz else None)
    return p


ROWS = [
    {"GeneSymbol": "SLC2A1", "Name": "NM_1(SLC2A1):c.1A>T (p.Arg93Trp)",
     "ClinicalSignificance": "Pathogenic"},
    {"GeneSymbol": "SLC2A1", "Name": "NM_1(SLC2A1):c.1A>T (p.Arg93Trp)",
     "ClinicalSignificance": "Pathogenic"},                      # the GRCh37/38 duplicate
    {"GeneSymbol": "SLC2A1", "Name": "NM_1(SLC2A1):c.2A>T (p.Ala10Val)",
     "ClinicalSignificance": "Benign"},
    {"GeneSymbol": "SLC2A1", "Name": "NM_1(SLC2A1):c.3A>T (p.Leu20Pro)",
     "ClinicalSignificance": "Uncertain significance"},
    {"GeneSymbol": "BRCA1", "Name": "NM_1(BRCA1):c.1A>T (p.Cys61Gly)",
     "ClinicalSignificance": "Pathogenic"},
]


def test_read_clinvar_keeps_only_the_genes_asked_for(tmp_path):
    cv = read_clinvar(_table(tmp_path, ROWS), {"SLC2A1"})
    assert set(cv.gene) == {"SLC2A1"}


def test_read_clinvar_drops_uncertain_and_deduplicates(tmp_path):
    cv = read_clinvar(_table(tmp_path, ROWS), {"SLC2A1"})
    assert len(cv) == 2                                   # the duplicate and the VUS are gone
    assert set(cv.label) == {"pathogenic", "benign"}
    assert cv[cv.label == "pathogenic"].iloc[0].pos == 93


def test_read_clinvar_reads_an_uncompressed_file_and_a_renamed_column(tmp_path):
    cv = read_clinvar(_table(tmp_path, ROWS, gz=False, sig_col="Germline classification"),
                      {"SLC2A1"})
    assert len(cv) == 2


def test_read_clinvar_says_what_is_missing(tmp_path):
    p = tmp_path / "bad.txt"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(p, sep="\t", index=False)
    with pytest.raises(SystemExit, match="gene column"):
        read_clinvar(p, {"SLC2A1"})


def test_read_clinvar_reports_an_unknown_gene(tmp_path):
    with pytest.raises(SystemExit, match="no rows"):
        read_clinvar(_table(tmp_path, ROWS), {"NOTAGENE"})


# ------------------------------------------------------------------ scoring
def _scored(sep=0.6, n=120, seed=0):
    """Pathogenic variants sit at more constrained columns by `sep`."""
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "gene": "G", "pos": np.arange(2 * n), "label": ["pathogenic"] * n + ["benign"] * n,
        "mu": np.concatenate([r.normal(-sep, 0.4, n), r.normal(0.0, 0.4, n)])})


def test_more_constrained_positions_score_as_pathogenic():
    """The direction matters: a negative mu means the yeast family says the site matters."""
    r = evaluate(_scored())
    assert r["auroc"] > 0.8
    assert r["auroc_lo"] > 0.5
    assert r["median_pathogenic"] < r["median_benign"]
    assert r["p_mannwhitney"] < 1e-6


def test_no_separation_gives_an_auroc_interval_covering_chance():
    r = evaluate(_scored(sep=0.0, seed=7))
    assert r["auroc_lo"] < 0.5 < r["auroc_hi"]
    assert r["p_mannwhitney"] > 0.01


def test_an_inverted_signal_is_reported_as_worse_than_chance():
    r = evaluate(_scored(sep=-0.6))
    assert r["auroc"] < 0.3


def test_too_few_of_either_class_is_refused_rather_than_reported():
    d = _scored()
    few = pd.concat([d[d.label == "pathogenic"].head(3), d[d.label == "benign"]])
    r = evaluate(few)
    assert "auroc" not in r and "at least 5" in r["reason"]


def test_the_variants_per_position_ratio_is_reported():
    """A position score cannot separate two variants at one residue; the ratio says how
    often that happens, which bounds what the AUROC can mean."""
    d = _scored()
    d.loc[:, "pos"] = d.pos // 2                 # two variants per position
    r = evaluate(d)
    assert r["variants_per_position"] == pytest.approx(2.0, abs=0.01)
