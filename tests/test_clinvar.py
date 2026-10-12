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
    # uncertain is now a label of its own: it is scored and plotted, never compared
    ("Uncertain significance", "uncertain"),
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


def _with_cons(n=150, link=1.0, seed=0):
    """Pathogenic variants sit at constrained columns; `link` sets how much of that
    constraint is just conservation."""
    r = np.random.default_rng(seed)
    cons = r.random(2 * n)
    base = np.concatenate([r.normal(-0.6, 0.3, n), r.normal(0.0, 0.3, n)])
    mu = -(link * (cons - 0.5) * 1.6) + (1 - link) * base + r.normal(0, 0.05, 2 * n)
    return pd.DataFrame({"gene": "G", "pos": np.arange(2 * n), "col": np.arange(2 * n),
                         "label": ["pathogenic"] * n + ["benign"] * n,
                         "mu": mu, "conservation": cons})


def test_conservation_control_credits_abundance_when_it_is_independent():
    from mpdms.clinvar import conservation_control
    d = _with_cons(link=0.0, seed=1)
    c = conservation_control(d)
    assert c["auroc_abundance"] > 0.75
    assert c["auroc_abundance_given_conservation"] > 0.7      # survives the control
    assert abs(c["corr_abundance_conservation"]) < 0.2


def test_conservation_control_exposes_a_signal_that_is_only_conservation():
    """The failure this control exists to catch."""
    from mpdms.clinvar import conservation_control
    r = np.random.default_rng(2)
    n = 150
    cons = np.concatenate([r.normal(0.8, 0.12, n), r.normal(0.35, 0.12, n)]).clip(0, 1)
    d = pd.DataFrame({"gene": "G", "pos": np.arange(2 * n), "col": np.arange(2 * n),
                      "label": ["pathogenic"] * n + ["benign"] * n,
                      "mu": -cons * 1.5 + r.normal(0, 0.02, 2 * n), "conservation": cons})
    c = conservation_control(d)
    assert c["auroc_abundance"] > 0.9                         # looks excellent
    assert c["auroc_conservation"] > 0.9                      # so does conservation
    assert abs(c["corr_abundance_conservation"]) > 0.9
    assert c["auroc_abundance_given_conservation"] < 0.6      # and nothing is left


def test_conservation_control_refuses_too_little_data():
    from mpdms.clinvar import conservation_control
    assert "reason" in conservation_control(_with_cons(n=4))


def test_column_conservation_is_one_when_every_member_agrees():
    from mpdms.clinvar import column_conservation
    aln = pd.DataFrame({"col": [1, 2, 3],
                        "A_aa": ["M", "K", "-"], "B_aa": ["M", "R", "L"],
                        "C_aa": ["M", "W", "-"]})
    c = column_conservation(aln, ["A", "B", "C"]).set_index("col")
    assert c.loc[1, "conservation"] == pytest.approx(1.0)     # all M
    assert c.loc[2, "conservation"] == pytest.approx(0.0)     # all different
    assert np.isnan(c.loc[3, "conservation"])                 # only one residue present


def _panel_scored(genes, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in genes:
        for k in range(40):
            path = k % 2 == 0
            rows.append({"gene": g, "pos": k + 1, "wt": "A", "mut": "V",
                         "label": "pathogenic" if path else "benign",
                         "col": k + 1, "wt_matches_uniprot": True,
                         "mu": rng.normal(-0.6 if path else -0.1, 0.25),
                         "tau": 0.2, "J": 4, "seg_type": "TM"})
    return pd.DataFrame(rows)


def test_one_query_gene_drops_the_per_gene_panel(tmp_path):
    """With a single gene that panel is panel (b)'s number redrawn as one bar."""
    import matplotlib
    matplotlib.use("Agg")
    from mpdms import plotting as P
    from mpdms.clinvar import evaluate, figure
    P.use_style("default")
    for genes, want in ((["SLC2A10"], 2), (["SLC2A10", "SLC2A1"], 3)):
        s = _panel_scored(genes)
        res = evaluate(s)
        by_gene = pd.DataFrame([{**evaluate(gg), "gene": g} for g, gg in s.groupby("gene")])
        figure(s, res, by_gene, tmp_path, "hexose")
        import matplotlib.image as mpimg
        png = tmp_path / "figures" / "c01_hexose_clinvar.png"
        png = png if png.exists() else tmp_path / "c01_hexose_clinvar.png"
        assert png.exists()
        # the two-panel figure is narrower than the three-panel one, which is the
        # observable consequence of the panel being dropped
        w = mpimg.imread(png).shape[1]
        if want == 2:
            narrow = w
        else:
            assert w > narrow


def test_the_per_variant_contributions_average_to_the_auroc():
    """The identity that makes the table trustworthy: AUROC is the mean of these."""
    from mpdms.clinvar import evaluate, rank_variants
    s = _panel_scored(["SLC2A10"], seed=3)
    r = rank_variants(s)
    auc = evaluate(s)["auroc"]
    for lab in ("pathogenic", "benign"):
        assert r.loc[r.label == lab, "pairs_beaten"].mean() == pytest.approx(auc, abs=1e-9)
    assert r.pairs_beaten.between(0, 1).all()
    assert r.constraint_rank.iloc[0] == 1
    assert r.mu.is_monotonic_increasing          # rank 1 is the most constrained


def test_the_variants_on_the_wrong_side_are_the_ones_named():
    from mpdms.clinvar import rank_variants
    s = _panel_scored(["SLC2A10"], seed=5)
    # plant one obvious miss: a pathogenic variant at a completely unconstrained column
    s.loc[s.index[0], ["label", "mu"]] = ["pathogenic", 0.9]
    r = rank_variants(s)
    worst = r.sort_values("pairs_beaten").iloc[0]
    assert worst.mu == pytest.approx(0.9)
    assert worst.outcome.startswith("missed")
    assert worst.pairs_beaten == pytest.approx(0.0)
    assert set(r.outcome) <= {"true positive", "true negative",
                              "missed (false negative)", "false alarm (false positive)"}
    assert r.attrs["youden_j"] > 0


def test_one_sided_input_returns_an_empty_table_rather_than_raising():
    from mpdms.clinvar import rank_variants
    s = _panel_scored(["SLC2A10"])
    assert not len(rank_variants(s[s.label == "benign"]))


def test_several_substitutions_at_one_residue_are_one_observation():
    """mu is a position score, so a gene whose variants cluster is scored on fewer units."""
    from mpdms.clinvar import collapse_to_positions, evaluate
    rows = []
    # three pathogenic substitutions at one well-predicted arginine, plus singletons
    for mut in "WPL":
        rows.append({"gene": "G", "pos": 132, "wt": "R", "mut": mut,
                     "label": "pathogenic", "mu": -0.12})
    for k, mu in enumerate([-0.20, -0.18, -0.15, -0.10, 0.03]):
        rows.append({"gene": "G", "pos": 200 + k, "wt": "A", "mut": "V",
                     "label": "pathogenic", "mu": mu})
    for k, mu in enumerate([-0.05, -0.02, 0.00, 0.01, 0.02, 0.04]):
        rows.append({"gene": "G", "pos": 300 + k, "wt": "A", "mut": "V",
                     "label": "benign", "mu": mu})
    s = pd.DataFrame(rows)
    at_pos = collapse_to_positions(s)
    assert len(at_pos) == 12 and (at_pos.label == "pathogenic").sum() == 6
    assert int(at_pos.loc[at_pos.pos == 132, "n_variants"].iloc[0]) == 3
    # the triple-counted residue inflates the variant-level number
    assert evaluate(s)["auroc"] > evaluate(at_pos)["auroc"]


def test_a_position_called_both_ways_is_dropped_rather_than_guessed():
    from mpdms.clinvar import collapse_to_positions
    s = pd.DataFrame([{"gene": "G", "pos": 7, "wt": "A", "mut": "V",
                       "label": "pathogenic", "mu": -0.2},
                      {"gene": "G", "pos": 7, "wt": "A", "mut": "T",
                       "label": "benign", "mu": -0.2},
                      {"gene": "G", "pos": 9, "wt": "A", "mut": "V",
                       "label": "benign", "mu": 0.0}])
    out = collapse_to_positions(s)
    assert list(out.pos) == [9]
    assert out.attrs["n_mixed_positions"] == 1


def test_a_coarse_conservation_baseline_is_counted():
    """Conservation from four aligned residues is nearly binary; the run must say so."""
    from mpdms.clinvar import conservation_control
    rng = np.random.default_rng(0)
    n = 60
    s = pd.DataFrame({"label": ["pathogenic"] * (n // 2) + ["benign"] * (n // 2),
                      "mu": np.concatenate([rng.normal(-0.3, 0.1, n // 2),
                                            rng.normal(0.0, 0.1, n // 2)]),
                      "conservation": rng.choice([0.42, 0.59, 1.0], n)})
    assert conservation_control(s)["n_conservation_levels"] == 3


def test_uncertain_variants_are_read_only_when_asked_for(tmp_path):
    from mpdms.clinvar import read_clinvar
    f = tmp_path / "cv.txt"
    f.write_text("GeneSymbol\tName\tClinicalSignificance\n"
                 "SLC2A10\tNM_1(SLC2A10):c.1A>G (p.Gly23Asp)\tPathogenic\n"
                 "SLC2A10\tNM_1(SLC2A10):c.2A>G (p.Ala106Ser)\tBenign\n"
                 "SLC2A10\tNM_1(SLC2A10):c.3A>G (p.Ser81Arg)\tUncertain significance\n"
                 "SLC2A10\tNM_1(SLC2A10):c.4A>G (p.Leu300Trp)\t"
                 "Conflicting interpretations of pathogenicity\n")
    assert set(read_clinvar(f, {"SLC2A10"}).label) == {"pathogenic", "benign"}
    both = read_clinvar(f, {"SLC2A10"}, include_uncertain=True)
    assert set(both.label) == {"pathogenic", "benign", "uncertain"}
    # conflicting is never kept: two submitters disagreeing is not the same as nobody calling
    assert 300 not in set(both.pos)


def test_a_vus_is_scored_but_never_counted_in_the_comparison():
    """The guard the whole design rests on: a VUS has no right answer to be flattered by."""
    from mpdms.clinvar import (collapse_to_positions, conservation_control, evaluate,
                               place_uncertain, rank_variants)
    s = _panel_scored(["SLC2A10"], seed=11)
    s["conservation"] = 0.6
    plain = {"auroc": evaluate(s)["auroc"],
             "cc": conservation_control(s)["auroc_abundance"],
             "n_ranked": len(rank_variants(s)),
             "n_pos": len(collapse_to_positions(s))}
    with_vus = pd.concat([s, s.head(12).assign(label="uncertain", pos=range(900, 912))],
                         ignore_index=True)
    known = with_vus[with_vus.label.isin(["benign", "pathogenic"])]
    assert evaluate(known)["auroc"] == pytest.approx(plain["auroc"])
    assert conservation_control(known)["auroc_abundance"] == pytest.approx(plain["cc"])
    assert len(rank_variants(known)) == plain["n_ranked"]
    assert len(collapse_to_positions(known)) == plain["n_pos"]
    placed = place_uncertain(with_vus)
    assert len(placed) == 12
    assert set(placed.label if "label" in placed else []) <= {"uncertain"}


def test_where_a_vus_falls_is_described_not_decided():
    from mpdms.clinvar import place_uncertain
    rows = [{"gene": "G", "pos": p, "wt": "A", "mut": "V", "label": "benign", "mu": m}
            for p, m in enumerate([-0.05, -0.02, 0.00, 0.02], start=1)]
    rows += [{"gene": "G", "pos": p, "wt": "A", "mut": "V", "label": "pathogenic", "mu": m}
             for p, m in enumerate([-0.20, -0.15, -0.12, -0.10], start=10)]
    rows += [{"gene": "G", "pos": p, "wt": "A", "mut": "V", "label": "uncertain", "mu": m}
             for p, m in enumerate([-0.40, -0.13, -0.06, 0.03], start=20)]
    out = place_uncertain(pd.DataFrame(rows)).set_index("pos")
    assert out.loc[20, "sits_with"] == "beyond every pathogenic"
    assert out.loc[21, "sits_with"] == "with the pathogenic set"
    assert out.loc[23, "sits_with"] == "with the benign set"
    assert out.loc[22, "sits_with"] == "between the two"
    assert out.loc[20, "beats_benign"] == 1.0 and out.loc[23, "beats_pathogenic"] == 0.0
    assert out.mu.is_monotonic_increasing


def test_the_figure_gains_a_panel_only_when_there_are_vus(tmp_path):
    import matplotlib.image as mpimg

    from mpdms import plotting as P
    from mpdms.clinvar import evaluate, figure
    P.use_style("default")
    s = _panel_scored(["SLC2A10"], seed=2)
    known = s[s.label.isin(["benign", "pathogenic"])]
    by_gene = pd.DataFrame([{**evaluate(known), "gene": "SLC2A10"}])
    figure(s, evaluate(known), by_gene, tmp_path, "a")
    plain = mpimg.imread(tmp_path / "a_clinvar.png"
                         if (tmp_path / "a_clinvar.png").exists()
                         else tmp_path / "c01_a_clinvar.png").shape[1]
    with_vus = pd.concat([s, s.head(8).assign(label="uncertain", pos=range(900, 908))],
                         ignore_index=True)
    figure(with_vus, evaluate(known), by_gene, tmp_path, "b")
    wide = mpimg.imread(tmp_path / "c01_b_clinvar.png").shape[1]
    assert wide > plain


def _four_predictors(n=80, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    return pd.DataFrame({
        "gene": "SLC2A10", "uniprot": "O95528",
        "pos": np.arange(n), "wt": "A", "mut": "V",
        "label": ["pathogenic"] * half + ["benign"] * half,
        # each predictor points a DIFFERENT way round on purpose
        "mu": np.r_[rng.normal(-0.30, 0.10, half), rng.normal(0.00, 0.10, half)],
        "esm1v": np.r_[rng.normal(-8.0, 1.0, half), rng.normal(-4.0, 1.0, half)],
        "am_pathogenicity": np.r_[rng.normal(0.80, 0.12, half),
                                  rng.normal(0.20, 0.12, half)].clip(0, 1),
        "conservation": np.r_[rng.normal(0.85, 0.12, half), rng.normal(0.55, 0.12, half)]})


def test_every_predictor_is_turned_the_right_way_round():
    """The one mistake that is silently catastrophic: a flipped sign gives 1 - AUROC."""
    from mpdms.clinvar import PREDICTORS, score_predictors
    d = _four_predictors()
    keys = [k for k, *_ in PREDICTORS]
    r = score_predictors(d, keys).query("scope == 'own'").set_index("key")
    for k in keys:
        assert r.loc[k, "auroc"] > 0.9, (k, r.loc[k, "auroc"])
    # and flipping one really does invert it, so the test above has teeth
    flipped = score_predictors(d.assign(am_pathogenicity=-d.am_pathogenicity),
                               keys).query("scope == 'own'").set_index("key")
    assert flipped.loc["am_pathogenicity", "auroc"] < 0.1


def test_the_shared_scope_uses_one_set_of_variants_for_every_predictor():
    from mpdms.clinvar import PREDICTORS, score_predictors
    d = _four_predictors(seed=2)
    d.loc[d.index[:10], "am_pathogenicity"] = np.nan      # AlphaMissense misses ten
    d.loc[d.index[-6:], "mu"] = np.nan                    # the family model misses six
    r = score_predictors(d, [k for k, *_ in PREDICTORS])
    shared = r[r.scope == "shared"]
    assert shared.n.nunique() == 1 and int(shared.n.iloc[0]) == 64
    own = r[r.scope == "own"].set_index("key")
    assert int(own.loc["am_pathogenicity", "n"]) == 70
    assert int(own.loc["esm1v", "n"]) == 80


def test_alphamissense_is_read_by_column_name_and_filtered_by_accession(tmp_path):
    from mpdms.clinvar import read_alphamissense
    f = tmp_path / "am.tsv"
    f.write_text("# AlphaMissense\n"
                 "am_pathogenicity\tprotein_variant\tuniprot_id\tam_class\n"
                 "0.91\tG23D\tO95528\tpathogenic\n"
                 "0.08\tA106S\tO95528\tbenign\n"
                 "0.50\tQ5R\tP11166\tambiguous\n"
                 "0.70\tnonsense\tO95528\tx\n")
    out = read_alphamissense(f, {"O95528"})
    assert list(out.columns) == ["uniprot", "wt", "pos", "mut", "am_pathogenicity"]
    assert len(out) == 2                      # the other accession and the junk row go
    assert set(out.pos) == {23, 106}
    assert out.loc[out.pos == 23, "am_pathogenicity"].iloc[0] == pytest.approx(0.91)
    assert out.wt.tolist() == ["G", "A"] and out.mut.tolist() == ["D", "S"]


def test_the_predictor_figure_draws_a_violin_per_predictor_plus_one_roc(tmp_path):
    from mpdms import plotting as P
    from mpdms.clinvar import PREDICTORS, figure_predictors, score_predictors
    P.use_style("default")
    d = _four_predictors(seed=5)
    r = score_predictors(d, [k for k, *_ in PREDICTORS])
    figure_predictors(d, r, tmp_path, "hexose", "SLC2A10")
    png = tmp_path / "c03_hexose_predictors.png"
    assert png.exists()
    cap = (tmp_path / "c03_hexose_predictors_caption.txt")
    if cap.exists():                       # paper style writes panel text to a caption
        assert "AlphaMissense" in cap.read_text()
