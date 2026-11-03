"""Transfer from a measured family to a protein measured separately."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms import transfer as T


def test_read_scores_takes_medians_per_position_and_drops_deletions(tmp_path):
    f = tmp_path / "v.csv"
    pd.DataFrame({
        "pos": [10, 10, 10, 11, 11, 12],
        "kind": ["missense"] * 5 + ["deletion"],
        "abundance": [-1.0, 0.0, 1.0, -0.5, -0.1, -9.0],
        "function": [-2.0, -1.0, 0.0, 0.2, 0.4, -9.0],
    }).to_csv(f, index=False)
    g = T.read_scores(f, ["abundance", "function"]).set_index("pos")
    assert g.loc[10, "abundance"] == 0.0            # median of -1, 0, 1
    assert g.loc[11, "function"] == pytest.approx(0.3)
    assert 12 not in g.index                         # the deletion is not a substitution
    assert g.loc[10, "n_variants"] == 3


def test_read_scores_names_the_column_it_could_not_find(tmp_path):
    f = tmp_path / "v.csv"
    pd.DataFrame({"pos": [1], "abundance": [0.0]}).to_csv(f, index=False)
    with pytest.raises(SystemExit, match="function"):
        T.read_scores(f, ["abundance", "function"])
    g = tmp_path / "noPos.csv"
    pd.DataFrame({"residue": [1], "abundance": [0.0]}).to_csv(g, index=False)
    with pytest.raises(SystemExit, match="pos"):
        T.read_scores(g, ["abundance"])


def test_boot_spearman_recovers_a_planted_correlation_and_brackets_it():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = 0.9 * x + rng.normal(0, 0.5, 200)
    r = T.boot_spearman(x, y, n_boot=300)
    assert r["rho"] > 0.7
    assert r["lo"] <= r["rho"] <= r["hi"]
    assert r["n"] == 200


def test_boot_spearman_refuses_too_few_aligned_sites():
    """Below a couple of dozen positions a correlation here is not worth drawing."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, T.MIN_SITES - 1)
    assert not np.isfinite(T.boot_spearman(x, x)["rho"])


def _aln(seqs: dict) -> pd.DataFrame:
    """A gapless alignment table of equal-length sequences, in the shape family uses."""
    n = len(next(iter(seqs.values())))
    out = {"col": np.arange(1, n + 1)}
    for g, s in seqs.items():
        out[f"{g}_pos"] = np.arange(1, n + 1, dtype=float)
        out[f"{g}_aa"] = list(s)
    return pd.DataFrame(out)


def test_pair_identity_counts_only_columns_where_both_have_a_residue():
    a = _aln({"A": "ACDEFGHIKL", "B": "ACDEFGHIKL"})
    a.loc[a.col > 7, "B_pos"] = np.nan              # B gapped at the last three
    r = T.pair_identity(a, "A", "B")
    assert r["aligned"] == 7 and r["identity"] == pytest.approx(1.0)


def test_x01_flags_a_query_no_closer_to_the_family_than_chance(tmp_path):
    """A twilight-zone query has to be called out, or x02 reads as biology."""
    fam = "ACDEFGHIKLMNPQRSTVWY" * 3
    near = fam[:55] + "AAAAA"                        # nearly identical to the family
    far = "WWWWWWWWWWWWWWWWWWWW" * 3                 # shares almost nothing
    a = _aln({"M1": fam, "M2": fam, "NEAR": near, "FAR": far})
    out = T.x01_alignment(a, ["M1", "M2"], ["NEAR", "FAR"], tmp_path)
    assert out["status"] == "ok"
    assert "FAR" in out["below_twilight"]
    assert "NEAR" not in out["below_twilight"]
    assert out["best_identity_to_family"]["NEAR"] > T.TWILIGHT
    assert (tmp_path / "x01_alignment_identity.csv").exists()


def test_x02_finds_a_planted_relationship_between_family_mu_and_the_query(tmp_path):
    rng = np.random.default_rng(2)
    n = 120
    cols = pd.DataFrame({"col": np.arange(1, n + 1), "mu": rng.normal(0, 1, n),
                         "mu_sd": 0.2, "tau": 0.3, "J": 3})
    maps = {"Q": {p: p for p in range(1, n + 1)}}
    site = pd.DataFrame({"pos": np.arange(1, n + 1),
                         "abundance": 0.8 * cols.mu + rng.normal(0, 0.4, n),
                         "function": rng.normal(0, 1, n)})       # unrelated on purpose
    out = T.x02_family_vs_query(cols, maps, {"Q": site},
                                ["abundance", "function"], tmp_path)
    t = pd.DataFrame(out["by_readout"]).set_index("score")
    assert t.loc["abundance", "rho"] > 0.6
    assert abs(t.loc["function", "rho"]) < 0.25
    assert t.loc["abundance", "lo"] > 0              # the interval excludes nothing-doing
    assert (tmp_path / "x02_family_vs_query.csv").exists()


def test_x02_says_so_rather_than_drawing_nothing_when_no_residue_maps(tmp_path):
    cols = pd.DataFrame({"col": [1, 2, 3], "mu": [0.0, 0.1, 0.2], "mu_sd": 0.1,
                         "tau": 0.1, "J": 3})
    site = pd.DataFrame({"pos": [1, 2, 3], "abundance": [0.0, 0.1, 0.2]})
    out = T.x02_family_vs_query(cols, {}, {"Q": site}, ["abundance"], tmp_path)
    assert out["status"] == "skipped"


def test_x03_refuses_a_member_that_is_not_in_the_family(tmp_path):
    long = pd.DataFrame({"protein": ["A"] * 3, "col": [1, 2, 3], "y": [0.0, 1.0, 2.0],
                         "s": [0.1, 0.1, 0.1]})
    site = pd.DataFrame({"pos": [1, 2, 3], "abundance": [0.0, 1.0, 2.0]})
    out = T.x03_member_vs_query(long, {"Q": {1: 1, 2: 2, 3: 3}}, {"Q": site},
                                ["abundance"], "NOPE", tmp_path)
    assert out["status"] == "skipped" and "NOPE" in out["reason"]
