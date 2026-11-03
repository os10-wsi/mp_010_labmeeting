"""Transfer of a yeast abundance model onto a human protein measured two ways.

The planted truth is the one the module claims to detect: a predictor that knows
abundance and nothing else. It must rank abundance above function, and it must report
no correlation with the residual, because the residual is the part of function that
abundance was regressed out of. A module that reported a residual correlation here
would be finding structure in noise, so that is asserted as an absence.
"""
import numpy as np
import pandas as pd
import pytest

from mpdms import transfer as T


def planted(tmp_path, n=300, seed=0, leak=0.0):
    """mu predicts abundance; function = abundance + an independent component.

    `leak` lets the predictor see a fraction of that independent component, which is
    the failure the residual test exists to catch.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n)                      # the abundance the family knows
    extra = rng.normal(size=n)                      # function beyond abundance
    abundance = truth + 0.3 * rng.normal(size=n)
    function = abundance + extra
    mu = truth + leak * extra + 0.3 * rng.normal(size=n)
    sites = pd.DataFrame({
        "pos": np.arange(1, n + 1), "wt": ["A"] * n,
        "median_abundance": abundance, "median_function": function,
        "median_z": extra, "functional": extra < np.quantile(extra, 0.05),
    })
    m = tmp_path / "d02_sites.csv"
    sites.to_csv(m, index=False)
    pred = pd.DataFrame({"pos": np.arange(1, n + 1), "wt": ["A"] * n, "mu": mu})
    p = tmp_path / "FAM_SLC22A1_residues.csv"
    pred.to_csv(p, index=False)
    return m, p


def test_abundance_model_ranks_abundance_above_function_and_misses_the_residual(tmp_path):
    m, p = planted(tmp_path)
    res = T.run(m, {"PF07690": p}, {}, "SLC22A1", tmp_path / "out")
    v = res["verdict"]["PF07690"]
    assert v["abundance_beats_function"], v
    assert v["residual_interval_covers_zero"], v
    assert abs(v["rho_residual"]) < 0.15
    assert v["rho_abundance"] > 0.7
    assert (tmp_path / "out" / "t01_predictor_against_measured.png").exists()
    assert (tmp_path / "out" / "t02_correlation_summary.png").exists()
    assert (tmp_path / "out" / "t02_correlations.csv").exists()


def test_a_predictor_that_does_see_the_residual_is_reported_as_such(tmp_path):
    """The same machinery with the leak open: the absence must stop being an absence."""
    m, p = planted(tmp_path, leak=1.0)
    res = T.run(m, {"PF07690": p}, {}, "SLC22A1", tmp_path / "out")
    v = res["verdict"]["PF07690"]
    assert not v["residual_interval_covers_zero"]
    assert v["rho_residual"] > 0.3


def test_correlations_are_over_positions_not_variants(tmp_path):
    """n is the number of mapped positions, which is what the interval is built on."""
    m, p = planted(tmp_path, n=120)
    res = T.run(m, {"PF07690": p}, {}, "SLC22A1", tmp_path / "out")
    assert {r["n"] for r in res["correlations"]} == {120}


def test_unmapped_positions_do_not_enter_the_correlation(tmp_path):
    m, p = planted(tmp_path, n=200)
    d = pd.read_csv(p)
    d.loc[d.pos > 80, "mu"] = np.nan
    d.to_csv(p, index=False)
    res = T.run(m, {"PF07690": p}, {}, "SLC22A1", tmp_path / "out")
    assert {r["n"] for r in res["correlations"]} == {80}


def test_too_few_mapped_positions_is_reported_not_guessed(tmp_path):
    m, p = planted(tmp_path, n=200)
    d = pd.read_csv(p)
    d.loc[d.pos > T.MIN_N - 1, "mu"] = np.nan
    d.to_csv(p, index=False)
    res = T.run(m, {"PF07690": p}, {}, "SLC22A1", tmp_path / "out")
    assert all(not np.isfinite(r["rho"]) for r in res["correlations"])
    assert res["verdict"]["PF07690"]["rho_abundance"] is None


def test_donor_file_accepts_the_alignment_map_column_names(tmp_path):
    m, _ = planted(tmp_path, n=150)
    donor = tmp_path / "aqr1.csv"
    pd.DataFrame({"slc22a1_pos": np.arange(1, 151),
                  "site_median": np.linspace(-1, 1, 150)}).to_csv(donor, index=False)
    res = T.run(m, {}, {"AQR1": donor}, "SLC22A1", tmp_path / "out")
    assert "AQR1" in res["verdict"]


def test_wrong_input_tables_fail_loudly(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"position": [1, 2], "value": [0.1, 0.2]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        T.read_measured(bad)
    with pytest.raises(SystemExit):
        T.read_prediction(bad)
    with pytest.raises(SystemExit):
        T.read_donor(bad)


def test_no_predictor_at_all_is_an_error(tmp_path):
    m, _ = planted(tmp_path, n=60)
    with pytest.raises(SystemExit):
        T.run(m, {}, {}, "SLC22A1", tmp_path / "out")


def test_bootstrap_interval_brackets_the_point_estimate(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = x + rng.normal(size=200)
    s = T.boot_spearman(x, y)
    assert s["lo"] < s["rho"] < s["hi"]
    assert s["n"] == 200
