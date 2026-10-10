"""Hierarchical pooling across a family. The core is checked against published posteriors."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms.bayes import (column_frame, coverage, hierarchical_normal,
                         log_predictive_density, pool_columns, predict_held_out,
                         score_predictions)

# Rubin (1981) eight schools, as tabulated in Gelman, BDA3 tables 5.2 and 5.3
EIGHT_Y = [28, 8, -3, 7, -1, 1, 18, 12]
EIGHT_S = [15, 10, 16, 11, 9, 11, 10, 18]
BDA_THETA = [11.4, 7.9, 6.0, 7.7, 5.1, 6.0, 10.4, 8.3]
BDA_THETA_SD = [8.4, 6.3, 7.7, 6.6, 6.1, 6.7, 6.8, 7.9]


def test_eight_schools_posterior_matches_the_textbook():
    r = hierarchical_normal(EIGHT_Y, EIGHT_S, seed=0)
    assert r["mu_mean"] == pytest.approx(7.9, abs=0.6)
    assert r["mu_sd"] == pytest.approx(5.1, abs=0.8)
    assert r["tau_median"] == pytest.approx(5.0, abs=1.5)
    assert np.allclose(r["theta_mean"], BDA_THETA, atol=0.5)
    assert np.allclose(r["theta_sd"], BDA_THETA_SD, atol=0.7)


def test_shrinkage_pulls_every_estimate_toward_the_family_mean():
    r = hierarchical_normal(EIGHT_Y, EIGHT_S, seed=0)
    y, th, mu = np.array(EIGHT_Y, float), r["theta_mean"], r["mu_mean"]
    assert np.all(np.abs(th - mu) <= np.abs(y - mu) + 1e-9)
    assert 0 < r["shrinkage"] < 1


def test_a_noisier_protein_is_shrunk_harder():
    """The point of pooling: a weak measurement borrows more from its relatives."""
    r = hierarchical_normal([2.0, 2.0], [0.2, 3.0], seed=0)
    pulled = np.abs(np.array([2.0, 2.0]) - r["mu_mean"]) - np.abs(r["theta_mean"] - r["mu_mean"])
    assert r["theta_sd"][0] < r["theta_sd"][1]
    assert pulled[1] >= pulled[0] - 1e-9


def test_agreeing_proteins_give_a_small_tau_and_a_tight_prediction():
    agree = hierarchical_normal([1.0, 1.02, 0.98, 1.01], [0.05] * 4, seed=1)
    differ = hierarchical_normal([0.0, 1.0, 2.0, 3.0], [0.05] * 4, seed=1)
    assert agree["tau_median"] < 0.2 < differ["tau_median"]
    assert agree["pred_sd"] < differ["pred_sd"]
    # the prediction for an unmeasured member is wider than the family mean, always
    for r in (agree, differ):
        assert r["pred_sd"] >= r["mu_sd"] - 1e-9


def test_prediction_for_a_new_member_is_wider_than_the_mean_it_centres_on():
    """The next protein scatters around mu by tau as well; var(pred) = var(mu) + E[tau^2]."""
    r = hierarchical_normal([0.0, 1.0, 2.0], [0.1] * 3, seed=2)
    assert r["pred_mean"] == pytest.approx(r["mu_mean"], abs=0.05)
    assert r["pred_sd"] > r["mu_sd"]
    assert r["pred_sd"] > r["tau_median"]
    assert r["pred_sd"] ** 2 == pytest.approx(r["mu_sd"] ** 2 + r["tau_median"] ** 2, rel=0.35)


def test_one_protein_cannot_be_pooled_and_says_so():
    r = hierarchical_normal([1.0], [0.3])
    assert r["J"] == 1 and "no pooling" in r["note"]
    assert r["mu_mean"] == 1.0 and r["pred_sd"] == 0.3


def test_empty_input_is_not_a_crash():
    assert hierarchical_normal([], [])["J"] == 0
    assert hierarchical_normal([1.0, np.nan], [0.1, 0.1])["J"] == 1
    assert hierarchical_normal([1.0, 2.0], [0.1, 0.0])["J"] == 1      # a zero SE is dropped


def test_p_positive_tracks_the_evidence():
    assert hierarchical_normal([2.0, 2.1, 1.9], [0.1] * 3, seed=3)["p_positive"] > 0.99
    assert hierarchical_normal([-2.0, -2.1], [0.1] * 2, seed=3)["p_positive"] < 0.01
    mixed = hierarchical_normal([-1.0, 1.0], [0.5] * 2, seed=3)["p_positive"]
    assert 0.2 < mixed < 0.8


def test_log_predictive_density_prefers_the_closer_prediction():
    near = log_predictive_density([1.0], [0.1], 1.0, 0.2, 0.2)
    far = log_predictive_density([1.0], [0.1], 3.0, 0.2, 0.2)
    assert near[0] > far[0]


def test_a_noisily_measured_protein_is_not_punished_for_being_noisy():
    """Its own error widens the predictive distribution, so the density does not collapse."""
    clean = log_predictive_density([2.0], [0.05], 1.0, 0.1, 0.1)[0]
    noisy = log_predictive_density([2.0], [2.00], 1.0, 0.1, 0.1)[0]
    assert noisy > clean


def test_coverage_is_calibrated_when_the_model_is_right():
    rng = np.random.default_rng(0)
    mu, tau, s = 0.4, 0.5, 0.2
    y = rng.normal(mu, np.sqrt(tau ** 2 + s ** 2), 4000)
    assert coverage(y, np.full(4000, s), mu, 0.0, tau, 0.90) == pytest.approx(0.90, abs=0.02)
    assert coverage(y, np.full(4000, s), mu, 0.0, tau, 0.50) == pytest.approx(0.50, abs=0.03)


def test_coverage_detects_overconfidence():
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 2000)
    assert coverage(y, np.full(2000, 0.01), 0.0, 0.0, 0.2, 0.90) < 0.6


# ----------------------------------------------------------------- pipeline
def _long(n_col=60, seed=0, outsider=False):
    """Four proteins measuring the same alignment columns; the last may disagree."""
    r = np.random.default_rng(seed)
    truth = r.normal(0, 0.6, n_col)
    rows = []
    for p in ("P1", "P2", "P3", "P4"):
        t = -truth if (outsider and p == "P4") else truth
        for c in range(n_col):
            rows.append({"col": c + 1, "protein": p, "pos": c + 1,
                         "y": t[c] + r.normal(0, 0.1), "s": 0.1, "n": 19})
    return pd.DataFrame(rows)


def test_pooling_recovers_the_shared_signal():
    long = _long()
    cols = pool_columns(long, min_proteins=4)
    assert len(cols) == 60
    assert cols.tau.median() < 0.2                   # the proteins agree, so tau is small


def test_held_out_prediction_is_good_when_the_family_agrees():
    long = _long(seed=2)
    pr = predict_held_out(long, "P4")
    sc = score_predictions(pr, float(long.y.mean()))
    assert sc["spearman"] > 0.9
    assert sc["elpd_gain"] > 0
    assert sc["coverage_90"] > 0.7


def test_held_out_prediction_collapses_for_a_member_that_disagrees():
    """The number that matters: the family must not claim to predict what it cannot.

    P1-P3 share a signal and P4 inverts it. Holding out P4 means predicting it from three
    proteins that agree with each other and not with it, which must come out negative.
    """
    long = _long(seed=3, outsider=True)
    good = score_predictions(predict_held_out(long, "P2"), float(long.y.mean()))
    bad = score_predictions(predict_held_out(long, "P4"), float(long.y.mean()))
    assert good["spearman"] > 0.5
    assert bad["spearman"] < -0.5
    assert bad["elpd_gain"] < good["elpd_gain"]


def test_column_frame_maps_positions_through_the_alignment():
    aln = pd.DataFrame({"col": [1, 2, 3], "A_pos": [1, 2, np.nan], "B_pos": [1, np.nan, 2]})
    est = {"A": pd.DataFrame({"pos": [1, 2], "y": [0.1, 0.2], "s": [.1, .1], "n": [19, 19]}),
           "B": pd.DataFrame({"pos": [1, 2], "y": [0.3, 0.4], "s": [.1, .1], "n": [19, 19]})}
    long = column_frame(aln, est)
    assert len(long) == 4
    assert set(long[long.col == 1].protein) == {"A", "B"}
    assert set(long[long.col == 2].protein) == {"A"}         # B has a gap there
    assert long[(long.col == 3) & (long.protein == "B")].y.iloc[0] == 0.4


def test_a_column_with_too_few_proteins_is_not_pooled():
    long = _long(n_col=5)
    long = long[~((long.col == 1) & (long.protein != "P1"))]
    cols = pool_columns(long, min_proteins=4)
    assert 1 not in set(cols.col) and 2 in set(cols.col)
