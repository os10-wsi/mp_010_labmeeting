"""Per-residue predictions for an unmeasured protein, from the family posterior."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms.predict import clinvar_crosstab, predict_residues, summarise


def _cols(n=10, mu=-0.9, pred_sd=0.2, seg="TM"):
    return pd.DataFrame({"col": np.arange(1, n + 1), "mu": mu, "mu_lo": mu - 0.2,
                         "mu_hi": mu + 0.2, "tau": 0.1, "pred_sd": pred_sd,
                         "J": 4, "seg_type": seg})


def test_a_confidently_constrained_column_is_called_a_loss():
    d = predict_residues("A" * 10, {i: i for i in range(1, 11)}, _cols(), threshold=-0.5)
    assert (d.call == "loss").all()
    assert (d.p_loss > 0.95).all()


def test_a_confidently_tolerant_column_is_called_tolerated():
    d = predict_residues("A" * 10, {i: i for i in range(1, 11)},
                         _cols(mu=0.0, pred_sd=0.1), threshold=-0.5)
    assert (d.call == "tolerated").all()


def test_a_column_the_family_disagrees_about_is_called_uncertain():
    """A wide predictive distribution must not produce a confident call either way."""
    d = predict_residues("A" * 10, {i: i for i in range(1, 11)},
                         _cols(mu=-0.5, pred_sd=3.0), threshold=-0.5)
    assert (d.call == "uncertain").all()
    assert np.allclose(d.p_loss, 0.5, atol=0.1)


def test_unmapped_residues_are_marked_not_guessed():
    d = predict_residues("A" * 10, {1: 1, 2: 2}, _cols(), threshold=-0.5)
    assert (d.call == "unmapped").sum() == 8
    assert d.loc[d.call == "unmapped", "p_loss"].isna().all()
    assert int(d.mapped.sum()) == 2


def test_the_threshold_moves_the_calls():
    m = {i: i for i in range(1, 11)}
    strict = predict_residues("A" * 10, m, _cols(mu=-0.6, pred_sd=0.2), threshold=-1.5)
    loose = predict_residues("A" * 10, m, _cols(mu=-0.6, pred_sd=0.2), threshold=-0.2)
    assert (strict.call == "loss").sum() < (loose.call == "loss").sum()


def test_the_calling_probability_moves_the_calls():
    m = {i: i for i in range(1, 11)}
    d_strict = predict_residues("A" * 10, m, _cols(mu=-0.7, pred_sd=0.3),
                                threshold=-0.5, call_at=0.99)
    d_loose = predict_residues("A" * 10, m, _cols(mu=-0.7, pred_sd=0.3),
                               threshold=-0.5, call_at=0.60)
    assert (d_strict.call == "loss").sum() <= (d_loose.call == "loss").sum()


def test_summary_counts_add_up_and_expected_losses_is_not_a_count():
    cols = pd.concat([_cols(5, mu=-1.2, pred_sd=0.2), _cols(5, mu=0.0, pred_sd=0.2)
                     .assign(col=range(6, 11), seg_type="loop")], ignore_index=True)
    d = predict_residues("A" * 12, {i: i for i in range(1, 11)}, cols, threshold=-0.5)
    s = summarise(d, "G", -0.5)
    assert s["n_residues"] == 12 and s["n_mapped"] == 10
    assert s["n_loss"] + s["n_tolerated"] + s["n_uncertain"] == 10
    assert s["n_loss_TM"] == 5 and s["n_loss_loop"] == 0
    # summing probabilities is the expected number, and need not equal the hard count
    assert s["expected_losses"] == pytest.approx(d.p_loss.sum())


def test_clinvar_crosstab_finds_an_enrichment_and_survives_an_empty_cell():
    d = predict_residues("A" * 20, {i: i for i in range(1, 21)},
                         pd.concat([_cols(10, mu=-1.2), _cols(10, mu=0.0)
                                    .assign(col=range(11, 21))], ignore_index=True),
                         threshold=-0.5)
    cv = pd.DataFrame({"gene": "G", "pos": list(range(1, 11)) + list(range(11, 21)),
                       "wt": "A", "mut": "V",
                       "label": ["pathogenic"] * 10 + ["benign"] * 10})
    c = clinvar_crosstab(d, cv, "G")
    assert c["n_pathogenic_at_loss"] == 10 and c["n_benign_at_loss"] == 0
    assert c["fisher_p"] < 1e-4
    assert not np.isfinite(c["odds_ratio"])
    assert np.isfinite(c["odds_ratio_haldane"]) and c["odds_ratio_haldane"] > 10


def test_clinvar_crosstab_says_when_nothing_lands():
    d = predict_residues("A" * 5, {}, _cols(), threshold=-0.5)
    cv = pd.DataFrame({"gene": "G", "pos": [1], "wt": "A", "mut": "V", "label": ["benign"]})
    assert "reason" in clinvar_crosstab(d, cv, "G")
