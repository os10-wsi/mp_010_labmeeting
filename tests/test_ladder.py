"""Forward selection scored out of sample: the ladder must stop, and must not cheat."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms.ladder import cv_score, equation, final_model, forward_ladder


def _data(n_fam=4, n_per=400, seed=0, noise=0.5):
    """Only `signal` and `helper` matter; `copy` duplicates signal; the rest are noise."""
    r = np.random.default_rng(seed)
    rows = []
    for f in range(n_fam):
        for i in range(n_per):
            sig, helper = r.normal(), r.normal()
            rows.append({
                "dataset_id": f"P{f}", "family": f"F{f}", "pos": i % 60,
                "signal": sig, "copy": sig + r.normal(0, 0.01), "helper": helper,
                "noise1": r.normal(), "noise2": r.normal(), "noise3": r.normal(),
                "score_z": 1.0 * sig + 0.6 * helper + r.normal(0, noise)})
    return pd.DataFrame(rows)


COLS = ["signal", "copy", "helper", "noise1", "noise2", "noise3"]


def test_the_ladder_picks_the_real_predictors_first():
    lad, chosen = forward_ladder(_data(), COLS, max_steps=4, min_gain=0.005)
    assert chosen[0] in ("signal", "copy")
    assert "helper" in chosen[:2]


def test_a_duplicate_feature_is_not_taken_twice():
    """`copy` is `signal` again; adding it buys nothing, so the ladder must stop."""
    lad, chosen = forward_ladder(_data(), COLS, max_steps=5, min_gain=0.005)
    assert not {"signal", "copy"} <= set(chosen)


def test_pure_noise_is_not_added():
    lad, chosen = forward_ladder(_data(), COLS, max_steps=6, min_gain=0.005)
    assert not {"noise1", "noise2", "noise3"} & set(chosen)


def test_the_ladder_stops_rather_than_running_to_max_steps():
    lad, chosen = forward_ladder(_data(), COLS, max_steps=6, min_gain=0.01)
    assert len(chosen) < 6
    assert len(lad) >= len(chosen)          # the step that failed is recorded, not kept


def test_min_gain_controls_how_many_features_are_kept():
    _, strict = forward_ladder(_data(), COLS, max_steps=6, min_gain=0.20)
    _, loose = forward_ladder(_data(), COLS, max_steps=6, min_gain=0.0)
    assert len(strict) <= len(loose)


def test_scoring_never_sees_the_held_out_family():
    """If it did, a feature that is pure noise shared within a family would look useful."""
    d = _data()
    d["family_id_leak"] = d.family.str[1:].astype(float)   # perfectly predicts the fold
    s = cv_score(d, ["family_id_leak"], split="family")
    assert abs(s["mean"]) < 0.2          # constant within each held-out fold: no ranking


def test_cv_score_reports_every_fold():
    s = cv_score(_data(), ["signal"], split="family")
    assert set(s["per_fold"]) == {"F0", "F1", "F2", "F3"}
    assert s["min"] <= s["mean"]


def test_final_model_reports_per_protein_coefficients_and_their_spread():
    d = _data()
    c = final_model(d, ["signal", "helper"])
    assert set(c.feature) == {"signal", "helper"}
    assert (c.pooled > 0).all()                      # both were built with positive effect
    assert "per_protein_sd" in c and "sign_agreement" in c
    assert (c.sign_agreement == 1.0).all()           # every protein agrees here


def test_sign_agreement_catches_a_coefficient_that_flips():
    d = _data()
    flip = d.dataset_id.isin(["P0", "P1"])
    d.loc[flip, "score_z"] = d.loc[flip, "score_z"] - 2.0 * d.loc[flip, "helper"]
    c = final_model(d, ["signal", "helper"]).set_index("feature")
    assert c.loc["helper", "sign_agreement"] < 0.75
    assert c.loc["signal", "sign_agreement"] == 1.0


def test_equation_lists_the_chosen_features_in_order():
    d = _data()
    c = final_model(d, ["signal", "helper"])
    eq = equation(c, ["signal", "helper"])
    assert eq.startswith("score_z ≈") and "z(signal)" in eq and "z(helper)" in eq
    assert eq.index("z(signal)") < eq.index("z(helper)")
