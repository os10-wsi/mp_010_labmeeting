"""The structure-informed family prior, and the comparison against ESM-1v.

The claims worth testing are the ones the figures rest on: that a prior with finite width
is what lets structure enter at all, that the regression recovers a planted effect, that
ESM is put on the right scale before it is compared, and that nothing the held-out protein
knows leaks into the model that scores it.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms.bayes import hierarchical_normal
from mpdms.famstruct import (blend, calibrate_esm, compare_models, fit_prior,
                             hierarchical_prior_mean, pool_with_prior, predict_held_out)


def test_a_wide_prior_is_the_flat_prior():
    """The proper prior must contain the model in `bayes` as its limit."""
    y, s = [0.4, -0.2, 0.1, 0.0], [0.1, 0.15, 0.1, 0.2]
    a = hierarchical_normal(y, s, seed=0)
    b = hierarchical_prior_mean(y, s, m=5.0, k=1e4, seed=0)
    assert b["mu_mean"] == pytest.approx(a["mu_mean"], abs=1e-3)
    assert b["mu_sd"] == pytest.approx(a["mu_sd"], rel=0.05)
    assert b["tau_median"] == pytest.approx(a["tau_median"], rel=0.1, abs=0.01)


def test_centring_on_the_prior_mean_under_a_flat_prior_does_nothing():
    """Why the proper prior is needed: shifting the data and shifting back is a no-op."""
    y, s, m = np.array([0.4, -0.2, 0.1]), np.array([0.1, 0.15, 0.1]), -0.3
    shifted = hierarchical_normal(y - m, s, seed=1)
    plain = hierarchical_normal(y, s, seed=1)
    assert shifted["mu_mean"] + m == pytest.approx(plain["mu_mean"], abs=1e-9)


def test_a_tight_prior_pulls_the_posterior_toward_the_structural_mean():
    y, s = [0.0, 0.1, -0.1], [0.3, 0.3, 0.3]
    flat = hierarchical_normal(y, s, seed=2)["mu_mean"]
    tight = hierarchical_prior_mean(y, s, m=-1.0, k=0.05, seed=2)["mu_mean"]
    loose = hierarchical_prior_mean(y, s, m=-1.0, k=1.0, seed=2)["mu_mean"]
    assert tight < loose < flat
    assert tight == pytest.approx(-1.0, abs=0.15)
    # and a tighter prior cannot make the answer less certain
    assert (hierarchical_prior_mean(y, s, -1.0, 0.05, seed=2)["mu_sd"]
            <= hierarchical_prior_mean(y, s, -1.0, 1.0, seed=2)["mu_sd"])


def _family(n_cols=120, n_prot=4, beta=-0.8, noise=0.25, seed=0):
    """A family whose column means really do depend on one structural feature."""
    rng = np.random.default_rng(seed)
    rsa = rng.uniform(0, 1, n_cols)
    truth = 0.2 + beta * rsa * -1.0          # buried (low rsa) = more negative
    rows = []
    for j in range(n_prot):
        for c in range(n_cols):
            s = 0.12
            rows.append({"col": c, "protein": f"P{j}", "pos": c + 1,
                         "y": rng.normal(truth[c], noise) + rng.normal(0, s),
                         "s": s, "n": 20, "seg_type": "TM" if rsa[c] < 0.4 else "loop"})
    feats = pd.DataFrame({"col": np.arange(n_cols), "rsa": rsa,
                          "abs_z": rng.normal(0, 1, n_cols),
                          "radial": rng.normal(0, 1, n_cols)})
    return pd.DataFrame(rows), feats, truth


def test_the_prior_regression_finds_a_planted_structural_effect():
    long, feats, _ = _family()
    fn, coefs = fit_prior(long, feats, ["rsa", "abs_z", "radial"])
    c = coefs.set_index("covariate")
    assert c.loc["rsa", "p"] < 1e-6
    assert c.loc["rsa", "coef"] > 0            # higher rsa, less negative effect
    assert c.loc["abs_z", "p"] > 0.01 and c.loc["radial", "p"] > 0.01
    assert fn.sigma > 0
    m = fn(feats.set_index("col"))
    assert m.notna().all() and len(m) == len(feats)


def test_a_feature_that_predicts_nothing_leaves_the_prior_wide():
    """The prior width must come from the data, so a useless regression is harmless."""
    long, feats, _ = _family(beta=0.0)
    fn, _ = fit_prior(long, feats, ["rsa"])
    spread = long.groupby("col").y.mean().std()
    assert fn.sigma == pytest.approx(spread, rel=0.25)


def test_structure_helps_when_it_is_real_and_does_not_hurt_when_it_is_not():
    real, feats, _ = _family(beta=-0.8, seed=3)
    null, nfeats, _ = _family(beta=0.0, seed=4)
    for long, f, expect in ((real, feats, True), (null, nfeats, False)):
        sc, _ = compare_models(long, f, ["rsa"])
        p = sc.pivot_table(index="held_out", columns="model", values="elpd_gain")
        d = float((p["structure"] - p["intercept only"]).mean())
        assert d > 0.02 if expect else d > -0.05


def test_the_held_out_protein_never_enters_its_own_prior():
    """Refitting without the held-out protein is the whole basis of the comparison."""
    long, feats, _ = _family(n_prot=3, seed=5)
    seen = []
    import mpdms.famstruct as F
    orig = F.fit_prior

    def spy(rest, f, covars):
        seen.append(set(rest.protein.unique()))
        return orig(rest, f, covars)

    F.fit_prior = spy
    try:
        compare_models(long, feats, ["rsa"])
    finally:
        F.fit_prior = orig
    for held, s in zip(sorted(long.protein.unique()), seen):
        assert held not in s


def test_esm_is_calibrated_to_the_assay_before_it_is_compared():
    rng = np.random.default_rng(0)
    esm = rng.normal(-6, 3, 400)
    y = -0.4 + 0.15 * esm + rng.normal(0, 0.2, 400)
    f, sd = calibrate_esm(pd.DataFrame({"esm": esm, "y": y}))
    assert f(np.array([-6.0]))[0] == pytest.approx(-0.4 + 0.15 * -6, abs=0.06)
    assert sd == pytest.approx(0.2, rel=0.2)
    assert calibrate_esm(pd.DataFrame({"esm": esm[:5], "y": y[:5]})) is None


def test_blending_two_predictions_is_never_less_certain_than_either():
    d = pd.DataFrame({"pred": [0.0, 1.0], "mu_sd": [0.2, 0.2], "tau": [0.3, 0.3],
                      "esm_pred": [1.0, np.nan], "esm_sd": [0.4, 0.4]})
    b = blend(d)
    fam_sd = np.sqrt(0.2 ** 2 + 0.3 ** 2)
    assert b.blend_sd[0] < min(fam_sd, 0.4)
    assert 0.0 < b.pred_blend[0] < 1.0            # between the two, nearer the tighter one
    assert b.pred_blend[0] < 0.5
    # a column with no ESM score falls back to the family answer untouched
    assert b.pred_blend[1] == 1.0 and b.blend_sd[1] == pytest.approx(fam_sd)


def test_the_four_models_are_all_scored_when_esm_is_available():
    long, feats, _ = _family(n_prot=3, seed=6)
    rng = np.random.default_rng(7)
    truth = long.groupby("col").y.mean()
    esm = pd.concat([pd.DataFrame({"col": truth.index, "protein": p,
                                   "esm": -6 + 4 * truth.to_numpy()
                                   + rng.normal(0, 1.0, len(truth))})
                     for p in long.protein.unique()], ignore_index=True)
    sc, preds = compare_models(long, feats, ["rsa"], esm)
    assert set(sc.model) == {"intercept only", "structure", "ESM-1v", "structure + ESM-1v"}
    assert sc.spearman.min() > 0
    assert set(preds.model) == set(sc.model)


def test_pooling_with_a_prior_returns_one_row_per_column_with_the_prior_recorded():
    long, feats, _ = _family(n_cols=40, seed=8)
    fn, _ = fit_prior(long, feats, ["rsa"])
    out = pool_with_prior(long, fn(feats.set_index("col")), fn.sigma, min_proteins=2)
    assert len(out) == 40
    assert out.prior_mean.notna().all()
    assert (out.pred_sd >= out.mu_sd - 1e-9).all()
    assert out.mu.between(out.mu_lo, out.mu_hi).mean() > 0.95


def test_predictions_fall_back_to_the_flat_model_without_a_prior():
    long, feats, _ = _family(n_cols=30, n_prot=3, seed=9)
    a = predict_held_out(long, "P0", None)
    assert not a.used_prior.any() and len(a) == 30
    fn, _ = fit_prior(long[long.protein != "P0"], feats, ["rsa"])
    b = predict_held_out(long, "P0", fn(feats.set_index("col")), fn.sigma)
    assert b.used_prior.all()
    assert not np.allclose(a.pred, b.pred)


def test_esm_uncertainty_is_measured_out_of_sample():
    """Within-protein correlated error makes the in-sample residual spread optimistic.

    Each protein gets its own offset, which a fit that has seen it absorbs and a fit that
    has not cannot. If the blend is handed the in-sample number it overweights ESM.
    """
    rng = np.random.default_rng(0)
    rows = []
    for j, off in enumerate([-0.5, 0.0, 0.5, 0.25]):
        e = rng.normal(-6, 3, 200)
        rows.append(pd.DataFrame({"protein": f"P{j}", "esm": e,
                                  "y": off + 0.15 * e + rng.normal(0, 0.1, 200)}))
    d = pd.concat(rows, ignore_index=True)
    _, sd = calibrate_esm(d)
    b, a = np.polyfit(d.esm, d.y, 1)
    in_sample = float(np.std(d.y - (a + b * d.esm), ddof=2))
    assert sd > in_sample
    # with one protein there is nothing to hold out, and the in-sample value stands
    one = d[d.protein == "P0"]
    _, sd1 = calibrate_esm(one)
    assert sd1 == pytest.approx(0.1, rel=0.3)


def test_a_wider_esm_spread_pulls_the_blend_back_toward_the_family():
    d = pd.DataFrame({"pred": [0.0], "mu_sd": [0.2], "tau": [0.3],
                      "esm_pred": [1.0], "esm_sd": [0.4]})
    near = blend(d).pred_blend[0]
    far = blend(d.assign(esm_sd=[1.2])).pred_blend[0]
    assert far < near                              # less trust in ESM, closer to 0
