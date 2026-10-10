"""Single-feature predictive power: the simplest thing to get wrong is the interval."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms.univariate import _order, pooled_table, spearman_ci, table


def _data(n_pos=120, per_pos=19, rho_strength=1.0, seed=0):
    r = np.random.default_rng(seed)
    rows = []
    for p in range(n_pos):
        site = r.normal()                       # a position effect shared by its variants
        for _ in range(per_pos):
            x = r.normal()
            rows.append({"dataset_id": "D1", "pos": p, "good": rho_strength * x + 0.3 * site,
                         "useless": r.normal(), "site_only": site,
                         "score_z": rho_strength * x + site + r.normal(0, 0.3)})
    return pd.DataFrame(rows)


def test_a_predictive_feature_is_found():
    d = _data()
    good = spearman_ci(d.good, d.score_z, d.pos, n_boot=200)
    assert good["rho"] > 0.5 and good["lo"] > 0


def test_the_interval_is_calibrated_on_a_feature_that_predicts_nothing():
    """A 95% interval is allowed to miss sometimes; across seeds it must mostly cover zero,
    and the correlation must stay small. Asserting one seed tests the seed, not the method."""
    covers, rhos = 0, []
    for seed in range(12):
        d = _data(seed=100 + seed)
        r = spearman_ci(d.useless, d.score_z, d.pos, n_boot=200, seed=seed)
        rhos.append(abs(r["rho"]))
        covers += int(r["lo"] <= 0 <= r["hi"])
    assert covers >= 10                      # ~95% coverage, allowing for 12 draws
    assert max(rhos) < 0.15 and np.mean(rhos) < 0.06


def test_the_interval_is_wider_than_a_variant_level_one_would_be():
    """Nineteen substitutions at a site are not nineteen independent measurements."""
    d = _data()
    clustered = spearman_ci(d.site_only, d.score_z, d.pos, n_boot=300)
    # resampling variants instead of positions: each row becomes its own cluster
    naive = spearman_ci(d.site_only, d.score_z, np.arange(len(d)), n_boot=300)
    assert (clustered["hi"] - clustered["lo"]) > 2 * (naive["hi"] - naive["lo"])


def test_interval_covers_the_point_estimate():
    d = _data()
    r = spearman_ci(d.good, d.score_z, d.pos, n_boot=200)
    assert r["lo"] <= r["rho"] <= r["hi"]


def test_too_little_data_returns_nan_rather_than_a_confident_number():
    d = _data(n_pos=2, per_pos=5)
    r = spearman_ci(d.good, d.score_z, d.pos, n_boot=50)
    assert np.isnan(r["rho"]) and r["n"] < 100


def test_a_constant_feature_predicts_nothing():
    d = _data()
    d["flat"] = 1.0
    r = spearman_ci(d.flat, d.score_z, d.pos, n_boot=50)
    assert np.isnan(r["rho"])


def test_table_covers_every_protein_and_feature():
    d = pd.concat([_data(seed=1).assign(dataset_id="D1"),
                   _data(seed=2).assign(dataset_id="D2")], ignore_index=True)
    t = table(d, ["good", "useless"], n_boot=60)
    assert set(t.dataset_id) == {"D1", "D2"}
    assert set(t.feature) == {"good", "useless"}
    assert len(t) == 4


def test_features_are_ranked_by_mean_absolute_correlation():
    t = pd.DataFrame({"feature": ["a", "b", "a", "b"], "rho": [0.8, -0.1, 0.7, 0.05]})
    assert _order(t) == ["a", "b"]           # sign must not decide the ranking


def test_ranking_uses_the_absolute_value_so_a_negative_feature_is_not_buried():
    t = pd.DataFrame({"feature": ["neg", "pos"], "rho": [-0.9, 0.2]})
    assert _order(t)[0] == "neg"


def test_pooled_table_reports_both_the_pooled_and_the_meta_estimate():
    d = pd.concat([_data(seed=3).assign(dataset_id="D1"),
                   _data(seed=4).assign(dataset_id="D2"),
                   _data(seed=5).assign(dataset_id="D3")], ignore_index=True)
    t = table(d, ["good"], n_boot=60)
    p = pooled_table(d, t, ["good"], n_boot=60)
    row = p.iloc[0]
    assert row.rho > 0.5
    assert row.meta_rho == pytest.approx(row.rho, abs=0.15)   # same data, two routes
    assert row.n_proteins == 3 and row.k == 3
    assert -1 <= row.meta_lo <= row.meta_rho <= row.meta_hi <= 1
