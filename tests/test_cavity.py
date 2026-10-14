"""Cavity constraint: the test, the spatial correction, and the effect size."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms import cavity as V


def test_spatial_blocks_are_about_the_length_scale_they_claim():
    """Blocks that count patches are only honest if they are patch-sized."""
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 100, (600, 3))          # a 100 A box
    lab = V.spatial_blocks(xyz, target_a=20.0)
    assert len(np.unique(lab)) > 4
    spans = []
    for g in np.unique(lab):
        c = xyz[lab == g]
        if len(c) > 2:
            spans.append(float(np.linalg.norm(c.max(0) - c.min(0))))
    # not residue-sized and not the whole protein: within a factor of about two
    assert 10.0 < float(np.median(spans)) < 55.0
    # a handful of points cannot be blocked and must not raise
    assert len(V.spatial_blocks(xyz[:3])) == 3


def test_cliffs_delta_sign_says_which_group_is_more_constrained():
    low, high = np.full(40, -2.0), np.full(40, +2.0)
    assert V.cliffs_delta(low, high) == pytest.approx(-1.0)
    assert V.cliffs_delta(high, low) == pytest.approx(+1.0)
    assert V.cliffs_delta(low, low) == pytest.approx(0.0)
    assert np.isnan(V.cliffs_delta([], low))


def _sites(n_block=12, per_block=25, cavity_effect=0.0, block_sd=0.0, seed=1):
    """Residues laid along a line in blocks, with an optional real cavity effect and an
    optional spatially correlated nuisance shared within each block."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_block):
        shared = rng.normal(0, block_sd)
        for j in range(per_block):
            x = b * 22.0 + j * 0.8
            # the cavity is a contiguous run of blocks, as a real one would be
            cav = 1.0 if b < n_block // 3 else 0.0
            rows.append({"pos": len(rows) + 1, "x": x, "y": 0.0, "z": 0.0,
                         "radial": 4.0 if cav else 12.0,
                         "is_cavity": cav, "is_tm_f": 1.0,
                         "environ": "cavity-lining" if cav else "lipid-facing",
                         "median_z": shared + cavity_effect * cav + rng.normal(0, 0.35)})
    t = pd.DataFrame(rows)
    t["block"] = V.spatial_blocks(t[["x", "y", "z"]].to_numpy(float), target_a=20.0)
    return t


def test_a_planted_cavity_effect_is_found():
    t = _sites(cavity_effect=-0.8)
    r = V.one_test(t, pd.Series(True, index=t.index), "all", "T")
    assert r["difference"] == pytest.approx(-0.8, abs=0.2)
    assert r["p_block_clustered"] < 0.05
    assert r["cliffs_delta"] < -0.4          # cavity residuals sit lower
    assert r["delta_lo"] < r["cliffs_delta"] < r["delta_hi"]


def test_nothing_is_found_when_there_is_nothing():
    t = _sites(cavity_effect=0.0)
    r = V.one_test(t, pd.Series(True, index=t.index), "all", "T")
    assert r["p_block_clustered"] > 0.05
    assert abs(r["cliffs_delta"]) < 0.2


def test_the_block_clustered_error_is_wider_than_the_naive_one():
    """The correction has to cost something, or it is not doing anything.

    With a nuisance shared inside each spatial patch and no real cavity effect, the
    naive test sees hundreds of independent residues and calls the difference real.
    Clustering on patches counts the dozen patches that actually vary.
    """
    t = _sites(cavity_effect=0.0, block_sd=0.9, seed=5)
    r = V.one_test(t, pd.Series(True, index=t.index), "all", "T")
    assert r["se_block_clustered"] > 2 * r["se_naive"]
    assert r["p_naive_ols"] < 0.05           # the mistake the correction prevents
    assert r["p_block_clustered"] > r["p_naive_ols"]


def test_the_radial_trend_recovers_a_gradient_with_no_threshold_in_it():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(400):
        rad = rng.uniform(2, 20)
        rows.append({"pos": i + 1, "x": rad * 3, "y": float(i % 7), "z": 0.0,
                     "radial": rad, "is_tm_f": 1.0, "is_cavity": float(rad < 9),
                     "environ": "cavity-lining" if rad < 9 else "lipid-facing",
                     "median_z": -1.2 + 0.08 * rad + rng.normal(0, 0.3)})
    t = pd.DataFrame(rows)
    t["block"] = V.spatial_blocks(t[["x", "y", "z"]].to_numpy(float), target_a=20.0)
    r = V.radial_trend(t, "T")
    assert r["slope_per_A"] == pytest.approx(0.08, abs=0.03)
    assert r["lo"] > 0                        # rises away from the pathway
    assert r["spearman"] > 0.4


def test_too_small_a_group_is_declined_rather_than_tested():
    t = _sites(n_block=4, per_block=3)
    assert V.one_test(t, pd.Series(True, index=t.index), "all", "T") is None
    assert V.radial_trend(t.iloc[:10], "T") is None
