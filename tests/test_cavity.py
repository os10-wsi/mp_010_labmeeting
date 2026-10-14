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
                         "median_residual": shared + cavity_effect * cav + rng.normal(0, 0.35)})
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
                     "median_residual": -1.2 + 0.08 * rad + rng.normal(0, 0.3)})
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


def test_the_value_tested_is_the_residual_not_the_z():
    """The user asked for the residual's own units, so that has to be the default.

    The two differ by one constant shared by the whole protein, so the p-value must not
    move when the column is rescaled - which is the reason it was safe to test the z in
    the first place.
    """
    assert V.VALUE == "median_residual"
    t = _sites(cavity_effect=-0.8)
    a = V.one_test(t, pd.Series(True, index=t.index), "all", "T")
    scaled = t.assign(median_residual=t.median_residual / 1.4826)
    b = V.one_test(scaled, pd.Series(True, index=t.index), "all", "T")
    assert b["p_mannwhitney"] == pytest.approx(a["p_mannwhitney"])
    assert b["p_block_clustered"] == pytest.approx(a["p_block_clustered"], rel=1e-6)
    assert b["cliffs_delta"] == pytest.approx(a["cliffs_delta"])
    assert b["difference"] == pytest.approx(a["difference"] / 1.4826)


def test_the_shape_tests_see_a_difference_the_centres_do_not():
    """A group with the same median and a heavier tail is still a different group."""
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1.0, 300)
    c = rng.normal(0, 2.5, 300)
    r = V.distribution_tests(a, c)
    assert r["p_mannwhitney"] > 0.05          # the centres agree
    assert r["p_brown_forsythe"] < 1e-6       # the spreads do not
    assert r["p_ks"] < 1e-4
    assert r["iqr_b"] > 2 * r["iqr_a"]


def test_the_within_block_permutation_ignores_a_between_patch_gradient():
    """A large-scale gradient that lines up with the cavity must not pass as an effect.

    Each block is entirely cavity or entirely not in `_sites`, so after within-block
    centring there is nothing left to permute and the test declines rather than
    reporting a difference it cannot see.
    """
    t = _sites(cavity_effect=-0.8)
    assert np.isnan(V.within_block_permutation(t, "median_residual"))
    # mix the classes inside each block and the same planted effect is found
    mixed = t.copy()
    mixed["is_cavity"] = (np.arange(len(mixed)) % 2).astype(float)
    mixed["median_residual"] = mixed.median_residual - 0.8 * mixed.is_cavity
    p = V.within_block_permutation(mixed, "median_residual", n=400, seed=2)
    assert p < 0.01


def test_holm_is_monotone_and_never_shrinks_a_p_value():
    adj = V.holm([0.01, 0.04, 0.5])
    assert adj == pytest.approx([0.03, 0.08, 0.5])
    assert all(b >= a for a, b in zip(adj, adj[1:]))
    assert V.holm([0.3, 0.3, 0.3]) == pytest.approx([0.9, 0.9, 0.9])
    assert V.holm([0.9, 0.9, 0.9]) == pytest.approx([1.0, 1.0, 1.0])


def test_pairwise_covers_every_pair_and_corrects_across_them():
    t = _sites(cavity_effect=-0.8)
    t["klass"] = np.where(t.is_cavity > 0.5, "cavity",
                          np.where(np.arange(len(t)) % 2, "lipid", "aqueous"))
    q = V.pairwise_tests(t, "T", value="median_residual")
    assert set(zip(q.a, q.b)) == {("cavity", "lipid"), ("cavity", "aqueous"),
                                  ("lipid", "aqueous")}
    assert (q.p_mannwhitney_holm >= q.p_mannwhitney - 1e-12).all()
    hit = q[(q.a == "cavity")]
    assert (hit.cliffs_delta < -0.4).all()
    assert (q[(q.a == "lipid")].p_mannwhitney_holm > 0.05).all()


def test_p_is_never_printed_as_zero():
    assert V.pstr(0.0).startswith("p <")
    assert V.pstr(0.0312) == "p = 0.0312"
    assert V.pstr(float("nan")) == "p n/a"
