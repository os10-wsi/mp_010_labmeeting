"""Burial and hydrophobicity: the geometry and the models, not the plotting."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms import burial as B
from mpdms.annot import KYTE_DOOLITTLE


def test_the_two_loop_sides_become_one_aqueous_class():
    """The lumen and the cytoplasm are both water; the comparison is against lipid."""
    env = pd.Series(["cytosolic loop", "lumenal loop", "lipid-facing",
                     "helix-helix core", "cavity-lining", None])
    got = B.burial_class(env)
    assert list(got[:2]) == ["aqueous", "aqueous"]
    assert list(got[2:5]) == ["lipid", "core", "cavity"]
    assert pd.isna(got.iloc[5])
    # and nothing silently lands outside the four classes the figures draw
    assert set(B.FROM_ENVIRON.values()) == set(B.BCLASSES)


def test_neighbour_hydrophobicity_skips_the_sequence_neighbours():
    """A residue's own helix turn would otherwise measure it against itself."""
    rows = [{"pos": 10, "aa": "A", "x": 0.0, "y": 0.0, "z": 0.0}]
    # four residues either side, all close in space - these must be excluded
    for k in list(range(6, 10)) + list(range(11, 15)):
        rows.append({"pos": k, "aa": "I", "x": 1.0, "y": 0.0, "z": 0.0})
    # one residue far away in sequence, 5 A off: this is the whole answer
    rows.append({"pos": 60, "aa": "K", "x": 5.0, "y": 0.0, "z": 0.0})
    # and one far away in space, which must not count either
    rows.append({"pos": 80, "aa": "W", "x": 40.0, "y": 0.0, "z": 0.0})
    rt = pd.DataFrame(rows)
    got = B.neighbour_hydrophobicity(rt, cutoff=8.0, exclude=4)
    assert got.iloc[0] == pytest.approx(KYTE_DOOLITTLE["K"])


def test_inter_helix_distance_measures_to_a_different_helix():
    """Being wrapped by your own backbone is not being at a packing interface."""
    rows, helix = [], {}
    for k, xoff in enumerate((0.0, 10.0)):
        for j in range(8):
            p = 1 + 8 * k + j
            rows.append({"pos": p, "aa": "L", "x": xoff, "y": 0.0, "z": 1.5 * j})
            helix[p] = f"TM{k + 1}"
    rt = pd.DataFrame(rows)
    d = B.inter_helix_distance(rt, helix)
    # own-helix neighbours are 1.5 A away and must be ignored
    assert d.min() == pytest.approx(10.0)
    assert d.notna().all()
    # a single helix has nothing to be an interface with
    assert B.inter_helix_distance(rt, {p: "TM1" for p in rt.pos}).isna().all()


def _planted(n_pos=40, per_pos=12, polar_slope=0.12, hydro_slope=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pos):
        base = rng.normal(0, 0.2)                     # the position's own tolerance
        for _ in range(per_pos):
            dkd = rng.uniform(-8, 8)
            arm = polar_slope if dkd < 0 else hydro_slope
            rows.append({"pos": p, "delta_kd": dkd,
                         "score_z": base + arm * dkd + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_hinge_slopes_find_a_one_sided_response():
    """Lipid-facing is the claim that only one direction of change is punished."""
    d = _planted(polar_slope=0.12, hydro_slope=0.0)
    h = B.hinge_slopes(d.delta_kd, d.score_z, d.pos)
    assert h["slope_polar_arm"] == pytest.approx(0.12, abs=0.02)
    assert h["slope_hydrophobic_arm"] == pytest.approx(0.0, abs=0.02)
    assert h["asymmetry"] == pytest.approx(h["slope_polar_arm"]
                                           + h["slope_hydrophobic_arm"])
    assert h["n_positions"] == 40


def test_a_symmetric_tent_has_no_asymmetry():
    """Core is the claim that both directions are punished, which must read as zero."""
    d = _planted(polar_slope=0.12, hydro_slope=-0.12)
    h = B.hinge_slopes(d.delta_kd, d.score_z, d.pos)
    assert h["asymmetry"] == pytest.approx(0.0, abs=0.03)
    assert h["slope_polar_arm"] > 0.08 and h["slope_hydrophobic_arm"] < -0.08


def test_the_asymmetry_carries_an_interval_that_knows_the_arms_are_correlated():
    """A contrast of two estimates needs their covariance, not their errors added."""
    one_sided = B.hinge_slopes(*[_planted(polar_slope=0.12, hydro_slope=0.0)[c]
                                 for c in ("delta_kd", "score_z", "pos")])
    tent = B.hinge_slopes(*[_planted(polar_slope=0.12, hydro_slope=-0.12)[c]
                            for c in ("delta_kd", "score_z", "pos")])
    # a genuinely one-sided response: the interval is clear of zero, and wide of it
    assert one_sided["asymmetry_lo"] > 0 and one_sided["asymmetry_p"] < 1e-6
    assert one_sided["asymmetry"] / one_sided["asymmetry_se"] > 10
    # a symmetric tent: both arms are large and the asymmetry is two orders smaller.
    # Its interval can still exclude zero - with 480 variants the estimate is precise
    # enough for a sampling wobble to be "significant" - so the test is on the size of
    # the asymmetry against the size of the arms, which is what the figure claims.
    assert abs(tent["asymmetry"]) < 0.02
    assert min(abs(tent["slope_polar_arm"]), abs(tent["slope_hydrophobic_arm"])) > 0.08
    # and the contrast's error is not the naive sum of the two arms' errors
    naive = tent["se_polar_arm"] + tent["se_hydrophobic_arm"]
    assert tent["asymmetry_se"] < naive


def test_a_straight_line_would_have_missed_the_tent():
    """The reason the hinge exists: one slope through a tent is about zero."""
    d = _planted(polar_slope=0.12, hydro_slope=-0.12)
    from scipy import stats as ss
    assert abs(ss.linregress(d.delta_kd, d.score_z).slope) < 0.02
    assert B.hinge_slopes(d.delta_kd, d.score_z, d.pos)["slope_polar_arm"] > 0.08


def test_clustering_refuses_a_standard_error_built_from_too_few_positions():
    """Thousands of variants at four positions are four observations, not thousands."""
    d = _planted(n_pos=4, per_pos=200)
    assert B.cluster_ols("score_z ~ delta_kd", d[["score_z", "delta_kd"]], d.pos) is None
    wide = _planted(n_pos=40, per_pos=20)
    m = B.cluster_ols("score_z ~ delta_kd", wide[["score_z", "delta_kd"]], wide.pos)
    assert m is not None
    # and clustering must widen the interval relative to pretending variants are units
    import statsmodels.formula.api as smf
    naive = smf.ols("score_z ~ delta_kd", data=wide).fit()
    assert m.bse["delta_kd"] > naive.bse["delta_kd"]


def _helix(aligned: bool, n: int = 20) -> pd.DataFrame:
    """A helix with a clear hydrophobic face, and intolerance on it or opposite it."""
    ang = np.deg2rad(B.PERIOD) * np.arange(n)
    wt = ["I" if np.cos(a) > 0 else "K" for a in ang]
    kd = np.array([KYTE_DOOLITTLE[a] for a in wt])
    kd = kd - kd.mean()
    # effect is most negative where the sequence is most hydrophobic, or the reverse
    eff = (-kd if aligned else kd) * 0.1
    return pd.DataFrame({"pos": np.arange(n), "wt": wt, "effect": eff})


def test_the_intolerant_face_is_found_on_the_hydrophobic_face():
    ph, pe, dphi = B.face_angles(_helix(aligned=True))
    assert abs(dphi) < np.deg2rad(10)
    assert np.isclose(np.angle(np.exp(1j * (ph - pe))), 0, atol=np.deg2rad(10))


def test_the_intolerant_face_opposite_the_hydrophobic_face_reads_as_180():
    _, _, dphi = B.face_angles(_helix(aligned=False))
    assert abs(abs(dphi) - np.pi) < np.deg2rad(10)


def test_the_resultant_separates_agreement_from_a_spread():
    assert B.rayleigh(np.zeros(8)) == pytest.approx(1.0)
    assert B.rayleigh(np.linspace(0, 2 * np.pi, 9)[:-1]) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(B.rayleigh(np.array([])))


def test_mismatch_beats_the_change_from_wild_type_when_mismatch_is_the_truth():
    """b09's claim: a residue is wrong for its surroundings, not wrong in the absolute."""
    rng = np.random.default_rng(3)
    n_pos, per_pos = 50, 15
    rows = []
    for p in range(n_pos):
        env = rng.uniform(-2.5, 3.0)
        wt_kd = env + rng.normal(0, 0.5)
        for _ in range(per_pos):
            mut_kd = rng.uniform(-4.5, 4.5)
            rows.append({"pos": p, "kd_env": env, "kd_mut": mut_kd,
                         "delta_kd": mut_kd - wt_kd,
                         "score_z": -abs(mut_kd - env) * 0.15 + rng.normal(0, 0.1)})
    d = pd.DataFrame(rows)
    d["mismatch"] = (d.kd_mut - d.kd_env).abs()
    d["abs_dkd"] = d.delta_kd.abs()
    a = B.cluster_ols("score_z ~ mismatch", d[["score_z", "mismatch"]], d.pos)
    b = B.cluster_ols("score_z ~ abs_dkd", d[["score_z", "abs_dkd"]], d.pos)
    assert a.rsquared > b.rsquared
    assert a.params["mismatch"] < 0


def test_glycine_and_proline_are_in_neither_wild_type_group():
    """Both change the backbone, so calling one polar or hydrophobic says nothing."""
    g = B.wt_group(pd.Series(list("GPILKDA")))
    assert pd.isna(g.iloc[0]) and pd.isna(g.iloc[1])
    assert list(g.iloc[2:]) == ["hydrophobic", "hydrophobic", "polar", "polar",
                               "hydrophobic"]
