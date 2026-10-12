"""Insertion energetics: the dsTβL scale, and the interaction the hypothesis is about."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy import stats as ss

from mpdms import insertion as I


def test_the_scale_is_the_published_one():
    """Twenty residues, relative to alanine, with the signs the paper reports."""
    assert len(I.DSTBL) == 20
    assert set(I.DSTBL) == set("ACDEFGHIKLMNPQRSTVWY")
    assert all(c == 0.0 for c in I.DSTBL["A"])          # the reference residue
    assert I.residue_dg("A", 0.0) == 0.0
    # hydrophobic residues favour the bilayer, charged and polar ones resist it
    for aa in "LIFVMW":
        assert I.residue_dg(aa, 0.0) < 0, aa
    for aa in "DEKRNQGPSH":
        assert I.residue_dg(aa, 0.0) > 0, aa
    assert I.residue_dg("L", 0.0) == pytest.approx(-1.92, abs=0.02)
    assert I.residue_dg("D", 0.0) == pytest.approx(1.54, abs=0.02)


def test_the_profiles_are_asymmetric_and_carry_the_positive_inside_rule():
    """Not a symmetrised approximation: the measured scale knows which side is which.

    Negative z is cytoplasmic in this convention, and lysine and arginine are much
    cheaper there - the positive-inside rule, sitting inside the scale as measured. It
    is why helix orientation cannot be ignored when a segment is laid out.
    """
    for aa in "KR":
        assert I.residue_dg(aa, -15.0) < I.residue_dg(aa, 15.0) - 1.0, aa
    assert I.residue_dg("R", -15.0) < 0 < I.residue_dg("R", 15.0)
    # leucine is favourable at every depth, which is what makes a helix of it insert
    assert max(I.residue_dg("L", z) for z in (-15, -7, 0, 7, 15)) < 0
    # the polynomial is not evaluated far outside the range it was fitted over
    assert I.residue_dg("D", 400.0) == I.residue_dg("D", I.HALF_WIDTH + 5)
    assert I.residue_dg("D", -400.0) == I.residue_dg("D", -I.HALF_WIDTH - 5)
    assert np.isnan(I.residue_dg("X", 0.0))


def test_a_segment_is_laid_across_the_membrane_the_way_the_scale_was_built():
    """linspace(-15, +15) over the residues, summed - the authors' own convention."""
    seq = "LAWVGGSILAVIGLFA"
    z = np.linspace(-15, 15, num=len(seq))
    assert np.allclose(I.segment_depths(len(seq)), z)
    assert I.segment_dg(seq) == pytest.approx(
        sum(I.residue_dg(a, zi) for a, zi in zip(seq, z)))
    # a helix pointing the other way is read the other way, and the asymmetric profiles
    # then give it a different energy - which is the reason orientation is carried at all
    assert np.allclose(I.segment_depths(len(seq), "out_in"), z[::-1])
    assert I.segment_dg("K" + "L" * 15, orientation="in_out") < \
        I.segment_dg("K" + "L" * 15, orientation="out_in")
    # a poly-leucine helix inserts readily; poly-alanine is the zero by construction
    assert I.segment_dg("L" * 20) < -20
    assert I.segment_dg("A" * 20) == 0.0
    # and one aspartate costs more in the middle than at the end
    mid, end = "L" * 10 + "D" + "L" * 9, "D" + "L" * 19
    assert I.segment_dg(mid) > I.segment_dg(end)


def _toy(n_helices=8, effect="null", seed=0, per_helix=60):
    """A protein whose helices span a range of ΔG, with a chosen truth."""
    rng = np.random.default_rng(seed)
    aas = list("AVLIMFWYCGPSTNQHKRDE")
    rows, hrows = [], []
    for h in range(n_helices):
        # helices from comfortable to marginal, by how much leucine they carry
        frac = h / (n_helices - 1)
        seq = "".join(rng.choice(list("LIFV"), 1)[0] if rng.random() > frac * 0.8
                      else rng.choice(list("ASTGN"), 1)[0] for _ in range(21))
        dg = I.segment_dg(seq)
        hrows.append({"helix": f"TM{h+1}", "start": 1 + 30 * h, "end": 21 + 30 * h,
                      "length": 21, "dg": dg, "dg_per_residue": dg / 21,
                      "mean_effect": 0.0, "median_effect": 0.0, "se_effect": 0.05,
                      "n_positions": 21, "mean_rsa": 0.2})
        z = I.segment_depths(21)
        for k in range(per_helix):
            i = int(rng.integers(0, 21))
            wt, mut = seq[i], aas[int(rng.integers(0, 20))]
            ddg = I.residue_dg(mut, z[i]) - I.residue_dg(wt, z[i])
            if effect == "interaction":      # the hypothesis: steeper where dG is higher
                mu = -0.3 - 0.02 * ddg * (dg + 20)
            elif effect == "main":           # ddG matters, but equally everywhere
                mu = -0.3 - 0.08 * ddg
            else:
                mu = -0.3
            rows.append({"pos": 1 + 30 * h + i, "wt": wt, "mut": mut,
                         "score_z": mu + rng.normal(0, 0.15), "rsa": rng.uniform(0, 0.6),
                         "zdepth": z[i], "helix": f"TM{h+1}", "helix_dg": dg,
                         "z_ideal": z[i]})
    h = pd.DataFrame(hrows)
    h["marginal"] = (h.dg >= h.dg.median()).astype(int)
    v = pd.DataFrame(rows)
    v["marginal"] = v.helix_dg.ge(h.dg.median()).astype(int)
    return h, I.add_ddg(v)


def test_the_interaction_is_recovered_when_it_is_there_and_not_when_it_is_not(tmp_path):
    """The calibration that makes a null result worth reporting."""
    got = {}
    for truth in ("interaction", "main", "null"):
        h, v = _toy(effect=truth, seed=1)
        r = I.i02_interaction({"P": v}, {"P": h}, ["P"], tmp_path)
        got[truth] = r["models"][0]
    assert got["interaction"]["interaction"] < 0
    assert got["interaction"]["interaction_p"] < 0.01
    # a pure main effect of ddG must NOT produce an interaction: that is the confound
    assert got["main"]["interaction_p"] > 0.05
    assert got["null"]["interaction_p"] > 0.05
    # the main effect is present in both of the first two, and is not the test
    assert got["main"]["ddg"] < 0 and got["interaction"]["ddg"] < 0


def test_ddg_is_antisymmetric_and_zero_for_no_change():
    d = pd.DataFrame({"pos": [1, 2, 3], "wt": list("LDA"), "mut": list("DLA"),
                      "score_z": [0.0] * 3, "z_ideal": [0.0] * 3,
                      "helix": ["TM1"] * 3, "helix_dg": [-20.0] * 3, "marginal": [0] * 3})
    out = I.add_ddg(d)
    assert out.ddg.iloc[0] == pytest.approx(-out.ddg.iloc[1])
    assert out.ddg.iloc[2] == 0.0
    assert out.ddg.iloc[0] > 0            # leucine to aspartate makes insertion harder


def test_structure_depths_are_used_only_when_asked_for_and_only_where_they_exist():
    d = pd.DataFrame({"pos": [1, 2], "wt": list("LL"), "mut": list("DD"),
                      "score_z": [0.0, 0.0], "z_ideal": [0.0, 0.0],
                      "zdepth": [12.0, np.nan], "helix": ["TM1"] * 2,
                      "helix_dg": [-20.0] * 2, "marginal": [0] * 2})
    assert list(I.add_ddg(d, "ideal").z_used) == [0.0, 0.0]
    z = list(I.add_ddg(d, "structure").z_used)
    assert z == [12.0, 0.0]               # the missing one falls back, it is not dropped
    assert I.add_ddg(d, "structure").ddg.iloc[0] < I.add_ddg(d, "ideal").ddg.iloc[0]


def test_marginal_is_decided_within_a_protein_not_across_proteins():
    h, _ = _toy(seed=4)
    assert set(h.marginal) == {0, 1}
    assert h.loc[h.dg.idxmax(), "marginal"] == 1      # the least insertable is marginal
    assert h.loc[h.dg.idxmin(), "marginal"] == 0


def test_standard_errors_are_clustered_by_helix(tmp_path):
    """The interaction varies between helices, so helices are the unit of replication.

    Giving each variant its own degree of freedom would make a twelve-helix protein look
    like thousands of observations of a quantity that really has twelve.
    """
    import statsmodels.formula.api as smf
    h, v = _toy(effect="interaction", seed=7)
    # the contract: what the module reports IS the clustered fit
    d = v.assign(rsa_=v.rsa)
    clustered = smf.ols("score_z ~ ddg * helix_dg + rsa_", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d.helix})
    got = I.i02_interaction({"P": v}, {"P": h}, ["P"], tmp_path)["models"][0]
    assert got["interaction_se"] == pytest.approx(clustered.bse["ddg:helix_dg"], rel=1e-6)
    assert got["n_helices"] == 8

    # and it is worth doing: give each helix a shock of its own, the way a real helix
    # differs from its neighbours for reasons ddG knows nothing about, and the naive
    # standard error understates the uncertainty on the term that varies between helices
    rng = np.random.default_rng(0)
    shock = {hx: rng.normal(0, 0.25) for hx in v.helix.unique()}
    w = d.assign(score_z=d.score_z + d.helix.map(shock))
    naive_s = smf.ols("score_z ~ ddg * helix_dg + rsa_", data=w).fit()
    clustered_s = smf.ols("score_z ~ ddg * helix_dg + rsa_", data=w).fit(
        cov_type="cluster", cov_kwds={"groups": w.helix})
    assert clustered_s.bse["helix_dg"] > 2 * naive_s.bse["helix_dg"]


def test_every_helix_gets_its_own_curve_and_the_median_is_of_helices(tmp_path):
    """No median split: the figure draws each helix and the median across them."""
    h, v = _toy(n_helices=12, effect="interaction", seed=3, per_helix=120)
    r = I.i03_helix_curves({"P": v}, {"P": h}, ["P"], tmp_path)
    per = pd.read_csv(tmp_path / "i03_helix_slopes.csv")
    assert len(per) == 12 and per.helix.nunique() == 12
    med = pd.read_csv(tmp_path / "i03_median_curve.csv")
    assert len(med) >= 3 and (med.n_helices >= 3).all()
    # both summaries are reported, and both weight every helix once
    assert {"mean_effect", "median_effect", "sem_across_helices",
            "pooled_over_variants"} <= set(med.columns)
    assert med.sem_across_helices.gt(0).all()
    # with no single helix out on its own the two summaries track each other, within
    # the spread of the helices themselves; where they do not, that is the signal the
    # second line exists to show (asserted on its own below)
    assert ((med.mean_effect - med.median_effect).abs()
            <= 2 * med.sem_across_helices + 1e-9).all()
    # and with a planted interaction, the less insertable helices have steeper slopes
    rho = ss.spearmanr(per.dg, per.slope)[0]
    assert rho < -0.4, rho
    assert "ρ(slope, ΔG)" in r["headline"]


def test_a_helix_with_too_little_data_is_left_out_rather_than_drawn_thin(tmp_path):
    h, v = _toy(n_helices=6, seed=5, per_helix=90)
    sparse = pd.concat([v[v.helix != "TM3"], v[v.helix == "TM3"].head(4)],
                       ignore_index=True)
    I.i03_helix_curves({"P": sparse}, {"P": h}, ["P"], tmp_path)
    per = pd.read_csv(tmp_path / "i03_helix_slopes.csv")
    assert "TM3" not in set(per.helix)
    assert len(per) == 5


def test_the_mean_follows_an_outlier_helix_and_the_median_does_not(tmp_path):
    """Why both lines are drawn: where they separate, a few helices carry the trend."""
    h, v = _toy(n_helices=12, effect="null", seed=8, per_helix=120)
    # one helix collapses at high ddG, the rest are flat
    hit = (v.helix == "TM5") & (v.ddg > 1)
    v.loc[hit, "score_z"] = v.loc[hit, "score_z"] - 2.0
    I.i03_helix_curves({"P": v}, {"P": h}, ["P"], tmp_path)
    m = pd.read_csv(tmp_path / "i03_median_curve.csv").sort_values("ddg")
    right = m.iloc[-1]
    assert right.mean_effect < right.median_effect - 0.05
    # and the median stays near where the eleven unaffected helices sit
    assert abs(right.median_effect - m.iloc[0].median_effect) < 0.15


def test_the_pooled_curve_averages_over_variants_not_helices(tmp_path):
    """The estimator the old figure used: every variant counts once, so coverage weighs."""
    h, v = _toy(n_helices=12, effect="null", seed=12, per_helix=60)
    # one helix is deeply covered and sits well below the rest
    extra = v[v.helix == "TM1"].sample(900, replace=True, random_state=0)
    extra = extra.assign(score_z=extra.score_z - 1.0)
    deep = pd.concat([v, extra], ignore_index=True)
    I.i04_pooled({"P": deep}, ["P"], tmp_path)
    pooled = pd.read_csv(tmp_path / "i04_pooled_curve.csv")
    I.i03_helix_curves({"P": deep}, {"P": h}, ["P"], tmp_path)
    by_helix = pd.read_csv(tmp_path / "i03_median_curve.csv")
    # the well-covered helix drags the pooled average down; the median over helices
    # barely notices it, which is the whole difference between the two figures
    assert pooled.mean_effect.mean() < by_helix.median_effect.mean() - 0.2
    assert int(pooled.n_variants.sum()) == len(deep)


def test_the_pooled_slope_error_is_clustered_by_helix(tmp_path):
    h, v = _toy(n_helices=12, effect="main", seed=13, per_helix=90)
    import statsmodels.formula.api as smf
    r = I.i04_pooled({"P": v}, ["P"], tmp_path)["by_protein"][0]
    f = smf.ols("score_z ~ ddg", data=v).fit(cov_type="cluster",
                                             cov_kwds={"groups": v.helix})
    assert r["se_clustered"] == pytest.approx(f.bse["ddg"], rel=1e-6)
    assert r["slope"] < 0 and r["n_helices"] == 12
    assert r["n_variants"] == len(v)
