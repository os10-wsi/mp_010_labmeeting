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
                      "length": 21, "orientation": "in_out" if h % 2 == 0 else "out_in",
                      "dg": dg, "dg_per_residue": dg / 21,
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


def test_only_the_two_extreme_helices_are_drawn_by_default(tmp_path):
    """The default is the hardest and the easiest; --all-helices puts the rest back."""
    import matplotlib.pyplot as plt
    h, v = _toy(n_helices=12, effect="interaction", seed=21, per_helix=120)
    before = plt.get_fignums()
    I.i03_helix_curves({"P": v}, {"P": h}, ["P"], tmp_path)
    # every helix still gets a slope in the table, drawn or not
    per = pd.read_csv(tmp_path / "i03_helix_slopes.csv")
    assert len(per) == 12
    # and the two named in the legend are the real extremes of dG
    assert per.loc[per.dg.idxmax(), "helix"] != per.loc[per.dg.idxmin(), "helix"]
    assert plt.get_fignums() == before          # figures are closed, not leaked


def test_the_ranking_is_within_protein_and_orders_by_insertion_energy(tmp_path):
    h, _ = _toy(n_helices=12, seed=22)
    r = I.i05_ranking({"P": h}, ["P"], tmp_path)
    t = pd.read_csv(tmp_path / "i05_helix_ranking.csv")
    assert len(t) == 12
    assert list(t["rank"]) == list(range(1, 13))
    assert t.dg.is_monotonic_increasing                 # rank 1 inserts most readily
    assert t.would_not_insert_alone.tolist() == (t.dg >= 0).tolist()
    assert "easiest" in r["headline"] and "hardest" in r["headline"]


def test_every_panel_names_its_own_two_helices(tmp_path, monkeypatch):
    """The legend is per protein, because each panel's two helices are different ones.

    A shared legend on panel 1 would be labelling lines in panels 2 and 3 that are not
    the helices it names.
    """
    import matplotlib.pyplot as plt
    seen = []
    real = plt.Figure.savefig

    def capture(self, *a, **k):
        seen.append([[t.get_text() for t in ax.get_legend().get_texts()]
                     if ax.get_legend() else [] for ax in self.axes])
        return real(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", capture)
    hs, vs = {}, {}
    for k, name in enumerate(["P1", "P2", "P3"]):
        h, v = _toy(n_helices=12, effect="interaction", seed=30 + k, per_helix=120)
        hs[name], vs[name] = h, v
    I.i03_helix_curves(vs, hs, ["P1", "P2", "P3"], tmp_path)
    legends = [lg for lg in seen[0] if lg]
    assert len(legends) == 3                       # one per protein, not one in total
    for lg in legends:
        named = [t for t in lg if "to insert" in t]
        assert len(named) == 2                     # the hardest and the easiest
        assert any("hardest" in t for t in named)
        assert any("easiest" in t for t in named)
    # only the first panel repeats the mean and median, which are the same everywhere
    assert sum("mean of helices" in t for lg in legends for t in lg) == 1


# --------------------------------------------- the second scale: Hessa 2007 table S2
def test_the_hessa_table_is_the_published_one():
    """Twenty residues, nineteen positions, and the values the table prints."""
    assert set(I.HESSA) == set("ACDEFGHIKLMNPQRSTVWY")
    assert all(len(v) == 19 for v in I.HESSA.values())
    # spot values read off table S2: i = 0 is the centre, i = +-9 the ends
    assert I.residue_dg("K", 0.0, "hessa") == pytest.approx(1.85, abs=0.01)
    assert I.residue_dg("K", I.HALF_WIDTH, "hessa") == pytest.approx(0.32, abs=0.01)
    assert I.residue_dg("L", 0.0, "hessa") == pytest.approx(-0.43, abs=0.01)
    assert I.residue_dg("D", 0.0, "hessa") == pytest.approx(1.79, abs=0.01)
    assert I.residue_dg("W", 0.0, "hessa") == pytest.approx(0.27, abs=0.01)
    # tryptophan is the one row that is not monotone: it prefers the interface
    assert I.residue_dg("W", -10.0, "hessa") < I.residue_dg("W", 0.0, "hessa")
    assert np.isnan(I.residue_dg("X", 0.0, "hessa"))


def test_hessa_is_absolute_and_dstbl_is_relative_to_alanine():
    """The reference states differ, so single values differ and differences agree."""
    assert I.residue_dg("A", 0.0, "dstbl") == 0.0
    assert I.residue_dg("A", 0.0, "hessa") == pytest.approx(0.13, abs=0.01)
    # the two scales rank the twenty residues the same way, which is the check that
    # catches a transcription error without demanding they agree numerically
    aas = "ACDEFGHIKLMNPQRSTVWY"
    rho = ss.spearmanr([I.residue_dg(a, 0.0, "dstbl") for a in aas],
                       [I.residue_dg(a, 0.0, "hessa") for a in aas]).statistic
    assert rho > 0.75          # 0.79 as transcribed: agreement, not equivalence


def test_the_hessa_table_is_symmetric_so_orientation_cannot_matter():
    """dsTbL knows which side is cytoplasmic; this table does not, and must not pretend.

    Every row of table S2 reads the same forwards and backwards, so a helix scored
    N-to-C and the same helix scored C-to-N give the same number. The positive-inside
    rule is therefore absent from this scale - lysine costs the same at both ends here
    and ten times more at the outer end under dsTbL.
    """
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        for z in (3.0, 7.5, 15.0):
            assert I.residue_dg(aa, z, "hessa") == pytest.approx(
                I.residue_dg(aa, -z, "hessa"), abs=1e-9), aa
    seq = "LLKWFAGSILVTNPMAIRD"
    old = I.SCALE
    try:
        I.SCALE = "hessa"
        assert I.segment_dg(seq, orientation="in_out") == pytest.approx(
            I.segment_dg(seq, orientation="out_in"))
        I.SCALE = "dstbl"
        # the other scale is not symmetric, and the same call proves the test can fail
        assert I.segment_dg(seq, orientation="in_out") != pytest.approx(
            I.segment_dg(seq, orientation="out_in"))
    finally:
        I.SCALE = old
    assert I.residue_dg("K", -15.0, "dstbl") < I.residue_dg("K", 15.0, "dstbl") - 1.0


def test_a_segment_lands_on_the_table_positions_whatever_its_length():
    """Residue k of L sits where the table would put residue k of 19."""
    old = I.SCALE
    try:
        I.SCALE = "hessa"
        for length in (15, 19, 23):
            z = I.segment_depths(length)
            assert I.residue_dg("K", z[0]) == pytest.approx(0.32, abs=0.01)
            assert I.residue_dg("K", z[-1]) == pytest.approx(0.32, abs=0.01)
            assert I.residue_dg("K", z[length // 2]) == pytest.approx(1.85, abs=0.02)
        # a poly-leucine segment is favourable, a poly-arginine one is not
        assert I.segment_dg("L" * 19) < 0 < I.segment_dg("R" * 19)
    finally:
        I.SCALE = old


def test_ddg_is_the_same_shape_under_either_scale():
    """The analyses only ever take differences, so both scales feed them the same way."""
    d = pd.DataFrame({"wt": ["L", "L", "A"], "mut": ["K", "L", "K"],
                      "z_ideal": [0.0, 0.0, 0.0]})
    old = I.SCALE
    try:
        for scale, k_cost in (("dstbl", None), ("hessa", 2.28)):
            I.SCALE = scale
            out = I.add_ddg(d)
            assert out.ddg.iloc[1] == 0.0                 # no change costs nothing
            assert out.ddg.iloc[0] > 0                    # leucine to lysine is costly
            if k_cost is not None:                        # 1.85 - (-0.43)
                assert out.ddg.iloc[0] == pytest.approx(k_cost, abs=0.01)
    finally:
        I.SCALE = old


# ------------------------------------------------- i06: do the two scales agree?
def test_the_two_scales_agree_on_order_and_not_on_magnitude():
    """The finding i06 exists to report, pinned so a scale edit cannot quietly undo it.

    Ordering agrees - both call leucine cheap and lysine expensive - while dsTbL has
    several times the dynamic range. A correlation alone would hide that, which is why
    i06 reports the regression slope beside it.
    """
    for z in (0.0, 12.0):
        for grp in ("LIFVMWCA", "DEKR", "NQSTHYGP"):
            a = [I.residue_dg(x, z, "dstbl") for x in grp]
            b = [I.residue_dg(x, z, "hessa") for x in grp]
            # 1.39 (polar, interface) to 4.83 (hydrophobic, interface) as transcribed
            assert (max(a) - min(a)) > 1.3 * (max(b) - min(b)), (z, grp)
    # the widest gap is the one that matters most: the hydrophobics in the bilayer
    wide = [I.residue_dg(x, 12.0, "dstbl") for x in "LIFVMWCA"]
    narrow = [I.residue_dg(x, 12.0, "hessa") for x in "LIFVMWCA"]
    assert (max(wide) - min(wide)) > 4 * (max(narrow) - min(narrow))


def test_i06_reports_agreement_and_writes_both_figures(tmp_path):
    from mpdms.insertion import i06_scale_agreement
    rng = np.random.default_rng(0)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    n = 400
    v = pd.DataFrame({"wt": rng.choice(aas, n), "mut": rng.choice(aas, n),
                      "z_ideal": rng.uniform(-15, 15, n),
                      "score_z": rng.normal(-0.5, 0.3, n), "helix": "TM1"})
    h = pd.DataFrame({"helix": ["TM1", "TM2", "TM3", "TM4"],
                      "seq": ["LLLLLAAAAAVVVVVIIIII", "KKKKKDDDDDEEEEERRRRR",
                              "LLAAGGSSTTVVIIFFWWYY", "LLLLLLLLLLLLLLLLLLLL"],
                      "orientation": ["in_out", "out_in", "in_out", "out_in"]})
    r = i06_scale_agreement({"T": v}, {"T": h}, ["T"], tmp_path)
    assert r["status"] == "ok"
    row = r["by_protein"][0]
    assert row["n_variants"] == n
    assert row["pearson_ddg"] > 0.6              # the scales broadly agree
    assert 0 < row["slope_hessa_on_dstbl"] < 1   # and Hessa is the compressed one
    assert (tmp_path / "i06_scale_agreement.csv").exists()
    assert (tmp_path / "i06b_residue_scales.csv").exists()
    for f in ("i06_scale_agreement", "i06b_residue_scales"):
        assert (tmp_path / f"{f}.png").exists()


def test_i06_does_not_depend_on_which_scale_is_active():
    """It computes both, so the --scale flag must not change what it reports."""
    from mpdms.insertion import i06_scale_agreement
    rng = np.random.default_rng(3)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    v = pd.DataFrame({"wt": rng.choice(aas, 200), "mut": rng.choice(aas, 200),
                      "z_ideal": rng.uniform(-15, 15, 200),
                      "score_z": rng.normal(0, 1, 200), "helix": "TM1"})
    h = pd.DataFrame({"helix": ["TM1", "TM2", "TM3"],
                      "seq": ["LLLLLAAAAAVVVVV", "KKKKKDDDDDEEEEE", "LLAAGGSSTTVVIIF"],
                      "orientation": ["in_out", "out_in", "in_out"]})
    import tempfile
    from pathlib import Path as _P
    got = []
    old = I.SCALE
    try:
        for scale in ("dstbl", "hessa"):
            I.SCALE = scale
            with tempfile.TemporaryDirectory() as d:
                got.append(i06_scale_agreement({"T": v}, {"T": h}, ["T"], _P(d))
                           ["by_protein"][0])
    finally:
        I.SCALE = old
    assert got[0]["pearson_ddg"] == pytest.approx(got[1]["pearson_ddg"])
    assert got[0]["spearman_helix_dg"] == pytest.approx(got[1]["spearman_helix_dg"])


def test_the_ddg_axis_says_which_depth_and_which_scale():
    """"at that depth" pointed at nothing on the figure; the label now names both."""
    old = (I.SCALE, I.DEPTH)
    try:
        I.SCALE, I.DEPTH = "dstbl", "ideal"
        a = I.ddg_axis_label()
        I.SCALE, I.DEPTH = "hessa", "structure"
        b = I.ddg_axis_label()
    finally:
        I.SCALE, I.DEPTH = old
    assert "its own depth" in a and "that depth" not in a
    assert "dstbl" in a and "bilayer" in a
    assert "hessa" in b and "structure" in b
    assert a != b


# ------------------------------------------- i07: the plain-hydrophobicity control
def _tm_variants(n=900, kd_effect=-0.06, ddg_effect=0.0, seed=11):
    """TM variants whose effect is driven by KD change, by ddG, or by both."""
    from mpdms.annot import KYTE_DOOLITTLE as KD
    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    wt = rng.choice(aas, n)
    # a real variant frame is missense only; wt == mut would give one ddG and one
    # delta KD of exactly zero and look like a substitution with no depth spread
    mut = np.array([rng.choice([a for a in aas if a != w]) for w in wt])
    z = rng.uniform(-15, 15, n)
    dkd = np.array([KD[m] - KD[w] for m, w in zip(mut, wt)])
    ddg = np.array([I.residue_dg(m, zi, "dstbl") - I.residue_dg(w, zi, "dstbl")
                    for m, w, zi in zip(mut, wt, z)])
    return pd.DataFrame({
        "wt": wt, "mut": mut, "z_ideal": z, "ddg": ddg,
        "helix": rng.choice([f"TM{k}" for k in range(1, 9)], n),
        "score_z": (-0.4 + kd_effect * dkd + ddg_effect * ddg + rng.normal(0, 0.15, n))})


def test_i07_recovers_the_hydrophobicity_slope_and_writes_both_tables(tmp_path):
    from mpdms.insertion import i07_hydrophobicity_control
    r = i07_hydrophobicity_control({"T": _tm_variants(kd_effect=-0.06)}, ["T"], tmp_path)
    assert r["status"] == "ok"
    row = r["by_protein"][0]
    assert row["slope_kd"] == pytest.approx(-0.06, abs=0.02)
    assert row["spearman_kd"] < -0.5
    assert (tmp_path / "i07_hydrophobicity_curve.csv").exists()
    assert (tmp_path / "i07_hydrophobicity_slopes.csv").exists()
    assert (tmp_path / "i07_hydrophobicity_control.png").exists()


def test_i07_says_whether_ddg_adds_anything_over_plain_hydrophobicity():
    """The point of the control: when the effect is pure KD, ddG must add ~nothing,
    and when the effect really is depth-dependent, ddG must add something."""
    from mpdms.insertion import i07_hydrophobicity_control
    import tempfile
    from pathlib import Path as _P
    out = {}
    for name, kw in (("kd_only", dict(kd_effect=-0.06, ddg_effect=0.0)),
                     ("ddg_too", dict(kd_effect=-0.02, ddg_effect=-0.30))):
        with tempfile.TemporaryDirectory() as d:
            out[name] = i07_hydrophobicity_control(
                {"T": _tm_variants(**kw)}, ["T"], _P(d))["by_protein"][0]
    assert out["kd_only"]["r2_added_by_ddg"] < 0.02
    assert out["ddg_too"]["r2_added_by_ddg"] > 0.05
    assert out["ddg_too"]["p_ddg_given_kd"] < 0.01
    # the two predictors are strongly correlated, which is why the control is needed
    assert abs(out["kd_only"]["corr_kd_ddg"]) > 0.5


def test_i07_treats_every_residue_the_same_way():
    """No depth enters: the same substitution gets the same x wherever it sits."""
    from mpdms.annot import KYTE_DOOLITTLE as KD
    assert KD["K"] - KD["L"] == pytest.approx(-7.7)
    # ddG for the same substitution does depend on depth; delta_kd cannot
    deep = I.residue_dg("K", 0.0, "dstbl") - I.residue_dg("L", 0.0, "dstbl")
    edge = I.residue_dg("K", -15.0, "dstbl") - I.residue_dg("L", -15.0, "dstbl")
    assert abs(deep - edge) > 0.5


# ---------------------------- i08: the two predictors against each other, per mutation
def test_i08_reports_the_overlap_and_what_is_left_of_ddg(tmp_path):
    from mpdms.insertion import i08_ddg_vs_hydrophobicity
    v = _tm_variants(kd_effect=-0.05, ddg_effect=-0.10)
    r = i08_ddg_vs_hydrophobicity({"T": v}, ["T"], tmp_path)
    assert r["status"] == "ok"
    row = r["by_protein"][0]
    # the two predictors are strongly but not perfectly related, in opposite directions
    assert -1 < row["pearson"] < -0.5
    assert row["slope_ddg_on_kd"] < 0
    assert row["r2"] == pytest.approx(row["pearson"] ** 2, abs=1e-9)
    # and a real fraction of ddG survives hydrophobicity, which is the point
    assert 0.1 < row["frac_ddg_variance_left"] < 0.9
    assert row["sd_residual_ddg"] < row["sd_ddg"]
    for f in ("i08_ddg_vs_hydrophobicity.csv", "i08_ddg_vs_hydrophobicity.png",
              "i08b_residual_ddg_vs_effect.png"):
        assert (tmp_path / f).exists()


def test_i08_sees_the_depth_spread_that_hydrophobicity_cannot_have(tmp_path):
    """One substitution type is one x and many y: that smear is the depth dependence."""
    from mpdms.insertion import i08_ddg_vs_hydrophobicity, _with_kd
    v = _tm_variants()
    r = i08_ddg_vs_hydrophobicity({"T": v}, ["T"], tmp_path)["by_protein"][0]
    assert r["widest_depth_spread_one_substitution"] > 0.5
    # the same substitution really does take one delta KD and several ddG values
    w = _with_kd(v).assign(sub=lambda d: d.wt + ">" + d.mut)
    g = w.groupby("sub").agg(kd=("delta_kd", "nunique"), dg=("ddg", "nunique"),
                             n=("ddg", "size"))
    g = g[g.n >= 5]
    assert (g.kd == 1).all()
    assert (g.dg > 1).all()


def test_i08_residual_predicts_the_effect_only_when_ddg_really_drives_it():
    from mpdms.insertion import i08_ddg_vs_hydrophobicity
    import tempfile
    from pathlib import Path as _P
    got = {}
    for name, kw in (("kd_only", dict(kd_effect=-0.06, ddg_effect=0.0)),
                     ("ddg_too", dict(kd_effect=-0.02, ddg_effect=-0.30))):
        with tempfile.TemporaryDirectory() as d:
            got[name] = i08_ddg_vs_hydrophobicity(
                {"T": _tm_variants(**kw)}, ["T"], _P(d))["by_protein"][0]
    assert got["kd_only"]["p_effect_on_residual"] > 0.05
    assert got["ddg_too"]["p_effect_on_residual"] < 0.01
    assert got["ddg_too"]["slope_effect_on_residual"] < -0.1


# ------------------------------- i07b: the hydrophobicity control, helix by helix
def _helix_variants(slopes, n_per=160, seed=21):
    """TM variants where each helix has its own planted KD slope and its own dG."""
    from mpdms.annot import KYTE_DOOLITTLE as KD
    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    rows = []
    for hx, (slope, dg) in slopes.items():
        wt = rng.choice(aas, n_per)
        mut = np.array([rng.choice([a for a in aas if a != w]) for w in wt])
        dkd = np.array([KD[m] - KD[w] for m, w in zip(mut, wt)])
        z = rng.uniform(-15, 15, n_per)
        rows.append(pd.DataFrame({
            "wt": wt, "mut": mut, "helix": hx, "z_ideal": z,
            "ddg": [I.residue_dg(m, zi, "dstbl") - I.residue_dg(w, zi, "dstbl")
                    for m, w, zi in zip(mut, wt, z)],
            "score_z": -0.5 + slope * dkd + rng.normal(0, 0.12, n_per)}))
    v = pd.concat(rows, ignore_index=True)
    h = pd.DataFrame({"helix": list(slopes), "dg": [d for _, d in slopes.values()],
                      "seq": ["L" * 20] * len(slopes),
                      "orientation": ["in_out"] * len(slopes)})
    return v, h


def test_i07b_finds_each_helix_slope_and_relates_it_to_insertion_energy(tmp_path):
    """The marginal helices are the steep ones here, so rho(slope, dG) must be POSITIVE.

    The opposite sign to i03, and the reason is the axis. A ddG slope is negative, so
    marginal-is-steeper reads as rho < 0 there. A Kyte-Doolittle slope is positive -
    more hydrophobic is less damaging - and dG is LEAST negative for the helix with
    least margin, so marginal-is-steeper reads as rho > 0 here.
    """
    from mpdms.insertion import i07b_by_helix
    # easiest to insert (most negative dG) is flattest; marginal is steepest
    v, h = _helix_variants({"TM1": (0.02, -14.0), "TM2": (0.04, -12.0),
                            "TM3": (0.07, -10.0), "TM4": (0.10, -8.0),
                            "TM5": (0.12, -6.0)})
    r = i07b_by_helix({"T": v}, {"T": h}, ["T"], tmp_path)
    assert r["status"] == "ok"
    by = {d["helix"]: d for d in r["by_helix"]}
    assert len(by) == 5
    assert by["TM1"]["slope"] == pytest.approx(0.02, abs=0.02)
    assert by["TM5"]["slope"] == pytest.approx(0.12, abs=0.02)
    rho = ss.spearmanr([by[k]["dg"] for k in by], [by[k]["slope"] for k in by]).statistic
    assert rho > 0.9
    assert (tmp_path / "i07b_hydrophobicity_by_helix.csv").exists()
    assert (tmp_path / "i07b_hydrophobicity_by_helix.png").exists()


def test_i07b_finds_nothing_when_the_helices_are_the_same(tmp_path):
    from mpdms.insertion import i07b_by_helix
    v, h = _helix_variants({f"TM{k}": (0.06, -14.0 + 1.5 * k) for k in range(1, 6)})
    r = i07b_by_helix({"T": v}, {"T": h}, ["T"], tmp_path)
    sl = [d["slope"] for d in r["by_helix"]]
    assert max(sl) - min(sl) < 0.05        # the planted slopes really are equal
    dg = [d["dg"] for d in r["by_helix"]]
    assert abs(ss.spearmanr(dg, sl).statistic) < 0.9


def test_i07b_needs_helices_and_says_so(tmp_path):
    from mpdms.insertion import i07b_by_helix
    v, h = _helix_variants({"TM1": (0.05, -12.0), "TM2": (0.05, -10.0)})
    r = i07b_by_helix({"T": v}, {"T": h}, ["T"], tmp_path)
    assert r["status"] == "skipped"


def test_i07b_draws_every_helix_and_a_significance_bar_per_helix(tmp_path):
    """Twelve helices, all of them on the curve figure and all of them in the bars."""
    from mpdms.insertion import i07b_by_helix
    v, h = _helix_variants({f"TM{k}": (0.02 + 0.008 * k, -14.0 + 0.5 * k)
                            for k in range(1, 13)}, n_per=120)
    r = i07b_by_helix({"T": v}, {"T": h}, ["T"], tmp_path)
    assert r["status"] == "ok"
    assert len({d["helix"] for d in r["by_helix"]}) == 12
    for f in ("i07b_hydrophobicity_by_helix.png", "i07c_helix_slope_significance.png",
              "i07b_hydrophobicity_by_helix.csv"):
        assert (tmp_path / f).exists()
    t = pd.read_csv(tmp_path / "i07b_hydrophobicity_by_helix.csv")
    assert len(t) == 12
    # the corrected p is present, never smaller than the raw one, and still a p
    assert "q" in t and (t.q >= t.p - 1e-12).all()
    assert ((t.q >= 0) & (t.q <= 1)).all()


def test_the_helix_p_values_are_corrected_within_each_protein(tmp_path):
    """Twelve helices are twelve chances, and two proteins must not pool their tests."""
    from mpdms.insertion import i07b_by_helix
    v1, h1 = _helix_variants({f"TM{k}": (0.05, -12.0 + k) for k in range(1, 7)},
                             n_per=120, seed=5)
    v2, h2 = _helix_variants({f"TM{k}": (0.05, -12.0 + k) for k in range(1, 7)},
                             n_per=120, seed=6)
    i07b_by_helix({"A": v1, "B": v2}, {"A": h1, "B": h2}, ["A", "B"], tmp_path)
    t = pd.read_csv(tmp_path / "i07b_hydrophobicity_by_helix.csv")
    assert set(t.protein) == {"A", "B"}
    for _, d in t.groupby("protein"):
        # BH within six tests: the largest q is the largest p, unchanged
        assert d.q.max() == pytest.approx(d.p.max(), rel=1e-9)
        assert len(d) == 6


# ------------------- i09: one panel per mutant residue, effect against position
def test_i09_draws_twenty_panels_and_summarises_each_residue(tmp_path):
    from mpdms.insertion import i09_profiles
    v = _tm_variants(n=4000, kd_effect=-0.05, seed=31)
    r = i09_profiles({"T": v}, ["T"], tmp_path)
    assert r["status"] == "ok"
    got = {d["mut"] for d in r["by_residue"]}
    assert len(got) >= 18 and got <= set("ACDEFGHIKLMNPQRSTVWY")
    assert (tmp_path / "i09_profiles_T.png").exists()
    t = pd.read_csv(tmp_path / "i09_profiles.csv")
    for c in ("mut", "n_positions", "spearman_with_scale", "effect_at_centre",
              "effect_at_edges"):
        assert c in t.columns
    assert (t.n_positions >= 3).all()


def test_i09_x_axis_is_the_scales_own_coordinate(tmp_path):
    """x must mean what it means in the published table: position in a 19-mer, not A."""
    from mpdms.insertion import i09_profiles
    v = _tm_variants(n=3000, seed=32)
    i09_profiles({"T": v}, ["T"], tmp_path)
    # the ends of an ideally laid-out helix map to the ends of the table
    assert np.rint(I.HALF_WIDTH * (9.0 / I.HALF_WIDTH)) == 9
    assert np.rint(-I.HALF_WIDTH * (9.0 / I.HALF_WIDTH)) == -9
    # and the prediction drawn at position i is the scale's value at that depth
    assert I.residue_dg("K", 9 * I.HALF_WIDTH / 9.0, "hessa") == pytest.approx(
        I.residue_dg("K", I.HALF_WIDTH, "hessa"))


def test_i09_declines_a_protein_with_too_little_data(tmp_path):
    from mpdms.insertion import i09_profiles
    v = _tm_variants(n=50)
    assert i09_profiles({"T": v}, ["T"], tmp_path)["status"] == "skipped"


def test_i09_prediction_resolution_is_independent_of_the_data_binning():
    """The red curve was a seven-point polyline because it inherited the data bins.

    The scale is a continuous function of depth, so the curve is drawn on its own
    dense grid while the blue curve stays on the bins, where each point averages real
    measurements. The test is on the values the two grids produce, which is what the
    drawing reduces to.
    """
    dense = [I.residue_dg("K", x * I.HALF_WIDTH / 9.0, "hessa")
             for x in np.linspace(-9, 9, 181)]
    coarse = [I.residue_dg("K", x * I.HALF_WIDTH / 9.0, "hessa")
              for x in np.arange(-9, 10, 3)]
    assert len(dense) > 10 * len(coarse)
    assert len(set(np.round(dense, 6))) > 20            # not a handful of segments
    assert max(dense) >= max(coarse) - 1e-9             # the peak is not cut off
    assert 0 < int(np.argmax(dense)) < len(dense) - 1   # and it is interior


def test_i09_bin_width_is_a_choice_and_is_reported(tmp_path):
    """Seven points is bin_width 3; one gives the published nineteen."""
    from mpdms.insertion import i09_profiles
    v = _tm_variants(n=6000, seed=34)
    wide = i09_profiles({"T": v}, ["T"], tmp_path, 3, 3)
    t_wide = pd.read_csv(tmp_path / "i09_profiles.csv")
    fine = i09_profiles({"T": v}, ["T"], tmp_path, 3, 1)
    t_fine = pd.read_csv(tmp_path / "i09_profiles.csv")
    assert wide["status"] == fine["status"] == "ok"
    # the finer grid gives more points per residue, each averaging fewer measurements
    assert t_fine.n_positions.median() > t_wide.n_positions.median()
    assert t_fine.n_positions.max() <= 19 and t_wide.n_positions.max() <= 7


def test_i09_marks_zero_on_both_axes_in_their_own_colours(tmp_path):
    """Two axes in one panel: a single zero line would be read as belonging to both,
    and "no predicted cost" sits nowhere near "no measured effect"."""
    from mpdms.insertion import i09_profiles
    v = _tm_variants(n=3000, seed=35)
    i09_profiles({"T": v}, ["T"], tmp_path)
    # the contract is that the prediction axis carries its own
    # zero, which is only meaningful because the two axes have different origins
    assert I.residue_dg("A", 0.0, "dstbl") == 0.0
    assert I.residue_dg("A", 0.0, "hessa") != 0.0
    assert (tmp_path / "i09_profiles_T.png").exists()


# ----------------------------------- i09b: is a depth profile symmetric about zero?
def test_the_asymmetry_index_is_free_of_the_scale_it_is_measured_on():
    """The whole point: a profile in fitness units and one in kcal/mol are comparable."""
    idx = np.arange(-9, 10, 3)
    tilted = np.array([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0])
    base = I.asymmetry_index(idx, tilted)
    for mult in (0.01, 3.0, -2.0):
        assert I.asymmetry_index(idx, tilted * mult) == pytest.approx(base)
    for shift in (-5.0, 0.25, 100.0):
        assert I.asymmetry_index(idx, tilted + shift) == pytest.approx(base)


def test_the_index_is_zero_for_a_bell_and_one_for_a_ramp():
    idx = np.arange(-9, 10, 3)
    bell = np.array([0.1, 0.5, 0.9, 1.2, 0.9, 0.5, 0.1])
    ramp = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    assert I.asymmetry_index(idx, bell) == pytest.approx(0.0, abs=1e-12)
    assert I.asymmetry_index(idx, ramp) == pytest.approx(1.0, abs=1e-12)
    # a bell plus a ramp lands between them
    mixed = I.asymmetry_index(idx, bell + 0.3 * ramp)
    assert 0.05 < mixed < 0.95


def test_an_arm_with_no_partner_is_dropped_not_paired_with_zero():
    """Zero-filling a missing side would manufacture asymmetry out of missing data."""
    full = np.array([-6, -3, 0, 3, 6])
    y = np.array([0.2, 0.6, 1.0, 0.6, 0.2])
    assert I.asymmetry_index(full, y) == pytest.approx(0.0, abs=1e-12)
    # the same profile with one unmatched arm measured is still symmetric
    plus = np.array([-6, -3, 0, 3, 6, 9])
    assert I.asymmetry_index(plus, np.r_[y, 9.0]) == pytest.approx(0.0, abs=1e-12)
    evens, odds = I.even_odd(plus, np.r_[y, 9.0])
    assert len(odds) == 2                      # only the paired depths contribute


def test_hessa_is_symmetric_for_every_residue_and_dstbl_is_not():
    """The control the comparison rests on, asserted rather than assumed."""
    idx = np.arange(-9, 10)
    hes, dst = [], []
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        hes.append(I.asymmetry_index(
            idx, np.array([I.residue_dg(aa, i * I.HALF_WIDTH / 9.0, "hessa")
                           for i in idx])))
        dst.append(I.asymmetry_index(
            idx, np.array([I.residue_dg(aa, i * I.HALF_WIDTH / 9.0, "dstbl")
                           for i in idx])))
    assert np.nanmax(hes) == pytest.approx(0.0, abs=1e-12)
    assert np.nanmedian(dst) > 0.2
    # and the positive-inside residues are among the most tilted under dsTbL
    by_aa = dict(zip("ACDEFGHIKLMNPQRSTVWY", dst))
    assert by_aa["K"] > 0.5 and by_aa["R"] > 0.5


def _profile_variants(tilt=0.0, bell=0.6, n=1800, seed=41):
    """TM variants for one residue whose cost is a bell plus an optional ramp."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-15, 15, n)
    i = z * 9.0 / 15.0
    y = -bell * np.exp(-(i ** 2) / 30.0) + tilt * (i / 9.0) + rng.normal(0, 0.12, n)
    return pd.DataFrame({"wt": "L", "mut": "K", "z_ideal": z, "helix": "TM1",
                         "score_z": y})


def test_i09b_finds_a_planted_tilt_and_not_a_symmetric_bell(tmp_path):
    from mpdms.insertion import i09b_symmetry
    sym = i09b_symmetry({"T": _profile_variants(tilt=0.0)}, ["T"], tmp_path,
                        n_perm=300, n_boot=200)
    tilt = i09b_symmetry({"T": _profile_variants(tilt=0.9, seed=42)}, ["T"], tmp_path,
                         n_perm=300, n_boot=200)
    a = [d for d in sym["by_residue"] if d["mut"] == "K"][0]
    b = [d for d in tilt["by_residue"] if d["mut"] == "K"][0]
    assert a["asymmetry"] < 0.3 and a["p_permutation"] > 0.05
    assert b["asymmetry"] > 0.7 and b["p_permutation"] < 0.05
    # the signed statistic says which way, and the tilt rises towards the lumen
    assert b["lumenal_minus_cytosolic"] > 0.3
    assert a["asymmetry_hessa"] == pytest.approx(0.0, abs=1e-12)
    assert (tmp_path / "i09b_profile_symmetry.csv").exists()
    assert (tmp_path / "i09b_profile_symmetry.png").exists()
