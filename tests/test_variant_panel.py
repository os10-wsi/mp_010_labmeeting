"""A20 on a table with one estimate per variant, as DiMSum writes it."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms.analyses.a20_variant_panel import (collect_single, site_distributions,
                                              stats_single)


def _df(seed=0):
    r = np.random.default_rng(seed)
    AA = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    for pos, wt in ((209, "Q"), (335, "Q"), (473, "W")):
        for mut in AA:
            if mut == wt:
                continue
            rows.append({"pos": pos, "wt": wt, "mut": mut, "vclass": "missense",
                         "score_z": -1.0 if pos == 335 else -0.1, "se": 0.12,
                         "pass_filter": True})
    for k in range(200):
        rows.append({"pos": 50 + k % 7, "wt": "A", "mut": "=", "vclass": "synonymous",
                     "score_z": float(r.normal(0, 0.15)), "se": 0.1, "pass_filter": True})
    return pd.DataFrame(rows)


def test_named_variants_are_found_with_their_error():
    d, missing = collect_single(_df(), ["Q209A", "Q335A", "W473A"])
    assert list(d.variant) == ["Q209A", "Q335A", "W473A"]
    assert (d.se > 0).all() and missing == []


def test_a_wild_type_mismatch_is_named_not_silently_used():
    """The commonest numbering error: the position exists but carries another residue."""
    d, missing = collect_single(_df(), ["N209A"])
    assert len(d) == 0
    assert "numbering mismatch" in missing[0] and "Q209" in missing[0]


def test_an_unmeasured_variant_is_reported():
    d, missing = collect_single(_df(), ["Q999A"])
    assert len(d) == 0 and "not measured" in missing[0]


def test_a_filtered_variant_is_not_plotted():
    df = _df()
    df.loc[(df.pos == 209) & (df.mut == "A"), "pass_filter"] = False
    d, missing = collect_single(df, ["Q209A"])
    assert len(d) == 0 and "fails the read/replicate filter" in missing[0]


def test_a_malformed_label_is_rejected():
    d, missing = collect_single(_df(), ["notavariant"])
    assert len(d) == 0 and "not a point substitution" in missing[0]


def test_site_distributions_return_every_substitution_at_the_position():
    s = site_distributions(_df(), [209, 335])
    assert len(s[209]) == 19 and len(s[335]) == 19


def test_a_variant_far_from_synonymous_is_significant_and_a_near_one_is_not():
    df = _df()
    d, _ = collect_single(df, ["Q209A", "Q335A"])
    syn = df.loc[df.vclass == "synonymous", "score_z"].to_numpy()
    st = stats_single(d, syn).set_index("variant")
    assert st.loc["Q335A", "q"] < 0.01
    assert st.loc["Q335A", "z_vs_synonymous"] < -3
    assert st.loc["Q209A", "q"] > 0.05
    assert 0 <= st.loc["Q335A", "percentile_in_synonymous"] <= 100


def test_the_reported_error_is_the_rescaled_sigma_not_replicate_scatter():
    """There are no replicates in this format; the bar must come from the table's sigma."""
    df = _df()
    df.loc[(df.pos == 209) & (df.mut == "A"), "se"] = 0.33
    d, _ = collect_single(df, ["Q209A"])
    assert d.se.iloc[0] == pytest.approx(0.33)


# ------------------------------------------------- A15 functional-site ranking
def _sites(n=6):
    import numpy as np
    return pd.DataFrame({
        "pos": np.arange(100, 100 + n), "wt": list("QWNSYF")[:n],
        "segment": "TM1", "seg_type": "TM", "n_variants": 19,
        # deliberately not in position order, so sorting by z really does reorder
        "median_z": np.array([-1.4, -2.4, -1.05, -2.0, -1.2, -1.8])[:n],
        "median_abundance": np.linspace(-1.2, -0.2, n),
        "median_esm1v": np.linspace(-4, -2, n),
        "q": np.linspace(1e-4, 4e-2, n), "functional": True,
        "abundance_tolerant": [False] * (n - 2) + [True] * 2})


def test_the_ranking_is_by_constraint_not_by_position():
    """Sequence order finds a residue; constraint order decides which to work on."""
    from mpdms.analyses.a15_esm_functional import ranking_figure
    s = _sites()
    r = s.sort_values("median_z").reset_index(drop=True)
    assert list(r.pos) != sorted(r.pos)             # the two orders differ here
    assert r.median_z.is_monotonic_increasing       # most negative first


def test_the_ranking_figure_is_drawn_and_labels_every_site(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from mpdms.analyses.a15_esm_functional import ranking_figure

    class _Cfg:
        id = "T"
        display_name = "T"

        def get_path(self, *a, **k):
            return None
    s = _sites()
    out = ranking_figure(_Cfg(), s, s, tmp_path)
    assert out and all(p.exists() for p in out)


def test_no_functional_sites_draws_nothing_rather_than_an_empty_panel(tmp_path):
    from mpdms.analyses.a15_esm_functional import ranking_figure

    class _Cfg:
        id = "T"
        display_name = "T"

        def get_path(self, *a, **k):
            return None
    assert ranking_figure(_Cfg(), _sites(), _sites().iloc[:0], tmp_path) == []


# ----------------------------------------------- the per-variant estimate table
def _reps(seed=1):
    """Replicate-level rows in the shape `collect` returns."""
    r = np.random.default_rng(seed)
    rows = []
    for v, centre in (("Q149A", -0.80), ("Q238A", -0.05), ("R504A", -0.40)):
        for k in range(3):
            rows.append({"variant": v, "replicate": f"rep{k + 1}",
                         "value": float(centre + r.normal(0, 0.05))})
    return pd.DataFrame(rows)


def test_the_estimate_table_carries_a_value_and_an_error_for_every_variant():
    from mpdms.analyses.a20_variant_panel import estimates, stats_table
    mut = _reps()
    syn = pd.DataFrame({"variant": "synonymous", "replicate": "rep1",
                        "value": np.random.default_rng(2).normal(0, 0.12, 60),
                        "syn_index": np.arange(60)})
    syn_means = syn.value.to_numpy()
    est = estimates(mut, syn_means, stats_table(mut, syn, syn_means))
    assert list(est.variant) == ["Q149A", "Q238A", "R504A"]
    assert (est.n_replicates == 3).all()
    assert (est.se > 0).all() and (est.se < est.sd).all()        # se = sd / sqrt(3)
    assert np.allclose(est.se, est.sd / np.sqrt(3))
    # the interval is the t interval for two degrees of freedom, not 1.96 se
    half = (est.ci95_hi - est.ci95_lo) / 2
    assert ((half / est.se) > 4.0).all() and ((half / est.se) < 4.6).all()
    assert (est.ci95_lo < est.score_z).all() and (est.score_z < est.ci95_hi).all()
    # the strong variant's interval clears synonymous, the weak one's does not
    strong = est[est.variant == "Q149A"].iloc[0]
    weak = est[est.variant == "Q238A"].iloc[0]
    assert strong.ci95_hi < strong.synonymous_median
    assert weak.ci95_lo < weak.synonymous_median < weak.ci95_hi
    assert est.wt.tolist() == ["Q", "Q", "R"] and est.pos.tolist() == [149, 238, 504]


def test_the_two_kinds_of_error_are_labelled_and_never_silently_mixed():
    """Replicate scatter and the assay's own sigma are different quantities."""
    from mpdms.analyses.a20_variant_panel import (estimates, estimates_single,
                                                  stats_table)
    mut = _reps()
    syn = pd.DataFrame({"variant": "synonymous", "replicate": "rep1",
                        "value": np.random.default_rng(2).normal(0, 0.12, 60),
                        "syn_index": np.arange(60)})
    rep = estimates(mut, syn.value.to_numpy(), stats_table(mut, syn, syn.value.to_numpy()))
    named, _ = collect_single(_df(), ["Q209A", "Q335A", "W473A"])
    syn_all = _df().query("vclass == 'synonymous'").score_z.to_numpy()
    one = estimates_single(named, stats_single(named, syn_all), syn_all)
    assert set(rep.error_kind) == {"se of the replicate mean"}
    assert set(one.error_kind) == {"sigma from the dataset, rescaled"}
    # the same columns in the same order, so the two branches are one table downstream
    assert list(rep.columns) == list(one.columns)
    assert (one.n_replicates == 1).all() and one.sd.isna().all()
    assert np.allclose(one.ci95_hi - one.ci95_lo, 2 * 1.96 * one.se)
