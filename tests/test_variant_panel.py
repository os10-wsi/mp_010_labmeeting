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
