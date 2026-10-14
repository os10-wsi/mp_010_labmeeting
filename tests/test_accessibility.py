"""Solvent accessibility: the geometry and the decompositions, not the plotting."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms import accessibility as A


# ------------------------------------------------------------- r03 the breakpoint
def _hinge(n=400, knee=0.22, below=-3.0, above=-0.2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = 0.5 + below * np.minimum(x, knee) + above * np.maximum(x - knee, 0)
    return x, y + rng.normal(0, 0.05, n)


def test_the_breakpoint_search_finds_a_planted_knee():
    x, y = _hinge(knee=0.22, below=-3.0, above=-0.2)
    f = A.segmented(x, y)
    # the residual surface is flat near the knee, so the estimate carries real
    # uncertainty - which is exactly why r03 bootstraps it rather than quoting a point
    assert f["breakpoint"] == pytest.approx(0.22, abs=0.06)
    assert f["below"] == pytest.approx(-3.0, abs=0.5)
    assert f["above"] == pytest.approx(-0.2, abs=0.3)
    # the hinge must beat the straight line it is being compared against
    assert f["rss"] < 0.5 * f["rss_linear"]


def test_a_straight_line_gives_the_hinge_nothing_to_find():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 400)
    y = 0.5 - 1.2 * x + rng.normal(0, 0.05, 400)
    f = A.segmented(x, y)
    assert f["below"] == pytest.approx(f["above"], abs=0.25)
    assert f["rss"] == pytest.approx(f["rss_linear"], rel=0.05)
    assert f["linear_slope"] == pytest.approx(-1.2, abs=0.05)


def test_the_breakpoint_is_never_proposed_where_there_is_no_data():
    """A knee outside the observed range would be fitted on nothing."""
    x, y = _hinge()
    f = A.segmented(x, y)
    assert x.min() < f["breakpoint"] < x.max()
    assert (x < f["breakpoint"]).sum() >= 10
    assert (x >= f["breakpoint"]).sum() >= 10


# ------------------------------------------------- r05 the variance decomposition
def test_shapley_shares_add_up_to_the_models_r_squared():
    rng = np.random.default_rng(2)
    n = 300
    a = rng.normal(0, 1, n)
    b = 0.8 * a + 0.6 * rng.normal(0, 1, n)      # deliberately correlated
    c = rng.normal(0, 1, n)
    d = pd.DataFrame({"a": a, "b": b, "c": c,
                      "y": 1.0 * a + 0.5 * b + 0.3 * c + rng.normal(0, 1, n)})
    out = A.lmg_shares(d, "y", ["a", "b", "c"])
    assert sum(out["share"].values()) == pytest.approx(out["r2"], abs=1e-10)
    assert all(v >= 0 for v in out["share"].values())


def test_a_predictor_that_duplicates_another_splits_the_credit():
    """Two copies of the same information must not each be given all of it."""
    rng = np.random.default_rng(3)
    n = 400
    a = rng.normal(0, 1, n)
    d = pd.DataFrame({"a": a, "a_copy": a + rng.normal(0, 1e-3, n),
                      "y": 2.0 * a + rng.normal(0, 1, n)})
    out = A.lmg_shares(d, "y", ["a", "a_copy"])
    assert out["share"]["a"] == pytest.approx(out["share"]["a_copy"], abs=0.01)
    # each explains half the shared R², and neither adds anything entering last
    assert out["share"]["a"] == pytest.approx(out["r2"] / 2, abs=0.01)
    assert out["unique"]["a"] < 0.01


def test_orthogonal_predictors_have_share_equal_to_unique():
    """With nothing shared there is nothing to apportion, so the two agree exactly.

    The predictors are orthogonalised rather than merely drawn independently: at any
    finite sample size two independent draws still correlate a little, and the identity
    being tested here is an algebraic one that only holds at exactly zero.
    """
    rng = np.random.default_rng(4)
    n = 500
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    a, b = a - a.mean(), b - b.mean()
    b = b - (b @ a) / (a @ a) * a                      # exactly orthogonal to a
    d = pd.DataFrame({"a": a, "b": b, "y": a + 0.5 * b + rng.normal(0, 1, n)})
    out = A.lmg_shares(d, "y", ["a", "b"])
    for k in ("a", "b"):
        assert out["share"][k] == pytest.approx(out["unique"][k], abs=1e-9)


def test_the_unique_contribution_never_exceeds_the_share_when_predictors_overlap():
    rng = np.random.default_rng(5)
    n = 300
    a = rng.normal(0, 1, n)
    b = 0.9 * a + 0.4 * rng.normal(0, 1, n)
    d = pd.DataFrame({"a": a, "b": b, "y": a + b + rng.normal(0, 1, n)})
    out = A.lmg_shares(d, "y", ["a", "b"])
    for k in ("a", "b"):
        assert out["unique"][k] <= out["share"][k] + 1e-9


# --------------------------------------------------------- r04 the neighbourhood
def test_neighbour_rsa_skips_the_sequence_neighbours():
    rows = [{"pos": 10, "rsa": 0.9, "x": 0.0, "y": 0.0, "z": 0.0}]
    for k in list(range(6, 10)) + list(range(11, 15)):
        rows.append({"pos": k, "rsa": 0.9, "x": 1.0, "y": 0.0, "z": 0.0})
    rows.append({"pos": 60, "rsa": 0.1, "x": 5.0, "y": 0.0, "z": 0.0})
    rows.append({"pos": 80, "rsa": 0.5, "x": 40.0, "y": 0.0, "z": 0.0})
    got = A.neighbour_rsa(pd.DataFrame(rows), cutoff=8.0, exclude=4)
    assert got.iloc[0] == pytest.approx(0.1)


# ------------------------------------------------- r02 splitting the exposed surface
def _slab_pdb(path, n=30, spacing=3.0):
    """An extended poly-alanine run along z, so some residues sit in the slab and some
    sit well outside it."""
    lines, serial = [], 1
    for i in range(1, n + 1):
        z = (i - 1) * spacing
        for name, dx, dy, el in ((" N  ", -1.2, 0.0, "N"), (" CA ", 0.0, 0.0, "C"),
                                 (" C  ", 1.2, 0.0, "C"), (" O  ", 1.4, 1.1, "O"),
                                 (" CB ", 0.0, 1.5, "C")):
            lines.append(f"ATOM  {serial:>5} {name:<4} ALA A{i:>4}    "
                         f"{dx:8.3f}{dy:8.3f}{z:8.3f}  1.00 90.00          {el:>2}\n")
            serial += 1
    path.write_text("".join(lines) + "END\n")
    return path


class _Cfg:
    def __init__(self, segs):
        self.segs = segs

    def get_path(self, key, default=None):
        if key == "topology.segments":
            return self.segs
        if key == "structure.membrane_normal":
            return "pca_tm_axes"
        if key == "structure.chain":
            return "A"
        return default


def test_the_exposed_surface_is_split_by_where_the_atoms_are(tmp_path):
    """An atom inside the bilayer slab is against lipid; one outside it is in water."""
    pdb = _slab_pdb(tmp_path / "slab.pdb")
    cfg = _Cfg([{"name": "TM1", "type": "TM", "start": 6, "end": 25,
                 "orientation": "in_out"}])
    g = A.split_accessibility(pdb, cfg)
    assert len(g) == 30
    half = g.attrs["half_thickness"]
    assert half > 0
    mid = g[g.pos == 15].iloc[0]
    end = g[g.pos == 1].iloc[0]
    # a residue in the middle of the span is wholly inside the slab
    assert mid.asa_lipid > 0 and mid.asa_water == pytest.approx(0.0, abs=1e-9)
    # a residue at the far end of the chain is wholly outside it
    assert end.asa_water > 0 and end.asa_lipid == pytest.approx(0.0, abs=1e-9)


def test_the_two_halves_add_back_up_to_the_whole_surface(tmp_path):
    """Splitting the area must not create or destroy any of it."""
    from mpdms.structure import residue_table
    pdb = _slab_pdb(tmp_path / "slab.pdb")
    cfg = _Cfg([{"name": "TM1", "type": "TM", "start": 6, "end": 25,
                 "orientation": "in_out"}])
    g = A.split_accessibility(pdb, cfg)
    at = A.atom_sasa(pdb)
    total = at.groupby("pos").sasa.sum()
    got = (g.set_index("pos").asa_lipid + g.set_index("pos").asa_water)
    assert np.allclose(got.sort_index().to_numpy(),
                       total.sort_index().to_numpy(), atol=1e-6)
    # and the relative versions are bounded, since Tien's max ASA is only a reference
    rt = residue_table(pdb)
    assert (g.rsa_lipid <= 1.0).all() and (g.rsa_water <= 1.0).all()
    assert len(rt) == 30


# ------------------------------------------------------------------- housekeeping
def test_the_added_columns_do_not_shadow_dataframe_methods():
    """`corr` and `skew` have both shadowed a method in this project already.

    Attribute access on a column whose name is also a DataFrame method hands back the
    bound method with no error at all, and the failure surfaces much later as a
    comparison between two functions.
    """
    added = ["rsa_env", "asa_lipid", "asa_water", "rsa_lipid", "rsa_water",
             "sd", "skewness", "frac_dead", "n_scored", "wt_struct", "rsa_alt"]
    clashes = [c for c in added if callable(getattr(pd.DataFrame, c, None))]
    assert clashes == []


def test_label_colour_is_chosen_from_the_fill_not_from_the_stack_order():
    assert A.readable_on("#440154") == "white"       # dark viridis purple
    assert A.readable_on("#5ec962") == "#222222"     # light viridis green
    assert A.readable_on("#ffffff") == "#222222"


# ------------------------------- r11: K/R, proline and glycine against RSA in the TMs
def _tm_vars(n_helix=8, per_helix=14, kr_slope=0.9, pro_slope=0.1, seed=4):
    """TM variants where K/R depends on burial and proline barely does."""
    rng = np.random.default_rng(seed)
    rows, pos = [], 0
    for h in range(n_helix):
        for _ in range(per_helix):
            pos += 1
            rsa = float(rng.uniform(0, 0.8))
            for mut, slope in (("K", kr_slope), ("R", kr_slope), ("P", pro_slope),
                               ("G", 0.5)):
                rows.append({"pos": pos, "wt": "L", "mut": mut, "rsa": rsa,
                             "is_tm": 1.0, "helix": f"TM{h + 1}",
                             "score_z": -1.2 + slope * rsa + rng.normal(0, 0.08)})
    # loop positions, which r11 must ignore entirely
    for k in range(60):
        pos += 1
        rows.append({"pos": pos, "wt": "L", "mut": "K", "rsa": 0.9, "is_tm": 0.0,
                     "helix": None, "score_z": 0.0})
    return pd.DataFrame(rows)


def test_r11_recovers_the_planted_slopes_and_uses_tm_positions_only(tmp_path):
    from mpdms.accessibility import r11_breakers_vs_rsa
    v = _tm_vars()
    r = r11_breakers_vs_rsa({"T": {"var": v}}, ["T"], tmp_path)
    assert r["status"] == "ok"
    by = {d["substitution"]: d for d in r["by_class"]}
    assert set(by) == {"K or R", "proline", "glycine"}
    assert by["K or R"]["slope"] == pytest.approx(0.9, abs=0.15)
    assert by["proline"]["slope"] == pytest.approx(0.1, abs=0.15)
    assert by["K or R"]["slope"] > by["proline"]["slope"]
    # a position is the unit, so the 112 TM positions count once each even though
    # K and R are both measured at every one of them
    assert by["K or R"]["n_positions"] == 112
    assert by["K or R"]["n_variants"] == 224
    # the 60 exposed loop positions would flatten every slope if they leaked in
    assert by["K or R"]["mean_effect"] < -0.5
    assert (tmp_path / "r11_breakers_vs_rsa.csv").exists()
    assert (tmp_path / "r11_breakers_vs_rsa.png").exists()


def test_r11_says_which_error_it_reports():
    """Eight helices cluster; three fall back, and the row has to admit it."""
    from mpdms.accessibility import r11_breakers_vs_rsa
    import tempfile
    from pathlib import Path as _P
    with tempfile.TemporaryDirectory() as d:
        many = r11_breakers_vs_rsa({"T": {"var": _tm_vars(n_helix=8)}}, ["T"], _P(d))
    with tempfile.TemporaryDirectory() as d:
        few = r11_breakers_vs_rsa({"T": {"var": _tm_vars(n_helix=3, per_helix=20)}},
                                  ["T"], _P(d))
    assert all(x["se_kind"] == "clustered by helix" for x in many["by_class"])
    assert all("too few helices" in x["se_kind"] for x in few["by_class"])
    assert all(np.isfinite(x["slope"]) for x in many["by_class"] + few["by_class"])


def test_r11_declines_a_class_it_cannot_measure(tmp_path):
    from mpdms.accessibility import r11_breakers_vs_rsa
    v = _tm_vars()
    v = v[v.mut != "P"]                       # no proline measured anywhere
    r = r11_breakers_vs_rsa({"T": {"var": v}}, ["T"], tmp_path)
    assert {d["substitution"] for d in r["by_class"]} == {"K or R", "glycine"}
