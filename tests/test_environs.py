"""Residue environments: the classification, the chemistry, and the clustering test."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms import environs as E


class _Cfg:
    id = "T"

    def __init__(self, segs):
        self.segs = segs

    def get_path(self, k, default=None):
        return {"topology.segments": self.segs}.get(k, default)


SEGS = [{"name": "N", "start": 1, "end": 5, "type": "loop", "side": "cytosolic"},
        {"name": "TM1", "start": 6, "end": 15, "type": "TM", "orientation": "in_out"},
        {"name": "L1", "start": 16, "end": 20, "type": "loop", "side": "lumenal"}]


def _pt(**over):
    n = 20
    t = pd.DataFrame({"pos": np.arange(1, n + 1), "is_tm": 0.0, "rsa": 0.5,
                      "radial": 10.0, "plddt": 95.0, "abs_z": 5.0})
    t.loc[t.pos.between(6, 15), "is_tm"] = 1.0
    for k, v in over.items():
        t[k] = v
    return t


def test_the_three_transmembrane_environments_are_separated_by_rsa_and_radius():
    t = _pt()
    t.loc[t.pos.isin([6, 7, 8]), "rsa"] = 0.02          # buried: helix-helix core
    t.loc[t.pos.isin([9, 10, 11]), "radial"] = 4.0      # exposed and near the axis: cavity
    t.loc[t.pos.isin([12, 13, 14, 15]), "radial"] = 18.0  # exposed and far out: lipid
    c = E.classify(t, _Cfg(SEGS))
    got = dict(zip(c.pos, c.environ))
    assert got[6] == "helix-helix core"
    assert got[9] == "cavity-lining"
    assert got[13] == "lipid-facing"
    assert got[1] == "cytosolic loop" and got[18] == "lumenal loop"
    # the radial cut is this protein's own median TM radius, and it is recorded
    assert c.attrs["radial_cut"] == pytest.approx(float(np.median(
        t.loc[t.pos.between(6, 15), "radial"])))


def test_a_residue_the_model_is_unsure_about_is_not_classified():
    t = _pt()
    t.loc[t.pos == 7, "plddt"] = 40.0
    c = E.classify(t, _Cfg(SEGS))
    assert pd.isna(c.loc[c.pos == 7, "environ"].iloc[0])
    assert c.attrs["n_low_plddt"] == 1


def test_a_position_outside_every_annotated_segment_is_left_alone():
    """Classes come from the topology, so a gap in it must not be guessed at."""
    t = _pt()
    c = E.classify(t, _Cfg(SEGS[:2]))          # positions 16-20 unannotated
    assert c.loc[c.pos > 15, "environ"].isna().all()


def test_substitution_direction_ignores_glycine_and_proline():
    wt = pd.Series(list("SLSLGP"))
    mut = pd.Series(list("LSSLLL"))
    d = E.direction(wt, mut)
    assert list(d[:4]) == ["polar → hydrophobic", "hydrophobic → polar",
                           "polar → polar", "hydrophobic → hydrophobic"]
    assert d.isna().iloc[4] and d.isna().iloc[5]


def _coords(tight: bool, seed=0):
    rng = np.random.default_rng(seed)
    bulk = rng.normal(0, 30, (200, 3))
    if tight:
        bulk[:10] = rng.normal(0, 2.5, (10, 3))
    return bulk


def test_a_planted_cluster_is_detected_and_a_scattered_set_is_not(monkeypatch):
    monkeypatch.setattr(E, "N_PERM", 2000)
    pool = np.arange(200)
    tight = E.network_stats(_coords(True), np.arange(10), pool, seed=1)
    loose = E.network_stats(_coords(False), np.arange(10), pool, seed=1)
    assert tight["p_clustered"] < 0.01 and tight["z"] < -2
    assert loose["p_clustered"] > 0.05
    assert tight["n_pairs"] == 45
    assert tight["observed_mean_distance"] < loose["observed_mean_distance"]


def test_the_matched_null_removes_clustering_the_topology_already_explains(monkeypatch):
    """Ten TM residues are close together because TM residues are close together."""
    monkeypatch.setattr(E, "N_PERM", 2000)
    rng = np.random.default_rng(2)
    xyz = np.vstack([rng.normal(0, 6, (40, 3)),        # a compact bundle: "TM"
                     rng.normal(0, 40, (160, 3))])     # loops, strung out
    strata = np.array(["TM"] * 40 + ["loop"] * 160)
    st = E.network_stats(xyz, np.arange(10), np.arange(200), strata=strata, seed=3)
    assert st["p_clustered"] < 0.01              # against any ten positions, yes
    assert st["p_clustered_matched"] > 0.05      # against ten other TM positions, no
    assert st["strata"] == "TM:10"


def test_the_matched_null_is_skipped_when_a_stratum_is_too_small(monkeypatch):
    monkeypatch.setattr(E, "N_PERM", 500)
    xyz = np.random.default_rng(0).normal(0, 20, (30, 3))
    strata = np.array(["TM"] * 8 + ["loop"] * 22)
    st = E.network_stats(xyz, np.arange(10), np.arange(30), strata=strata)
    assert "p_clustered_matched" not in st       # only 8 TM positions, 9 were picked


def test_the_command_runs_end_to_end_and_writes_what_it_says_it_does(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "N_PERM", 300)
    rc = E.main(["configs/SYN1.yaml", "configs/SYN2.yaml", "configs/SYN3.yaml",
                 "--outdir", str(tmp_path)])
    assert rc == 0
    import json
    out = json.loads((tmp_path / "environ.json").read_text())
    assert set(out) == set(E.ANALYSES)
    assert [k for k, v in out.items() if v["status"] == "error"] == []
    for f in ("e01_environment_by_protein.csv", "e01_environment_contrasts.csv",
              "e02_burial_slopes.csv", "e03_hydrophobicity_slopes.csv",
              "e04_direction_by_protein.csv", "e05_contacts.csv", "e07_helix_face.csv",
              "e08_depth_profile.csv", "e09_substitution_matrix.csv",
              "e10_charge_depth.csv", "positions.csv"):
        assert (tmp_path / f).exists(), f
    figs = {p.stem for p in tmp_path.glob("*.png")}
    assert {"e01_environment", "e02a_burial_depth", "e02b_burial_slope_by_depth",
            "e03_hydrophobicity_burial", "e04_substitution_direction", "e05_contacts",
            "e07_helix_face", "e08_membrane_map", "e09_substitution_matrix",
            "e10_charge_depth"} <= figs
    # SYN1 is the only fixture with ESM scores, so it is the only one with a network
    assert (tmp_path / "structures" / "SYN1" / "e06_network.pml").exists()
    assert {"e06a_esm_residual_network", "e06b_residual_distances",
            "e06c_clustering_null"} <= figs
    pos = pd.read_csv(tmp_path / "positions.csv")
    assert set(pos.environ.dropna()) <= set(E.CLASSES)


def test_panels_follow_the_order_the_configs_were_given_not_the_alphabet():
    """Which protein sits where is a decision about the figure, so order is the caller's."""
    fig, pairs, axes = E.grid(["Z", "A", "M", "B"], cols=2)
    assert [n for n, _ in pairs] == ["Z", "A", "M", "B"]
    assert pairs[0][1] is axes[0][0] and pairs[1][1] is axes[0][1]
    assert pairs[2][1] is axes[1][0] and pairs[3][1] is axes[1][1]
    import matplotlib.pyplot as plt
    plt.close(fig)
    # an odd number leaves the spare panel blank rather than reflowing the rest
    fig, pairs, axes = E.grid(["Z", "A", "M"], cols=2)
    assert not axes[1][1].get_visible()
    plt.close(fig)


def test_a_helix_with_one_intolerant_face_scores_higher_than_a_uniform_one():
    """The moment is about the arrangement, not about how intolerant a helix is."""
    n = 18
    ang = np.deg2rad(100.0) * np.arange(n)
    faced = -1.0 * (np.cos(ang) > 0)                 # one face dead, the other fine
    uniform = np.full(n, -1.0)                       # every residue dead
    noisy = np.random.default_rng(0).normal(-0.5, 0.3, n)
    assert E.helix_moment(faced) > 0.5
    assert np.isnan(E.helix_moment(uniform))         # nothing to arrange: no face
    assert E.helix_moment(noisy) < E.helix_moment(faced)
    # and it is blind to a constant offset, since only the arrangement should matter
    assert E.helix_moment(faced) == pytest.approx(E.helix_moment(faced - 3.0))


def test_the_colour_of_an_effect_is_the_projects_red_white_blue_scale():
    import numpy as np
    dead, neutral = E.effect_color(-1.0), E.effect_color(0.0)
    assert dead[0] > 0.4 and dead[2] < 0.1          # dark red
    assert np.allclose(neutral[:3], 1.0, atol=0.02)  # white at zero
