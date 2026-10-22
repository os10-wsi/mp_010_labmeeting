"""A predicted stability change against fitness: the merge, and the split correlations."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms import stability as S


def _ddg_csv(tmp_path, positions, wt, name="ThermoMPNN_inference.csv", offset=0):
    rows = [{"position": p + offset, "wildtype": wt[p], "mutation": m, "ddG_pred": 0.1}
            for p in positions for m in "AVLI"]
    f = tmp_path / name
    pd.DataFrame(rows).to_csv(f, index_label="")
    return f


def test_read_ddg_takes_the_spellings_these_files_come_with(tmp_path):
    f = tmp_path / "x.csv"
    pd.DataFrame({"position": [1, 2], "wildtype": ["A", "C"], "mutation": ["V", "W"],
                  "ddG_pred": [0.5, -0.2]}).to_csv(f, index_label="")
    d = S.read_ddg(f)
    assert list(d.columns) == ["position", "wt", "mut", "ddg"]
    g = tmp_path / "y.csv"
    pd.DataFrame({"Pos": [1], "WT_AA": ["A"], "Mut": ["V"], "ddG": [0.5]}).to_csv(g)
    assert list(S.read_ddg(g).columns) == ["position", "wt", "mut", "ddg"]


def test_read_ddg_names_the_column_it_could_not_find(tmp_path):
    f = tmp_path / "bad.csv"
    pd.DataFrame({"position": [1], "wildtype": ["A"], "mutation": ["V"]}).to_csv(f)
    with pytest.raises(SystemExit, match="ddg"):
        S.read_ddg(f)


def _var(seq, n_mut=6, seed=0, effect=None):
    """A variant table of the pipeline's shape: pos, wt, mut, score_z, bclass, rsa."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, w in enumerate(seq, start=1):
        for m in "AVLIFWKR"[:n_mut]:
            rows.append({"pos": i, "wt": w, "mut": m,
                         "score_z": float(rng.normal(-0.4, 0.3)),
                         "bclass": "core" if i % 2 else "lipid",
                         "rsa": 0.05 if i % 2 else 0.45})
    d = pd.DataFrame(rows)
    return d[d.wt != d.mut].reset_index(drop=True)


# deliberately not a repeat of one motif: a periodic sequence has no unique offset
SEQ = ("MKWVTFALLLASGATGAFSRGVRDNYCPLVKEHMDSTQIWNFYKLPGAVCDEHMTRQNWYFI"
       "GQDHLPSTNECIVRWAYMFKDGTLHSPQNVRECWIYAFMKTDGLHSPNQVRECWIYAFMKT")


def test_find_offset_recovers_a_known_shift():
    var = _var(SEQ)
    ddg = pd.DataFrame([{"position": i - 7, "wt": w, "mut": "A"}
                        for i, w in enumerate(SEQ, start=1)])
    off, frac = S.find_offset(var, ddg)
    assert off == 7 and frac == pytest.approx(1.0)


def test_find_offset_prefers_the_offset_that_matches_more_positions():
    """A short perfect overlap must not beat the real alignment one residue short of it.

    Real inference files disagree at the odd position - a modelled residue, a cleaved
    terminus - so ranking offsets by the FRACTION that agree hands the answer to any
    tiny overlap that happens to be clean.
    """
    var = _var(SEQ)
    ddg = pd.DataFrame([{"position": i, "wt": w, "mut": "A"}
                        for i, w in enumerate(SEQ, start=1)])
    ddg.loc[ddg.position == 5, "wt"] = "C" if SEQ[4] != "C" else "W"   # one bad residue
    off, frac = S.find_offset(var, ddg)
    assert off == 0
    assert frac == pytest.approx(1 - 1 / len(SEQ), abs=1e-6)


def test_attach_ddg_refuses_a_numbering_it_cannot_trust():
    """A wrong offset yields a full table of predictions for the wrong residues."""
    var = _var(SEQ)
    rng = np.random.default_rng(1)
    scrambled = "".join(rng.permutation(list("ACDEFGHIKLMNPQRSTVWY"))[0] for _ in SEQ)
    ddg = pd.DataFrame([{"position": i, "wt": w, "mut": "A", "ddg": 0.0}
                        for i, w in enumerate(scrambled, start=1)])
    with pytest.raises(SystemExit, match="wild-type"):
        S.attach_ddg({"var": var}, ddg)


def test_attach_ddg_drops_the_rows_whose_wild_type_disagrees():
    var = _var(SEQ)
    ddg = pd.DataFrame([{"position": i, "wt": w, "mut": m, "ddg": 0.3}
                        for i, w in enumerate(SEQ, start=1) for m in "AVLIFWKR"])
    ddg.loc[ddg.position == 3, "wt"] = "C" if SEQ[2] != "C" else "W"
    merged, info = S.attach_ddg({"var": var}, ddg)
    assert info["offset"] == 0
    assert info["n_wt_mismatch"] > 0
    assert (merged.pos != 3).all()


def _pair(n_pos, rho_strength, seed=0, bclass="core", rsa=0.05):
    """Positions whose fitness and ddG are coupled by `rho_strength`."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pos):
        for _ in range(8):
            y = rng.normal(0, 1)
            rows.append({"pos": p, "score_z": y, "bclass": bclass, "rsa": rsa,
                         "ddg": rho_strength * y + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_corr_with_ci_recovers_a_planted_correlation():
    d = _pair(40, -2.0, seed=2)
    r = S.corr_with_ci(d, n_boot=300)
    assert r["r"] < -0.6
    assert r["lo"] <= r["r"] <= r["hi"]
    assert r["n_positions"] == 40


def test_corr_with_ci_clusters_on_positions_not_variants():
    """The interval must count residues. Repeating each variant must not narrow it."""
    d = _pair(20, -1.0, seed=3)
    wide = S.corr_with_ci(d, n_boot=400, seed=5)
    doubled = pd.concat([d, d], ignore_index=True)       # same residues, twice the rows
    same = S.corr_with_ci(doubled, n_boot=400, seed=5)
    assert same["n_positions"] == wide["n_positions"]
    # duplicating variants within the same positions buys almost no precision
    assert same["se"] > 0.6 * wide["se"]


def test_corr_with_ci_refuses_a_group_too_small_to_mean_anything():
    d = _pair(3, -2.0)
    assert not np.isfinite(S.corr_with_ci(d)["r"])


def test_difference_in_corr_finds_a_planted_gap_between_two_classes():
    core = _pair(40, -2.5, seed=6, bclass="core", rsa=0.05)
    lipid = _pair(40, 0.0, seed=7, bclass="lipid", rsa=0.45)
    d = S.difference_in_corr(core, lipid, n_boot=600)
    assert d["r_a"] < -0.6 and abs(d["r_b"]) < 0.2
    assert d["difference"] < -0.4
    assert d["lo"] < d["difference"] < d["hi"]
    assert d["p"] < 0.05


def test_difference_in_corr_finds_nothing_when_both_classes_agree():
    a = _pair(40, -1.5, seed=8)
    b = _pair(40, -1.5, seed=9, bclass="lipid", rsa=0.45)
    d = S.difference_in_corr(a, b, n_boot=600)
    assert abs(d["difference"]) < 0.25
    assert d["p"] > 0.05


def test_the_bootstrap_p_is_never_reported_as_exactly_zero():
    """With a finite number of draws the smallest honest p is 2/(n+1), not 0."""
    core = _pair(60, -6.0, seed=10)
    lipid = _pair(60, 0.0, seed=11, bclass="lipid", rsa=0.45)
    d = S.difference_in_corr(core, lipid, n_boot=200)
    assert 0 < d["p"] <= 1
    assert d["p"] >= 2 / (d["n_boot"] + 1)


def test_t02_writes_its_own_files_and_keeps_the_classes_apart(tmp_path):
    core = _pair(30, -2.5, seed=12, bclass="core", rsa=0.05)
    lipid = _pair(30, 0.0, seed=13, bclass="lipid", rsa=0.45)
    lipid["pos"] = lipid.pos + 100
    data = {"SYN": {"var": pd.concat([core, lipid], ignore_index=True),
                    "merge": {"offset": 0, "wt_agreement": 1.0}}}
    out = S.t02_by_class(data, ["SYN"], tmp_path)
    assert out["status"] == "ok"
    t = pd.DataFrame(out["by_class"]).set_index("bclass")
    assert t.loc["core", "r"] < -0.6
    assert abs(t.loc["lipid", "r"]) < 0.2
    d = pd.DataFrame(out["differences"])
    assert (d.contrast == "core − lipid").any()
    assert (tmp_path / "t02_correlation_by_class.csv").exists()
    assert (tmp_path / "t02_class_differences.csv").exists()
    # the forest and the scatter behind it are both written, under their own names
    assert (tmp_path / "t02a_correlation_by_class.png").exists()
    assert (tmp_path / "t02b_scatter_by_class.png").exists()


def test_the_scatter_takes_its_numbers_from_the_table_it_is_given(tmp_path):
    """The figure must not recompute a correlation: one source, so they cannot drift.

    It is handed a stats table that disagrees with the data on purpose; drawing it
    without complaint is the evidence that the panel annotation comes from the table
    t02 wrote and not from a second calculation inside the plotting code.
    """
    core = _pair(30, -2.5, seed=16, bclass="core", rsa=0.05)
    lipid = _pair(30, 0.0, seed=17, bclass="lipid", rsa=0.45)
    lipid["pos"] = lipid.pos + 100
    data = {"SYN": {"var": pd.concat([core, lipid], ignore_index=True),
                    "merge": {"offset": 0, "wt_agreement": 1.0}}}
    stats = pd.DataFrame([{"protein": "SYN", "bclass": "core", "r": -0.99,
                           "lo": -1.0, "hi": -0.98, "n": 240},
                          {"protein": "SYN", "bclass": "lipid", "r": 0.42,
                           "lo": 0.3, "hi": 0.5, "n": 240}])
    before = stats.copy()
    S._class_scatter(data, ["SYN"], tmp_path, stats, "spearman")
    assert (tmp_path / "t02b_scatter_by_class.png").exists()
    pd.testing.assert_frame_equal(stats, before)


def test_the_scatter_survives_a_class_with_no_variants(tmp_path):
    core = _pair(30, -2.0, seed=18, bclass="core", rsa=0.05)
    data = {"SYN": {"var": core, "merge": {"offset": 0, "wt_agreement": 1.0}}}
    stats = pd.DataFrame([{"protein": "SYN", "bclass": "core", "r": -0.7,
                           "lo": -0.8, "hi": -0.6, "n": 240},
                          {"protein": "SYN", "bclass": "aqueous", "r": np.nan,
                           "lo": np.nan, "hi": np.nan, "n": 0}])
    S._class_scatter(data, ["SYN"], tmp_path, stats, "spearman")
    assert (tmp_path / "t02b_scatter_by_class.png").exists()


def test_t03_contrasts_buried_against_exposed(tmp_path):
    buried = _pair(30, -2.5, seed=14, bclass="core", rsa=0.05)
    exposed = _pair(30, 0.0, seed=15, bclass="lipid", rsa=0.45)
    exposed["pos"] = exposed.pos + 100
    data = {"SYN": {"var": pd.concat([buried, exposed], ignore_index=True),
                    "merge": {"offset": 0, "wt_agreement": 1.0}}}
    out = S.t03_by_rsa(data, ["SYN"], tmp_path)
    c = pd.DataFrame(out["contrasts"])
    assert len(c) == 1 and c.difference.iloc[0] < -0.4 and c.p.iloc[0] < 0.05
    assert (tmp_path / "t03_correlation_by_rsa.csv").exists()
