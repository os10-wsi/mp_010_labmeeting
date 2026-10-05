"""Soluble reference loader and the membrane-vs-soluble comparison."""
import numpy as np
import pandas as pd
import pytest

from mpdms.annot import BIOLOGICAL
from mpdms.reference import load_soluble

AA = "ACDEFGHIKLMNPQRSTVWY"


def _reference(tmp_path, domains=(("SH3", 1.0), ("PDZ", 4.0)), hyd_slope=-0.06,
               with_rsa=True, controls=True, seed=0):
    """A soluble reference whose domains differ in raw dynamic range."""
    rng = np.random.default_rng(seed)
    rows = []
    for dom, scale in domains:
        for i in range(1, 31):
            wt = AA[rng.integers(0, 20)]
            rsa = rng.uniform(0, 0.8)
            for mut in AA:
                if mut == wt:
                    continue
                dh = BIOLOGICAL[mut] - BIOLOGICAL[wt]
                y = -0.25 + hyd_slope * dh - 0.5 * (1 - rsa) + rng.normal(0, 0.15)
                r = {"domain": dom, "position": i, "wt_aa": wt, "mut_aa": mut,
                     "mut_type": "missense", "fitness": scale * y}
                if with_rsa:
                    r["rsa"] = rsa
                rows.append(r)
        if controls:
            for i in range(1, 16):
                for cls, val in (("synonymous", 0.0), ("stop", -1.0)):
                    r = {"domain": dom, "position": i, "wt_aa": "L",
                         "mut_aa": "L" if cls == "synonymous" else "*", "mut_type": cls,
                         "fitness": scale * (val + rng.normal(0, 0.08))}
                    if with_rsa:
                        r["rsa"] = 0.3
                    rows.append(r)
    p = tmp_path / "ref.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_load_soluble_normalises_each_domain_on_its_own_controls():
    """A domain with 4x the raw range must not dominate: both land on syn 0, stop -1."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        d = load_soluble(_reference(Path(td)))
        med = d.groupby(["dataset", "vclass"]).score_z.median()
        for dom in ("SH3", "PDZ"):
            assert med[(dom, "synonymous")] == pytest.approx(0, abs=0.05)
            assert med[(dom, "nonsense")] == pytest.approx(-1, abs=0.05)
        assert d.attrs["n_datasets_kept"] == 2


def test_load_soluble_drops_domains_without_controls(tmp_path):
    p = _reference(tmp_path, controls=False)
    with pytest.raises(ValueError, match="synonymous and nonsense"):
        load_soluble(p)


def test_load_soluble_names_the_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="could not find column"):
        load_soluble(p)


def _membrane(hyd_slope=+0.27, seed=1):
    """A membrane protein whose TM helices like hydrophobicity and whose loops do not."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, 121):
        tm = 30 <= i <= 50 or 70 <= i <= 90
        wt = AA[rng.integers(0, 20)]
        for mut in AA:
            if mut == wt:
                continue
            dh = BIOLOGICAL[mut] - BIOLOGICAL[wt]
            y = (-0.6 + hyd_slope * dh) if tm else (-0.05 + 0.02 * dh)
            rows.append({"pos": i, "wt": wt, "mut": mut, "vclass": "missense", "pass_filter": True,
                         "score_z": y + rng.normal(0, 0.15),
                         "seg_type": "TM" if tm else "loop"})
    return pd.DataFrame(rows)


def _cfg():
    from mpdms.config import Config
    return Config.wrap({"id": "M", "display_name": "M", "topology": {"segments": [
        {"name": "TM1", "start": 30, "end": 50, "type": "TM", "orientation": "in_out"},
        {"name": "TM2", "start": 70, "end": 90, "type": "TM", "orientation": "out_in"}]}})


def test_a25_detects_the_hydrophobicity_sign_flip(tmp_path):
    """The headline comparison: greasier is bad in solution, good in the bilayer."""
    from mpdms.analyses.a25_soluble_comparison import combined, property_slopes
    sol = load_soluble(_reference(tmp_path, hyd_slope=-0.06))
    d = combined(_membrane(hyd_slope=+0.27), _cfg(), sol)
    r = property_slopes(d, "dhyd")
    assert r["testable"] and r["sign_flip"], r
    assert r["slope_soluble"] < 0 and r["slopes"]["membrane, TM"] > 0
    assert r["p_interaction_membrane, TM"] < 0.01


def test_a25_reports_no_flip_when_both_dislike_hydrophobicity(tmp_path):
    from mpdms.analyses.a25_soluble_comparison import combined, property_slopes
    sol = load_soluble(_reference(tmp_path, hyd_slope=-0.06))
    d = combined(_membrane(hyd_slope=-0.06), _cfg(), sol)
    r = property_slopes(d, "dhyd")
    assert r["testable"] and not r["sign_flip"], r


def test_a25_substitution_difference_matrix_is_oriented_tm_minus_soluble(tmp_path):
    from mpdms.analyses.a25_soluble_comparison import combined, substitution_matrices
    sol = load_soluble(_reference(tmp_path))
    d = combined(_membrane(), _cfg(), sol)
    ma, mb, diff, st = substitution_matrices(d, "membrane, TM", "soluble")
    ok = diff.notna()
    assert ok.to_numpy().sum() > 20
    sub = (ma - mb)[ok]
    assert np.allclose(sub.to_numpy()[ok.to_numpy()], diff.to_numpy()[ok.to_numpy()], equal_nan=True)


def test_a25_non_tm_group_comes_from_the_membrane_protein_itself(tmp_path):
    """The internal control must be this protein's loops, not the reference."""
    from mpdms.analyses.a25_soluble_comparison import combined
    sol = load_soluble(_reference(tmp_path))
    d = combined(_membrane(), _cfg(), sol)
    assert set(d[d.group == "membrane, non-TM"].dataset) == {"M"}
    assert set(d[d.group == "soluble"].dataset) == {"SH3", "PDZ"}
    assert (d[d.group == "membrane, TM"].pos.between(30, 50)
            | d[d.group == "membrane, TM"].pos.between(70, 90)).all()
