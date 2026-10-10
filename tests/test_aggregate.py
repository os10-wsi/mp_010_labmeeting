"""Aggregate transmembrane questions: the estimators must recover a planted effect and,
just as importantly, must not invent one that is not there."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms.aggregate import g01_positive_inside, g02_depth_profile, g03_aromatic_belt, nested_variance
from mpdms.stats import meta_analysis

AA = "ACDEFGHIKLMNPQRSTVWY"


def _frame(name, seed, kr_slope=0.0, de_slope=0.0, belt=0.0, n_helix=6, hl=16):
    """A TM-only variant frame of the shape aggregate.tm_frame produces."""
    r = np.random.default_rng(seed)
    rows = []
    for h in range(1, n_helix + 1):
        for i in range(hl):
            x = i / (hl - 1)
            pos = h * 100 + i
            # aromatics occur throughout, not only at the ends: the belt estimate needs
            # aromatic positions in the core as well to have anything to contrast with
            wt = "W" if i in (0, hl - 1) else str(r.choice(list(AA)))
            for m in AA:
                if m == wt:
                    continue
                eff = r.normal(-0.5, 0.25)
                if m in "KR":
                    eff += kr_slope * x
                if m in "DE":
                    eff += de_slope * x
                if wt in "WYF" and (x <= 0.18 or x >= 0.82):
                    eff += belt
                rows.append({"protein": name, "tm": f"TM{h}", "helix": f"{name}:TM{h}",
                             "n_helix": float(h), "pos": pos, "wt": wt, "mut": m,
                             "x": x, "in_membrane": True, "score_z": eff,
                             "charge": "basic" if m in "KR" else "acidic"})
    return pd.DataFrame(rows)


def _set(seed0=0, **kw):
    return {f"P{i}": _frame(f"P{i}", seed0 + i, **kw) for i in range(1, 5)}


def test_positive_inside_recovers_a_planted_interaction(tmp_path):
    """K/R made steadily worse toward the lumen, D/E flat: the interaction is the slope."""
    frames = _set(kr_slope=-0.7, de_slope=0.0)
    r = g01_positive_inside(frames, pd.DataFrame(), tmp_path)
    assert r["status"] == "ok"
    assert r["meta_mu"] == pytest.approx(-0.7, abs=0.12)
    assert r["mixed_coef"] == pytest.approx(-0.7, abs=0.12)
    assert r["meta_hi"] < 0 and r["mixed_p"] < 1e-6
    assert r["agree_in_sign"]
    assert r["n_helices"] == 24


def test_positive_inside_finds_nothing_when_there_is_nothing(tmp_path):
    """A charge penalty with no depth dependence must not read as positive-inside."""
    frames = _set(seed0=50, kr_slope=0.0, de_slope=0.0)
    r = g01_positive_inside(frames, pd.DataFrame(), tmp_path)
    assert r["status"] == "ok"
    assert abs(r["meta_mu"]) < 0.1
    assert r["meta_lo"] < 0 < r["meta_hi"]        # the interval covers no effect
    assert r["mixed_p"] > 0.01


def test_a_shared_charge_penalty_is_not_mistaken_for_asymmetry(tmp_path):
    """Both charges worsening with depth is depth, not a positive-inside signal."""
    frames = _set(seed0=90, kr_slope=-0.7, de_slope=-0.7)
    r = g01_positive_inside(frames, pd.DataFrame(), tmp_path)
    assert abs(r["meta_mu"]) < 0.12               # the interaction, not the main effect


def test_depth_profile_agrees_with_the_per_helix_route(tmp_path):
    """g02 pools per protein and g01 per helix; on the same data they must not disagree."""
    frames = _set(seed0=20, kr_slope=-0.5, de_slope=0.0)
    a = g01_positive_inside(frames, pd.DataFrame(), tmp_path)
    b = g02_depth_profile(frames, pd.DataFrame(), tmp_path)
    assert b["status"] == "ok"
    assert b["meta_mu"] == pytest.approx(-0.5, abs=0.12)
    assert b["meta_mu"] == pytest.approx(a["meta_mu"], abs=0.12)


def test_aromatic_belt_recovers_a_planted_belt(tmp_path):
    frames = _set(seed0=30, belt=-0.6)
    r = g03_aromatic_belt(frames, pd.DataFrame(), tmp_path)
    assert r["status"] == "ok"
    assert r["meta_mu"] == pytest.approx(-0.6, abs=0.15)
    assert r["meta_hi"] < 0


def test_aromatic_belt_is_null_without_one(tmp_path):
    frames = _set(seed0=70, belt=0.0)
    r = g03_aromatic_belt(frames, pd.DataFrame(), tmp_path)
    assert abs(r["meta_mu"]) < 0.12
    assert r["meta_lo"] < 0 < r["meta_hi"]


def test_variance_partition_puts_the_variance_where_it_was_planted():
    """All of the spread between proteins; almost none within a position."""
    rows = []
    for k, name in enumerate(["A", "B", "C", "D"]):
        for h in range(3):
            for p in range(8):
                for m in range(10):
                    rows.append({"protein": name, "helix": f"{name}:TM{h}", "pos": h * 100 + p,
                                 "score_z": k * 1.0 + 1e-6 * m})
    f = nested_variance(pd.DataFrame(rows))
    tot = sum(f.values())
    assert f["protein"] / tot > 0.99
    assert f["substitution (residual)"] / tot < 0.01


def test_meta_analysis_reports_disagreement_as_heterogeneity():
    agree = meta_analysis([0.5, 0.5, 0.5, 0.5], [0.01] * 4)
    differ = meta_analysis([0.0, 0.5, 1.0, 1.5], [0.01] * 4)
    assert agree["I2"] < 1 and agree["tau2"] == pytest.approx(0, abs=1e-9)
    assert differ["I2"] > 90 and differ["tau2"] > 0.1
    assert differ["mu"] == pytest.approx(0.75, abs=1e-6)
    # with the units scattered, the interval must be wider despite the same within-unit SEs
    assert (differ["hi"] - differ["lo"]) > 5 * (agree["hi"] - agree["lo"])


def test_meta_analysis_weights_by_precision():
    r = meta_analysis([0.0, 1.0], [0.0001, 1.0])
    assert r["mu"] < 0.05                      # pulled to the precise estimate


def test_meta_analysis_handles_degenerate_input():
    assert meta_analysis([], [])["k"] == 0
    assert meta_analysis([1.0, np.nan], [0.1, 0.1])["k"] == 1
    assert meta_analysis([1.0, 2.0], [0.1, -1.0])["k"] == 1     # a non-positive variance is dropped
