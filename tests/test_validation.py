"""Validation battery and benchmark harness: do they detect real signal and stay quiet otherwise."""
import numpy as np
import pandas as pd
import pytest

from mpdms import benchmark as B
from mpdms.analyses.a16_assay_validation import k12_pseudosymmetry
from mpdms.config import Config


def _mfs_cfg(n_tm=12, start=20, helix_len=21, gap=12):
    segs, p = [], start
    for i in range(n_tm):
        segs.append({"name": f"TM{i + 1}", "start": p, "end": p + helix_len - 1, "type": "TM",
                     "orientation": "in_out" if i % 2 == 0 else "out_in"})
        p += helix_len + gap
    return Config.wrap({"id": "MFS", "display_name": "MFS", "topology": {"segments": segs}}), segs


def _profile_df(segs, shape_fn, rng, noise=0.05, second_half_shape=None):
    """Build a variant table whose per-position mean follows shape_fn(rel_pos) per helix."""
    half = len(segs) // 2
    rows = []
    for k, s in enumerate(segs):
        fn = shape_fn if k < half else (second_half_shape or shape_fn)
        L = s["end"] - s["start"] + 1
        for j, pos in enumerate(range(s["start"], s["end"] + 1)):
            f = (pos - s["start"]) / (L - 1)
            rel = (2 * f - 1) if s["orientation"] == "in_out" else (1 - 2 * f)
            mu = fn(rel, k % half, j)
            for _ in range(6):
                rows.append({"pos": pos, "score_z": mu + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def test_k12_detects_pseudosymmetry():
    """Two bundles built from the same per-helix shapes must be flagged as correlated."""
    rng = np.random.default_rng(0)
    cfg, segs = _mfs_cfg()
    shape = lambda rel, hx, j: -0.8 - 0.5 * np.cos(rel * np.pi) + 0.35 * np.sin(hx + 2 * rel)  # noqa: E731
    mis = _profile_df(segs, shape, rng)
    r = k12_pseudosymmetry(None, cfg, mis, None)
    assert r["testable"] and r["passed"], r
    assert r["value"] > 0.5 and r["p"] < 0.05


def test_k12_quiet_when_bundles_unrelated():
    """Independent random shapes per helix must not produce a symmetry call."""
    rng = np.random.default_rng(1)
    cfg, segs = _mfs_cfg()
    offs = rng.normal(0, 1, len(segs))
    shape = lambda rel, hx, j: -0.8 + 0.6 * np.sin(3 * rel + offs[hx])  # noqa: E731
    second = lambda rel, hx, j: -0.8 + 0.6 * np.sin(3 * rel + offs[hx + 6] + 2.0)  # noqa: E731
    mis = _profile_df(segs, shape, rng, second_half_shape=second)
    r = k12_pseudosymmetry(None, cfg, mis, None)
    assert r["testable"]
    assert not r["passed"] or r["p"] > 0.01, r


def test_k12_not_testable_for_small_proteins():
    cfg, segs = _mfs_cfg(n_tm=4)
    mis = _profile_df(segs, lambda rel, hx, j: -0.5, np.random.default_rng(2))
    r = k12_pseudosymmetry(None, cfg, mis, None)
    assert not r["testable"] and "TM helices" in r["note"]


# ------------------------------------------------------------------- splits
def _multi(n_proteins=4, n_pos=40, rule=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_proteins):
        shift = rng.normal(0, 0.3)
        for p in range(1, n_pos + 1):
            rel = -1 + 2 * (p % 21) / 20
            for m in "ACDEFGHIKLMNPQRSTVWY":
                dh = rng.normal()
                mu = (-0.8 * abs(rel) - 0.4 * dh + shift) if rule else rng.normal(0, 0.5)
                rows.append({"dataset_id": f"P{g}", "pos": p, "wt": "A", "mut": m,
                             "abs_rel": abs(rel), "rel_pos": rel, "delta_biological": dh,
                             "seg_type": "TM" if abs(rel) < 0.8 else "loop",
                             "score_z": mu + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_protein_split_holds_out_whole_proteins():
    d = _multi()
    for name, tr, te in B.split_folds(d, "protein"):
        assert set(d.loc[tr].dataset_id).isdisjoint(d.loc[te].dataset_id)
        assert set(d.loc[te].dataset_id) == {name}


def test_position_split_keeps_sites_together():
    d = _multi()
    for _, tr, te in B.split_folds(d, "position", n_folds=5):
        a = set(zip(d.loc[tr].dataset_id, d.loc[tr].pos))
        b = set(zip(d.loc[te].dataset_id, d.loc[te].pos))
        assert a.isdisjoint(b)


def test_site_mean_collapses_across_the_position_split():
    """site_mean looks strong on a random split and must fail when positions are held out."""
    d = _multi()
    rnd = B.evaluate(d, [B.SiteMean()], split="random", n_folds=4).spearman.mean()
    pos = B.evaluate(d, [B.SiteMean()], split="position", n_folds=4).spearman.mean()
    assert rnd > 0.3 and abs(pos) < 0.15


# -------------------------------------------------------------- transfer
def test_lopo_transfer_when_a_shared_rule_exists():
    d = _multi(rule=True)
    res = B.evaluate(d, [B.Baseline("ridge", features=["abs_rel", "delta_biological"])], split="protein")
    assert res.spearman.min() > 0.5, res


def test_lopo_transfer_absent_when_no_shared_rule():
    d = _multi(rule=False)
    res = B.evaluate(d, [B.Baseline("ridge", features=["abs_rel", "delta_biological"])], split="protein")
    assert abs(res.spearman.mean()) < 0.15, res


def test_score_column_auto_orients_sign():
    d = _multi()
    d["ddg"] = -d.score_z + np.random.default_rng(0).normal(0, 0.1, len(d))  # inverted convention
    res = B.evaluate(d, [B.ScoreColumn("ddg", sign="auto")], split="protein")
    assert res.spearman.mean() > 0.8


@pytest.mark.parametrize("how", ["random", "position", "protein", "segment"])
def test_every_split_covers_the_data(how):
    d = _multi()
    seen = set()
    for _, tr, te in B.split_folds(d, how, n_folds=3):
        assert len(tr) and len(te)
        seen |= set(te)
    assert len(seen) == len(d)


def test_metrics_report_discrimination():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    m = B.metrics(y, y + rng.normal(0, 0.2, 500))
    assert m["spearman"] > 0.9 and m["auroc_deleterious"] > 0.9
    m0 = B.metrics(y, rng.normal(0, 1, 500))
    assert abs(m0["spearman"]) < 0.2


# ------------------------------------------- A18: burial vs substitution cost
def _rsa_positions(kr_slope, pro_slope, seed=0, n_helices=8, n_pos=20):
    """Positions whose K/R and Pro effects depend on RSA with the given slopes."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_helices):
        for i in range(n_pos):
            rsa = rng.uniform(0, 0.7)
            rows.append({"pos": h * 100 + i, "helix": f"TM{h}", "orientation": "in_out", "rsa": rsa,
                         "mean_KR": -1.0 + kr_slope * rsa + rng.normal(0, 0.12), "n_KR": 2,
                         "mean_Pro": -1.0 + pro_slope * rsa + rng.normal(0, 0.12), "n_Pro": 1})
    return pd.DataFrame(rows)


def test_rsa_slope_difference_detected():
    """K/R relieved by exposure, proline flat: the interaction must fire."""
    from mpdms.analyses.a18_helix_kr_vs_pro import rsa_stats
    r = rsa_stats(_rsa_positions(kr_slope=1.2, pro_slope=0.0, seed=1))
    assert r["mean_KR"]["slope"] > 0.8 and r["mean_KR"]["p_slope"] < 0.01
    assert abs(r["mean_Pro"]["slope"]) < 0.3
    assert r["interaction"]["delta"] > 0.6 and r["interaction"]["p"] < 0.01


def test_rsa_slope_difference_absent_when_slopes_match():
    from mpdms.analyses.a18_helix_kr_vs_pro import rsa_stats
    r = rsa_stats(_rsa_positions(kr_slope=0.6, pro_slope=0.6, seed=2))
    assert abs(r["interaction"]["delta"]) < 0.3 and r["interaction"]["p"] > 0.05


# ------------------------------------------- A19: published residues vs screen
def _sites(functional_pos, n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, n + 1):
        func = p in functional_pos
        rows.append({"pos": p, "wt": "Q", "segment": "TM1", "n_variants": 18,
                     "median_abundance": 0.1 if func else rng.uniform(-1.2, 0.2),
                     "median_z": -2.5 if func else rng.uniform(-0.8, 0.8),
                     "q": 0.001 if func else 0.6, "functional": func,
                     "abundance_tolerant": True if func else False})
    return pd.DataFrame(rows)


def test_a19_wt_mismatch_drops_the_entry():
    """A paper's numbering that disagrees with the sequence must be refused, not used."""
    from mpdms.analyses.a19_literature import check_numbering
    seq = "M" + "Q" * 50          # Q at 2..51, M at 1
    ok, bad = check_numbering([{"pos": 10, "wt": "Q"}, {"pos": 1, "wt": "R"}, {"pos": 999, "wt": "Q"}], seq)
    assert [e["pos"] for e in ok] == [10] and ok[0]["wt_checked"]
    assert len(bad) == 2 and "numbering mismatch" in bad[0] and "outside" in bad[1]


def test_a19_marks_abundance_tolerant_hits():
    """A published residue the screen calls functional but abundance-tolerant is the gold case."""
    from mpdms.analyses.a19_literature import residue_table
    s = _sites({149, 504})
    t = residue_table([{"pos": 149, "wt": "Q", "substitution": "Q149A"},
                       {"pos": 60, "wt": "Q", "substitution": "Q60A"}], s)
    assert t.loc[t.pos == 149, "verdict"].iloc[0] == "functional, abundance-tolerant"
    assert t.loc[t.pos == 60, "verdict"].iloc[0] != "functional, abundance-tolerant"


def test_a19_uncovered_residue_is_not_scored_as_a_miss():
    from mpdms.analyses.a19_literature import enrichment, residue_table
    s = _sites({149})
    t = residue_table([{"pos": 149, "wt": "Q"}, {"pos": 5000, "wt": "R"}], s)
    assert t.loc[t.pos == 5000, "verdict"].iloc[0] == "not covered by the screen"
    e = enrichment(t, s)
    assert e["n_published_covered"] == 1 and e["n_hit"] == 1 and e["underpowered"]


def test_a19_curated_file_parses_and_qdr2_claims_nothing():
    """Guard against someone inventing QDR2 residues: there is no published set."""
    from mpdms.analyses.a19_literature import load_literature
    aqr1, qdr2 = load_literature("AQR1"), load_literature("qdr2")
    assert aqr1 and {e["pos"] for e in aqr1["measured"]} == {149, 238, 504}
    assert all(e["confidence"] in {"verified", "unverified"} for e in aqr1["measured"])
    assert qdr2 is not None and qdr2["measured"] == []
