"""A13 helix-level statistics: detect a real surface/core effect, stay quiet under the null."""
import numpy as np
import pandas as pd

from mpdms.analyses.a13_helix_burial import class_stats, position_stats


def _toy(effect, rng, n_helices=12, L=21):
    rows = []
    for h in range(n_helices):
        cls = "surface" if h % 2 else "core"
        rsa = (0.4 if cls == "surface" else 0.2) + rng.normal(0, 0.02)
        hmean = -0.8 + (effect if cls == "surface" else 0) + rng.normal(0, 0.05)
        for i in range(L):
            rel = -1 + 2 * i / (L - 1)
            rows.append({"dataset_id": "X", "helix": f"TM{h}", "rel_pos": rel, "abs_rel": abs(rel),
                         "fit_rest": hmean + 0.3 * abs(rel) + rng.normal(0, 0.1), "confident": True,
                         "class_median": cls, "class_tertile": cls, "helix_rsa": rsa})
    return pd.DataFrame(rows)


def test_class_effect_detected():
    rng = np.random.default_rng(0)
    r = class_stats(_toy(0.4, rng), "fit_rest", "class_median", rng)
    assert r["surface_minus_core"] > 0.3 and r["p_perm"] < 0.01


def test_class_null_not_significant():
    rng = np.random.default_rng(1)
    r = class_stats(_toy(0.0, rng), "fit_rest", "class_median", rng)
    assert r["p_perm"] > 0.05


def test_position_effect_ends_more_tolerant():
    rng = np.random.default_rng(2)
    r = position_stats(_toy(0.0, rng), "fit_rest", rng)
    assert r["spearman_absrel"] > 0.3 and r["p_perm_within_helix"] < 0.01


# ------------------------------------- A22: topology-class and per-helix violins
def _topo_cfg(n_tm=4, start=20, helix_len=21, loop=15):
    from mpdms.config import Config
    segs, p, side = [], 1, "cytosolic"
    segs.append({"name": "Nterm", "start": 1, "end": start - 1, "type": "loop", "side": side})
    p = start
    for i in range(n_tm):
        segs.append({"name": f"TM{i + 1}", "start": p, "end": p + helix_len - 1, "type": "TM",
                     "orientation": "in_out" if i % 2 == 0 else "out_in"})
        p += helix_len
        side = "lumenal" if i % 2 == 0 else "cytosolic"
        segs.append({"name": f"L{i + 1}", "start": p, "end": p + loop - 1, "type": "loop", "side": side})
        p += loop
    return Config.wrap({"id": "T", "display_name": "T", "topology": {"segments": segs}}), segs


def _topo_df(segs, tm_effect=-0.9, loop_effect=-0.05, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for s in segs:
        for pos in range(s["start"], s["end"] + 1):
            mu = tm_effect if s["type"] == "TM" else loop_effect
            for mut in "ACDEFGHIKL":
                rows.append({"pos": pos, "wt": "V", "mut": mut, "vclass": "missense",
                             "pass_filter": True, "score_z": mu + rng.normal(0, 0.15)})
    return pd.DataFrame(rows)


def test_a22_separates_transmembrane_from_both_loop_sides():
    from mpdms.analyses.a22_topology_violins import classify, compare, summary
    cfg, segs = _topo_cfg()
    d = classify(_topo_df(segs), cfg)
    assert set(d.topo_class) == {"Transmembrane", "Intracellular", "Extracellular"}
    st = summary(d, "topo_class", ["Transmembrane", "Intracellular", "Extracellular"]).set_index("group")
    assert st.loc["Transmembrane", "median"] < -0.7
    assert st.loc["Intracellular", "median"] > -0.3 and st.loc["Extracellular", "median"] > -0.3
    t = compare(d, "topo_class", ["Transmembrane", "Intracellular", "Extracellular"]).set_index(["a", "b"])
    assert t.loc[("Transmembrane", "Intracellular"), "q"] < 0.01
    assert t.loc[("Intracellular", "Extracellular"), "q"] > 0.05   # the two sides are alike here


def test_a22_orders_helices_from_n_to_c_not_alphabetically():
    """TM10 must follow TM9, which a string sort would not do."""
    from mpdms.analyses.a22_topology_violins import _helix_key
    names = ["TM10", "TM2", "TM1", "TM11", "TM9"]
    assert sorted(names, key=_helix_key) == ["TM1", "TM2", "TM9", "TM10", "TM11"]


def test_a22_tests_positions_rather_than_variants():
    """10 variants per position must not inflate n tenfold in the comparison."""
    from mpdms.analyses.a22_topology_violins import classify, compare
    cfg, segs = _topo_cfg()
    d = classify(_topo_df(segs), cfg)
    t = compare(d, "topo_class", ["Transmembrane", "Intracellular"])
    n_tm_pos = d[d.topo_class == "Transmembrane"].pos.nunique()
    assert int(t.iloc[0].n_pos_a) == n_tm_pos
    assert int(t.iloc[0].n_pos_a) * 10 == int((d.topo_class == "Transmembrane").sum())
