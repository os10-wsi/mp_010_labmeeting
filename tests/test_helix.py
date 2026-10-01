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
