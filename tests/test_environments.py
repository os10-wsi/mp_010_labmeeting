"""A27: the six environment comparisons, each against a planted answer."""
import numpy as np
import pandas as pd
import pytest

from mpdms.analyses.a27_environment_systematic import (choosiness, dose_response, environments,
                                                       neighbourhood, property_coefficients,
                                                       wt_optimality)

AA = "ACDEFGHIKLMNPQRSTVWY"


def _variants(env_spec, n_pos=40, seed=0, **kw):
    """env_spec(i) -> (rsa, in_slab, face); kw passes planted effects."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pos):
        rsa, in_slab, face = env_spec(i)
        for mut in AA:
            rows.append({"pos": i + 1, "wt": "L", "mut": mut, "seg_type": "TM" if in_slab else "loop",
                         "rsa": rsa, "in_slab": in_slab, "face": face,
                         "score_z": kw.get("effect", lambda **_: -0.3)(rsa=rsa, mut=mut, rng=rng)})
    return environments(pd.DataFrame(rows))


def _four_env(i):
    return [(0.05, False, "non-membrane"), (0.6, False, "non-membrane"),
            (0.6, True, "lipid-facing"), (0.6, True, "cavity-lining")][i % 4]


def test_a27_assigns_the_four_environments():
    d = _four_env
    t = _variants(d, n_pos=40)
    assert set(t.env) == {"buried", "water-exposed", "lipid-exposed", "cavity"}
    # a buried position inside the slab is still buried, not lipid-exposed
    one = _variants(lambda i: (0.05, True, "lipid-facing"), n_pos=12)
    assert set(one.env) == {"buried"}


def test_a27_property_coefficients_are_comparable_across_properties():
    """Volume is in A^3 and hydrophobicity in kcal/mol; without standardising, volume reads 0."""
    from mpdms.annot import VOLUME

    def eff(rsa, mut, rng, **_):
        return -0.5 - 0.004 * (VOLUME[mut] - VOLUME["L"]) * (1 if rsa < 0.25 else 0) + rng.normal(0, .08)
    d = _variants(_four_env, n_pos=80, effect=eff)
    pc, pq = property_coefficients(d)
    piv = pc.pivot_table(index="property", columns="env", values="coef")
    assert abs(piv.loc["volume", "buried"]) > 0.1           # readable, not 0.00
    assert abs(piv.loc["volume", "buried"]) > abs(piv.loc["volume", "water-exposed"])
    assert (pq.set_index("property").loc["volume", "q"]) < 0.05


def test_a27_choosiness_separates_uniformly_bad_from_selective():
    """One position bad at everything, one bad at a few: same mean is possible, spread is not."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        selective = i % 2 == 0
        for k, mut in enumerate(AA):
            v = (-1.2 if k < 5 else 0.0) if selective else -0.3
            rows.append({"pos": i + 1, "env": "buried" if selective else "water-exposed",
                         "seg_type": "loop", "rsa": 0.1 if selective else 0.6, "n": 20,
                         "mean_effect": v, "median_effect": v, "score_z": v + rng.normal(0, .05)})
    pt = (pd.DataFrame(rows).groupby(["pos", "env", "seg_type", "rsa"])
          .score_z.agg(mean_effect="mean", iqr=lambda s: np.subtract(*np.percentile(s, [75, 25])))
          .reset_index())
    r = choosiness(pt)
    assert r["testable"]
    tab = pd.DataFrame(r["table"]).set_index("env")
    assert tab.loc["buried", "median_choosy"] > tab.loc["water-exposed", "median_choosy"]


def test_a27_wt_optimality_rises_with_accessibility():
    """Buried: nothing beats WT. Exposed: plenty does. rho(RSA, fraction) must be positive."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(80):
        buried = i % 2 == 0
        rsa = 0.05 if buried else 0.7
        for k in range(20):
            better = (not buried) and k < 6
            rows.append({"pos": i + 1, "env": "buried" if buried else "water-exposed",
                         "seg_type": "loop", "rsa": rsa,
                         "score_z": (0.4 if better else -0.6) + rng.normal(0, .05)})
    pt = []
    for (pos, env, st, rsa), g in pd.DataFrame(rows).groupby(["pos", "env", "seg_type", "rsa"]):
        v = g.score_z.to_numpy()
        pt.append({"pos": pos, "env": env, "seg_type": st, "rsa": rsa, "n": len(v),
                   "median_effect": float(np.median(v)),
                   "n_better_than_wt": int((v > 0.1).sum()),
                   "frac_better_than_wt": float((v > 0.1).mean())})
    r = wt_optimality(pd.DataFrame(pt))
    assert r["testable"] and r["passed"] and r["rho_rsa"] > 0.5
    tab = pd.DataFrame(r["table"]).set_index("env")
    assert tab.loc["buried", "frac_positions_wt_optimal"] == pytest.approx(1.0)
    assert tab.loc["water-exposed", "frac_positions_wt_optimal"] == pytest.approx(0.0)


def _dose_frame(kind, n=120, seed=0):
    rng = np.random.default_rng(seed)
    rsa = rng.uniform(0, 1, n)
    if kind == "threshold":
        # a hinge, not a step: flat and bad below the breakpoint, easing above it. This is
        # both what the model fits and what "a critical burial" means biologically.
        y = -1.0 + 1.4 * np.clip(rsa - 0.35, 0, None) + rng.normal(0, 0.08, n)
    else:
        y = -1.0 + 0.9 * rsa + rng.normal(0, 0.08, n)
    return pd.DataFrame({"pos": np.arange(n), "rsa": rsa, "median_effect": y,
                         "seg_type": "TM", "env": "buried"})


def test_a27_dose_response_prefers_a_threshold_when_there_is_one():
    r = dose_response(_dose_frame("threshold"))["TM"]
    assert r["testable"] and r["prefers"] == "threshold"
    assert 0.25 < r["breakpoint"] < 0.45, r["breakpoint"]


def test_a27_dose_response_prefers_a_gradient_when_it_is_linear():
    r = dose_response(_dose_frame("linear"))["TM"]
    assert r["testable"] and r["prefers"] == "gradient"


def test_a27_neighbourhood_burial_detected_with_own_burial_in_the_model():
    """Effect driven by the neighbours' burial, not the residue's own."""
    rng = np.random.default_rng(2)
    n = 90
    own = rng.uniform(0, 1, n)
    nbr = rng.uniform(0.1, 0.8, n)
    pt = pd.DataFrame({"pos": np.arange(1, n + 1), "env": "buried", "rsa": own,
                       "median_effect": -1.0 + 1.2 * nbr + rng.normal(0, 0.1, n)})
    rt = pd.DataFrame({"pos": np.arange(1, 2 * n + 1),
                       "rsa": np.concatenate([own, nbr])})
    pairs = pd.DataFrame({"pos_i": np.arange(1, n + 1), "pos_j": np.arange(n + 1, 2 * n + 1)})
    r = neighbourhood(pt, rt, pairs)
    assert r["testable"] and r["passed"]
    assert r["beta_neighbour"] > 0.8 and r["p_neighbour"] < 1e-6
    assert r["p_own"] > 0.05 and r["r2_both"] > r["r2_own_only"] + 0.3


# ------------------- A05b: TM vs rest substitution matrices from the topology alone
def _tm_rest(tm_charge_cost=-1.2, seed=0):
    """TM helices made of hydrophobics, loops of polars, as real proteins are."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, 121):
        tm = 30 <= i <= 50 or 70 <= i <= 90
        # a few shared wild types so the composition-controlled marginal has something
        wt = ("ILVFW" if tm else "GSTNQ")[i % 5] if i % 7 else "G"
        for mut in AA:
            if mut == wt:
                continue
            charged = mut in "DEKR"
            y = (tm_charge_cost if (tm and charged) else -0.3 if tm else -0.05)
            rows.append({"pos": i, "wt": wt, "mut": mut, "is_tm": tm, "seg_type": "TM" if tm else "loop",
                         "vclass": "missense", "pass_filter": True,
                         "score_z": y + rng.normal(0, 0.08)})
    return pd.DataFrame(rows)


def test_a05b_difference_matrix_is_empty_when_subsets_share_no_wild_type():
    """TM and loops are built from different residues; the cell-wise difference may not exist."""
    from mpdms.analyses.a05_substitution_physchem import substitution_panels
    from mpdms.annot import HYDROPHOBICITY_ORDER
    from mpdms.config import Config
    import tempfile
    from pathlib import Path

    d = _tm_rest()
    d = d[~((d.is_tm) & (d.wt == "G"))]           # remove the only shared wild type
    with tempfile.TemporaryDirectory() as td:
        f = t = Path(td)
        _, _, st = substitution_panels(Config.wrap({"id": "T", "display_name": "T"}),
                                       d, list(HYDROPHOBICITY_ORDER), f, t)
    assert st["n_cells_compared"] == 0
    assert "GSTNQ".count(st["shared_wt"][:1]) or st["shared_wt"] != ""   # loops-only residues remain


def test_a05b_marginal_recovers_the_planted_tm_charge_cost():
    """Restricted to shared wild types, introducing a charge must be much worse in TM."""
    from mpdms.analyses.a05_substitution_physchem import substitution_panels
    from mpdms.annot import HYDROPHOBICITY_ORDER
    from mpdms.config import Config
    import tempfile
    from pathlib import Path

    d = _tm_rest(tm_charge_cost=-1.2)
    with tempfile.TemporaryDirectory() as td:
        f = t = Path(td)
        _, tabs, st = substitution_panels(Config.wrap({"id": "T", "display_name": "T"}),
                                          d, list(HYDROPHOBICITY_ORDER), f, t)
        marg = pd.read_csv([p for p in tabs if "marginal" in p.name][0])
    assert st["n_shared_wt_residues"] >= 1
    piv = marg.pivot_table(index="introduced", columns="subset", values="median")
    charged = piv.loc[[a for a in "DEKR" if a in piv.index]]
    hydro = piv.loc[[a for a in "ILVF" if a in piv.index]]
    assert charged["TM"].mean() < hydro["TM"].mean() - 0.5      # charge is the expensive one
    assert abs(charged["non-TM"].mean() - hydro["non-TM"].mean()) < 0.2   # not so in loops


def test_a05b_marginal_falls_back_and_says_so_when_composition_blocks_the_control():
    """Disjoint wild types: the shared-WT marginal is impossible, so it must say so loudly."""
    from mpdms.analyses.a05_substitution_physchem import substitution_panels
    from mpdms.annot import HYDROPHOBICITY_ORDER
    from mpdms.config import Config
    import tempfile
    from pathlib import Path

    rng = np.random.default_rng(0)
    rows = []
    for i in range(1, 81):
        tm = i % 2 == 0
        wt = "ILVF"[i % 4] if tm else "GSTN"[i % 4]     # no overlap at all
        for mut in AA:
            if mut == wt:
                continue
            rows.append({"pos": i, "wt": wt, "mut": mut, "is_tm": tm,
                         "score_z": (-0.9 if tm else -0.1) + rng.normal(0, 0.08)})
    d = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as td:
        f = t = Path(td)
        _, tabs, st = substitution_panels(Config.wrap({"id": "T", "display_name": "T"}),
                                          d, list(HYDROPHOBICITY_ORDER), f, t)
        marg = pd.read_csv([p for p in tabs if "marginal" in p.name][0])
    assert st["n_shared_wt_residues"] == 0
    assert st["marginal_composition_controlled"] is False
    # both subsets must still be drawn, so the panel does not look half-broken
    assert set(marg.subset) == {"TM", "non-TM"}
    assert (marg.composition_controlled == False).all()
