"""Splits and the learning curve: the honesty of a predictive claim lives here."""
import numpy as np
import pandas as pd
import pytest

from mpdms.benchmark import Baseline, assign_families, learning_curve, split_folds


def _toy(n_per=200, seed=0):
    """Four datasets; A1/A2 are near-copies of one another, B and C are not."""
    r = np.random.default_rng(seed)
    rows = []
    for ds, shift in [("A1", 0.0), ("A2", 0.02), ("B", 0.9), ("C", -0.8)]:
        for i in range(n_per):
            x = r.normal()
            rows.append({"dataset_id": ds, "pos": i % 40, "score_z": x + shift,
                         "delta_biological": x, "rsa": r.random(), "is_tm": float(i % 2)})
    d = pd.DataFrame(rows)
    d["family"] = d.dataset_id.map({"A1": "A1+A2", "A2": "A1+A2", "B": "B", "C": "C"})
    return d


def test_family_split_holds_out_every_member_of_the_family():
    d = _toy()
    folds = {n: (tr, te) for n, tr, te in split_folds(d, "family")}
    assert set(folds) == {"A1+A2", "B", "C"}
    tr, te = folds["A1+A2"]
    assert set(d.loc[te].dataset_id) == {"A1", "A2"}
    assert not set(d.loc[tr].dataset_id) & {"A1", "A2"}       # no relative left in training


def test_protein_split_leaves_the_paralog_in_training():
    """The leak the family split exists to close; asserted so it cannot be mistaken."""
    d = _toy()
    folds = {n: (tr, te) for n, tr, te in split_folds(d, "protein")}
    tr, _ = folds["A1"]
    assert "A2" in set(d.loc[tr].dataset_id)


def test_family_split_needs_the_column():
    d = _toy().drop(columns="family")
    with pytest.raises(ValueError, match="family"):
        list(split_folds(d, "family"))


def test_assign_families_groups_near_identical_sequences_only():
    seqs = {"P1": "MKLWTTAVGCLAVSAGGLFFGYDTGVISGA" * 5,
            "P2": "MKLWTTAVGCLAVSAGGLFFGYDTGVISGV" * 5,
            "P3": "PQRSTPQRSTPQRSTPQRSTPQRSTPQRST" * 5}
    fam, ident = assign_families(seqs, 0.40)
    assert fam["P1"] == fam["P2"] != fam["P3"]
    assert ident[("P1", "P2")] > 0.9 and ident[("P1", "P3")] < 0.4


def test_assign_families_is_transitive():
    """A close to B and B close to C puts all three in one group, even if A and C are not."""
    base = "MKLWTTAVGCLAVSAGGLFFGYDTGVISGA" * 5
    seqs = {"A": base,
            "B": base.replace("A", "V", 40),
            "C": base.replace("A", "V", 80)}
    fam, _ = assign_families(seqs, 0.40)
    assert len(set(fam.values())) == 1


def test_assign_families_tolerates_a_missing_sequence():
    fam, _ = assign_families({"X": "MKLWTT" * 20, "Y": ""}, 0.40)
    assert fam["Y"] == "Y"


def test_learning_curve_never_trains_on_the_held_out_family():
    d = _toy()
    lc = learning_curve(d, lambda: Baseline("ridge", features=["delta_biological", "rsa", "is_tm"]),
                        group="family", repeats=2, seed=1)
    assert set(lc.held_out) == {"A1+A2", "B", "C"}
    assert lc.k_train_groups.min() == 1 and lc.k_train_groups.max() == 2
    # with 3 families, holding one out leaves at most 2 to train on
    assert (lc.k_train_groups <= 2).all()
    assert "spearman" in lc.columns


def test_learning_curve_improves_as_families_are_added():
    """The point of the curve: more training groups should not make it worse on average."""
    r = np.random.default_rng(3)
    rows = []
    for g in range(6):
        for i in range(300):
            x, z = r.normal(), r.normal()
            rows.append({"dataset_id": f"P{g}", "family": f"P{g}", "pos": i % 50,
                         "delta_biological": x, "rsa": z, "is_tm": float(i % 2),
                         "score_z": 1.2 * x - 0.8 * z + r.normal(0, 0.6)})
    d = pd.DataFrame(rows)
    lc = learning_curve(d, lambda: Baseline("gbm", features=["delta_biological", "rsa", "is_tm"]),
                        group="family", repeats=3, seed=0)
    g = lc.groupby("k_train_groups").spearman.mean()
    assert g.loc[5] >= g.loc[1] - 0.02
