"""Pairwise identity: the context that says whether a pooled result can mean anything."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms.identity import cluster_order, pairwise_identity

BASE = "MKLWTTAVGCLAVSAGGLFFGYDTGVISGAMPFIQKEFNLSSLQEGLVVSSLLVGA" * 2


def _mutate(seq, n, seed=0):
    r = np.random.default_rng(seed)
    s = list(seq)
    for i in r.choice(len(s), n, replace=False):
        s[i] = str(r.choice(list("ACDEFGHIKLMNPQRSTVWY")))
    return "".join(s)


def test_identity_is_one_on_the_diagonal_and_symmetric():
    M = pairwise_identity({"A": BASE, "B": _mutate(BASE, 20)})
    assert np.allclose(np.diag(M.to_numpy()), 1.0)
    assert M.loc["A", "B"] == M.loc["B", "A"]
    assert 0 < M.loc["A", "B"] < 1


def test_more_mutations_mean_less_identity():
    M = pairwise_identity({"A": BASE, "near": _mutate(BASE, 10, 1),
                           "far": _mutate(BASE, 60, 2)})
    assert M.loc["A", "near"] > M.loc["A", "far"]


def test_an_unrelated_sequence_scores_low():
    r = np.random.default_rng(3)
    other = "".join(r.choice(list("ACDEFGHIKLMNPQRSTVWY"), len(BASE)))
    M = pairwise_identity({"A": BASE, "X": other})
    assert M.loc["A", "X"] < 0.35


def test_identity_is_normalised_by_the_shorter_sequence():
    """A long extension must not read as divergence when the shared part is conserved."""
    M = pairwise_identity({"short": BASE, "extended": BASE + "G" * 200})
    assert M.loc["short", "extended"] > 0.95


def test_a_missing_sequence_is_nan_not_zero():
    M = pairwise_identity({"A": BASE, "empty": ""})
    assert np.isnan(M.loc["A", "empty"])


def test_clustering_puts_relatives_next_to_each_other():
    seqs = {"A1": BASE, "A2": _mutate(BASE, 8, 4), "A3": _mutate(BASE, 12, 5)}
    r = np.random.default_rng(6)
    other = "".join(r.choice(list("ACDEFGHIKLMNPQRSTVWY"), len(BASE)))
    seqs["B1"] = other
    seqs["B2"] = _mutate(other, 8, 7)
    order = cluster_order(pairwise_identity(seqs))
    pos = {g: i for i, g in enumerate(order)}
    a = sorted(pos[g] for g in ("A1", "A2", "A3"))
    b = sorted(pos[g] for g in ("B1", "B2"))
    assert a == list(range(a[0], a[0] + 3))       # the three relatives are contiguous
    assert b == list(range(b[0], b[0] + 2))


def test_cluster_order_keeps_every_protein():
    seqs = {n: _mutate(BASE, 10 * i, i) for i, n in enumerate("ABCD")}
    order = cluster_order(pairwise_identity(seqs))
    assert sorted(order) == sorted(seqs)
