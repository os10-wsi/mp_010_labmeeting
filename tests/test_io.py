"""Loader and normalisation: syn median -> 0, stop median -> -1, several input formats."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mpdms.config import Config, defaults
from mpdms.io import load_dataset, parse_variant_string


def _cfg(tmp_path: Path, table: pd.DataFrame, **over) -> Config:
    src = tmp_path / "fitness_estimation.tsv"
    table.to_csv(src, sep="\t", index=False)
    d = defaults()
    d.update(id="TST1", display_name="TST1")
    d["source"]["path"] = str(src)
    d["filters"]["min_reads"] = 0
    d["filters"]["min_replicates"] = 1
    for k, v in over.items():
        d[k].update(v)
    cfg = Config.wrap(d)
    return cfg


def _toy(rng, n_pos=40):
    rows = []
    for p in range(1, n_pos + 1):
        for m, f in [("A", -0.5), ("*", -2.0), ("M", 0.3), ("L", -1.0)]:
            rows.append({"Pos": p, "WT_AA": "M", "Mut": m, "fitness1": f + rng.normal(0, .01),
                         "fitness2": f + rng.normal(0, .01)})
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch, tmp_path):
    import mpdms.io as io
    monkeypatch.setattr(io.Config, "resolve", lambda self, p: None if not p else (
        tmp_path / Path(p).name if str(p).startswith("data/processed") else Path(p)))


def test_normalisation_anchors(tmp_path):
    rng = np.random.default_rng(0)
    df = load_dataset(_cfg(tmp_path, _toy(rng)))
    ok = df.pass_filter
    assert abs(df.loc[ok & (df.vclass == "synonymous"), "score_z"].median()) < 1e-9
    assert abs(df.loc[ok & (df.vclass == "nonsense"), "score_z"].median() + 1) < 1e-9
    # linear: A (-0.5 raw) sits at (-0.5-0.3)/(0.3+2.0) = -0.348
    assert abs(df.loc[df.mut == "A", "score_z"].median() + 0.8 / 2.3) < 0.02
    for r in ("rep1", "rep2"):
        assert abs(df.loc[ok & (df.vclass == "nonsense"), r].median() + 1) < 1e-9


def test_sign_flip(tmp_path):
    rng = np.random.default_rng(1)
    t = _toy(rng)
    t[["fitness1", "fitness2"]] *= -1
    df = load_dataset(_cfg(tmp_path, t, assay={"higher_is_better": False}))
    assert df.loc[df.vclass == "nonsense", "score_z"].median() == pytest.approx(-1)


def test_hgvs_input(tmp_path):
    rng = np.random.default_rng(2)
    t = _toy(rng)
    three = {"A": "Ala", "*": "Ter", "M": "=", "L": "Leu"}
    t["variant"] = [f"p.Met{p}{three[m]}" for p, m in zip(t.Pos, t.Mut)]
    df = load_dataset(_cfg(tmp_path, t.drop(columns=["Pos", "WT_AA", "Mut"])))
    assert set(df.vclass) == {"missense", "synonymous", "nonsense"}


def test_min_reads_filter(tmp_path):
    rng = np.random.default_rng(3)
    t = _toy(rng)
    t["input1"] = 100
    t["input2"] = 100
    t.loc[t.Pos <= 5, ["input1", "input2"]] = 2
    df = load_dataset(_cfg(tmp_path, t, filters={"min_reads": 10, "min_replicates": 2}))
    assert not df.loc[df.pos <= 5, "pass_filter"].any()
    assert df.loc[df.pos > 5, "pass_filter"].all()


@pytest.mark.parametrize("s,exp", [("M1A", (1, "M", "A")), ("p.Leu23Pro", (23, "L", "P")),
                                   ("p.(Trp5Ter)", (5, "W", "*")), ("p.Gly7=", (7, "G", "="))])
def test_parse_variant(s, exp):
    assert parse_variant_string(s) == exp


def test_dimsum_aa_seq_input(tmp_path):
    wt = "MKLVA"
    wt_nt = "ATGAAACTGGTTGCT"
    rows = [{"aa_seq": wt, "nt_seq": wt_nt, "WT": True, "Nham_aa": 0, "fitness1_uncorr": 0.0, "fitness2_uncorr": 0.0}]
    rows.append({"aa_seq": wt, "nt_seq": "ATGAAGCTGGTTGCT", "WT": False, "Nham_aa": 0,
                 "fitness1_uncorr": 0.05, "fitness2_uncorr": -0.05})  # synonymous at codon 2
    for i in range(1, 5):
        for m, f in [("*", -2.0), ("P", -1.0), ("A" if wt[i] != "A" else "G", -0.2)]:
            s = wt[:i] + m + wt[i + 1:]
            rows.append({"aa_seq": s, "nt_seq": "", "WT": False, "Nham_aa": 1, "fitness1_uncorr": f, "fitness2_uncorr": f})
    for k in range(6):  # a few more synonymous rows so the 0 anchor is not a fallback
        rows.append(dict(rows[1], fitness1_uncorr=0.01 * k, fitness2_uncorr=-0.01 * k))
    df = load_dataset(_cfg(tmp_path, pd.DataFrame(rows)))
    assert (df.vclass == "nonsense").sum() == 4
    syn = df[df.vclass == "synonymous"]
    assert set(syn.pos) == {2} and syn.codon.iloc[0] == "AAG"
    assert df.attrs["wt_row_score_raw"] == 0.0
