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


def _user_format_table(rng):
    """Mirrors the lab's fitness_estimation.tsv: spaced column names, synonymous rows with
    empty pos/wt aa/mut aa (aa_ham = 0), boolean wt/stop flags, raw + rescaled replicates."""
    from mpdms.io import CODON_TABLE
    aa2codon = {}
    for c, a in CODON_TABLE.items():
        aa2codon.setdefault(a, c)
    prot = "MKLVAGSTWE"
    cds = "".join(aa2codon[a] for a in prot)
    syn_alt = {"L": "CTT", "V": "GTC", "A": "GCC", "G": "GGC", "S": "TCC"}
    rows = []

    def row(wt, pos, mut, aa_ham, aa, nt_ham, nt, f, flag_wt="", flag_stop=""):
        r = {"wt aa": wt, "pos": pos, "mut aa": mut, "aa_ham": aa_ham, "aa_seq": aa, "nt_ham": nt_ham, "nt_seq": nt}
        for k in (1, 2, 3):
            r[f"input{k}"] = 20
            r[f"output{k}"] = 20
        r["wt"], r["stop"] = flag_wt, flag_stop
        for k in (1, 2, 3):
            r[f"raw_fitness_rep{k}"] = f + rng.normal(0, .02)
            r[f"rescaled_fitness_rep{k}"] = f + rng.normal(0, .02)
        r["mean fitness"] = f
        r["fitness sd"] = 0.02
        return r

    rows.append(row("", "", "", 0, prot, 0, cds, 0.0, flag_wt=True))
    for i, a in enumerate(prot):
        p = i + 1
        if a in syn_alt:  # synonymous: no pos / wt aa / mut aa
            nt = cds[:i * 3] + syn_alt[a] + cds[i * 3 + 3:]
            rows.append(row("", "", "", 0, prot, 1, nt, 0.05))
        for m, f in (("P", -0.6), ("*", -2.0)):
            aa = prot[:i] + m + prot[i + 1:]
            codon = "TAA" if m == "*" else "CCT"
            nt = cds[:i * 3] + codon + cds[i * 3 + 3:]
            rows.append(row(a, p, m, 1, aa, 3, nt, f, flag_stop=(m == "*")))
    return pd.DataFrame(rows), prot


def test_lab_fitness_estimation_format(tmp_path):
    rng = np.random.default_rng(5)
    t, prot = _user_format_table(rng)
    cfg = _cfg(tmp_path, t, filters={"min_reads": 5, "min_replicates": 2})
    df = load_dataset(cfg)
    cols = df.attrs["columns_detected"]
    assert cols["position"] == "pos" and cols["wt_aa"] == "wt aa" and cols["mut_aa"] == "mut aa"
    assert cols["score"] == "mean fitness" and cols["se"] == "fitness sd"
    assert cols["replicates"] == [f"rescaled_fitness_rep{k}" for k in (1, 2, 3)]
    syn = df[df.vclass == "synonymous"]
    assert sorted(syn.pos) == [3, 4, 5, 6, 7]                      # L3 V4 A5 G6 S7
    assert list(syn.wt) == [prot[p - 1] for p in sorted(syn.pos)]
    assert (df.vclass == "nonsense").sum() == len(prot)
    assert df.attrs["wt_row_score_raw"] == 0.0
    assert df.loc[df.vclass == "synonymous", "score_z"].median() == pytest.approx(0, abs=1e-9)
    assert df.loc[df.vclass == "nonsense", "score_z"].median() == pytest.approx(-1, abs=1e-9)


def test_synonymous_recovered_without_wt_row(tmp_path):
    """No WT row: the WT CDS is rebuilt as the consensus of all variant sequences."""
    rng = np.random.default_rng(6)
    t, _ = _user_format_table(rng)
    t = t.iloc[1:]
    df = load_dataset(_cfg(tmp_path, t))
    assert sorted(df[df.vclass == "synonymous"].pos) == [3, 4, 5, 6, 7]


# ------------------------------------------- DiMSum fitness_singles column naming
def test_dimsum_fitness_singles_columns_are_recognised():
    """Pos / WT_AA / Mut / Nham_aa / STOP / mean_count / fitness / sigma, as DiMSum writes them."""
    from mpdms.io import CANDIDATES, _find
    cols = ["Pos", "WT_AA", "Mut", "nt_seq", "aa_seq", "Nham_nt", "Nham_aa", "Nmut_codons",
            "STOP", "STOP_readthrough", "mean_count", "fitness", "sigma"]
    got = {k: _find(cols, v) for k, v in CANDIDATES.items()}
    assert got["position"] == "Pos" and got["wt_aa"] == "WT_AA" and got["mut_aa"] == "Mut"
    assert got["score"] == "fitness" and got["se"] == "sigma"
    assert got["aa_ham"] == "Nham_aa" and got["stop_flag"] == "STOP"
    assert got["n_reads"] == "mean_count"


def test_missing_controls_raise_a_normalisation_warning(tmp_path):
    """No synonymous variants means the scale is anchored on missense: must be said out loud."""
    import pandas as pd

    from mpdms.config import Config, validate
    df = pd.DataFrame({"pos": [1, 2], "wt": ["A", "A"], "mut": ["C", "D"],
                       "score_z": [0.1, -0.2], "vclass": ["missense", "missense"]})
    df.attrs["normalization"] = {"score": {"n_syn": 0, "n_stop": 3,
                                           "syn_source": "missense_median_FALLBACK",
                                           "stop_source": "nonsense_median"}}
    cfg = Config.wrap({"id": "X", "display_name": "X",
                       "source": {"path": str(tmp_path / "x.tsv")},
                       "protein": {"sequence": None}, "topology": {"segments": []}})
    (tmp_path / "x.tsv").write_text("pos\n1\n")
    w = validate(cfg, df, strict=False)
    assert any("NORMALISATION FALLBACK" in x for x in w), w
    assert any("synonymous n=0" in x for x in w)


def test_no_warning_when_both_anchors_are_real(tmp_path):
    import pandas as pd

    from mpdms.config import Config, validate
    df = pd.DataFrame({"pos": [1, 2], "wt": ["A", "A"], "mut": ["C", "D"],
                       "score_z": [0.1, -0.2], "vclass": ["missense", "missense"]})
    df.attrs["normalization"] = {"score": {"n_syn": 40, "n_stop": 30,
                                           "syn_source": "synonymous_median",
                                           "stop_source": "nonsense_median"}}
    cfg = Config.wrap({"id": "X", "display_name": "X",
                       "source": {"path": str(tmp_path / "x.tsv")},
                       "protein": {"sequence": None}, "topology": {"segments": []}})
    (tmp_path / "x.tsv").write_text("pos\n1\n")
    assert not any("FALLBACK" in x for x in validate(cfg, df, strict=False))
