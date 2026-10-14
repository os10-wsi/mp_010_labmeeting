"""ESM-1v masked-marginal bookkeeping (model-free) and the A15 functional-site call."""
import pytest
import numpy as np

from mpdms.esm_score import AA20, masked_marginals, to_long, windows


def test_windows_cover_long_sequences():
    w = windows(2500)
    assert w[0][0] == 0 and w[-1][1] == 2500
    assert all(b - a <= 1022 for a, b in w)
    assert all(w[i + 1][0] < w[i][1] for i in range(len(w) - 1))  # overlapping


def test_masked_marginals_relative_to_wt():
    seq = "MKLVAG" * 200  # 1200 aa -> two windows
    seen = []

    def fake(items):
        seen.extend(len(s) for s, _ in items)
        # log p is 0 for alanine, -1 elsewhere
        return np.array([[0.0 if a == "A" else -1.0 for a in AA20] for _ in items])

    m = masked_marginals(seq, fake, batch=16)
    assert m.shape == (1200, 20) and max(seen) <= 1022
    wt_cols = [AA20.index(a) for a in seq]
    assert np.allclose(m[np.arange(1200), wt_cols], 0)          # wt -> wt scores 0
    assert np.allclose(m[seq.index("K"), AA20.index("A")], 1)    # K->A favoured
    tab = to_long(seq[:6], [m[:6]], [1])
    assert len(tab) == 6 * 19 and set(tab.columns) >= {"pos", "wt", "mut", "esm1v"}


def test_functional_list_is_complete_wrapped_and_in_sequence_order():
    """Panels (b)/(c) only label the top few; the list panel must carry every residue."""
    import numpy as np
    import pandas as pd

    from mpdms.analyses.a15_esm_functional import functional_list
    rng = np.random.default_rng(0)
    pos = sorted(rng.choice(np.arange(1, 531), 130, replace=False))
    f = pd.DataFrame({"pos": pos, "wt": list("ACDEFGHIKLMNPQRSTVWY") * 6 + ["A"] * 10,
                      "abundance_tolerant": rng.random(130) < 0.4})
    lines = functional_list(f)
    joined = " ".join(lines)
    assert all(f"{r.wt}{r.pos}" in joined for r in f.itertuples())        # nothing dropped
    assert all(len(ln) <= 140 for ln in lines)                            # wrapped to width
    order = [int(tok.rstrip("*")[1:]) for tok in joined.split()]
    assert order == sorted(order)                                         # sequence order
    assert joined.count("*") == int(f.abundance_tolerant.sum())           # tolerant marked
    assert functional_list(f.iloc[:0]) == ["none"]


# ------------------------------- finding the right protein in a proteome-wide table
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _proteome(target_seq, target_acc="TARGET", n_decoys=60, seed=0):
    """A table of many proteins, one of which is the target, as a whole-proteome file is."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_decoys):
        acc = f"DECOY{k:03d}"
        seq = "".join(rng.choice(list(AA20), rng.integers(150, 400)))
        for pos in range(1, len(seq) + 1, 5):
            for mut in rng.choice(list(AA20), 3, replace=False):
                rows.append({"id": acc, "mutation": f"{seq[pos - 1]}{pos}{mut}",
                             "mean": float(rng.normal(-5, 2))})
    for pos in range(1, len(target_seq) + 1):
        for mut in rng.choice(list(AA20), 4, replace=False):
            rows.append({"id": target_acc, "mutation": f"{target_seq[pos - 1]}{pos}{mut}",
                         "mean": float(rng.normal(-5, 2))})
    return pd.DataFrame(rows)


def _seq(n=320, seed=7):
    import numpy as np
    return "".join(np.random.default_rng(seed).choice(list(AA20), n))


def test_accession_found_by_sequence_in_a_proteome_table(tmp_path):
    """The accession need not be known: the mutation strings carry the wild-type residues."""
    from mpdms.esm_score import match_accession, read_bulk_table
    seq = _seq()
    p = tmp_path / "proteome.csv"
    _proteome(seq, target_acc="Q12345").to_csv(p, index=False)
    acc, info = match_accession(read_bulk_table(p), seq)
    assert acc == "Q12345"
    assert info["agreement"] == 1.0 and info["positions"] == len(seq)
    assert info["runner_up_agreement"] < 0.5          # chance agreement for the decoys


def test_no_accession_returned_when_the_protein_is_absent(tmp_path):
    """Importing another protein's scores would look normal downstream, so refuse instead."""
    from mpdms.esm_score import match_accession, read_bulk_table
    seq = _seq()
    p = tmp_path / "proteome.csv"
    t = _proteome(seq, target_acc="Q12345")
    t[t.id != "Q12345"].to_csv(p, index=False)
    acc, info = match_accession(read_bulk_table(p), seq)
    assert acc is None and "below" in info["reason"]


def test_match_ignores_accessions_covering_too_few_positions(tmp_path):
    """A tiny entry can agree perfectly by luck; it must not outrank the real protein."""
    import pandas as pd
    from mpdms.esm_score import match_accession, read_bulk_table
    seq = _seq()
    t = _proteome(seq, target_acc="Q12345", n_decoys=5)
    tiny = pd.DataFrame([{"id": "TINY", "mutation": f"{seq[i]}{i + 1}A", "mean": -5.0}
                         for i in range(4)])          # 4 positions, all correct
    p = tmp_path / "proteome.csv"
    pd.concat([t, tiny]).to_csv(p, index=False)
    acc, info = match_accession(read_bulk_table(p), seq)
    assert acc == "Q12345"


def test_streaming_match_agrees_with_the_in_memory_one(tmp_path):
    """Streaming is only worth having if it gives the same answer as loading the table."""
    from mpdms.esm_score import match_accession, match_accession_streaming, read_bulk_table
    seq = _seq()
    p = tmp_path / "proteome.csv"
    _proteome(seq, target_acc="Q12345", n_decoys=6).to_csv(p, index=False)
    whole, winfo = match_accession(read_bulk_table(p), seq)
    for chunk in (7, 101, 10_000):                    # chunk boundaries must not matter
        acc, info = match_accession_streaming(p, seq, chunksize=chunk)
        assert acc == whole == "Q12345"
        assert info["positions"] == winfo["positions"]
        assert info["agreement"] == pytest.approx(winfo["agreement"])
        assert info["n_variants"] == winfo["n_variants"]


def test_streaming_extract_returns_only_that_protein(tmp_path):
    from mpdms.esm_score import read_bulk_table, rows_for_accession
    seq = _seq()
    p = tmp_path / "proteome.csv"
    _proteome(seq, target_acc="Q12345", n_decoys=4).to_csv(p, index=False)
    want = read_bulk_table(p).query("acc == 'Q12345'").reset_index(drop=True)
    got = rows_for_accession(p, "Q12345", chunksize=137)
    assert set(got.acc) == {"Q12345"}
    assert len(got) == len(want)
    assert got.sort_values(["pos", "mut"]).esm1v.tolist() == \
           want.sort_values(["pos", "mut"]).esm1v.tolist()


def test_mutation_parsing_matches_the_regex_it_replaced():
    """Slicing is quicker than the regex, so it must agree with it, junk rows included."""
    import pandas as pd
    from mpdms.esm_score import MUT_RE, parse_mutations
    s = pd.Series(["M1A", "K123R", "W1000Y", "", "x", "nonsense", "M1", "1A2", "A12b", "QQ3R"])
    ref = s.str.extract(MUT_RE)
    got = parse_mutations(s)
    assert got.pos.notna().tolist() == pd.to_numeric(ref[1]).notna().tolist()
    keep = got.pos.notna()
    assert got.loc[keep, "wt"].tolist() == ref.loc[keep, 0].tolist()
    assert got.loc[keep, "mut"].tolist() == ref.loc[keep, 2].tolist()
    assert got.loc[keep, "pos"].tolist() == pd.to_numeric(ref.loc[keep, 1]).tolist()


def test_a_protein_with_no_config_can_be_scored_by_accession(tmp_path, monkeypatch):
    """A human family member has no config, but still has to be scored the same way."""
    import pandas as pd

    from mpdms import esm_score as E
    seq = "MAKTLVIGAGSLSWFPRQDEKNMHYC" * 2
    calls = {}

    def fake_fetch(acc):
        calls["acc"] = acc
        return seq

    def fake_score(s, models, device, batch):
        calls["len"] = len(s)
        return pd.DataFrame({"pos": [1, 1], "wt": ["M", "M"], "mut": ["A", "V"],
                             "esm1v": [-1.0, -2.0]})

    monkeypatch.setattr("mpdms.clinvar.fetch_sequence", fake_fetch)
    monkeypatch.setattr(E, "score_sequence", fake_score)
    monkeypatch.setattr(E, "REPO_ROOT", tmp_path)
    E.main(["--human", "slc2a10=O95528"])
    assert calls == {"acc": "O95528", "len": len(seq)}
    out = tmp_path / "data" / "external" / "SLC2A10" / "SLC2A10_esm1v.csv"
    assert out.exists()
    # the file lands where the clinvar reader expects it, and reads back
    from mpdms.analyses.a15_esm_functional import read_esm
    assert len(read_esm(out)) == 2


def test_a_local_fasta_works_too_and_a_bad_spec_is_refused(tmp_path, monkeypatch):
    import pandas as pd
    import pytest as _pytest

    from mpdms import esm_score as E
    f = tmp_path / "q.fasta"
    f.write_text(">sp|X|QUERY\nMAKTLV\nIGAGSL\n")
    seen = {}
    monkeypatch.setattr(E, "score_sequence",
                        lambda s, m, d, b: seen.setdefault("seq", s) and None
                        or pd.DataFrame({"pos": [1], "wt": ["M"], "mut": ["A"],
                                         "esm1v": [-1.0]}))
    monkeypatch.setattr(E, "REPO_ROOT", tmp_path)
    E.main(["--fasta", f"QUERY={f}"])
    assert seen["seq"] == "MAKTLVIGAGSL"      # header dropped, lines joined
    with _pytest.raises(SystemExit):
        E.main(["--fasta", "nogene"])
    with _pytest.raises(SystemExit):
        E.main([])


def test_an_ensemble_table_with_a_mutation_column_is_read():
    """The layout a batch ESM-1v run writes: id, mutation, delta_logp_1..5, mean."""
    import pandas as pd

    from mpdms.analyses.a15_esm_functional import read_esm
    import tempfile
    import pathlib
    t = pathlib.Path(tempfile.mkdtemp()) / "ens.csv"
    pd.DataFrame({"Unnamed: 0": [0, 1, 2], "id": ["P40107"] * 3,
                  "mutation": ["M1A", "M1C", "junk"],
                  "delta_logp_1": [-7.0, -8.5, 0.0],
                  "mean": [-6.17, -7.73, 0.0]}).to_csv(t, index=False)
    e = read_esm(t)
    assert len(e) == 2                       # the unparseable row is dropped
    assert e.attrs["source_column"] == "mean"
    assert list(e.wt) == ["M", "M"] and list(e.mut) == ["A", "C"]
    assert list(e.pos) == [1, 1]
    assert e.esm1v.iloc[0] == pytest.approx(-6.17)


def test_mean_alone_is_not_taken_as_a_score():
    """`mean` is too generic a name to claim without the layout that identifies it."""
    import pandas as pd
    import pathlib
    import tempfile

    from mpdms.analyses.a15_esm_functional import read_esm
    t = pathlib.Path(tempfile.mkdtemp()) / "x.csv"
    pd.DataFrame({"pos": [1, 2], "mut": ["A", "C"], "mean": [1.0, 2.0]}).to_csv(t,
                                                                                index=False)
    assert read_esm(t) is None


def test_highlight_positions_come_from_the_config_or_the_environment(monkeypatch):
    from mpdms.analyses.a15_esm_functional import highlight_positions

    class Cfg:
        def __init__(self, v):
            self.v = v

        def get_path(self, key, default=None):
            return self.v if key == "plotting.highlight_positions" else default

    monkeypatch.delenv("MPDMS_HIGHLIGHT", raising=False)
    assert highlight_positions(Cfg([370, 335, 336, 335])) == [335, 336, 370]
    assert highlight_positions(Cfg(None)) == []
    # the environment overrides the config entirely, for a one-off run
    monkeypatch.setenv("MPDMS_HIGHLIGHT", "473, 209")
    assert highlight_positions(Cfg([1, 2])) == [209, 473]
    # and a value that is not a number is dropped with a note, not a crash
    monkeypatch.setenv("MPDMS_HIGHLIGHT", "473,Q335A,")
    assert highlight_positions(Cfg(None)) == [473]


def test_a_wide_sequence_track_is_exempt_from_squaring(tmp_path, monkeypatch):
    """MPDMS_SQUARE must not put 600 residues in a box as tall as it is wide."""
    import matplotlib.pyplot as plt

    from mpdms import plotting as P
    monkeypatch.setenv("MPDMS_SQUARE", "1")
    fig, (a, b) = plt.subplots(1, 2)
    P.no_square(b)
    c = fig.add_axes([0.1, 0.01, 0.8, 0.05])
    c.axis("off")
    P.save(fig, tmp_path, "mixed")
    assert a.get_box_aspect() == 1          # an ordinary panel is squared
    assert b.get_box_aspect() is None       # the exempt track is not
    assert c.get_box_aspect() is None       # nor is a text panel with its frame off


def test_the_structure_colour_range_is_anchored_to_the_calling_threshold(monkeypatch):
    """A percentile rule would be set by the constrained tail it is meant to show."""
    import importlib
    from mpdms.analyses import a15_esm_functional as A15
    assert A15.colour_range() == pytest.approx(3 * abs(A15.Z_SITE))
    # it is symmetric by construction, so white lands exactly on a residual of zero
    assert A15.colour_range() > 0
    monkeypatch.setenv("MPDMS_Z_COLOR_RANGE", "5.5")
    assert A15.colour_range() == pytest.approx(5.5)
    monkeypatch.setenv("MPDMS_Z_COLOR_RANGE", "-5.5")       # sign is taken care of
    assert A15.colour_range() == pytest.approx(5.5)
    monkeypatch.setenv("MPDMS_Z_COLOR_RANGE", "not-a-number")
    assert A15.colour_range() == pytest.approx(3 * abs(A15.Z_SITE))
    monkeypatch.delenv("MPDMS_Z_COLOR_RANGE")
    # and it follows the threshold, so the colours cannot contradict the calls
    monkeypatch.setenv("MPDMS_Z_SITE", "2.0")
    reloaded = importlib.reload(A15)
    try:
        assert reloaded.Z_SITE == pytest.approx(-2.0)
        assert reloaded.colour_range() == pytest.approx(6.0)
    finally:
        monkeypatch.delenv("MPDMS_Z_SITE")
        importlib.reload(A15)


def test_a_cif_structure_is_parsed_rather_than_silently_emptied(tmp_path):
    """PDBParser on a .cif returns zero residues with no error, and PDBIO then writes a
    seven-byte file containing only "END" - which is what reached the user."""
    import warnings
    from Bio.PDB import MMCIFIO, PDBIO, PDBParser
    warnings.filterwarnings("ignore")
    pdb = tmp_path / "m.pdb"
    lines = []
    for i in range(1, 31):
        lines.append(f"ATOM  {i:>5}  CA  ALA A{i:>4}    "
                     f"{i * 3.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C\n")
    pdb.write_text("".join(lines) + "END\n")
    st = PDBParser(QUIET=True).get_structure("s", str(pdb))
    cif = tmp_path / "m.cif"
    w = MMCIFIO(); w.set_structure(st); w.save(str(cif))

    # the failure mode, pinned so it cannot come back unnoticed
    empty = PDBParser(QUIET=True).get_structure("x", str(cif))
    assert sum(1 for _ in empty.get_residues()) == 0
    out = tmp_path / "empty.pdb"
    io = PDBIO(); io.set_structure(empty); io.save(str(out))
    assert out.stat().st_size < 20

    # what the project's own loader does with the same file
    from mpdms.structure import load_structure
    good = load_structure(cif, "A")[0]
    assert sum(1 for _ in good.get_residues()) == 30
