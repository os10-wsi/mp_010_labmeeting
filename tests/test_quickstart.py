"""quickstart: one fitness file plus a gff3 should need nothing else."""
import numpy as np
import pandas as pd
import pytest

from mpdms.quickstart import classify_counts, gene_from_filename, wt_sequence

WT = "MSEFATSRVESGSQQTSIHSTPIVQKLETDESPIQTKSEYTNAELPAKPIAAYWTVICLCLMIAFGGFVFGWDTGTISGF"


@pytest.mark.parametrize("name,gene", [
    ("fitness_singles_hxt2.txt", "HXT2"),
    ("fitness_singles_HXT2.tsv", "HXT2"),
    ("fitness_single_aqr1.txt", "AQR1"),
    ("qdr2_fitness.tsv", "QDR2"),
    ("fitness_pho84.txt", "PHO84"),
    ("something_else.tsv", "SOMETHING_ELSE"),
])
def test_gene_inferred_from_the_filename(name, gene):
    from pathlib import Path
    assert gene_from_filename(Path(name)) == gene


def _table(n_syn=5, n_stop=4, corrupt_one=False):
    """A DiMSum-shaped table whose aa_seq is the mutant sequence, as the real one is."""
    rows = []
    for pos in range(2, 40):
        for mut in "ACDEFG":
            wt = WT[pos - 1]
            if mut == wt:
                continue
            rows.append({"Pos": pos, "WT_AA": wt, "Mut": mut, "Nham_aa": 1, "STOP": "FALSE",
                         "aa_seq": WT[:pos - 1] + mut + WT[pos:], "fitness": -0.3, "sigma": 0.2})
    for k in range(n_syn):
        pos = 5 + k
        rows.append({"Pos": pos, "WT_AA": WT[pos - 1], "Mut": WT[pos - 1], "Nham_aa": 0,
                     "STOP": "FALSE", "aa_seq": WT, "fitness": 0.0, "sigma": 0.2})
    for k in range(n_stop):
        pos = 20 + k
        rows.append({"Pos": pos, "WT_AA": WT[pos - 1], "Mut": "*", "Nham_aa": 1, "STOP": "TRUE",
                     "aa_seq": WT[:pos - 1] + "*" + WT[pos:], "fitness": -1.0, "sigma": 0.3})
    d = pd.DataFrame(rows)
    if corrupt_one:
        d.loc[0, "aa_seq"] = "X" + d.loc[0, "aa_seq"][1:]
    return d


def test_wild_type_sequence_is_recovered_from_the_mutant_sequences():
    """aa_seq is the MUTANT protein; reverting each row's own substitution gives the WT."""
    d = _table()
    seq, info = wt_sequence(d, "Pos", "WT_AA", "Mut", "aa_seq")
    assert info["ok"] and seq == WT
    assert info["length"] == len(WT) and info["agreement"] == pytest.approx(1.0)


def test_sequence_recovery_reports_disagreement_rather_than_picking_quietly():
    d = _table(corrupt_one=True)
    seq, info = wt_sequence(d, "Pos", "WT_AA", "Mut", "aa_seq")
    assert seq == WT                      # consensus still right
    assert info["n_distinct"] > 1 and info["agreement"] < 1.0   # but the disagreement is reported


def test_variant_classes_counted_from_hamming_and_stop_flag():
    from mpdms.io import CANDIDATES, _find
    d = _table(n_syn=5, n_stop=4)
    cols = {k: _find(list(d.columns), v) for k, v in CANDIDATES.items()}
    c = classify_counts(d, cols)
    assert c["synonymous"] == 5 and c["nonsense"] == 4
    assert c["missense"] == len(d) - 9


def test_a_stop_is_never_counted_as_synonymous():
    """A stop row has Nham_aa = 1, but a mis-specified table could mark both."""
    from mpdms.io import CANDIDATES, _find
    d = _table(n_syn=0, n_stop=3)
    d.loc[d.Mut == "*", "Nham_aa"] = 0           # pathological but seen in the wild
    cols = {k: _find(list(d.columns), v) for k, v in CANDIDATES.items()}
    c = classify_counts(d, cols)
    assert c["nonsense"] == 3 and c["synonymous"] == 0


def test_quickstart_writes_a_config_and_fasta_without_a_network(tmp_path, monkeypatch):
    import mpdms.quickstart as Q
    from mpdms.config import load_config
    monkeypatch.setattr(Q, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    src = tmp_path / "fitness_singles_test1.txt"
    _table().to_csv(src, sep="\t", index=False)
    cfg_path, counts = Q.build(src, "TEST1", None, 0)
    assert cfg_path.exists() and counts["synonymous"] == 5
    cfg = load_config(cfg_path)
    assert cfg.id == "TEST1"
    fasta = tmp_path / "data" / "external" / "TEST1" / "TEST1.fasta"
    assert fasta.exists() and fasta.read_text().split("\n")[1] == WT


def test_hxt2_sites_load_with_wild_types_for_the_numbering_check():
    """The nine sites must come through with wt residues so A19/A21 can verify numbering."""
    from mpdms.analyses.a20_variant_panel import wanted_sites
    from mpdms.config import Config
    cfg = Config.wrap({"id": "HXT2", "display_name": "HXT2", "protein": {"gene": "HXT2"}})
    sites, wt, marked = wanted_sites(cfg)
    assert sites == [59, 61, 198, 201, 316, 331, 363, 366, 368]
    assert wt == {59: "L", 61: "L", 198: "F", 201: "L", 316: "V",
                  331: "N", 363: "A", 366: "F", 368: "A"}
    assert marked == {}          # positions only: no allele to ring in A21


def test_hxt2_wild_types_match_the_recovered_sequence():
    """Guards the numbering: these came off the sequence recovered from the DMS table."""
    from pathlib import Path

    from mpdms.analyses.a19_literature import check_numbering, load_literature
    f = Path("data/external/HXT2/HXT2.fasta")
    if not f.exists():
        pytest.skip("HXT2 fasta not present in this checkout")
    seq = f.read_text().split("\n")[1]
    entries = load_literature("HXT2")["measured"]
    ok, bad = check_numbering(entries, seq)
    assert not bad and len(ok) == 9


def test_a_different_sequence_for_the_same_gene_is_not_swapped_in_silently(tmp_path, monkeypatch, capsys):
    """Every config for a gene points at one fasta; replacing it renumbers finished work."""
    import pandas as pd
    import mpdms.quickstart as Q

    monkeypatch.setattr(Q, "REPO_ROOT", tmp_path)
    ext = tmp_path / "data" / "external" / "GENEX"
    ext.mkdir(parents=True)
    (ext / "GENEX.fasta").write_text(">GENEX\nMKKKKKKKKKKKKKKKKKKKK\n")
    (tmp_path / "configs").mkdir()

    seq = "MAAAAAAAAAAAAAAAAAAAA"
    rows = [{"Pos": p, "WT_AA": seq[p - 1], "Mut": "G", "aa_seq": seq[:p - 1] + "G" + seq[p:],
             "fitness": -0.3, "sigma": 0.1} for p in range(2, len(seq) + 1)]
    src = tmp_path / "fitness_singles_genex.txt"
    pd.DataFrame(rows).to_csv(src, sep="\t", index=False)

    Q.build(src, "GENEX", None, 0)
    out = capsys.readouterr().out
    assert "already held a DIFFERENT sequence" in out
    assert (ext / "GENEX.fasta.previous").read_text().split("\n")[1].startswith("MKKK")
    assert (ext / "GENEX.fasta").read_text().split("\n")[1] == seq
