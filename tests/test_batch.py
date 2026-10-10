"""Pairing a folder of DiMSum output into one run per protein."""
from pathlib import Path

from mpdms.batch import discover, find_companion, name_candidates


def _farm(tmp_path: Path, names) -> Path:
    for n in names:
        (tmp_path / n).write_text("x")
    return tmp_path


REAL = ["aqr1.cif", "aqr1.gff3", "fitness_singles_aqr1_003.txt",
        "fitness_singles_hxt1.txt", "fitness_singles_hxt2.txt", "fitness_singles_hxt3.txt",
        "fitness_singles_hxt7.txt", "hxt1.cif", "hxt1.gff3", "hxt2.cif", "hxt2.gff3",
        "hxt3.cif", "hxt3.gff3", "hxt7.cif", "hxt7.gff3"]


def test_pairs_a_real_dimsum_folder(tmp_path):
    got = {e["gene"]: e for e in discover(_farm(tmp_path, REAL))}
    assert set(got) == {"AQR1", "HXT1", "HXT2", "HXT3", "HXT7"}
    for gene, e in got.items():
        assert e["gff3"].name == f"{gene.lower()}.gff3"
        assert e["structure"].name == f"{gene.lower()}.cif"
    # the run suffix belongs to the file, not to the gene: _literature.yaml is keyed on AQR1
    assert got["AQR1"]["fitness"].name == "fitness_singles_aqr1_003.txt"


def test_run_suffix_is_only_trimmed_when_the_companions_say_so(tmp_path):
    """HXT2_2 with its own hxt2_2.gff3 is a second protein, not HXT2."""
    d = _farm(tmp_path, ["fitness_singles_hxt2_2.txt", "hxt2_2.gff3", "hxt2_2.cif"])
    e, = discover(d)
    assert e["gene"] == "HXT2_2"
    assert e["gff3"].name == "hxt2_2.gff3"


def test_missing_companions_are_reported_not_guessed(tmp_path):
    d = _farm(tmp_path, ["fitness_singles_hxt9.txt", "hxt1.gff3", "hxt1.cif"])
    e, = discover(d)
    assert e["gene"] == "HXT9" and e["gff3"] is None and e["structure"] is None


def test_an_ambiguous_prefix_is_not_resolved(tmp_path):
    """hxt1.gff3 and hxt10.gff3 both start with hxt1; picking either would be a guess."""
    d = _farm(tmp_path, ["fitness_singles_hxt1x.txt", "hxt1.gff3", "hxt10.gff3"])
    e, = discover(d)
    assert e["gff3"] is None


def test_decorated_companion_names_still_pair(tmp_path):
    d = _farm(tmp_path, ["fitness_singles_hxt2.txt", "hxt2_deeptmhmm.gff3", "hxt2_AF_model.cif"])
    e, = discover(d)
    assert e["gff3"].name == "hxt2_deeptmhmm.gff3"
    assert e["structure"].name == "hxt2_AF_model.cif"


def test_name_candidates_trims_run_and_replicate_suffixes():
    assert name_candidates("AQR1_003") == ["aqr1_003", "aqr1"]
    assert name_candidates("HXT2") == ["hxt2"]
    assert name_candidates("PHO84_rep2_v3")[-1] == "pho84"


def test_find_companion_prefers_an_exact_name_over_a_prefix(tmp_path):
    d = _farm(tmp_path, ["hxt2.gff3", "hxt2_old.gff3"])
    assert find_companion(d, "HXT2", (".gff3",)).name == "hxt2.gff3"
