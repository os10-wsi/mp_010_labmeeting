"""TMHMM annotation -> topology segments, and the alignment position mapping."""
import numpy as np
import pandas as pd
import pytest

from mpdms.family import _to_table, align_sequences, aligned_table
from mpdms.topology import parse_table, segments_from_annotation


def test_segments_orientation_alternates_from_inside_outside_labels():
    ann = "i" * 10 + "M" * 20 + "o" * 8 + "M" * 20 + "i" * 12
    segs, n_term = segments_from_annotation(ann)
    assert n_term == "cytosolic"
    tms = [s for s in segs if s["type"] == "TM"]
    assert [s["orientation"] for s in tms] == ["in_out", "out_in"]
    assert (tms[0]["start"], tms[0]["end"]) == (11, 30)
    assert (tms[1]["start"], tms[1]["end"]) == (39, 58)
    loops = [s for s in segs if s["type"] in ("loop", "soluble")]
    assert [s["side"] for s in loops] == ["cytosolic", "lumenal", "cytosolic"]


def test_n_terminus_outside_flips_every_orientation():
    ann = "o" * 10 + "M" * 20 + "i" * 8 + "M" * 20 + "o" * 10
    segs, n_term = segments_from_annotation(ann)
    assert n_term == "lumenal"
    assert [s["orientation"] for s in segs if s["type"] == "TM"] == ["out_in", "in_out"]


def test_segments_cover_every_residue_without_gaps():
    ann = "i" * 5 + "M" * 21 + "o" * 70 + "M" * 19 + "i" * 30
    segs, _ = segments_from_annotation(ann)
    cov = []
    for s in sorted(segs, key=lambda x: x["start"]):
        cov.append((s["start"], s["end"]))
    assert cov[0][0] == 1 and cov[-1][1] == len(ann)
    assert all(b[0] == a[1] + 1 for a, b in zip(cov, cov[1:]))
    assert any(s["type"] == "soluble" for s in segs)   # the 70-residue stretch


@pytest.mark.parametrize("text", [
    "q\tTMHMM2.0\toutside\t1\t10\nq\tTMHMM2.0\tTMhelix\t11\t30\nq\tTMHMM2.0\tinside\t31\t40\n",
    "# q Length: 40\nq\toutside\t1\t10\nq\tTMhelix\t11\t30\nq\tinside\t31\t40\n",
])
def test_parse_tmhmm_and_deeptmhmm_tables(text):
    ann = parse_table(text, 40)
    assert ann == "o" * 10 + "M" * 20 + "i" * 10


def test_parse_table_rejects_unrecognised_input():
    with pytest.raises(ValueError):
        parse_table("nothing useful here\n", 10)


def test_alignment_positions_skip_gaps():
    t = _to_table(["A", "B"], ["MKL-VA", "MK-LVA"])
    np.testing.assert_array_equal(t.A_pos.to_numpy(float), [1, 2, 3, np.nan, 4, 5])
    np.testing.assert_array_equal(t.B_pos.to_numpy(float), [1, 2, np.nan, 3, 4, 5])
    assert t.A_aa.tolist() == list("MKL-VA") and t.B_aa.tolist() == list("MK-LVA")


def test_pairwise_alignment_recovers_a_known_correspondence():
    a = "MKLVAGSTWEQQRTYIPLM"
    b = a[:8] + "D" + a[9:]          # one substitution, no indels
    t = align_sequences({"A": a, "B": b})
    both = t[t.A_aa.ne("-") & t.B_aa.ne("-")]
    assert len(both) == len(a)
    assert (both.A_pos.to_numpy() == both.B_pos.to_numpy()).all()


def test_aligned_table_flags_tm_and_identity():
    t = _to_table(["A", "B"], ["MKLVA", "MKIVA"])
    tabs = {}
    for g in ("A", "B"):
        tabs[g] = pd.DataFrame({"pos": range(1, 6), "seg_type": ["loop", "TM", "TM", "TM", "loop"],
                                "helix": [None, "TM1", "TM1", "TM1", None], "ss": list("CHHHC"),
                                "plddt": 90.0, "mean_all": [0, -1, -.5, -.8, 0],
                                "mean_P": np.nan, "mean_G": np.nan, "mean_KR": np.nan, "n_variants": 10,
                                **{f"sub_{c}": np.nan for c in "ACDEFGHIKLMNPQRSTVWY"}})
    at = aligned_table(t, tabs)
    assert at.both_residues.all()
    assert at.identical.tolist() == [True, True, False, True, True]
    assert at.both_tm.tolist() == [False, True, True, True, False]
    assert at.both_helical.tolist() == [False, True, True, True, False]


def test_from_file_beats_a_stale_cache(tmp_path):
    """Naming a gff3 must override a cached annotation, not be silently ignored."""
    from mpdms.topology import annotation
    seq = "M" * 60
    cache = tmp_path / "c.tmhmm"
    cache.write_text("# cached\n" + "i" * 60 + "\n")          # cache says: no TM helices
    gff = tmp_path / "TMRs.gff3"                               # the file says: one TM, 20-39
    gff.write_text("\n".join(["# test", "p\tTMhelix\t20\t39"]) + "\n")
    ann, src = annotation(seq, "p", source="auto", from_file=gff, cache=cache)
    assert src.startswith("file:"), f"cache won over --from-file: {src}"
    assert set(ann[19:39]) == {"M"} and "M" not in ann[:19]

    ann2, src2 = annotation(seq, "p", source="auto", from_file=None, cache=cache)
    assert src2 == "cache" and "M" not in ann2                 # still used when no file is given
