"""The B-factor-coloured structures: what is in the column, and what is in the script."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms.analyses import heatmap_structure as H


def test_ranges_collapses_residue_numbers_the_way_pymol_wants():
    assert H._ranges([1, 2, 3, 7, 9, 10]) == "1-3+7+9-10"
    assert H._ranges([5]) == "5"
    assert H._ranges([]) == ""
    assert H._ranges([3, 1, 2]) == "1-3"          # order of the input does not matter


def _df(n_pos=10):
    """Missense, synonymous and nonsense over ten positions, with prolines and a K/R."""
    rng = np.random.default_rng(0)
    rows = []
    for pos in range(1, n_pos + 1):
        for mut in "ACDEFGHIKLMNPQRSTVWY":
            rows.append({"pos": pos, "wt": "L", "mut": mut, "vclass": "missense",
                         "score_z": float(rng.normal(-0.4, 0.1)), "pass_filter": True})
    return pd.DataFrame(rows)


def test_the_profiles_separate_proline_from_the_positive_charges():
    prof = H.position_profiles(_df())
    for c in ("mean_missense", "proline", "lys_arg", "glycine"):
        assert c in prof.columns
    assert prof.proline.notna().all() and prof.lys_arg.notna().all()
    # lys_arg pools two substitutions, proline is one
    assert set(H.STRUCTURE_KEYS) == {"mean_missense", "proline", "lys_arg"}


def _cfg(tmp_path, pdb):
    class C:
        id, display_name = "T", "T"
        def resolve(self, p):
            return p
        def get_path(self, k, d=None):
            return {"structure.path": pdb, "structure.chain": "A",
                    "structure.numbering_offset": 0}.get(k, d)
    return C()


def _pdb(tmp_path, n=40):
    """Big enough to clear the writer's minimum-size guard, which exists because an
    empty parse used to produce a file containing nothing but END."""
    lines = []
    for i in range(1, n + 1):
        lines.append(f"ATOM  {i:5d}  CA  LEU A{i:4d}    "
                     f"{i * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 50.00           C")
    lines.append("END")
    p = tmp_path / "s.pdb"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_each_quantity_gets_a_structure_whose_b_factor_is_the_value(tmp_path):
    from mpdms.structure import load_structure
    prof = H.position_profiles(_df())
    got = H.write_structure_colouring(_cfg(tmp_path, _pdb(tmp_path)), prof, tmp_path)
    names = {p.name for p in got}
    for key in H.STRUCTURE_KEYS:
        assert f"structure_{key}.pdb" in names and f"structure_{key}.pml" in names
        st = load_structure(tmp_path / f"structure_{key}.pdb", "A")[0]
        b = {r.id[1]: r["CA"].bfactor for r in st.get_residues() if "CA" in r}
        for pos, v in prof[key].dropna().items():
            if pos in b:
                assert b[pos] == pytest.approx(float(v), abs=5e-3)


def test_an_unmeasured_residue_is_not_given_a_value_that_reads_as_no_effect(tmp_path):
    """A sentinel inside the fitness scale would paint untested residues as neutral."""
    from mpdms.structure import load_structure
    prof = H.position_profiles(_df())
    pdb = _pdb(tmp_path, n=40)                     # residues beyond the data
    H.write_structure_colouring(_cfg(tmp_path, pdb), prof, tmp_path)
    st = load_structure(tmp_path / "structure_proline.pdb", "A")[0]
    b = {r.id[1]: r["CA"].bfactor for r in st.get_residues() if "CA" in r}
    assert b[12] == pytest.approx(H.UNSCORED)
    assert H.UNSCORED < -10                        # far outside the score_z range
    # and the script does not try to colour them
    pml = (tmp_path / "structure_proline.pml").read_text()
    assert "resi 12" not in pml.replace("resi 12-", "")


def test_the_script_builds_and_saves_a_session(tmp_path):
    H.write_structure_colouring(_cfg(tmp_path, _pdb(tmp_path)),
                                H.position_profiles(_df()), tmp_path)
    pml = (tmp_path / "structure_lys_arg.pml").read_text()
    assert "load structure_lys_arg.pdb" in pml
    assert "show cartoon" in pml
    assert "save structure_lys_arg.pse" in pml     # the session is the deliverable
    assert "B-factor = the value" in pml           # and the column is documented
    assert "_scored," in pml                       # a selection of measured residues


def test_a_structure_that_parses_empty_is_refused_rather_than_written(tmp_path):
    """The .cif-through-PDBParser failure wrote a seven-byte file and said nothing."""
    bad = tmp_path / "empty.pdb"
    bad.write_text("END\n")
    got = H.write_structure_colouring(_cfg(tmp_path, bad),
                                      H.position_profiles(_df()), tmp_path)
    assert got == []
    assert not list(tmp_path.glob("structure_*.pdb"))
