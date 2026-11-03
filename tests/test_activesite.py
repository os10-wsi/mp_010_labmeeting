"""The active-site test, on structures whose geometry is known because it was planted.

Three things can go wrong quietly here and each gets its own assertion: the distance
can be measured to the wrong atoms, the effect size can come back inverted while the
p-value stays right, and the transmembrane control can be dropped so that a result
driven by topology alone is reported as a result about the binding site.
"""
import numpy as np
import pandas as pd
import pytest

from mpdms import activesite as A


def write_pdb(path, residues, ligand=None, lig_name="LIG"):
    """A minimal PDB. `residues` is {pos: {atom_name: (x, y, z)}}."""
    lines, n = [], 0
    for pos in sorted(residues):
        for name, (x, y, z) in residues[pos].items():
            n += 1
            lines.append(f"ATOM  {n:5d}  {name:<3s} ALA A{pos:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
                         f"{name[0]:>2s}")
    for i, (x, y, z) in enumerate(ligand or [], 1):
        n += 1
        lines.append(f"HETATM{n:5d}  C{i:<2d} {lig_name:>3s} A 900    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_distance_is_to_the_nearest_heavy_atom_and_side_chain_is_separate(tmp_path):
    """A residue whose side chain points away is far by CB and near by backbone."""
    p = write_pdb(tmp_path / "s.pdb",
                  {1: {"N": (0, 0, 0), "CA": (1, 0, 0), "C": (2, 0, 0),
                       "O": (2, -6, 0), "CB": (1, 10, 0)}},
                  ligand=[(1, 3, 0)])
    d = A.ligand_distances(p, "LIG")
    assert len(d) == 1
    assert d.d_any.iloc[0] == pytest.approx(3.0, abs=1e-3)       # CA -> ligand
    assert d.d_sidechain.iloc[0] == pytest.approx(7.0, abs=1e-3)  # CB -> ligand
    assert d.d_sidechain.iloc[0] > d.d_any.iloc[0]


def test_glycine_has_no_side_chain_distance(tmp_path):
    p = write_pdb(tmp_path / "g.pdb",
                  {1: {"N": (0, 0, 0), "CA": (1, 0, 0), "C": (2, 0, 0),
                       "O": (2, -6, 0)}},
                  ligand=[(1, 4, 0)])
    d = A.ligand_distances(p, "LIG")
    assert np.isnan(d.d_sidechain.iloc[0])
    assert d.d_any.iloc[0] == pytest.approx(4.0, abs=1e-3)


def test_missing_ligand_is_an_error_not_an_empty_result(tmp_path):
    p = write_pdb(tmp_path / "n.pdb", {1: {"CA": (0, 0, 0)}}, ligand=[(5, 0, 0)])
    with pytest.raises(SystemExit):
        A.ligand_distances(p, "NOPE")


def _planted(tmp_path, n=60, n_called=10, called_near=True):
    """Called sites placed beside the ligand (or not), everything else far away."""
    rng = np.random.default_rng(0)
    coords, called = {}, set(range(1, n_called + 1))
    for pos in range(1, n + 1):
        near = pos in called and called_near
        r = 3.0 + rng.uniform(0, 1.5) if near else 20.0 + rng.uniform(0, 10)
        coords[pos] = {"CA": (r, 0.0, float(pos) * 0.01)}
    p = write_pdb(tmp_path / "p.pdb", coords, ligand=[(0, 0, 0)])
    sites = pd.DataFrame({"pos": list(range(1, n + 1)),
                          "wt": ["A"] * n,
                          "median_z": [-2.0 if i in called else 0.0
                                       for i in range(1, n + 1)],
                          "functional": [i in called for i in range(1, n + 1)]})
    csv = tmp_path / "d02_sites.csv"
    sites.to_csv(csv, index=False)
    return csv, p


def test_planted_binding_site_is_detected(tmp_path):
    csv, pdb = _planted(tmp_path)
    res = A.run(csv, [(pdb, "LIG")], "TEST", None, None, tmp_path / "out")
    whole = res["e01"]["tests"][0]
    assert whole["p"] < 1e-5
    assert whole["median_called"] < whole["median_other"]
    # the effect size must point the same way as the medians, not the other way
    assert whole["prob_called_closer"] > 0.9
    near = [s for s in res["e01"]["shells"] if s["cut"] == 5.0][0]
    assert near["called_near"] == near["called_total"]
    assert near["other_near"] == 0
    assert (tmp_path / "out" / "e01_sites_ligand_distance.csv").exists()
    assert (tmp_path / "out" / "e01_ligand_distance.png").exists()


def test_no_signal_when_called_sites_are_not_near_the_ligand(tmp_path):
    """The same machinery on a null: called sites scattered like everything else."""
    csv, pdb = _planted(tmp_path, called_near=False)
    res = A.run(csv, [(pdb, "LIG")], "TEST", None, None, tmp_path / "out")
    assert res["e01"]["tests"][0]["p"] > 0.05


def test_effect_size_is_asserted_against_the_medians(monkeypatch):
    """If the common-language effect size is ever flipped, the module must refuse.

    This is the failure the sign of a Mann-Whitney U invites: the p-value stays
    correct and only the reported effect size inverts, so nothing downstream notices.
    """
    t = pd.DataFrame({"d_min": [1.0, 1.2, 1.1, 30.0, 31.0, 32.0, 33.0],
                      "functional": [True, True, True, False, False, False, False]})
    ok = A._rank_test(t, "x")
    assert ok["prob_called_closer"] > 0.9

    class Flipped:
        pvalue, statistic = 1e-6, 12.0       # the un-inverted U, i.e. the wrong way

    monkeypatch.setattr(A.ss, "mannwhitneyu", lambda *a, **k: Flipped())
    with pytest.raises(AssertionError):
        A._rank_test(t, "x")


def test_transmembrane_control_is_reported_separately(tmp_path):
    csv, pdb = _planted(tmp_path)
    res = A.run(csv, [(pdb, "LIG")], "TEST", None, None, tmp_path / "out")
    scopes = [x["scope"] for x in res["e01"]["tests"]]
    assert scopes == ["whole protein", "transmembrane only"]
    # with no UniProt accession nothing is marked TM, so the control must say so
    # rather than silently reporting the whole-protein number twice
    assert res["e01"]["tests"][1].get("status") == "too few residues"
