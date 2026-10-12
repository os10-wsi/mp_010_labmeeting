"""The ten structural analyses: the geometry, not the plotting."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms import architecture as A


def test_kabsch_recovers_a_known_rotation():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 10, (12, 3))
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    b = a @ rot.T + np.array([3.0, -2.0, 7.0])
    r, ca, cb, rms = A.kabsch(a, b)
    assert rms == pytest.approx(0.0, abs=1e-8)
    assert np.allclose(r, rot, atol=1e-8)
    # a reflection must not be used to fake a fit
    mirrored = a * np.array([1.0, 1.0, -1.0])
    assert A.kabsch(a, mirrored)[3] > 1.0


def test_the_bundle_pairing_is_searched_not_assumed():
    """A fold whose halves are shuffled must be found, not forced into TM1-TM7."""
    rng = np.random.default_rng(1)
    # six helices in a ring, then the same ring again, permuted and displaced
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    base = np.c_[12 * np.cos(ang), 12 * np.sin(ang), np.zeros(6)]
    perm = [2, 0, 1, 4, 5, 3]
    rows, helix = [], {}
    pos = 1
    for k in range(12):
        c = base[k] if k < 6 else base[perm[k - 6]] + np.array([0.0, 0.0, 30.0])
        for j in range(10):
            rows.append({"pos": pos, "x": c[0] + rng.normal(0, .2),
                         "y": c[1] + rng.normal(0, .2), "z": c[2] + j * 1.5})
            helix[pos] = f"TM{k + 1}"
            pos += 1
    rt = pd.DataFrame(rows)
    segs = [{"name": f"TM{k+1}", "type": "TM", "start": 1 + 10 * k, "end": 10 + 10 * k}
            for k in range(12)]

    class Cfg:
        def get_path(self, key, default=None):
            return segs if key == "topology.segments" else default

    sup = A.superpose_bundles({"rt": rt, "helix": helix, "cfg": Cfg()})
    got = {a: b for a, b in sup["pairing"]}
    # helix 6+k sits at base[perm[k]], so the helix at base[0] is TM8, not TM7
    assert got["TM1"] == "TM8"
    assert got["TM2"] == "TM9" and got["TM3"] == "TM7"
    assert not sup["canonical"]
    assert sup["rmsd_centroid"] < sup["rmsd_centroid_canonical"]
    assert sup["rmsd_residue"] < 3.0


def test_salt_bridges_need_the_charged_atoms_not_the_backbone(tmp_path):
    """A lysine backbone near a glutamate is not a salt bridge."""
    pdb = tmp_path / "t.pdb"
    def atom(i, name, resn, resi, xyz, el):
        return (f"ATOM  {i:>5} {name:<4} {resn} A{resi:>4}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 90.00          {el:>2}\n")
    lines = [atom(1, " N  ", "GLU", 1, (0, 0, 0), "N"), atom(2, " CA ", "GLU", 1, (1, 0, 0), "C"),
             atom(3, " C  ", "GLU", 1, (2, 0, 0), "C"), atom(4, " O  ", "GLU", 1, (2, 1, 0), "O"),
             atom(5, " OE1", "GLU", 1, (3, 0, 0), "O"),
             atom(6, " N  ", "LYS", 5, (3.5, 0, 0), "N"),   # backbone N, close
             atom(7, " CA ", "LYS", 5, (9, 0, 0), "C"), atom(8, " C  ", "LYS", 5, (10, 0, 0), "C"),
             atom(9, " O  ", "LYS", 5, (10, 1, 0), "O"),
             atom(10, " NZ ", "LYS", 5, (20, 0, 0), "N"),   # charged N, far
             atom(11, " N  ", "ARG", 9, (30, 0, 0), "N"), atom(12, " CA ", "ARG", 9, (31, 0, 0), "C"),
             atom(13, " C  ", "ARG", 9, (32, 0, 0), "C"), atom(14, " O  ", "ARG", 9, (32, 1, 0), "O"),
             atom(15, " NH1", "ARG", 9, (3.0, 2.0, 0), "N")]  # charged N, close
    pdb.write_text("".join(lines) + "END\n")
    sb = A.salt_bridges(pdb)
    assert list(sb.pos_base) == [9]            # the arginine, not the lysine
    assert sb.distance.iloc[0] == pytest.approx(2.0, abs=0.01)


def _ring(n=60, patch=True, seed=0):
    """Positions on a line, with or without a contiguous block of intolerance."""
    rng = np.random.default_rng(seed)
    xyz = np.c_[np.arange(n) * 3.0, np.zeros(n), np.zeros(n)]
    v = rng.normal(0, 0.2, n)
    if patch:
        v[20:35] -= 1.5
    return xyz, v


def test_morans_i_is_unbiased_under_the_null():
    """Its expectation is -1/(n-1), so a bare threshold would be a coin toss."""
    xyz = np.c_[np.arange(60) * 3.0, np.zeros(60), np.zeros(60)]
    v = np.array([A.morans_i(xyz, np.random.default_rng(s).normal(0, 0.2, 60))
                  for s in range(200)])
    assert v.mean() == pytest.approx(-1 / 59, abs=0.015)


def _p_spatial(xyz, v, seed=0, n=300):
    rng = np.random.default_rng(seed)
    obs = A.morans_i(xyz, v)
    null = np.array([A.morans_i(xyz, rng.permutation(v)) for _ in range(n)])
    return float(((null >= obs).sum() + 1) / (n + 1))


def test_a_patch_of_constraint_is_detected_and_scattered_noise_is_not():
    """Calibration over seeds, not one draw.

    A single noise draw can genuinely be clustered - seed 0 here gives p = 0.017 for
    data with no patch in it at all - so asserting one seed comes out null tests the
    weather rather than the method. What has to hold is that the patch is always found
    and the false positive rate is near the nominal one.
    """
    hits = [_p_spatial(*_ring(patch=True, seed=s), seed=s) for s in range(12)]
    nulls = [_p_spatial(*_ring(patch=False, seed=s), seed=s) for s in range(12)]
    assert max(hits) < 0.01                      # the patch is found every time
    assert sum(p < 0.05 for p in nulls) <= 2     # and noise rarely looks like one
    xyz, v = _ring(patch=True)
    c = A.correlogram(xyz, v)
    # correlation is strongest at short range and decays
    assert c.iloc[0]["r"] > c.iloc[-1]["r"]
    assert c.iloc[0]["r"] > 0.2


def test_the_correlogram_column_is_not_called_corr():
    """`corr` on a DataFrame is a method; naming a column that hides it silently."""
    xyz, v = _ring()
    c = A.correlogram(xyz, v)
    assert "r" in c.columns and "corr" not in c.columns
    assert callable(c.corr)                      # still the method, unshadowed


def test_weighted_contacts_rank_a_buried_centre_above_an_edge():
    xyz = np.array([[0., 0., 0.], [4., 0., 0.], [-4., 0., 0.], [0., 4., 0.],
                    [0., -4., 0.], [40., 40., 40.]])
    w = A.weighted_contacts(xyz)
    assert w[0] == w.max()        # the centre of the cluster
    assert w[5] == w.min()        # the outlier
