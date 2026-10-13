"""Two homologues and the internal pseudo-symmetry: the maths, not the plotting."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpdms import homology as H


# -------------------------------------------------------------- the ceiling
def _assay(n_pos=120, n_var=16, noise=0.4, seed=0):
    """Positions with a true mean each, measured with per-variant noise."""
    rng = np.random.default_rng(seed)
    true = rng.normal(0, 1, n_pos)
    rows = []
    for p in range(n_pos):
        for _ in range(n_var):
            rows.append({"pos": p, "score_z": true[p] + rng.normal(0, noise)})
    return pd.DataFrame(rows), true


def test_split_half_reliability_is_high_when_the_assay_is_clean():
    d, _ = _assay(noise=0.2)
    assert H.split_half_reliability(d, 60) > 0.95


def test_split_half_reliability_falls_as_the_noise_rises():
    clean = H.split_half_reliability(_assay(noise=0.2, seed=1)[0], 60)
    noisy = H.split_half_reliability(_assay(noise=4.0, seed=1)[0], 60)
    assert clean > noisy + 0.2
    assert 0.0 <= noisy <= 1.0


def test_the_ceiling_caps_a_correlation_between_two_noisy_copies():
    """Two noisy measurements of the SAME quantity must not exceed their own ceiling."""
    rng = np.random.default_rng(2)
    n_pos, n_var, noise = 150, 10, 1.5
    true = rng.normal(0, 1, n_pos)
    def measure(seed):
        r = np.random.default_rng(seed)
        rows = [{"pos": p, "score_z": true[p] + r.normal(0, noise)}
                for p in range(n_pos) for _ in range(n_var)]
        return pd.DataFrame(rows)
    a, b = measure(10), measure(11)
    cap = H.ceiling(H.split_half_reliability(a, 60), H.split_half_reliability(b, 60))
    from scipy import stats as ss
    obs = ss.spearmanr(a.groupby("pos").score_z.mean(),
                       b.groupby("pos").score_z.mean())[0]
    # the biology is identical, so the observed correlation should sit at the ceiling
    assert obs == pytest.approx(cap, abs=0.08)
    assert cap < 0.99          # and the ceiling must be below one, or it says nothing


def test_the_ceiling_is_the_geometric_mean_of_the_two_reliabilities():
    assert H.ceiling(0.81, 0.64) == pytest.approx(0.72)
    assert np.isnan(H.ceiling(np.nan, 0.5))


# ------------------------------------------------------------ the membrane frame
def test_the_membrane_normal_is_recovered_from_the_depths():
    rng = np.random.default_rng(3)
    n = np.array([0.3, -0.5, 0.81])
    n = n / np.linalg.norm(n)
    xyz = rng.normal(0, 15, (200, 3))
    rt = pd.DataFrame(xyz, columns=["x", "y", "z"])
    rt["zdepth"] = xyz @ n - 4.0            # an offset the basis must not care about
    got, u, v = H.normal_basis(rt)
    assert abs(abs(got @ n) - 1.0) < 1e-8
    # u and v must be a clean basis across the membrane
    for w in (u, v):
        assert abs(w @ got) < 1e-8
        assert np.linalg.norm(w) == pytest.approx(1.0)
    assert abs(u @ v) < 1e-8


# ---------------------------------------------------------- the symmetry pairing
def _bundle(perm=(0, 1, 2, 3, 4, 5), n_res=12, seed=4):
    """Twelve helices: a ring of six, then the same ring again, displaced and permuted.

    The ring is deliberately IRREGULAR. A regular hexagon maps onto itself under every
    rotation, so a Kabsch fit on six centroids would be degenerate and the search would
    return an arbitrary pairing - which is a real limitation of centroid-only matching
    and not something the fixture should hide.
    """
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    rad = np.array([11.0, 14.0, 18.0, 12.5, 16.0, 20.0])
    base = np.c_[rad * np.cos(ang), rad * np.sin(ang), np.zeros(6)]
    rows, helix, pos = [], {}, 1
    for k in range(12):
        # the second bundle sits BESIDE the first in the plane of the membrane, as the
        # two lobes of an MFS transporter do, not stacked on top of it
        c = base[k] if k < 6 else base[perm[k - 6]] + np.array([34.0, 0.0, 1.0])
        for j in range(n_res):
            rows.append({"pos": pos, "aa": "L", "plddt": 95.0,
                         "x": c[0] + rng.normal(0, .15),
                         "y": c[1] + rng.normal(0, .15),
                         "z": c[2] + (j - n_res / 2) * 1.5})
            helix[pos] = f"TM{k + 1}"
            pos += 1
    rt = pd.DataFrame(rows)
    rt["zdepth"] = rt.z
    segs = [{"name": f"TM{k+1}", "type": "TM", "start": 1 + n_res * k,
             "end": n_res * (k + 1), "orientation": "in_out" if k % 2 == 0 else "out_in"}
            for k in range(12)]

    class Cfg:
        id = "FAKE"
        def get_path(self, key, default=None):
            return segs if key == "topology.segments" else default

    return {"rt": rt, "helix": helix, "cfg": Cfg()}


def test_the_canonical_pairing_is_found_when_the_fold_is_canonical():
    sym = H.symmetry_pairs(_bundle())
    assert sym["canonical"]
    assert [(p["helix_a"], p["helix_b"]) for p in sym["pairs"]] == [
        (f"TM{k}", f"TM{k + 6}") for k in range(1, 7)]
    assert all(p["rmsd"] < 1.0 for p in sym["pairs"])


def test_a_shuffled_second_bundle_is_found_rather_than_forced():
    """The textbook pairing is an assumption; the structure gets to disagree with it."""
    sym = H.symmetry_pairs(_bundle(perm=(2, 0, 1, 4, 5, 3)))
    got = {a: b for a, b in [(p["helix_a"], p["helix_b"]) for p in sym["pairs"]]}
    assert not sym["canonical"]
    # helix 6+k sits where base[perm[k]] is, so TM1 at base[0] pairs with TM8
    assert got["TM1"] == "TM8"
    assert got["TM2"] == "TM9" and got["TM3"] == "TM7"


def test_the_canonical_pairing_can_be_forced():
    sym = H.symmetry_pairs(_bundle(perm=(2, 0, 1, 4, 5, 3)), canonical=True)
    assert sym["canonical"]
    assert [(p["helix_a"], p["helix_b"]) for p in sym["pairs"]] == [
        (f"TM{k}", f"TM{k + 6}") for k in range(1, 7)]


def test_every_paired_helix_gets_a_residue_correspondence_of_equal_length():
    sym = H.symmetry_pairs(_bundle())
    for p in sym["pairs"]:
        assert len(p["pos_a"]) == len(p["pos_b"]) == p["n"]
        assert set(p["pos_a"]).isdisjoint(p["pos_b"])


# ------------------------------------------------------- the helix correspondence
def test_helix_columns_drops_a_column_whose_two_helices_disagree():
    """An extra predicted segment in one protein must not shift every label by one."""
    t = pd.DataFrame({
        "A_helix": ["TM1"] * 6 + ["TM2"] * 6 + ["TM2"],
        "B_helix": ["TM1"] * 6 + ["TM2"] * 6 + ["TM5"],   # one stray column
        "A_mean_all": np.linspace(-1, 0, 13),
        "B_mean_all": np.linspace(-1, 0, 13),
        "both_tm": True,
    })
    d = H.helix_columns(t, "A", "B")
    assert len(d) == 12
    assert set(d.helix) == {"TM1", "TM2"}
    assert sorted({(r["A_helix"], r["B_helix"]) for _, r in d.iterrows()}) == [
        ("TM1", "TM1"), ("TM2", "TM2")]


# ------------------------------------------------------------- small helpers
def test_blosum_bins_identity_similarity_and_difference():
    assert H.blosum_bin("A", "A") == "identical"
    assert H.blosum_bin("I", "L").startswith("similar")        # BLOSUM62 I/L = +2
    assert H.blosum_bin("D", "W").startswith("dissimilar")     # BLOSUM62 D/W = -4
    assert H.blosum_bin("A", "-") == "unscored"


def test_auroc_counts_ties_as_half_and_is_symmetric_about_a_half():
    assert H.auroc([3, 4, 5], [0, 1, 2]) == pytest.approx(1.0)
    assert H.auroc([0, 1, 2], [3, 4, 5]) == pytest.approx(0.0)
    assert H.auroc([1, 1, 1], [1, 1, 1]) == pytest.approx(0.5)


def test_bootstrap_rho_brackets_the_point_estimate():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 200)
    y = x + rng.normal(0, 1, 200)
    rho, lo, hi = H.bootstrap_rho(x, y, n=300)
    assert lo < rho < hi
    assert lo > 0                      # a real correlation, so the interval clears zero
    assert H.bootstrap_rho([1, 2], [1, 2])[0] is not None


# ---------------------------------------------- the two symmetry analyses end to end
def _fake_bits(tmp_path, seed=7, mate_strength=0.9):
    """A twelve-helix protein whose two bundles agree to a controlled degree."""
    b = _bundle(seed=seed)
    sym = H.symmetry_pairs(b)
    rng = np.random.default_rng(seed)
    eff = {}
    for pr in sym["pairs"]:
        for x, y in zip(pr["pos_a"], pr["pos_b"]):
            base = rng.normal(-0.5, 0.35)
            eff[x] = base + rng.normal(0, 0.1)
            eff[y] = mate_strength * base + (1 - mate_strength) * rng.normal(-0.5, 0.35)
    pdb = tmp_path / "fake.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  "
                   "1.00 90.00           C\nEND\n")
    mate = {}
    for pr in sym["pairs"]:
        for x, y in zip(pr["pos_a"], pr["pos_b"]):
            mate[x] = y
            mate[y] = x
    b.update(symmetry=sym, effect=eff, mate=mate, path=pdb, offset=0,
             reliability=0.9,
             pos=pd.DataFrame({"pos": list(eff), "is_tm": 1.0}))
    return b


def test_h04_scores_every_symmetry_pair_separately(tmp_path):
    bits = {"FAKE": _fake_bits(tmp_path)}
    out = H.h04_symmetry_pairs(bits, ["FAKE"], tmp_path)
    assert out["status"] == "ok"
    d = pd.DataFrame(out["by_pair"])
    assert len(d) == 6                        # one row per structurally equivalent pair
    assert set(d.pair) == {f"TM{k}–TM{k + 6}" for k in range(1, 7)}
    # the bundles were built to agree, so every pair should correlate positively
    assert (d.spearman > 0.4).all()
    assert (tmp_path / "h04_symmetry_pairs.csv").exists()


def test_h04_sees_a_bundle_pair_that_does_not_agree(tmp_path):
    """With nothing planted, the typical pair finds nothing.

    The MEDIAN is the assertion, not the maximum: a helix pair carries only a dozen or
    so measured positions, so one pair out of six reaching |ρ| = 0.7 by chance is
    ordinary. That is why h04 reports a bootstrap interval for every pair rather than a
    bare correlation, and why a single striking pair is not a finding on its own.
    """
    weak = {"FAKE": _fake_bits(tmp_path, mate_strength=0.0)}
    strong = {"FAKE": _fake_bits(tmp_path, mate_strength=0.9)}
    dw = pd.DataFrame(H.h04_symmetry_pairs(weak, ["FAKE"], tmp_path)["by_pair"])
    ds = pd.DataFrame(H.h04_symmetry_pairs(strong, ["FAKE"], tmp_path)["by_pair"])
    assert dw.spearman.abs().median() < 0.35
    assert ds.spearman.median() > dw.spearman.median() + 0.4
    # and the intervals are wide enough that the stray pair is not called significant
    stray = dw.loc[dw.spearman.abs().idxmax()]
    assert stray.hi - stray.lo > 0.4


def test_h05_draws_the_bundle_and_writes_the_structure_scripts(tmp_path):
    bits = {"FAKE": _fake_bits(tmp_path)}
    rho = pd.DataFrame(H.h04_symmetry_pairs(bits, ["FAKE"], tmp_path)["by_pair"])
    out = H.h05_bundle_map(bits, ["FAKE"], tmp_path, rho)
    assert out["status"] == "ok"
    m = pd.read_csv(tmp_path / "h05_bundle_map.csv")
    assert len(m) == 6
    # the second bundle is the first translated by 34 A across the membrane, so a
    # faithful projection must put every mate exactly that far away in the same
    # direction - this checks the drawing is the structure, not a cartoon of it
    assert np.allclose(m.x_b - m.x_a, (m.x_b - m.x_a).iloc[0], atol=0.6)
    assert np.allclose(m.y_b - m.y_a, (m.y_b - m.y_a).iloc[0], atol=0.6)
    assert np.hypot((m.x_b - m.x_a).iloc[0],
                    (m.y_b - m.y_a).iloc[0]) == pytest.approx(34.0, abs=0.6)
    assert (tmp_path / "structures" / "FAKE" / "h05_bundles.pml").exists()
    assert (tmp_path / "structures" / "FAKE" / "h05_bundles.cxc").exists()


def test_h06_puts_the_mate_and_the_orthologue_on_one_axis(tmp_path):
    """The quartet: self, symmetry mate, orthologue, and the orthologue's mate."""
    a_bits = _fake_bits(tmp_path, seed=7)
    b_bits = _fake_bits(tmp_path, seed=8)
    # make protein B's effects a noisy copy of A's at the same positions, so the
    # orthologue should be the closest partner of the three
    rng = np.random.default_rng(9)
    b_bits["effect"] = {p: v + rng.normal(0, 0.05) for p, v in a_bits["effect"].items()}
    pos = sorted(a_bits["effect"])
    t = pd.DataFrame({"A_pos": pos, "B_pos": pos, "both_residues": True})
    out = H.h06_fold_or_lineage(t, "A", "B", {"A": a_bits, "B": b_bits}, tmp_path)
    assert out["status"] == "ok"
    d = pd.DataFrame(out["comparisons"]).set_index("column")
    assert set(d.index) == {"mate", "ortholog", "mate_of_ortholog"}
    assert d.loc["ortholog", "spearman"] > d.loc["mate", "spearman"]
    assert d.loc["ortholog", "spearman"] > 0.9
    # and all three are measured on exactly the same positions
    assert d.n.nunique() == 1


# ----------------------------------------- the alignment-based analyses end to end
def _fake_alignment(n_helix=12, per_helix=14, agree=0.8, seed=11):
    """An aligned table for two homologues whose agreement is planted at `agree`."""
    rng = np.random.default_rng(seed)
    rows = []
    pos = 1
    for h in range(n_helix):
        level = rng.normal(-0.55, 0.22)            # the helix's own level
        for _ in range(per_helix):
            base = level + rng.normal(0, 0.3)
            other = agree * base + (1 - agree) * rng.normal(-0.55, 0.3)
            same = rng.random() < 0.4
            aa_a = "AILVFSTNQDEKRGPWYHMC"[rng.integers(0, 20)]
            aa_b = aa_a if same else "AILVFSTNQDEKRGPWYHMC"[rng.integers(0, 20)]
            rows.append({"A_pos": pos, "B_pos": pos, "A_helix": f"TM{h+1}",
                         "B_helix": f"TM{h+1}", "A_aa": aa_a, "B_aa": aa_b,
                         "A_mean_all": base, "B_mean_all": other,
                         "both_tm": True, "both_loop": False, "both_residues": True,
                         "identical": aa_a == aa_b})
            pos += 1
    # a tail of loop columns, so the within-class null has two classes to work with
    for _ in range(60):
        base = rng.normal(-0.1, 0.2)
        rows.append({"A_pos": pos, "B_pos": pos, "A_helix": None, "B_helix": None,
                     "A_aa": "S", "B_aa": "S",
                     "A_mean_all": base,
                     "B_mean_all": agree * base + (1 - agree) * rng.normal(-0.1, 0.2),
                     "both_tm": False, "both_loop": True, "both_residues": True,
                     "identical": True})
        pos += 1
    return pd.DataFrame(rows)


def _alignment_bits(t, reliability=0.92):
    """The little of `bits` that the alignment-based analyses actually touch."""
    def one(col):
        rng = np.random.default_rng(abs(hash(col)) % 1000)
        var = pd.DataFrame([{"pos": int(p), "score_z": v + rng.normal(0, 0.25)}
                            for p, v in zip(t[f"{col}_pos"], t[f"{col}_mean_all"])
                            for _ in range(12)])

        class Cfg:
            id = col
            def get_path(self, key, default=None):
                if key == "topology.segments":
                    return [{"name": f"TM{k+1}", "type": "TM",
                             "start": 1 + 14 * k, "end": 14 * (k + 1)}
                            for k in range(12)]
                return default

        return {"variants": var, "reliability": reliability, "cfg": Cfg()}
    return {"A": one("A"), "B": one("B")}


def test_h01_reports_the_correlation_against_both_a_null_and_a_ceiling(tmp_path):
    t = _fake_alignment(agree=0.8)
    out = H.h01_whole_protein(t, "A", "B", _alignment_bits(t), tmp_path)
    assert out["status"] == "ok"
    d = pd.DataFrame(out["subsets"]).set_index("subset")
    assert d.loc["all aligned", "spearman"] > 0.5
    assert d.loc["all aligned", "p_within_class_null"] < 0.01
    # the fraction of the ceiling is the number the figure leads with, so it must exist
    assert 0 < d.loc["all aligned", "fraction_of_ceiling"] <= 1.2
    assert (tmp_path / "h01_whole_protein.csv").exists()


def test_h01_finds_nothing_when_the_two_proteins_are_unrelated(tmp_path):
    """Topology alone would still give a positive correlation; the null must absorb it."""
    t = _fake_alignment(agree=0.0)
    out = H.h01_whole_protein(t, "A", "B", _alignment_bits(t), tmp_path)
    d = pd.DataFrame(out["subsets"]).set_index("subset")
    assert d.loc["all aligned", "p_within_class_null"] > 0.05


def test_h02_and_h03_answer_different_questions(tmp_path):
    """A conserved helix LEVEL does not imply a conserved pattern inside the helix."""
    rng = np.random.default_rng(12)
    rows, pos = [], 1
    for h in range(12):
        level = rng.normal(-0.55, 0.3)
        for _ in range(14):
            rows.append({"A_pos": pos, "B_pos": pos, "A_helix": f"TM{h+1}",
                         "B_helix": f"TM{h+1}", "A_aa": "L", "B_aa": "L",
                         # same level in both, but the within-helix pattern is independent
                         "A_mean_all": level + rng.normal(0, 0.3),
                         "B_mean_all": level + rng.normal(0, 0.3),
                         "both_tm": True, "both_loop": False, "both_residues": True,
                         "identical": True})
            pos += 1
    t = pd.DataFrame(rows)
    bits = _alignment_bits(t)
    h3 = H.h03_helix_means(t, "A", "B", bits, tmp_path)
    h2 = H.h02_per_helix(t, "A", "B", bits, tmp_path)
    means = pd.DataFrame(h3["by_helix"])
    from scipy import stats as ss
    assert ss.spearmanr(means.mean_a, means.mean_b)[0] > 0.7     # levels agree
    per = pd.DataFrame(h2["by_helix"])
    assert abs(per.spearman.median()) < 0.35                     # patterns do not


def test_h08_separates_the_identity_bins(tmp_path):
    t = _fake_alignment(agree=0.8)
    out = H.h08_identity(t, "A", "B", _alignment_bits(t), tmp_path)
    assert out["status"] == "ok"
    d = pd.DataFrame(out["bins"])
    assert set(d.bin) <= set(H.IDENT_ORDER)
    assert (d.n >= 15).all()
    # the agreement was planted independently of identity, so no bin should stand out
    assert d.spearman.max() - d.spearman.min() < 0.45


def test_h10_transfer_beats_predicting_the_mean(tmp_path):
    t = _fake_alignment(agree=0.85)
    out = H.h10_transfer(t, "A", "B", _alignment_bits(t), tmp_path)
    assert out["status"] == "ok"
    d = pd.DataFrame(out["rows"])
    both = d[d.subset == "all aligned"]
    assert len(both) == 2                      # both directions are reported
    assert (both.rmse < both.sd_of_target).all()
    assert (both.auroc_worst_tenth > 0.7).all()
