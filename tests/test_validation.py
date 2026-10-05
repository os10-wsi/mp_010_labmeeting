"""Validation battery and benchmark harness: do they detect real signal and stay quiet otherwise."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as ss

from mpdms import benchmark as B
from mpdms.analyses.a16_assay_validation import k12_pseudosymmetry
from mpdms.config import Config


def _mfs_cfg(n_tm=12, start=20, helix_len=21, gap=12):
    segs, p = [], start
    for i in range(n_tm):
        segs.append({"name": f"TM{i + 1}", "start": p, "end": p + helix_len - 1, "type": "TM",
                     "orientation": "in_out" if i % 2 == 0 else "out_in"})
        p += helix_len + gap
    return Config.wrap({"id": "MFS", "display_name": "MFS", "topology": {"segments": segs}}), segs


def _profile_df(segs, shape_fn, rng, noise=0.05, second_half_shape=None):
    """Build a variant table whose per-position mean follows shape_fn(rel_pos) per helix."""
    half = len(segs) // 2
    rows = []
    for k, s in enumerate(segs):
        fn = shape_fn if k < half else (second_half_shape or shape_fn)
        L = s["end"] - s["start"] + 1
        for j, pos in enumerate(range(s["start"], s["end"] + 1)):
            f = (pos - s["start"]) / (L - 1)
            rel = (2 * f - 1) if s["orientation"] == "in_out" else (1 - 2 * f)
            mu = fn(rel, k % half, j)
            for _ in range(6):
                rows.append({"pos": pos, "score_z": mu + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def test_k12_detects_pseudosymmetry():
    """Two bundles built from the same per-helix shapes must be flagged as correlated."""
    rng = np.random.default_rng(0)
    cfg, segs = _mfs_cfg()
    shape = lambda rel, hx, j: -0.8 - 0.5 * np.cos(rel * np.pi) + 0.35 * np.sin(hx + 2 * rel)  # noqa: E731
    mis = _profile_df(segs, shape, rng)
    r = k12_pseudosymmetry(None, cfg, mis, None)
    assert r["testable"] and r["passed"], r
    assert r["value"] > 0.5 and r["p"] < 0.05


def test_k12_quiet_when_bundles_unrelated():
    """Independent random shapes per helix must not produce a symmetry call."""
    rng = np.random.default_rng(1)
    cfg, segs = _mfs_cfg()
    offs = rng.normal(0, 1, len(segs))
    shape = lambda rel, hx, j: -0.8 + 0.6 * np.sin(3 * rel + offs[hx])  # noqa: E731
    second = lambda rel, hx, j: -0.8 + 0.6 * np.sin(3 * rel + offs[hx + 6] + 2.0)  # noqa: E731
    mis = _profile_df(segs, shape, rng, second_half_shape=second)
    r = k12_pseudosymmetry(None, cfg, mis, None)
    assert r["testable"]
    assert not r["passed"] or r["p"] > 0.01, r


def test_k12_not_testable_for_small_proteins():
    cfg, segs = _mfs_cfg(n_tm=4)
    mis = _profile_df(segs, lambda rel, hx, j: -0.5, np.random.default_rng(2))
    r = k12_pseudosymmetry(None, cfg, mis, None)
    assert not r["testable"] and "TM helices" in r["note"]


# ------------------------------------------------------------------- splits
def _multi(n_proteins=4, n_pos=40, rule=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_proteins):
        shift = rng.normal(0, 0.3)
        for p in range(1, n_pos + 1):
            rel = -1 + 2 * (p % 21) / 20
            for m in "ACDEFGHIKLMNPQRSTVWY":
                dh = rng.normal()
                mu = (-0.8 * abs(rel) - 0.4 * dh + shift) if rule else rng.normal(0, 0.5)
                rows.append({"dataset_id": f"P{g}", "pos": p, "wt": "A", "mut": m,
                             "abs_rel": abs(rel), "rel_pos": rel, "delta_biological": dh,
                             "seg_type": "TM" if abs(rel) < 0.8 else "loop",
                             "score_z": mu + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_protein_split_holds_out_whole_proteins():
    d = _multi()
    for name, tr, te in B.split_folds(d, "protein"):
        assert set(d.loc[tr].dataset_id).isdisjoint(d.loc[te].dataset_id)
        assert set(d.loc[te].dataset_id) == {name}


def test_position_split_keeps_sites_together():
    d = _multi()
    for _, tr, te in B.split_folds(d, "position", n_folds=5):
        a = set(zip(d.loc[tr].dataset_id, d.loc[tr].pos))
        b = set(zip(d.loc[te].dataset_id, d.loc[te].pos))
        assert a.isdisjoint(b)


def test_site_mean_collapses_across_the_position_split():
    """site_mean looks strong on a random split and must fail when positions are held out."""
    d = _multi()
    rnd = B.evaluate(d, [B.SiteMean()], split="random", n_folds=4).spearman.mean()
    pos = B.evaluate(d, [B.SiteMean()], split="position", n_folds=4).spearman.mean()
    assert rnd > 0.3 and abs(pos) < 0.15


# -------------------------------------------------------------- transfer
def test_lopo_transfer_when_a_shared_rule_exists():
    d = _multi(rule=True)
    res = B.evaluate(d, [B.Baseline("ridge", features=["abs_rel", "delta_biological"])], split="protein")
    assert res.spearman.min() > 0.5, res


def test_lopo_transfer_absent_when_no_shared_rule():
    d = _multi(rule=False)
    res = B.evaluate(d, [B.Baseline("ridge", features=["abs_rel", "delta_biological"])], split="protein")
    assert abs(res.spearman.mean()) < 0.15, res


def test_score_column_auto_orients_sign():
    d = _multi()
    d["ddg"] = -d.score_z + np.random.default_rng(0).normal(0, 0.1, len(d))  # inverted convention
    res = B.evaluate(d, [B.ScoreColumn("ddg", sign="auto")], split="protein")
    assert res.spearman.mean() > 0.8


@pytest.mark.parametrize("how", ["random", "position", "protein", "segment"])
def test_every_split_covers_the_data(how):
    d = _multi()
    seen = set()
    for _, tr, te in B.split_folds(d, how, n_folds=3):
        assert len(tr) and len(te)
        seen |= set(te)
    assert len(seen) == len(d)


def test_metrics_report_discrimination():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    m = B.metrics(y, y + rng.normal(0, 0.2, 500))
    assert m["spearman"] > 0.9 and m["auroc_deleterious"] > 0.9
    m0 = B.metrics(y, rng.normal(0, 1, 500))
    assert abs(m0["spearman"]) < 0.2


# ------------------------------------------- A18: burial vs substitution cost
def _rsa_positions(kr_slope, pro_slope, seed=0, n_helices=8, n_pos=20):
    """Positions whose K/R and Pro effects depend on RSA with the given slopes."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_helices):
        for i in range(n_pos):
            rsa = rng.uniform(0, 0.7)
            rows.append({"pos": h * 100 + i, "helix": f"TM{h}", "orientation": "in_out", "rsa": rsa,
                         "mean_KR": -1.0 + kr_slope * rsa + rng.normal(0, 0.12), "n_KR": 2,
                         "mean_Pro": -1.0 + pro_slope * rsa + rng.normal(0, 0.12), "n_Pro": 1})
    return pd.DataFrame(rows)


def test_rsa_slope_difference_detected():
    """K/R relieved by exposure, proline flat: the interaction must fire."""
    from mpdms.analyses.a18_helix_kr_vs_pro import rsa_stats
    r = rsa_stats(_rsa_positions(kr_slope=1.2, pro_slope=0.0, seed=1))
    assert r["mean_KR"]["slope"] > 0.8 and r["mean_KR"]["p_slope"] < 0.01
    assert abs(r["mean_Pro"]["slope"]) < 0.3
    assert r["interaction"]["delta"] > 0.6 and r["interaction"]["p"] < 0.01


def test_rsa_slope_difference_absent_when_slopes_match():
    from mpdms.analyses.a18_helix_kr_vs_pro import rsa_stats
    r = rsa_stats(_rsa_positions(kr_slope=0.6, pro_slope=0.6, seed=2))
    assert abs(r["interaction"]["delta"]) < 0.3 and r["interaction"]["p"] > 0.05


# ------------------------------------------- A19: published residues vs screen
def _sites(functional_pos, n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, n + 1):
        func = p in functional_pos
        rows.append({"pos": p, "wt": "Q", "segment": "TM1", "n_variants": 18,
                     "median_abundance": 0.1 if func else rng.uniform(-1.2, 0.2),
                     "median_z": -2.5 if func else rng.uniform(-0.8, 0.8),
                     "q": 0.001 if func else 0.6, "functional": func,
                     "abundance_tolerant": True if func else False})
    return pd.DataFrame(rows)


def test_a19_wt_mismatch_drops_the_entry():
    """A paper's numbering that disagrees with the sequence must be refused, not used."""
    from mpdms.analyses.a19_literature import check_numbering
    seq = "M" + "Q" * 50          # Q at 2..51, M at 1
    ok, bad = check_numbering([{"pos": 10, "wt": "Q"}, {"pos": 1, "wt": "R"}, {"pos": 999, "wt": "Q"}], seq)
    assert [e["pos"] for e in ok] == [10] and ok[0]["wt_checked"]
    assert len(bad) == 2 and "numbering mismatch" in bad[0] and "outside" in bad[1]


def test_a19_marks_abundance_tolerant_hits():
    """A published residue the screen calls functional but abundance-tolerant is the gold case."""
    from mpdms.analyses.a19_literature import residue_table
    s = _sites({149, 504})
    t = residue_table([{"pos": 149, "wt": "Q", "substitution": "Q149A"},
                       {"pos": 60, "wt": "Q", "substitution": "Q60A"}], s)
    assert t.loc[t.pos == 149, "verdict"].iloc[0] == "functional, abundance-tolerant"
    assert t.loc[t.pos == 60, "verdict"].iloc[0] != "functional, abundance-tolerant"


def test_a19_uncovered_residue_is_not_scored_as_a_miss():
    from mpdms.analyses.a19_literature import enrichment, residue_table
    s = _sites({149})
    t = residue_table([{"pos": 149, "wt": "Q"}, {"pos": 5000, "wt": "R"}], s)
    assert t.loc[t.pos == 5000, "verdict"].iloc[0] == "not covered by the screen"
    e = enrichment(t, s)
    assert e["n_published_covered"] == 1 and e["n_hit"] == 1 and e["underpowered"]


def test_a19_curated_file_parses_and_qdr2_claims_nothing():
    """Guard against someone inventing QDR2 residues: there is no published set."""
    from mpdms.analyses.a19_literature import load_literature
    aqr1, qdr2 = load_literature("AQR1"), load_literature("qdr2")
    assert aqr1 and {e["pos"] for e in aqr1["measured"]} == {149, 238, 504}
    assert all(e["confidence"] in {"verified", "unverified"} for e in aqr1["measured"])
    assert qdr2 is not None and qdr2["measured"] == []


# ------------------------------------------- A20: named variants vs synonymous
def _panel_frame(effects, n_syn=200, sd=0.08, seed=0):
    """Three replicates per named variant, plus a synonymous cloud with real spread."""
    rng = np.random.default_rng(seed)
    rows = []
    for (pos, wt, mut), eff in effects.items():
        r = {"pos": pos, "wt": wt, "mut": mut, "vclass": "missense", "pass_filter": True, "score_z": eff}
        r.update({f"rep{k}": eff + rng.normal(0, sd) for k in (1, 2, 3)})
        rows.append(r)
    for i in range(n_syn):
        v = rng.normal(0, 0.15)
        r = {"pos": i + 1, "wt": "A", "mut": "A", "vclass": "synonymous", "pass_filter": True, "score_z": v}
        r.update({f"rep{k}": v + rng.normal(0, sd) for k in (1, 2, 3)})
        rows.append(r)
    return pd.DataFrame(rows)


def _panel_cfg():
    return Config.wrap({"id": "T", "display_name": "T", "protein": {"gene": "AQR1"}})


def test_a20_finds_a_real_effect_and_spares_the_null_ones():
    from mpdms.analyses.a20_variant_panel import collect, stats_table
    df = _panel_frame({(149, "Q", "A"): -0.05, (238, "Q", "A"): -0.90, (504, "R", "A"): -0.02})
    mut, syn, missing = collect(df, _panel_cfg(), ["Q149A", "Q238A", "R504A"])
    assert not missing and mut.variant.nunique() == 3 and len(mut) == 9
    st = stats_table(mut, syn, df[df.vclass == "synonymous"].score_z.to_numpy())
    vs = st[st.kind == "vs_synonymous"].set_index("a")
    assert vs.loc["Q238A", "q"] < 0.05, "planted 0.9-unit effect missed"
    assert vs.loc["Q149A", "q"] > 0.05 and vs.loc["R504A", "q"] > 0.05, "null variants called"
    assert abs(vs.loc["Q238A", "z_vs_syn_variants"]) > 4


def test_a20_empirical_p_is_primary_and_respects_its_floor():
    """The Welch test on correlated replicates reads too significant; p must be the empirical one."""
    from mpdms.analyses.a20_variant_panel import collect, stats_table
    df = _panel_frame({(238, "Q", "A"): -0.90}, n_syn=50)
    mut, syn, _ = collect(df, _panel_cfg(), ["Q238A"])
    st = stats_table(mut, syn, df[df.vclass == "synonymous"].score_z.to_numpy())
    r = st[st.kind == "vs_synonymous"].iloc[0]
    assert r.p == pytest.approx(1 / 51, rel=1e-6)        # at the floor set by 50 synonymous variants
    assert r.p_welch_replicates < r.p                     # and the parametric one is more optimistic


def test_a20_reports_what_it_could_not_plot():
    from mpdms.analyses.a20_variant_panel import collect
    df = _panel_frame({(149, "Q", "A"): -0.05})
    mut, _, missing = collect(df, _panel_cfg(), ["Q149A", "R504A", "W149A", "rubbish"])
    assert mut.variant.unique().tolist() == ["Q149A"]
    assert any("not measured" in m for m in missing)                  # absent from the data
    assert any("numbering mismatch" in m and "W149A" in m for m in missing)   # wrong wt at that position
    assert any("not a point substitution" in m for m in missing)


def test_a20_pairwise_reports_its_detection_limit():
    from mpdms.analyses.a20_variant_panel import collect, stats_table
    df = _panel_frame({(149, "Q", "A"): -0.05, (504, "R", "A"): -0.02})
    mut, syn, _ = collect(df, _panel_cfg(), ["Q149A", "R504A"])
    st = stats_table(mut, syn, df[df.vclass == "synonymous"].score_z.to_numpy())
    pair = st[st.kind == "mutant_pair"]
    assert len(pair) == 1 and pair.iloc[0].q > 0.05            # 0.03 apart: not detectable
    assert pair.iloc[0].min_detectable_difference > 0.1        # and the figure says so


# ----------------------------------- A21: every substitution at a site vs ESM-1v
def _variant_rows(spec, seed=0):
    """spec: {pos: (wt, {mut: z})}. Builds an a15_variants-shaped frame."""
    rng = np.random.default_rng(seed)
    rows = []
    for pos, (wt, muts) in spec.items():
        for mut, z in muts.items():
            rows.append({"pos": pos, "wt": wt, "mut": mut, "score_z": rng.normal(-0.1, 0.1),
                         "segment": "TM1", "seg_type": "TM", "esm1v": -4 + z,
                         "esm_expected": -4.0, "residual": z, "z": z, "functional": z <= -1.5})
    return pd.DataFrame(rows)


def test_a21_counts_substitutions_below_the_threshold():
    from mpdms.analyses.a21_site_substitutions import site_table
    spec = {149: ("Q", {"A": -2.4, "G": -1.9, "L": -0.2, "S": 0.4, "W": -1.5}),
            504: ("R", {"A": -0.3, "G": -0.1, "L": 0.2, "S": 0.1, "W": -0.4})}
    t = site_table(_variant_rows(spec), [149, 504], {149: "A", 504: "A"}).set_index("pos")
    assert t.loc[149, "n_below_threshold"] == 3                 # -2.4, -1.9 and -1.5 (inclusive)
    assert set(t.loc[149, "substitutions_below"]) == {"A", "G", "W"}
    assert t.loc[504, "n_below_threshold"] == 0
    assert t.loc[149, "marked_below_threshold"] and not t.loc[504, "marked_below_threshold"]
    assert t.loc[149, "marked_z"] == pytest.approx(-2.4)


def test_a21_separates_a_site_carried_by_one_substitution():
    """A site whose median is flat but whose marked allele is extreme must stay visible."""
    from mpdms.analyses.a21_site_substitutions import site_table
    spec = {238: ("Q", {"A": -3.0, **{m: 0.1 for m in "GLSVTIFYWCNDEKRHMP"}})}
    t = site_table(_variant_rows(spec), [238], {238: "A"}).iloc[0]
    assert t.median_z > -0.5 and t.n_below_threshold == 1       # the site looks tolerant overall
    assert t.marked_below_threshold and t.marked_z == pytest.approx(-3.0)


def test_a21_site_with_no_measured_substitutions_is_dropped_not_faked():
    from mpdms.analyses.a21_site_substitutions import site_table
    t = site_table(_variant_rows({149: ("Q", {"A": -2.0, "G": -1.0})}), [149, 999], {149: "A"})
    assert list(t.pos) == [149]


def test_a21_local_refit_absorbs_what_the_sites_share():
    """The point of A21b: a shift common to all the sites is soaked up by the local fit."""
    from mpdms.analyses.a21_site_substitutions import add_local_z
    rng = np.random.default_rng(0)
    rows = []
    for pos in (149, 238, 504):
        for mut in "ACDEFGHIKLMNPQRSTVW":
            x = rng.uniform(-0.3, 0.1)
            rows.append({"pos": pos, "wt": "Q", "mut": mut, "score_z": x,
                         "esm1v": -4 + 2 * x - 3.0 + rng.normal(0, 0.3),   # -3.0 shared by all sites
                         "esm_expected": -4 + 2 * x, "residual": -3.0, "z": -8.0})
    v, how, scale = add_local_z(pd.DataFrame(rows))
    assert "LOESS" in how and len(v) == 57
    assert abs(v.z_local.median()) < 0.5, "shared offset should vanish in the local fit"
    assert v.z.median() < -5, "the global z still sees it"
    assert scale > 0


def test_a21_local_fit_falls_back_to_a_line_when_there_are_too_few_points():
    from mpdms.analyses.a21_site_substitutions import local_fit
    x = np.linspace(-1, 1, 12)
    f, how = local_fit(x, 2 * x + 1)
    assert "linear" in how
    assert f(np.array([0.0, 0.5])) == pytest.approx([1.0, 2.0], abs=1e-6)


def test_a21_local_z_still_finds_a_substitution_that_stands_out_from_its_peers():
    from mpdms.analyses.a21_site_substitutions import add_local_z, site_table
    rng = np.random.default_rng(1)
    rows = []
    for pos in (149, 238, 504):
        for mut in "ACDEFGHIKLMNPQRSTVW":
            x = rng.uniform(-0.3, 0.1)
            e = -4 + 2 * x + rng.normal(0, 0.2)
            if (pos, mut) == (149, "A"):
                e -= 3.0                                   # one genuine outlier among the 57
            rows.append({"pos": pos, "wt": "Q", "mut": mut, "score_z": x, "esm1v": e,
                         "segment": "TM1", "esm_expected": -4 + 2 * x,
                         "residual": e - (-4 + 2 * x), "z": 0.0})
    v, _, _ = add_local_z(pd.DataFrame(rows))
    t = site_table(v, [149], {149: "A"}, zcol="z_local", rescol="residual_local").iloc[0]
    assert t.marked_below_threshold and t.marked_z < -3


def test_a20_samples_synonymous_for_display_without_weakening_the_test():
    """20 drawn for the plot; the null must stay the full set or nothing can reach significance."""
    from mpdms.analyses.a20_variant_panel import SYN_DISPLAY_N, collect, sample_for_display, stats_table
    df = _panel_frame({(238, "Q", "A"): -0.90}, n_syn=200)
    mut, syn, _ = collect(df, _panel_cfg(), ["Q238A"])
    shown = sample_for_display(syn)
    assert shown.syn_index.nunique() == SYN_DISPLAY_N
    assert len(shown) == SYN_DISPLAY_N * 3                    # whole variants, all replicates
    assert sample_for_display(syn).equals(shown)              # fixed seed, reproducible
    syn_means = df[df.vclass == "synonymous"].score_z.to_numpy()
    r = stats_table(mut, syn, syn_means)[lambda d: d.kind == "vs_synonymous"].iloc[0]
    assert r.n_b == 200 and r.p_floor == pytest.approx(1 / 201)   # full set, not the 20


def test_a20_sample_keeps_everything_when_there_is_little_synonymous_data():
    from mpdms.analyses.a20_variant_panel import collect, sample_for_display
    df = _panel_frame({(238, "Q", "A"): -0.9}, n_syn=7)
    _, syn, _ = collect(df, _panel_cfg(), ["Q238A"])
    assert sample_for_display(syn).syn_index.nunique() == 7


def test_a20_percentiles_locate_variants_in_the_missense_distribution():
    from mpdms.analyses.a20_variant_panel import percentiles
    rng = np.random.default_rng(0)
    rows = [{"pos": i % 400 + 1, "wt": "L", "mut": "V", "vclass": "missense",
             "pass_filter": True, "score_z": v} for i, v in enumerate(rng.normal(-0.45, 0.38, 2000))]
    rows += [{"pos": 238, "wt": "Q", "mut": "A", "vclass": "missense", "pass_filter": True, "score_z": -1.6},
             {"pos": 149, "wt": "Q", "mut": "A", "vclass": "missense", "pass_filter": True, "score_z": 0.3}]
    pc = percentiles(pd.DataFrame(rows), ["Q238A", "Q149A", "R504A"]).set_index("variant")
    assert "R504A" not in pc.index                 # absent from the data, not invented
    assert pc.loc["Q238A", "percentile"] < 5       # near the bottom of the missense distribution
    assert pc.loc["Q149A", "percentile"] > 95
    assert pc.loc["Q238A", "rank_from_bottom"] < pc.loc["Q149A", "rank_from_bottom"]


# ------------------------------- A21/A20: sites named as residues, not substitutions
def test_wanted_sites_accepts_bare_residues_and_substitutions():
    """PHO84's list mixes Phe160 with D358N; both must contribute their position."""
    from mpdms.analyses.a20_variant_panel import wanted_sites
    cfg = Config.wrap({"id": "PHO84", "display_name": "PHO84", "protein": {"gene": "PHO84"}})
    sites, wt, marked = wanted_sites(cfg)
    assert sites == [160, 168, 178, 179, 358, 392, 473, 492]
    assert wt == {160: "F", 168: "R", 178: "D", 179: "Y", 358: "D", 392: "V", 473: "E", 492: "K"}
    # one allele at 358 -> marked; three at 492 and two at 178 -> no single allele to point at
    assert marked == {358: "N"}


def test_wanted_sites_leaves_the_single_allele_genes_alone():
    from mpdms.analyses.a20_variant_panel import wanted_sites
    cfg = Config.wrap({"id": "AQR1", "display_name": "AQR1", "protein": {"gene": "AQR1"}})
    sites, _, marked = wanted_sites(cfg)
    assert sites == [149, 238, 504] and marked == {149: "A", 238: "A", 504: "A"}


def test_wanted_sites_follows_an_explicit_config_list():
    from mpdms.analyses.a20_variant_panel import wanted_sites
    cfg = Config.wrap({"id": "X", "display_name": "X", "protein": {"gene": "NOPE"},
                       "report": {"variant_panel": ["D178N", "K492A"]}})
    sites, wt, marked = wanted_sites(cfg)
    assert sites == [178, 492] and wt == {178: "D", 492: "K"} and marked == {178: "N", 492: "A"}


def test_a20_panel_still_uses_substitutions_only():
    """A20 plots alleles, so bare residues cannot become columns."""
    from mpdms.analyses.a20_variant_panel import wanted_variants
    cfg = Config.wrap({"id": "PHO84", "display_name": "PHO84", "protein": {"gene": "PHO84"}})
    assert wanted_variants(cfg) == ["D178N", "D178E", "D358N", "K492A", "K492Q", "K492E"]


# ---------------------------------- A23: canonical membrane-protein predictions
def _tm_cfg(n_tm=4, start=20, helix_len=21, gap=12):
    segs, p = [], start
    for i in range(n_tm):
        segs.append({"name": f"TM{i + 1}", "start": p, "end": p + helix_len - 1, "type": "TM",
                     "orientation": "in_out" if i % 2 == 0 else "out_in"})
        p += helix_len + gap
    return Config.wrap({"id": "T", "display_name": "T", "topology": {"segments": segs}}), segs


def test_a23_flank_follows_helix_orientation_not_structure_orientation():
    """in_out means the N-terminal half is cytosolic; out_in is the mirror."""
    from mpdms.analyses.a23_membrane_canon import flank
    cfg, segs = _tm_cfg(n_tm=2)
    a, b = segs[0], segs[1]
    d = pd.DataFrame({"pos": [a["start"], a["end"], b["start"], b["end"]]})
    f = flank(d, cfg).tolist()
    assert f == [True, False, False, True], f


def test_a23_snorkel_detects_a_v_in_kr_but_not_in_the_control():
    from mpdms.analyses.a23_membrane_canon import snorkel
    rng = np.random.default_rng(0)
    rows = []
    for pos in range(1, 120):
        z = rng.uniform(0, 18)
        for mut in "KR":                       # V shape: worst at the centre
            rows.append({"pos": pos, "wt": "L", "mut": mut, "seg_type": "TM", "absz": z,
                         "score_z": -1.2 + 0.006 * z ** 2 + rng.normal(0, 0.1)})
        for mut in "LI":                       # control: flat in depth
            rows.append({"pos": pos, "wt": "A", "mut": mut, "seg_type": "TM", "absz": z,
                         "score_z": -0.3 + rng.normal(0, 0.1)})
    r = snorkel(pd.DataFrame(rows))
    assert r["testable"] and r["passed"], r
    assert r["delta_curvature"] > 0 and r["p"] < 0.01


def test_a23_snorkel_quiet_when_both_classes_share_the_same_depth_shape():
    from mpdms.analyses.a23_membrane_canon import snorkel
    rng = np.random.default_rng(1)
    rows = []
    for pos in range(1, 120):
        z = rng.uniform(0, 18)
        for mut, wt, base in (("K", "L", -1.2), ("R", "L", -1.2), ("L", "A", -0.3), ("I", "A", -0.3)):
            rows.append({"pos": pos, "wt": wt, "mut": mut, "seg_type": "TM", "absz": z,
                         "score_z": base + 0.006 * z ** 2 + rng.normal(0, 0.1)})
    r = snorkel(pd.DataFrame(rows))
    assert r["testable"] and not r["passed"], r


def test_a23_positive_inside_reads_the_sign_of_each_prediction():
    from mpdms.analyses.a23_membrane_canon import positive_inside
    rng = np.random.default_rng(0)
    rows = []
    for pos in range(1, 120):
        cyt = pos % 2 == 0
        for mut in "KR":                       # K/R cheaper on the cytosolic flank
            rows.append({"pos": pos, "wt": "L", "mut": mut, "seg_type": "TM", "cytosolic_flank": cyt,
                         "score_z": (-0.3 if cyt else -1.1) + rng.normal(0, 0.1)})
        for mut in "DE":                       # D/E the mirror image
            rows.append({"pos": pos, "wt": "L", "mut": mut, "seg_type": "TM", "cytosolic_flank": cyt,
                         "score_z": (-1.1 if cyt else -0.3) + rng.normal(0, 0.1)})
    t = positive_inside(pd.DataFrame(rows)).set_index("mutation")
    assert bool(t.loc["K/R", "as_predicted"]) and bool(t.loc["D/E", "as_predicted"])
    assert t.q.max() < 0.01


def test_a23_aromatic_belt_fires_against_the_usual_gradient():
    from mpdms.analyses.a23_membrane_canon import aromatic_belt
    rng = np.random.default_rng(0)
    rows = []
    for pos in range(1, 160):
        z = rng.uniform(0, 18)
        iface = z > 8.0
        for mut in "AG":
            # W/Y worse at the interface; hydrophobic WT shows the usual opposite gradient
            rows.append({"pos": pos, "wt": "W", "mut": mut, "seg_type": "TM", "absz": z,
                         "score_z": (-1.2 if iface else -0.4) + rng.normal(0, 0.1)})
            rows.append({"pos": pos + 1000, "wt": "L", "mut": mut, "seg_type": "TM", "absz": z,
                         "score_z": (-0.4 if iface else -1.2) + rng.normal(0, 0.1)})
    r = aromatic_belt(pd.DataFrame(rows))
    assert r["testable"] and r["passed"] and r["difference"] < 0
    assert r["control_gradient"] > 0          # the control runs the other way, as it should


# ------------------------------- A24: structural determinants of tolerance
def test_a24_contact_order_helper_averages_sequence_separation():
    from mpdms.structure import contact_order
    pairs = pd.DataFrame({"pos_i": [10, 10, 20], "pos_j": [15, 60, 25]})
    co = contact_order(pairs, pd.Index([10, 15, 20, 25, 60, 99]))
    assert co[10] == pytest.approx((5 + 50) / 2)
    assert co[15] == pytest.approx(5) and co[60] == pytest.approx(50)
    assert np.isnan(co[99])                       # no contacts -> NaN, not zero


def _struct_variants(n_pos=120, seed=0, cavity_tolerant=True):
    """Positions in a membrane slab with a face label and a planted effect per face."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pos):
        face = ["protein-facing", "lipid-facing", "cavity-lining"][i % 3]
        base = {"protein-facing": -1.0, "lipid-facing": -0.15,
                "cavity-lining": -0.1 if cavity_tolerant else -1.0}[face]
        for mut in "KRAVLG":
            charged = mut in "KR"
            rows.append({"pos": i + 1, "wt": "L", "mut": mut, "vclass": "missense", "pass_filter": True,
                         "seg_type": "TM", "in_slab": True, "face": face, "absz": rng.uniform(0, 14),
                         "contacts": 20 if face == "protein-facing" else 8,
                         "contact_order": rng.uniform(20, 110),
                         "dvol": 0.0, "score_z": base - (0.4 if charged else 0.0) + rng.normal(0, 0.08)})
    d = pd.DataFrame(rows)
    d["sub_class"] = np.where(d.mut.isin(list("KR")), "to charged", "to hydrophobic")
    return d


def test_a24_facing_classes_separate_as_planted():
    from mpdms.analyses.a24_structural_tolerance import facing_table, facing_tests
    d = _struct_variants()
    t = facing_table(d).set_index(["face", "sub_class"])
    assert t.loc[("protein-facing", "to hydrophobic"), "median"] < -0.8
    assert t.loc[("lipid-facing", "to hydrophobic"), "median"] > -0.4
    ft = facing_tests(d)
    sig = ft[(ft.sub_class == "to hydrophobic") & (ft.a == "lipid-facing") & (ft.b == "protein-facing")]
    assert len(sig) and sig.iloc[0].q < 0.01 and sig.iloc[0].difference > 0


def test_a24_charge_penalty_applies_in_every_facing_class():
    """The planted charge cost is face-independent, so no facing contrast should invent one."""
    from mpdms.analyses.a24_structural_tolerance import facing_table
    t = facing_table(_struct_variants()).set_index(["face", "sub_class"])
    for face in ("protein-facing", "lipid-facing", "cavity-lining"):
        d = t.loc[(face, "to charged"), "median"] - t.loc[(face, "to hydrophobic"), "median"]
        assert d == pytest.approx(-0.4, abs=0.1), (face, d)


def test_a24_interface_matrix_is_symmetric_and_skips_self_pairs():
    from mpdms.analyses.a24_structural_tolerance import interface_matrix
    cfg = Config.wrap({"id": "T", "display_name": "T", "topology": {"segments": [
        {"name": "TM1", "start": 1, "end": 20, "type": "TM"},
        {"name": "TM2", "start": 30, "end": 50, "type": "TM"}]}})
    pairs = pd.DataFrame({"pos_i": [5, 6, 7, 8, 9], "pos_j": [35, 36, 37, 38, 39]})
    d = pd.DataFrame({"pos": list(range(1, 51)), "score_z": -0.5, "in_slab": True})
    t, mat = interface_matrix(d, pairs, cfg)
    assert list(t.helix_a) == ["TM1"] and list(t.helix_b) == ["TM2"]
    assert mat.loc["TM1", "TM2"] == mat.loc["TM2", "TM1"]
    assert np.isnan(mat.loc["TM1", "TM1"])        # a helix is not its own interface


# ------------------- A26: hydrophobicity preference vs solvent accessibility
def _slope_frame(tm_slope, nontm_slope, rsa_effect=0.0, seed=0, n=60):
    """Per-position slopes for two classes, optionally with a real RSA gradient."""
    rng = np.random.default_rng(seed)
    rows = []
    for cls, base in (("TM", tm_slope), ("non-TM", nontm_slope)):
        for i in range(n):
            # TM positions are a little less accessible, as in a real membrane protein
            rsa = rng.uniform(0.0, 0.6) if cls == "TM" else rng.uniform(0.2, 0.9)
            rows.append({"pos": len(rows) + 1, "cls": cls, "rsa": rsa,
                         "slope": base + rsa_effect * rsa + rng.normal(0, 0.03),
                         "slope_se": 0.02, "surface": rsa >= 0.25})
    return pd.DataFrame(rows)


def test_a26_flags_a_pooled_correlation_that_is_only_the_class_split():
    """TM and non-TM differ on both axes, so pooling invents a correlation. Must be flagged."""
    from mpdms.analyses.a26_hydrophobicity_burial import confounded, fit
    t = _slope_frame(tm_slope=0.30, nontm_slope=0.0, rsa_effect=0.0)
    st = fit(t)
    assert st["all"]["p"] < 0.05                      # pooled looks significant
    assert st["TM"]["p"] > 0.05 and st["non-TM"]["p"] > 0.05   # neither class is
    assert confounded(st)


def test_a26_does_not_flag_a_real_within_class_gradient():
    from mpdms.analyses.a26_hydrophobicity_burial import confounded, fit
    t = _slope_frame(tm_slope=0.30, nontm_slope=0.0, rsa_effect=-0.5)
    st = fit(t)
    assert st["TM"]["p"] < 0.05 and st["TM"]["spearman"] < 0
    assert not confounded(st)


def test_a26_slope_sign_means_more_hydrophobic_is_better():
    """A position where greasier substitutions score higher must get a positive slope."""
    from mpdms.analyses.a26_hydrophobicity_burial import position_slopes
    from mpdms.annot import BIOLOGICAL
    rng = np.random.default_rng(0)
    rows = []
    for mut in "ACDEFGHIKLMNPQRSTVWY":
        if mut == "L":
            continue
        rows.append({"pos": 10, "wt": "L", "mut": mut, "vclass": "missense", "pass_filter": True,
                     "score_z": 0.3 * (BIOLOGICAL[mut] - BIOLOGICAL["L"]) + rng.normal(0, 0.05)})
    d = pd.DataFrame(rows)
    d["dhyd"] = d.mut.map(BIOLOGICAL) - d.wt.map(BIOLOGICAL)
    r = ss.linregress(d.dhyd, d.score_z)
    assert r.slope > 0.2 and r.pvalue < 1e-6
