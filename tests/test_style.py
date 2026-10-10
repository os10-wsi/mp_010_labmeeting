"""Paper style: it must restyle without changing the default or losing information."""
import matplotlib.pyplot as plt
import pytest

from mpdms import plotting as P
from mpdms.config import Config


@pytest.fixture(autouse=True)
def _restore_default():
    yield
    P.use_style("default")


def _fig():
    fig, axes = plt.subplots(1, 2)
    for ax, lab in zip(axes, ["(a) Sites", "(b) Variants"]):
        ax.set_title(lab, loc="left")
    return fig


def test_use_style_switches_and_restores_rcparams():
    P.use_style("paper")
    assert P.paper() and plt.rcParams["font.size"] == 7
    P.use_style("default")
    assert not P.paper() and plt.rcParams["font.size"] == 8
    with pytest.raises(ValueError):
        P.use_style("nature-ish")


def test_panel_letters_replace_prefixed_titles():
    P.use_style("paper")
    fig = _fig()
    assert P._relabel_panels(fig) == [("a", "Sites"), ("b", "Variants")]
    assert all(a.get_title(loc="left") == "" for a in fig.axes)
    assert {t.get_text() for a in fig.axes for t in a.texts} == {"a", "b"}
    plt.close(fig)


def test_panel_letters_come_from_the_analysis_not_from_position():
    """Re-deriving letters would renumber panels the headlines already refer to."""
    P.use_style("paper")
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("(c) third panel", loc="left")
    axes[1].set_title("no prefix here", loc="left")
    assert P._relabel_panels(fig) == [("c", "third panel")]
    assert axes[1].get_title(loc="left") == "no prefix here"   # untouched
    plt.close(fig)


def test_colorbars_are_never_lettered():
    P.use_style("paper")
    fig, ax = plt.subplots()
    ax.set_title("(a) Sites", loc="left")
    sc = ax.scatter([0, 1], [0, 1], c=[0, 1])
    fig.colorbar(sc, ax=ax)
    assert P._relabel_panels(fig) == [("a", "Sites")]
    plt.close(fig)


def test_caption_keeps_the_title_stamp_and_panel_text(tmp_path):
    P.use_style("paper")
    cfg = Config.wrap({"id": "X", "display_name": "X", "topology": {"source": "TMHMM (file:a.gff3)"}})
    fig = _fig()
    P.title(fig, cfg, "abundance vs ESM-1v")
    assert fig._suptitle is None                      # no title drawn on the artwork
    P.save(fig, tmp_path, "f", cfg, "A15")
    cap = (tmp_path / "f_caption.txt").read_text()
    assert "abundance vs ESM-1v" in cap and "(a) Sites" in cap
    assert "TMHMM (file:a.gff3)" in cap and "A15" in cap   # provenance preserved off-figure


def test_default_style_still_draws_title_and_stamp(tmp_path):
    cfg = Config.wrap({"id": "X", "display_name": "X"})
    fig = _fig()
    P.title(fig, cfg, "something")
    P.save(fig, tmp_path, "f", cfg, "A15")
    assert not (tmp_path / "f_caption.txt").exists()
    assert (tmp_path / "f.png").exists() and (tmp_path / "f.pdf").exists()


# ------------------------------------------------- journal mark idioms
def test_density_by_class_draws_one_curve_per_class():
    import numpy as np
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    P.density_by_class(ax, {"missense": rng.normal(-0.5, .3, 400),
                            "synonymous": rng.normal(0, .1, 200),
                            "nonsense": rng.normal(-1, .2, 100)})
    assert len(ax.lines) == 3 and ax.get_ylabel() == "Density"
    assert all("n = " in ln.get_label() for ln in ax.lines)
    plt.close(fig)


def test_density_by_class_survives_degenerate_groups():
    """A class with one value, or all-identical values, must not kill the panel."""
    import numpy as np
    fig, ax = plt.subplots()
    P.density_by_class(ax, {"missense": np.array([0.2]),            # too few for a KDE
                            "synonymous": np.zeros(50),             # zero variance
                            "nonsense": np.random.default_rng(0).normal(-1, .2, 80)})
    assert len(ax.lines) + len(ax.collections) >= 1
    plt.close(fig)


def test_violin_box_marks_every_group_with_a_median():
    import numpy as np
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    g = {"a": rng.normal(0, 1, 100), "b": rng.normal(1, 1, 100), "c": rng.normal(-1, 1, 100)}
    P.violin_box(ax, g, points=10)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    white = [ln for ln in ax.lines if ln.get_marker() == "o" and ln.get_markerfacecolor() == "white"]
    assert len(white) == 3
    plt.close(fig)


def test_violin_box_skips_empty_groups_without_shifting_labels():
    import numpy as np
    fig, ax = plt.subplots()
    P.violin_box(ax, {"a": np.random.default_rng(0).normal(0, 1, 50), "empty": np.array([])})
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a"]
    plt.close(fig)


def test_scatter_fit_puts_the_statistic_in_the_corner_and_the_fit_in_red():
    import numpy as np
    x = np.linspace(-1, 1, 200)
    fig, ax = plt.subplots()
    P.scatter_fit(ax, x, 2 * x, fit=lambda q: 2 * q, stat="Spearman's rho = 0.99")
    assert any("rho" in t.get_text() for t in ax.texts)
    assert any(ln.get_color() == P.FIT_RED for ln in ax.lines)
    plt.close(fig)


def test_a19_density_panel_separates_published_from_the_rest(tmp_path):
    """A19 panel (a) is the reference's annotated-vs-other density comparison."""
    import numpy as np
    import pandas as pd

    from mpdms.analyses.a19_literature import figure, residue_table
    P.use_style("paper")
    rng = np.random.default_rng(0)
    n = 300
    sites = pd.DataFrame({"pos": np.arange(1, n + 1), "wt": "Q", "segment": "TM1", "n_variants": 18,
                          "median_abundance": rng.uniform(-1.2, 0.2, n),
                          "median_z": rng.normal(0, 0.8, n), "q": rng.uniform(0, 1, n)})
    pub = [10, 20, 30, 40, 50]
    sites.loc[sites.pos.isin(pub), "median_z"] = -2.5
    sites["functional"] = (sites.q < 0.05) & (sites.median_z <= -1.0)
    sites["abundance_tolerant"] = sites.median_abundance >= -0.5
    t = residue_table([{"pos": p_, "wt": "Q", "substitution": f"Q{p_}A"} for p_ in pub], sites)
    out = figure(Config.wrap({"id": "X", "display_name": "X"}), t, sites, tmp_path)
    assert any(str(p_).endswith(".png") for p_ in out)
    cap = (tmp_path / "a19_literature_caption.txt").read_text()
    assert "(a)" in cap and "(b)" in cap          # both panels lettered and captioned


def test_replicate_grid_fills_the_upper_triangle_with_r(tmp_path):
    """Fig 1c idiom: correlations in the upper triangle rather than blank panels."""
    import numpy as np
    import pandas as pd

    from mpdms.analyses.qc import replicate_grid
    rng = np.random.default_rng(0)
    n = 400
    base = rng.normal(-0.4, 0.5, n)
    df = pd.DataFrame({"pass_filter": True, "pos": np.arange(n), "vclass": "missense",
                       "rep1": base + rng.normal(0, .1, n), "rep2": base + rng.normal(0, .1, n),
                       "rep3": base + rng.normal(0, .1, n)})
    figs, cor = replicate_grid(df, Config.wrap({"id": "X", "display_name": "X"}),
                               ["rep1", "rep2", "rep3"], tmp_path)
    assert len(cor) == 3 and cor.pearson.min() > 0.8
    assert any(str(p_).endswith(".png") for p_ in figs)


def test_series_colours_are_fixed_in_order_not_cycled():
    """A protein dropping out of a figure must not repaint the others."""
    from mpdms.plotting import series_colors
    assert series_colors(3) == series_colors(5)[:3]
    assert series_colors(8)[:4] == series_colors(4)


def test_series_colours_extend_past_the_fixed_list():
    from mpdms.plotting import SERIES, series_colors
    n = len(SERIES) + 4
    c = series_colors(n)
    assert len(c) == n and len(set(c)) == n
    assert all(x.startswith("#") for x in c)


def test_signed_quantities_keep_a_diverging_map_not_viridis():
    """Viridis has no neutral midpoint, so using it for data centred on zero would put a
    strong hue where 'no effect' belongs."""
    import numpy as np
    from mpdms.plotting import diverging_cmap, sequential_cmap
    d = diverging_cmap()
    mid = np.array(d(0.5)[:3])
    assert np.ptp(mid) < 0.12                      # the midpoint is near-neutral
    lo, hi = np.array(d(0.0)[:3]), np.array(d(1.0)[:3])
    assert np.argmax(lo) != np.argmax(hi)        # two different hues at the poles
    v = sequential_cmap()
    assert np.ptp(np.array(v(0.5)[:3])) > 0.2      # viridis is emphatically not neutral there


def test_sequential_map_is_monotone_in_lightness():
    import numpy as np
    from mpdms.plotting import sequential_cmap
    v = sequential_cmap()
    lum = [0.2126 * r + 0.7152 * g + 0.0722 * b
           for r, g, b, _ in (v(i / 20) for i in range(21))]
    assert all(b >= a - 1e-3 for a, b in zip(lum, lum[1:]))


def test_titles_and_axis_labels_are_bold_in_both_styles():
    import matplotlib.pyplot as plt
    from mpdms import plotting as P
    old = P.STYLE
    try:
        for style in ("default", "paper"):
            P.use_style(style)
            assert plt.rcParams["axes.titleweight"] == "bold", style
            assert plt.rcParams["axes.labelweight"] == "bold", style
    finally:
        P.use_style(old)
