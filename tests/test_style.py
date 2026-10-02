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
