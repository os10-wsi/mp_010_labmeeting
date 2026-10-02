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
