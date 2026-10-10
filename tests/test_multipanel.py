"""Cross-protein panel figures: one panel per protein, comparable across panels."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mpdms.multipanel import PANELS, NotAvailable, build, harmonise


def _two_axes(lims):
    fig, axes = plt.subplots(1, len(lims))
    for ax, (lo, hi) in zip(np.atleast_1d(axes), lims):
        ax.plot([lo, hi], [lo, hi])
    return fig, list(np.atleast_1d(axes))


def test_harmonise_gives_every_panel_the_same_limits():
    """Eight panels that each auto-scaled cannot be compared, which is the whole point."""
    fig, axes = _two_axes([(-1, 1), (-5, 5)])
    harmonise(axes, "both")
    assert len({ax.get_xlim() for ax in axes}) == 1
    assert len({ax.get_ylim() for ax in axes}) == 1
    lo, hi = axes[0].get_xlim()
    assert lo <= -5 and hi >= 5          # the union, not the first panel's range
    plt.close(fig)


def test_harmonise_x_leaves_y_alone():
    fig, axes = _two_axes([(-1, 1), (-5, 5)])
    before = [ax.get_ylim() for ax in axes]
    harmonise(axes, "x")
    assert len({ax.get_xlim() for ax in axes}) == 1
    assert [ax.get_ylim() for ax in axes] == before
    plt.close(fig)


def test_harmonise_is_a_noop_for_a_single_panel():
    fig, axes = _two_axes([(-1, 1)])
    before = axes[0].get_xlim()
    harmonise(axes, "both")
    assert axes[0].get_xlim() == before
    plt.close(fig)


class _Cfg:
    """Enough of a config for the driver; the panels themselves are stubbed out."""
    def __init__(self, name):
        self.id = name
        self.display_name = name

    def get_path(self, *a, **k):
        return None

    def resolve(self, p):
        return None


def test_a_protein_that_cannot_supply_a_panel_keeps_its_place(monkeypatch):
    """Dropping it would shuffle the grid, so the panel says why instead."""
    def draw(ax, df, cfg, shared):
        if cfg.id == "B":
            raise NotAvailable("no ESM-1v scores")
        ax.plot([0, 1], [0, 1])
        return {}

    monkeypatch.setitem(PANELS, "_t", (draw, 2, (2.0, 2.0), "test", None))
    loaded = [(_Cfg(n), pd.DataFrame()) for n in "ABC"]
    fig, drawn, missing = build("_t", loaded)
    assert len(drawn) == 2 and missing == [("B", "no ESM-1v scores")]
    titles = [ax.get_title(loc="left") for ax in fig.axes[:3]]
    assert titles[0].startswith("A") and titles[2].startswith("C")   # B's slot still there
    assert any("no ESM-1v scores" in t.get_text() for t in fig.axes[1].texts)
    plt.close(fig)


def test_unused_slots_are_hidden_not_left_as_empty_frames(monkeypatch):
    monkeypatch.setitem(PANELS, "_t", (lambda ax, df, cfg, sh: ax.plot([0, 1], [0, 1]) and {},
                                       4, (2.0, 2.0), "test", None))
    loaded = [(_Cfg(n), pd.DataFrame()) for n in "AB"]
    fig, drawn, _ = build("_t", loaded)
    assert [ax.get_visible() for ax in fig.axes] == [True, True, False, False]
    plt.close(fig)


def test_the_prepass_result_reaches_every_panel(monkeypatch):
    """a05b needs one colour scale over all proteins, computed before any panel is drawn."""
    seen = []

    def draw(ax, df, cfg, shared):
        seen.append(shared["lim"])
        ax.plot([0, 1], [0, 1])
        return {}

    monkeypatch.setitem(PANELS, "_t", (draw, 2, (2.0, 2.0), "test", lambda loaded: {"lim": 2.5}))
    fig, _, _ = build("_t", [(_Cfg(n), pd.DataFrame()) for n in "AB"])
    assert seen == [2.5, 2.5]
    plt.close(fig)


def test_every_registered_panel_has_a_usable_spec():
    for key, (draw, ncols, size, title, pre) in PANELS.items():
        assert callable(draw) and ncols >= 1 and len(size) == 2 and title
        assert pre is None or callable(pre)


def test_density_legend_can_omit_per_protein_counts():
    """One shared legend must not carry one protein's n as if it were all of them."""
    from mpdms import plotting as P
    fig, ax = plt.subplots()
    P.density_by_class(ax, {"missense": np.random.default_rng(0).normal(size=200)}, counts=False)
    assert ax.get_legend_handles_labels()[1] == ["Missense"]
    ax.clear()
    P.density_by_class(ax, {"missense": np.random.default_rng(0).normal(size=200)})
    assert "n = 200" in ax.get_legend_handles_labels()[1][0]
    plt.close(fig)
