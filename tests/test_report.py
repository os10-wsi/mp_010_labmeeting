

def test_report_binds_the_figures_the_paper_style_actually_wrote(tmp_path):
    """A paper run writes to figures_paper/; binding figures/ would pick up stale artwork."""
    import mpdms.plotting as P
    from mpdms.report_cli import figure_dir
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "a01_dynamic_range.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "figures_paper").mkdir()
    (tmp_path / "figures_paper" / "a01_dynamic_range.pdf").write_bytes(b"%PDF-1.4\n")
    old = P.STYLE
    try:
        P.use_style("paper")
        assert figure_dir(tmp_path).name == "figures_paper"
        P.use_style("default")
        assert figure_dir(tmp_path).name == "figures"
    finally:
        P.use_style(old)


def test_report_falls_back_when_the_preferred_style_produced_nothing(tmp_path):
    import mpdms.plotting as P
    from mpdms.report_cli import figure_dir
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "a01_dynamic_range.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "figures_paper").mkdir()                     # exists but empty
    old = P.STYLE
    try:
        P.use_style("paper")
        assert figure_dir(tmp_path).name == "figures"
    finally:
        P.use_style(old)


def test_merging_no_figures_returns_false_rather_than_raising(tmp_path):
    """A --only subset can leave nothing to bind; that is a note, not a crash."""
    from mpdms.report_cli import merge_pdfs
    out = tmp_path / "empty.pdf"
    assert merge_pdfs([], out) is False
    assert not out.exists()
