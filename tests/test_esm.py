"""ESM-1v masked-marginal bookkeeping (model-free) and the A15 functional-site call."""
import pytest
import numpy as np

from mpdms.esm_score import AA20, masked_marginals, to_long, windows


def test_windows_cover_long_sequences():
    w = windows(2500)
    assert w[0][0] == 0 and w[-1][1] == 2500
    assert all(b - a <= 1022 for a, b in w)
    assert all(w[i + 1][0] < w[i][1] for i in range(len(w) - 1))  # overlapping


def test_masked_marginals_relative_to_wt():
    seq = "MKLVAG" * 200  # 1200 aa -> two windows
    seen = []

    def fake(items):
        seen.extend(len(s) for s, _ in items)
        # log p is 0 for alanine, -1 elsewhere
        return np.array([[0.0 if a == "A" else -1.0 for a in AA20] for _ in items])

    m = masked_marginals(seq, fake, batch=16)
    assert m.shape == (1200, 20) and max(seen) <= 1022
    wt_cols = [AA20.index(a) for a in seq]
    assert np.allclose(m[np.arange(1200), wt_cols], 0)          # wt -> wt scores 0
    assert np.allclose(m[seq.index("K"), AA20.index("A")], 1)    # K->A favoured
    tab = to_long(seq[:6], [m[:6]], [1])
    assert len(tab) == 6 * 19 and set(tab.columns) >= {"pos", "wt", "mut", "esm1v"}


def test_functional_list_is_complete_wrapped_and_in_sequence_order():
    """Panels (b)/(c) only label the top few; the list panel must carry every residue."""
    import numpy as np
    import pandas as pd

    from mpdms.analyses.a15_esm_functional import functional_list
    rng = np.random.default_rng(0)
    pos = sorted(rng.choice(np.arange(1, 531), 130, replace=False))
    f = pd.DataFrame({"pos": pos, "wt": list("ACDEFGHIKLMNPQRSTVWY") * 6 + ["A"] * 10,
                      "abundance_tolerant": rng.random(130) < 0.4})
    lines = functional_list(f)
    joined = " ".join(lines)
    assert all(f"{r.wt}{r.pos}" in joined for r in f.itertuples())        # nothing dropped
    assert all(len(ln) <= 140 for ln in lines)                            # wrapped to width
    order = [int(tok.rstrip("*")[1:]) for tok in joined.split()]
    assert order == sorted(order)                                         # sequence order
    assert joined.count("*") == int(f.abundance_tolerant.sum())           # tolerant marked
    assert functional_list(f.iloc[:0]) == ["none"]


# ------------------------------- finding the right protein in a proteome-wide table
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _proteome(target_seq, target_acc="TARGET", n_decoys=60, seed=0):
    """A table of many proteins, one of which is the target, as a whole-proteome file is."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_decoys):
        acc = f"DECOY{k:03d}"
        seq = "".join(rng.choice(list(AA20), rng.integers(150, 400)))
        for pos in range(1, len(seq) + 1, 5):
            for mut in rng.choice(list(AA20), 3, replace=False):
                rows.append({"id": acc, "mutation": f"{seq[pos - 1]}{pos}{mut}",
                             "mean": float(rng.normal(-5, 2))})
    for pos in range(1, len(target_seq) + 1):
        for mut in rng.choice(list(AA20), 4, replace=False):
            rows.append({"id": target_acc, "mutation": f"{target_seq[pos - 1]}{pos}{mut}",
                         "mean": float(rng.normal(-5, 2))})
    return pd.DataFrame(rows)


def _seq(n=320, seed=7):
    import numpy as np
    return "".join(np.random.default_rng(seed).choice(list(AA20), n))


def test_accession_found_by_sequence_in_a_proteome_table(tmp_path):
    """The accession need not be known: the mutation strings carry the wild-type residues."""
    from mpdms.esm_score import match_accession, read_bulk_table
    seq = _seq()
    p = tmp_path / "proteome.csv"
    _proteome(seq, target_acc="Q12345").to_csv(p, index=False)
    acc, info = match_accession(read_bulk_table(p), seq)
    assert acc == "Q12345"
    assert info["agreement"] == 1.0 and info["positions"] == len(seq)
    assert info["runner_up_agreement"] < 0.5          # chance agreement for the decoys


def test_no_accession_returned_when_the_protein_is_absent(tmp_path):
    """Importing another protein's scores would look normal downstream, so refuse instead."""
    from mpdms.esm_score import match_accession, read_bulk_table
    seq = _seq()
    p = tmp_path / "proteome.csv"
    t = _proteome(seq, target_acc="Q12345")
    t[t.id != "Q12345"].to_csv(p, index=False)
    acc, info = match_accession(read_bulk_table(p), seq)
    assert acc is None and "below" in info["reason"]


def test_match_ignores_accessions_covering_too_few_positions(tmp_path):
    """A tiny entry can agree perfectly by luck; it must not outrank the real protein."""
    import pandas as pd
    from mpdms.esm_score import match_accession, read_bulk_table
    seq = _seq()
    t = _proteome(seq, target_acc="Q12345", n_decoys=5)
    tiny = pd.DataFrame([{"id": "TINY", "mutation": f"{seq[i]}{i + 1}A", "mean": -5.0}
                         for i in range(4)])          # 4 positions, all correct
    p = tmp_path / "proteome.csv"
    pd.concat([t, tiny]).to_csv(p, index=False)
    acc, info = match_accession(read_bulk_table(p), seq)
    assert acc == "Q12345"


def test_streaming_match_agrees_with_the_in_memory_one(tmp_path):
    """Streaming is only worth having if it gives the same answer as loading the table."""
    from mpdms.esm_score import match_accession, match_accession_streaming, read_bulk_table
    seq = _seq()
    p = tmp_path / "proteome.csv"
    _proteome(seq, target_acc="Q12345", n_decoys=6).to_csv(p, index=False)
    whole, winfo = match_accession(read_bulk_table(p), seq)
    for chunk in (7, 101, 10_000):                    # chunk boundaries must not matter
        acc, info = match_accession_streaming(p, seq, chunksize=chunk)
        assert acc == whole == "Q12345"
        assert info["positions"] == winfo["positions"]
        assert info["agreement"] == pytest.approx(winfo["agreement"])
        assert info["n_variants"] == winfo["n_variants"]


def test_streaming_extract_returns_only_that_protein(tmp_path):
    from mpdms.esm_score import read_bulk_table, rows_for_accession
    seq = _seq()
    p = tmp_path / "proteome.csv"
    _proteome(seq, target_acc="Q12345", n_decoys=4).to_csv(p, index=False)
    want = read_bulk_table(p).query("acc == 'Q12345'").reset_index(drop=True)
    got = rows_for_accession(p, "Q12345", chunksize=137)
    assert set(got.acc) == {"Q12345"}
    assert len(got) == len(want)
    assert got.sort_values(["pos", "mut"]).esm1v.tolist() == \
           want.sort_values(["pos", "mut"]).esm1v.tolist()


def test_mutation_parsing_matches_the_regex_it_replaced():
    """Slicing is quicker than the regex, so it must agree with it, junk rows included."""
    import pandas as pd
    from mpdms.esm_score import MUT_RE, parse_mutations
    s = pd.Series(["M1A", "K123R", "W1000Y", "", "x", "nonsense", "M1", "1A2", "A12b", "QQ3R"])
    ref = s.str.extract(MUT_RE)
    got = parse_mutations(s)
    assert got.pos.notna().tolist() == pd.to_numeric(ref[1]).notna().tolist()
    keep = got.pos.notna()
    assert got.loc[keep, "wt"].tolist() == ref.loc[keep, 0].tolist()
    assert got.loc[keep, "mut"].tolist() == ref.loc[keep, 2].tolist()
    assert got.loc[keep, "pos"].tolist() == pd.to_numeric(ref.loc[keep, 1]).tolist()
