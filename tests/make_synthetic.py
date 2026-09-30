"""Generate two contrasting synthetic DMS datasets (+ fake structures) for testing.

SYN1: 4-TM protein, pos/WT_AA/Mut columns, 3 replicates, DiMSum-like counts.
SYN2: 1-TM protein with a soluble domain, HGVS variant strings, 2 replicates.

    python tests/make_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mpdms.annot import BIOLOGICAL, CHARGE  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
HYD = "LIVFAMLIVFGALLIVAFCW"
POLAR = "SNDEKRQTGPHSEKDNRQST"
THREE = dict(A="Ala", R="Arg", N="Asn", D="Asp", C="Cys", Q="Gln", E="Glu", G="Gly", H="His", I="Ile",
             L="Leu", K="Lys", M="Met", F="Phe", P="Pro", S="Ser", T="Thr", W="Trp", Y="Tyr", V="Val")


def make_seq(layout, rng):
    seq = []
    for kind, n in layout:
        pool = HYD if kind == "TM" else POLAR
        seq += list(rng.choice(list(pool), n))
    seq[0] = "M"
    return "".join(seq)


def build_pdb(seq, layout, path):
    """Idealised model: TM helices antiparallel along z on a circle, other segments as
    helices/extended strands placed between them. Chain breaks are fine for our purposes."""
    import PeptideBuilder
    from PeptideBuilder import Geometry
    from Bio.PDB import PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain

    chain = Chain("A")
    resid, i0, ntm = 1, 0, 0
    tm_total = sum(k == "TM" for k, _ in layout)
    for k, n in layout:
        frag = seq[i0:i0 + n]
        geos = []
        for j, aa in enumerate(frag):
            g = Geometry.geometry(aa)
            if k in ("TM", "helix"):
                g.phi, g.psi_im1 = -57, -47
            else:
                g.phi, g.psi_im1 = -120, 130
            geos.append(g)
        st = PeptideBuilder.initialize_res(geos[0])
        for g in geos[1:]:
            PeptideBuilder.add_residue(st, g)
        res = list(st[0]["A"])

        ca = np.array([r["CA"].coord for r in res])
        c = ca.mean(0)
        v = np.linalg.svd(ca - c)[2][0]
        if np.dot(v, ca[-1] - ca[0]) < 0:
            v = -v
        if k == "TM":
            target = np.array([0, 0, 1.0]) * (1 if ntm % 2 == 0 else -1)
            ang = 2 * np.pi * ntm / max(tm_total, 1)
            centre = np.array([10 * np.cos(ang), 10 * np.sin(ang), 0])
            ntm += 1
        else:
            target = np.array([1.0, 0, 0])
            side = 1 if ntm % 2 == 0 else -1
            centre = np.array([25 + 3 * i0 / 10, 0, side * (22 + n / 4)])
        # rotation v -> target
        a, b = v / np.linalg.norm(v), target
        cr = np.cross(a, b)
        s, cth = np.linalg.norm(cr), np.dot(a, b)
        if s < 1e-8:
            R = np.eye(3) if cth > 0 else -np.eye(3)
        else:
            K = np.array([[0, -cr[2], cr[1]], [cr[2], 0, -cr[0]], [-cr[1], cr[0], 0]])
            R = np.eye(3) + K + K @ K * ((1 - cth) / s ** 2)
        for r in res:
            for at in r:
                at.coord = ((at.coord - c) @ R.T + centre).astype("f")
            r.detach_parent()
            r.id = (" ", resid, " ")
            for at in r:
                at.bfactor = 90.0 if k in ("TM", "helix") else 50.0
            chain.add(r)
            resid += 1
        i0 += n
    m = Model(0)
    m.add(chain)
    s = Structure("syn")
    s.add(m)
    io = PDBIO()
    io.set_structure(s)
    io.save(str(path))


def simulate(seq, layout, rng, n_reps=3, stop_present=True, depth=150):
    kinds = [k for k, n in layout for _ in range(n)]
    L = len(seq)
    # position sensitivity: TM > helix > loop, periodic buried face in TMs
    sens = np.array([{"TM": 0.55, "helix": 0.35, "strand": 0.35}.get(k, 0.02) for k in kinds])
    for i, k in enumerate(kinds):
        if k == "TM":
            sens[i] *= 0.6 + 0.8 * (np.cos(2 * np.pi * i / 3.6) > 0)
    sens += rng.gamma(1.0, 0.04, L)
    sens[:3] = 0.02
    rows = []
    for p in range(1, L + 1):
        wt = seq[p - 1]
        k = kinds[p - 1]
        muts = list(AA) + (["*"] if stop_present else [])
        for m in muts:
            if m == wt:
                effect = 0.0; mm = wt
            elif m == "*":
                effect = -1.0 if p < L - 8 else -0.1
                mm = "*"
            else:
                mm = m
                dh = BIOLOGICAL[m] - BIOLOGICAL[wt]
                effect = -sens[p - 1] * (0.6 + (0.9 * max(0, -dh) if k == "TM" else 0.3 * abs(dh) / 3))
                if m == "P" and k in ("TM", "helix"):
                    effect -= 0.6
                if CHARGE[m] != 0 and k == "TM":
                    effect -= 0.25
                effect += rng.normal(0, 0.05)
                effect = max(effect, -1.3)
            rows.append((p, wt, mm, effect))
        # a couple of extra synonymous codon variants
        for _ in range(rng.integers(0, 2)):
            rows.append((p, wt, wt, 0.0))
    df = pd.DataFrame(rows, columns=["pos", "wt", "mut", "true"])
    df = df.sample(frac=0.93, random_state=1).sort_index()  # missingness
    # a dead tiling block
    dead = (df.pos > int(L * 0.62)) & (df.pos < int(L * 0.62) + 6)
    df = df[~dead | (rng.random(len(df)) < 0.2)]
    raw_scale, raw_syn = 2.3, 0.05
    inp = rng.lognormal(np.log(depth), 1.0, (len(df), n_reps)).round()
    out = {}
    for r in range(n_reps):
        noise = rng.normal(0, 1, len(df)) * np.sqrt(4 / (inp[:, r] + 1) + 1 / 25) * 1.0
        out[f"rescaled_fitness_rep{r + 1}"] = raw_syn + raw_scale * df["true"].to_numpy() + noise + rng.normal(0, 0.03)
        f = np.exp(out[f"rescaled_fitness_rep{r + 1}"])
        out[f"input{r + 1}"] = inp[:, r]
        out[f"output{r + 1}"] = rng.poisson(inp[:, r] * f)
    df = df.reset_index(drop=True)
    for k, v in out.items():
        df[k] = v
    reps = [f"rescaled_fitness_rep{r + 1}" for r in range(n_reps)]
    df["rescaled_fitness"] = df[reps].mean(axis=1)
    df["rescaled_sigma"] = df[reps].std(axis=1) / np.sqrt(n_reps)
    return df


def to_lab_format(df, seq, rng):
    """Write the lab's fitness_estimation.tsv layout (spaced names, empty pos for synonymous)."""
    from mpdms.io import CODON_TABLE
    by_aa = {}
    for c, a in CODON_TABLE.items():
        by_aa.setdefault(a, []).append(c)
    cds = [rng.choice(by_aa[a]) for a in seq]
    out = []
    for r in df.itertuples():
        i = r.pos - 1
        if r.mut == r.wt:  # synonymous: alternative codon, no pos/wt/mut
            alts = [c for c in by_aa[r.wt] if c != cds[i]]
            if not alts:
                continue
            cod, aa_seq, wt_, pos_, mut_, aa_ham = rng.choice(alts), seq, "", "", "", 0
        else:
            cod = by_aa[r.mut][0]
            aa_seq = seq[:i] + r.mut + seq[i + 1:]
            wt_, pos_, mut_, aa_ham = r.wt, r.pos, r.mut, 1
        nt = "".join(cds[:i]) + cod + "".join(cds[i + 1:])
        row = {"wt aa": wt_, "pos": pos_, "mut aa": mut_, "aa_ham": aa_ham, "aa_seq": aa_seq,
               "nt_ham": sum(a != b for a, b in zip(cod, cds[i])), "nt_seq": nt}
        for k in (1, 2, 3):
            row[f"input{k}"] = getattr(r, f"input{k}")
        for k in (1, 2, 3):
            row[f"output{k}"] = getattr(r, f"output{k}")
        row["wt"], row["stop"] = "", (r.mut == "*") or ""
        for k in (1, 2, 3):
            row[f"raw_fitness_rep{k}"] = getattr(r, f"rescaled_fitness_rep{k}") * 1.3
        for k in (1, 2, 3):
            row[f"rescaled_fitness_rep{k}"] = getattr(r, f"rescaled_fitness_rep{k}")
        row["mean fitness"] = r.rescaled_fitness
        row["fitness sd"] = r.rescaled_sigma
        out.append(row)
    wt_row = dict(out[0], **{"wt aa": "", "pos": "", "mut aa": "", "aa_ham": 0, "aa_seq": seq, "nt_ham": 0,
                              "nt_seq": "".join(cds), "wt": True, "stop": "", "mean fitness": 0.0})
    return pd.DataFrame([wt_row] + out)


def main():
    rng = np.random.default_rng(7)
    # SYN1: 4-TM, N-term cytosolic
    lay1 = [("loop", 30), ("TM", 21), ("loop", 12), ("TM", 20), ("loop", 25), ("helix", 18), ("loop", 10),
            ("TM", 22), ("loop", 9), ("TM", 21), ("loop", 40)]
    seq1 = make_seq(lay1, rng)
    d1 = simulate(seq1, lay1, rng, n_reps=3)
    out1 = ROOT / "data/raw/SYN1/fitness_estimation"
    out1.mkdir(parents=True, exist_ok=True)
    to_lab_format(d1, seq1, rng).to_csv(out1 / "fitness_estimation.tsv", sep="\t", index=False)
    ext1 = ROOT / "data/external/SYN1"
    ext1.mkdir(parents=True, exist_ok=True)
    (ext1 / "SYN1.fasta").write_text(f">SYN1\n{seq1}\n")
    build_pdb(seq1, lay1, ext1 / "AF-SYN1-F1-model_v4.pdb")

    # SYN2: 1-TM + soluble domain, HGVS strings, 2 replicates
    lay2 = [("loop", 15), ("TM", 20), ("loop", 20), ("helix", 25), ("loop", 8), ("strand", 7), ("loop", 5),
            ("strand", 7), ("loop", 10), ("helix", 20), ("loop", 30)]
    seq2 = make_seq(lay2, rng)
    d2 = simulate(seq2, lay2, rng, n_reps=2, depth=60)
    d2["variant"] = [f"p.{THREE[w]}{p}{'=' if m == w else ('Ter' if m == '*' else THREE[m])}"
                     for p, w, m in zip(d2.pos, d2.wt, d2.mut)]
    out2 = ROOT / "data/raw/SYN2/fitness_estimation"
    out2.mkdir(parents=True, exist_ok=True)
    d2.drop(columns=["pos", "wt", "mut", "true"]).to_csv(out2 / "fitness_estimation.tsv", sep="\t", index=False)
    ext2 = ROOT / "data/external/SYN2"
    ext2.mkdir(parents=True, exist_ok=True)
    (ext2 / "SYN2.fasta").write_text(f">SYN2\n{seq2}\n")
    build_pdb(seq2, lay2, ext2 / "AF-SYN2-F1-model_v4.pdb")
    print("wrote", out1, out2)


if __name__ == "__main__":
    main()
