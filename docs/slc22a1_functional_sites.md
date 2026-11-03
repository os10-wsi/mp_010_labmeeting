# SLC22A1: the sites that lose function beyond their abundance

Table S1 of the SLC22A1 survivability library, 11,573 variants, three score columns:
`GFP_score` (abundance), `SM73_1_score` (function under selection) and `SM73_0_score`
(the same selection with nothing selecting). Both assays are put on the pipeline's
two-point scale, 0 at the synonymous median and −1 at the dead median, with in-frame
deletions as the dead anchor because the library contains no nonsense variants.

```bash
python -m mpdms dual data/raw/SLC22A1/slc22a1_table_s1.xlsx --id SLC22A1 \
    --abundance GFP_score --function SM73_1_score --control SM73_0_score
```

## The selection is real and the two assays agree

| | |
| --- | --- |
| missense / deletion / synonymous | 10,507 / 543 / 522 |
| abundance, wild type → dead | +0.012 → −2.063 |
| function, wild type → dead | −0.032 → **+1.489** |
| function against abundance | ρ = **+0.75** (n = 9,715) |
| no-selection control against abundance | ρ = **−0.06** (n = 9,803) |

The dead anchor sits *above* wild type in the function assay. This is a survival
selection: a broken transporter does not import the cytotoxin and the cell lives. The
two-point rescaling handles that through the sign of its denominator, so nothing is
flipped by hand. The control panel is the licence to read the first one as biology —
the same selection with no selection applied separates nothing.

## 28 positions lose more function than their abundance explains

A LOESS of function on abundance (`frac` 0.3), residuals in MAD-scaled units, a
one-sided signed-rank test per position on its own variants, Benjamini–Hochberg across
540 positions, and a called site needs both `q < 0.05` and a median residual
`z ≤ −1.5`. 28 positions qualify.

> F32 · D149 · S153 · N156 · G210 · W217 · Y240 · Q241 · T245 · L248 · W354 · Y361 ·
> G363 · I365 · L366 · F379 · A383 · E386 · A390 · C436 · R439 · L451 · G465 · D474 ·
> G476 · G477 · T480 · R486

## They are the substrate site

```bash
python -m mpdms activesite --sites outputs/_dual/d02_sites.csv --id SLC22A1 \
    --uniprot O15245 --chain A \
    --structure data/external/SLC22A1/8sc3.cif:ZVJ \
    --structure data/external/SLC22A1/8sc6.cif:VIB \
    --structure data/external/SLC22A1/8ju0.cif:SNL
```

Three ligand-bound cryo-EM structures of human OCT1, three different compounds in the
same pocket: fenoterol (8SC3), thiamine (8SC6), spironolactone (8JU0). Each residue is
placed at its closest approach to any of them.

| scope | called | rest | Mann–Whitney |
| --- | --- | --- | --- |
| whole protein | 6.1 Å (n 28) | 16.4 Å (n 419) | p = 2.2 × 10⁻¹² |
| transmembrane only | 6.0 Å (n 26) | 12.2 Å (n 225) | p = 2.0 × 10⁻⁹ |

| shell | called | other | odds ratio | Fisher |
| --- | --- | --- | --- | --- |
| ≤ 5 Å | 9 / 28 | 13 / 419 | 14.8 | 1.1 × 10⁻⁶ |
| ≤ 8 Å | 18 / 28 | 47 / 419 | 14.2 | 3.9 × 10⁻¹⁰ |
| ≤ 10 Å | 24 / 28 | 81 / 419 | 25.0 | 7.8 × 10⁻¹³ |

The TM-only row is the one that carries the argument. A called site is in a helix
nearly by construction and helix residues sit nearer the cavity than loop residues
whatever the biology, so the comparison has to be made inside the membrane, and it
survives there.

Nearest first: **Y361** (2.5 Å), **Q241** (2.6 Å), **F32** (3.2 Å), **W217** (3.2 Å),
**E386** (3.3 Å), **W354** (3.3 Å), A390 (4.0 Å), T245 (4.1 Å), **D474** (4.2 Å).

## Where they sit in the fold

| helix | called sites |
| --- | --- |
| TM1 | F32 |
| TM2 | S153, N156 |
| TM4 | G210, W217 |
| TM5 | Y240, Q241, T245, L248 |
| TM7 | W354, Y361, G363, I365, L366 |
| TM8 | F379, A383, E386, A390 |
| TM10 | C436, R439, L451 |
| TM11 | G465, D474, G476, G477, T480 |
| outside a TM segment | D149, R486 |

All 26 called sites that fall in a TM helix fall in **TM1, 2, 4, 5, 7, 8, 10 and 11** —
the eight helices that line the translocation pathway of an MFS fold. None is in TM3,
6, 9 or 12, the support helices, which between them hold 84 tested positions
(Fisher p = 1.5 × 10⁻⁵).

Within a helix the spacing is 3, 4 or 7 residues — TM8 runs 379, 383, 386, 390 and TM5
runs 241, 245, 248 — which is one face of an α-helix, the face pointing at the cavity.

The two sites outside the helices are the two furthest from any ligand: D149 (13.3 Å),
immediately before TM2, and R486 (18.0 Å), immediately after TM11. Both are at the
extracellular ends of cavity-lining helices rather than in the pocket, which is where
a gating residue would be rather than a binding one. They are the part of this result
that is not explained by proximity.

## What is not done here

The AQR1 half of the comparison and the family-wide Bayesian transfer both need the
yeast DMS tables, which are not in the repository (`data/raw` is ignored). The code
paths are `mpdms family` for aligned-position correlation, `mpdms bayes` then
`mpdms predict` for the pooled family model, and `mpdms transfer` to score that model
against the three measured quantities here. See `data/alignments/README.md` before
mapping AQR1 onto SLC22A1 position by position: at 21% identity the pairwise column
map fails the topology check that `bayes` applies to its own alignment.
