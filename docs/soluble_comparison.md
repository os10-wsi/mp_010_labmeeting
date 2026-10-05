# Comparing the membrane set to a soluble-protein abundance DMS

## Why this comparison, and what the control is

A25 is three-way, not two-way:

| group | what it is |
|---|---|
| `soluble` | the published reference: abundance DMS of soluble domains |
| `membrane, non-TM` | the loops and termini of **your own** protein |
| `membrane, TM` | its transmembrane helices |

The middle group is the point. It shares the assay, the host, the library and the
normalisation with the TM group, so:

- a difference that shows up against **both** non-TM and soluble is about the
  transmembrane environment;
- a difference that shows up against **soluble only** is about the experiment —
  different reporter, different library, different protein size — and is not evidence
  about membranes.

Without that control, every soluble-vs-membrane difference is confounded with the fact
that two different labs ran two different screens.

## The reference as actually supplied

Supplementary Table 2 of the domainome paper needed three things the first version of the
loader did not do, all now handled:

1. **Tab-delimited**, with a `.txt` extension. The delimiter is sniffed from the header.
2. **Multi-mutant rows.** Of 601,597 rows only 71,940 are single variants; the rest carry
   no single (position, wt, mut) and are multiples or wild-type replicates. Only singles
   are kept, and the number dropped is reported.
3. **No synonymous variants at all.** Its zero is the wild type, not a synonymous median.
   Dropping domains for want of synonymous controls would have discarded every one, so a
   domain with nonsense but no synonymous is rescaled on the anchor it has.

With those, all 50 domains load — 65,197 missense and 3,364 nonsense singles — each with
a nonsense median of exactly −1.000.

## The reference dataset

The natural comparator is an **aPCA abundance screen of soluble domains**, because aPCA
reports cellular abundance of a folded protein in yeast — the same quantity, in the same
host, as these screens. The human domainome dataset (Beltran et al., *Nature*; the paper
whose figures prompted this) is the obvious choice: hundreds of domains, stop and
synonymous controls in every one, and the same normalised-fitness convention.

I could not download it from this container — journal and data hosts are blocked — so
the loader takes whatever file you supply and the columns are matched by alias. Get the
variant-level supplementary table from the paper, then:

```yaml
# configs/<ID>_fitness.yaml
reference:
  soluble_path: data/external/domainome_variants.csv
```

```bash
python -m mpdms run configs/AQR1_fitness.yaml --only a25 --style paper
```

## Columns

Matched case-insensitively; the first alias found wins.

| needed | accepted names |
|---|---|
| domain id | `dataset`, `domain`, `domain_id`, `protein`, `gene`, `pdb`, `id` |
| position | `pos`, `position`, `wt_pos`, `aa_pos`, `site` |
| wild type | `wt`, `wt_aa`, `wild_type`, `aa_wt` |
| mutant | `mut`, `mut_aa`, `mutant`, `aa_mut`, `alt` |
| score | `score_z`, `normalized_fitness`, `fitness`, `score`, `nscore`, `mean_fitness` |
| class *(optional)* | `vclass`, `mut_type`, `variant_class`, `class`, `type` |
| RSA *(optional)* | `rsa`, `rel_sasa`, `relative_sasa`, `rsasa` |

Only position, mutant and score are required. Without a class column, `*` or `X` is read
as nonsense and `mut == wt` as synonymous. Without RSA, panel (d) of the burial
comparison is skipped with a note rather than guessed at.

If a column cannot be matched the loader raises and prints the first 25 column names it
did see, so adding an alias is a one-line change in `mpdms/reference.py`.

## Normalisation

**Every domain is normalised on its own synonymous and nonsense medians** before
anything is pooled, and a domain lacking either control is dropped and counted. This is
not optional: domains differ severalfold in raw dynamic range, and pooling first would
let the widest-range domains set the shape of the whole reference distribution. A test
asserts that two domains differing fourfold in raw range land on the same normalised
medians.

## What the panels test

- **(a) Distributions.** The three groups as densities, with Mann–Whitney on position
  medians. The expectation is TM ≪ soluble ≈ non-TM.
- **(b) Introduced residue classes.** Proline, glycine, charged, aromatic, small, each
  with a bootstrap CI, in all three groups. `tm_minus_soluble` in the table is the
  quantity of interest.
- **(c) Hydrophobicity.** Regress effect on Δhydrophobicity (von Heijne biological
  scale) in each group, cluster-robust by position, and test the interaction.

  I expected the soluble slope to be mildly *negative* — greasier is bad in solution,
  aggregation-prone — so that the comparison would be a **sign flip**. On the real
  domainome reference it is **slightly positive** (about +0.04): across folded soluble
  domains, replacing a residue with a more hydrophobic one is on average mildly
  tolerated. So the sign-flip framing does not hold for this comparator, and the
  statistic to quote is the **interaction** — how much steeper the TM slope is — not a
  change of sign. `sign_flip` is still reported, and should be read as a bonus rather
  than the expectation.
- **(d–f) Substitution matrices.** 20×20 mean effect in each group and their difference.
  Cells with fewer than three variants are left blank rather than drawn from one
  measurement.

## What is *not* comparable

Absolute dynamic range, because the reporters and libraries differ. Every comparison here
is of distribution shape, of contrasts computed within each dataset, or of ranks — never
of raw magnitudes between datasets. Nonsense variants are also not comparable: a stop in
a 40-residue domain and a stop in a 12-TM transporter are different experiments.
