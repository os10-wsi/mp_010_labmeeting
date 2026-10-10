# mpdms — membrane-protein DMS lab-meeting pipeline

This pipeline takes one DMS folder per yeast membrane protein (a `fitness_estimation.tsv`
plus the three QC PDFs) through these steps:

1. **QC.** Read-count filter, replicate correlations, and fitness distributions by variant
   class (missense blue, synonymous green, stop red).
2. **Per-protein normalisation.** Median(stop) is set to −1, median(synonymous) to 0, and
   every variant is rescaled linearly between them. Each replicate is normalised on its own
   anchors.
3. **DMS heatmap.** −1 is dark red, 0 is white and anything above 0 is light blue. Below
   every wrapped row of the heatmap are two **SSDraw** tracks, drawn on the gene's AlphaFold
   model: the mean missense effect at each position (stops excluded), and the effect of
   proline at each position.
4. **The prompt-pack analyses A01–A12**, each with one figure, one tidy table and one
   `stats.json` row, then a Markdown one-pager per dataset and a cross-dataset summary slide.

## Start here

One file plus its DeepTMHMM gff3, one command:

```bash
python -m mpdms quickstart /path/to/fitness_singles_hxt2.txt --gff3 /path/to/hxt2.gff3
```

That infers the gene from the filename, recovers the wild-type sequence from the table's
own `aa_seq` column (so nothing is fetched and no FASTA is needed), writes the config,
reads the topology from the gff3, runs every analysis, and builds the report. It finishes
by listing what ran, what skipped, and what would unlock each skip.

Add a structure when you have one — it is what most of the skips are waiting for:

```bash
python -m mpdms quickstart fitness_singles_hxt2.txt --gff3 hxt2.gff3 --structure hxt2.pdb
```

Useful flags: `--gene NAME` to override the inferred name, `--no-run` to write the config
and stop, `--only a05,a22` to run a subset, `--style default` for slide-style figures.

Everything below is the longer route, for when a dataset needs something the one-liner
cannot guess.

## Quick start

```bash
pip install -r requirements.lock      # plus `mkdssp` (conda install -c salilab dssp) if you want DSSP
                                      # (pydssp is used otherwise)

# 1. one config per dataset: gene name from the folder name, UniProt sequence + topology,
#    AlphaFold model -> data/external/<GENE>/
python -m mpdms init /path/to/dms_root          # finds every fitness_estimation.tsv below it
# or: python -m mpdms init /path/to/SEC61/fitness_estimation --gene SEC61

# 2. curate configs/<GENE>.yaml (topology, filters, assay.type), then
python run_all.py                               # all datasets x all analyses
python run_all.py configs/SEC61.yaml --only qc,heatmap
```

`init` works out the gene from the nearest path component that looks like a yeast gene
(`SEC61`, `PMA1`) or an ORF id (`YAL001C`). Retrieval uses UniProt (taxon 559292) and the
AlphaFold DB, and caches everything in `data/external/<GENE>/`. On a machine without
internet access, put `<ACC>.fasta`, `<ACC>.uniprot.json` and `AF-<ACC>-F1-model_v*.pdb` in
that folder yourself and run `init --offline`.

For an offline demo on two synthetic datasets, run `tests/run_demo.sh`.

## Input table

The lab's `fitness_estimation.tsv` layout is supported as is:

```
wt aa  pos  mut aa  aa_ham  aa_seq  nt_ham  nt_seq  input1..3  output1..3  wt  stop
raw_fitness_rep1..3  rescaled_fitness_rep1..3  mean fitness  fitness sd
```

- Replicates are read from `rescaled_fitness_rep1..3` and the mean from `mean fitness`.
  Normalisation is linear, so using the rescaled rather than the raw columns does not change
  `score_z`.
- Synonymous rows have `aa_ham = 0` and an empty `pos`/`wt aa`/`mut aa`. Their position and
  codon are recovered from `nt_seq`, compared with the WT CDS (the `nt_ham = 0` / `wt` row,
  or the consensus of all variants if there is no WT row). They form the 0 anchor.
- The boolean `stop` column forces the nonsense class, and the `wt` row is kept aside.
- Rows with more than one amino-acid change are dropped.

Other layouts are auto-detected too: `Pos`/`WT_AA`/`Mut`, HGVS strings (`p.Met1Ala`), DiMSum
`aa_seq` tables, `fitness1_uncorr`, and so on. Any column can be pinned under `columns:` in
the config.

Filters: a replicate value is used only if its input count is at least `filters.min_reads`.
A variant passes if it has `filters.min_replicates` usable replicates. The QC figure
`qc_read_threshold` shows replicate r and the fraction of variants kept across thresholds,
so you can choose `min_reads` for each dataset.

## Outputs

```
outputs/<GENE>/figures/   qc_*.pdf, heatmap.pdf, heatmap_ssdraw.pdf, ssdraw_mean_effect.pdf,
                          ssdraw_proline.pdf, a01…a12_*.pdf   (PDF with embedded fonts + 300-dpi PNG)
outputs/<GENE>/tables/    tidy CSVs; structure_{mean_missense,proline}.pdb (B-factor = score)
                          + .pml PyMOL scripts that use the heatmap colours; a10 PyMOL selection
outputs/<GENE>/stats.json one row per analysis: headline, metrics, caveats
outputs/_cross/           stats.json, summary.csv, summary.pdf  (slide 1; † = caveat fired)
reports/<GENE>.md         auto-assembled one-pager
data/processed/<GENE>.parquet   canonical table (score_raw, score_z, rep1..N, pass_filter, topology)
```

Every figure is stamped with the dataset id and the git commit.

## Layout

```
mpdms/config.py      config schema, defaults (configs/_template.yaml) and validation
mpdms/io.py          load_dataset(): column detection, variant parsing, filters, normalisation
mpdms/annot.py       topology, hydrophobicity/volume/helix/charge scales (with sources), motifs
mpdms/stats.py       circular-shift test, block bootstrap, BH-FDR, ICC, runs test
mpdms/structure.py   residue table, DSSP/pydssp SS, RSA, contacts, membrane frame
mpdms/ssdraw_tracks.py  SSDraw geometry drawn into matplotlib axes with a fixed colour scale
mpdms/fetch.py, init_dataset.py   UniProt/AlphaFold retrieval and config generation
mpdms/analyses/      qc, heatmap_structure, a01 … a12
configs/_motifs.yaml extensible motif regexes for A09
```

## Where the code departs from the prompt pack

- **A07 significance.** Circularly shifting a profile leaves its power spectrum unchanged,
  so a circular-shift null cannot test spectral power. Surrogates are instead AR(1) series
  matched to each segment's mean, variance and lag-1 autocorrelation. That keeps the
  autocorrelation structure the pack wanted preserved.
- **A02 missingness.** For the same reason, clustering of missing cells is tested with a
  runs test plus a shuffle null on the lag-1 autocorrelation of per-position coverage.
- **A08 membrane frame.** `pca_tm_axes` uses the mean of the per-TMD helix axes, sign-aligned
  by orientation, rather than one PCA over all TM Cα atoms. The single PCA gives a wrong
  normal for bundles that are wider than they are tall. For exact depths, give a PPM/OPM
  oriented model and set `membrane_normal: ppm`.
- **A10.** Holding out whole segments can give negative R² when a protein has only one or two
  TMDs. A 20-residue block CV is reported alongside, and a caveat fires.
- **A12** is skipped unless codon-level synonymous data exist. For the codon-adaptiveness
  term, provide `evolution.codon_usage` (csv `codon,w`).
- **Topology.** When UniProt has no TM annotation, TMs are predicted from hydropathy with the
  positive-inside rule, and the config and report are flagged "NOT curated".

## TM helix burial analyses (A13, A14; opt-in)

Run these for selected proteins only. The command below produces one output set per
protein and one pooled across them, with protein as a stratum:

```bash
python -m mpdms helix configs/TPO3.yaml configs/FEN2.yaml configs/QDR2.yaml
# per protein  -> outputs/<GENE>/figures/a13*, a14*
# pooled       -> outputs/_helix_TPO3_FEN2_QDR2/
```

- **A13a: position along the helix.** Fitness is plotted against position, from −1 at the
  cytosolic end to +1 at the lumenal end. Plots are made for all helices and for surface vs
  core helices, using both a median split and a tertile split.
- **A13b: surface vs core helices.** Burial is the relative solvent-accessible surface area
  (RSA) on the AlphaFold model, with no membrane present. A helix is "surface" or "core"
  by its median RSA. The helix is the unit of analysis: labels are permuted across helices
  within each protein, Hedges' g is reported, and a mixed model includes a random
  intercept per helix. Helix RSA is also compared with helix mean fitness as a continuous
  variable.
- **A13c: each helix independently.** Small multiples per helix, a ranking of helices, and
  a test of each helix against the protein's other TM helices (BH-FDR corrected).
- **A14: hydrophobic TM residues (A V L I F M W C).** Lipid-facing (RSA > 0.25) vs buried
  (RSA < 0.10), crossed with helix class. Substitutions are split into rest, → polar,
  → hydrophobic, → Pro and → Gly. Includes a facing × class × substitution model and
  within-cell slopes for hydrophobicity change vs volume change.

Fitness is always split into Pro, Gly and the rest (missense excluding Pro/Gly), and each
is plotted separately. Residues with pLDDT < 70 are excluded.
`a13_sensitivity_boundaries.csv` repeats the headline statistics with helix boundaries
taken from DSSP instead of UniProt.

## Abundance vs ESM-1v: functional sites (A15)

```bash
pip install fair-esm torch                    # once; on a GPU node install the CUDA build of torch
export TORCH_HOME=/lustre/.../torch_cache     # weights are ~2.6 GB per model; keep them off $HOME
python -m mpdms esm configs/QDR2.yaml configs/AQR1.yaml          # all 5 ESM-1v models (masked marginal)
python -m mpdms esm configs/QDR2.yaml --models 1                  # quicker: one model
python run_all.py configs/QDR2.yaml configs/AQR1.yaml --only a15
```

If you already have a table of precomputed ESM-1v scores for many proteins (columns
`id, mutation, delta_logp_1..5, mean`, keyed by UniProt accession), import from it instead
of running ESM:

```bash
python -m mpdms esm configs/QDR2.yaml configs/AQR1.yaml --from-table ../all_esm1v_predictions_with_mean.csv
# --id ACCESSION overrides the accession taken from protein.uniprot in the config
```

The table is streamed in chunks, so a proteome-wide one (tens of millions of rows) costs a
scan rather than its full size in memory. If the config names no accession, one extra pass
over the accession and mutation columns identifies the protein from its sequence — the
mutation strings carry their wild-type residues — and the answer is written back to
`protein.uniprot`, so every later run is a single filtering pass.

### Bayesian pooling across a family

```bash
python -m mpdms bayes \
  --family PF07690=configs/AQR1_fitness.yaml,configs/QDR2_fitness.yaml,configs/TNA1_fitness.yaml,configs/TPO3.yaml,configs/FEN2.yaml \
  --family PF00083=configs/PHO84_fitness.yaml,configs/HXT1.yaml,configs/HXT2.yaml,configs/HXT3.yaml,configs/HXT7.yaml,configs/ITR1.yaml
```

Each family is aligned, and every alignment column gets a hierarchical normal model:

    y_j | theta_j ~ Normal(theta_j, s_j^2)      protein j's measured effect, with its error
    theta_j | mu, tau ~ Normal(mu, tau^2)       the family's spread around a shared value

`mu` is what the family shares at that column, `tau` is how much its members genuinely
differ, and Normal(mu, tau^2) is what the family says about a member nobody has measured.
The posterior is computed exactly on a grid over tau, so there is no sampler and nothing
to diagnose; the implementation reproduces the eight-schools posterior of Gelman BDA3
tables 5.2 and 5.3 to within rounding, which is what the tests assert.

Aggregate predictive power is then leave-one-protein-out: refit every column without one
member and score the prediction against it by rank correlation, by expected log predictive
density against predicting one number for the whole family, and by whether the 50% and 90%
credible intervals actually contain 50% and 90% of what happened. The last of these is the
one that catches a model that looks accurate and is overconfident.

### What does each feature predict on its own?

Before any model combines them:

```bash
python -m mpdms univariate configs/*.yaml
```

One figure per protein ranking every feature by its rank correlation with the measured
effect, and one figure putting all the proteins together: the pooled ranking beside a
feature x protein grid. A feature strong in one protein and absent in the others is a fact
about that protein, not about membrane proteins, and the grid is where that shows.

Intervals come from resampling **positions**, not variants. Nineteen substitutions at one
site are nineteen measurements of that site, so a variant-level interval is roughly three
times too narrow and every feature looks significant.

### All proteins in one figure

```bash
python -m mpdms panel configs/AQR1_fitness.yaml configs/HXT1.yaml ... --only a01,a05,a05b,a06,a15,a22a,a22b
```

One figure per analysis with one panel per protein, in `outputs/_panels/`. Each panel is
drawn by the analysis's own code, so a panel here and the single-protein figure cannot
drift apart. Axis limits, colour scales and the legend are shared across the panels of a
figure — eight panels that each auto-scaled to their own data are the fastest way to read
a difference that is not there. A protein that cannot supply an analysis (no ESM table, no
topology) gets a panel saying so and keeps its place in the grid.

| key | one panel per protein shows |
|---|---|
| `a01` | score distributions by variant class |
| `a05` | standardised physicochemical coefficients, TM vs rest |
| `a05b` | the TM − rest substitution matrix, on one colour scale |
| `a06` | charge introduced across the membrane, both orientations pooled |
| `a15` | abundance vs ESM-1v with the functional sites picked out |
| `a22a` | missense fitness by topology class |
| `a22b` | missense fitness per transmembrane helix |

`--ncols` overrides the layout and `--style default` adds the figure title back.

### A whole folder at once

A folder of DiMSum output, straight from the pipeline:

```
fitness_singles_aqr1_003.txt  aqr1.gff3  aqr1.cif
fitness_singles_hxt2.txt      hxt2.gff3  hxt2.cif   ...
```

```bash
python -m mpdms batch /path/to/dimsum_default --esm-table all_esm1v_predictions_with_mean.csv
python -m mpdms batch /path/to/dimsum_default --dry-run     # check the pairing first
```

Each fitness table is paired with the gff3 and structure whose names match it, allowing for a
run or replicate suffix on the fitness file: `fitness_singles_aqr1_003.txt` pairs with
`aqr1.gff3` and `aqr1.cif`, and the gene stays `AQR1` so the sites in `configs/_literature.yaml`
still find it. A suffix is only trimmed when the companions agree, so `hxt2_2` with its own
`hxt2_2.gff3` stays a separate protein. Proteins missing a gff3 are listed and skipped rather
than run on a hydropathy guess; `--require-structure` does the same for the structure. One
protein failing does not stop the rest.

### Where the -1 end of the scale comes from

By default `score_z = (x - median(syn)) / (median(syn) - median(nonsense))`, so a stop codon
sits at -1. `normalization.lower_anchor` moves that floor:

| value | -1 means |
|---|---|
| `nonsense` (default) | the median nonsense variant |
| `missense_min` | the single worst missense variant measured |
| `missense_p1` | the 1st percentile of missense, the same idea but not hostage to one well |

It is a linear rescale, so ranks, correlations, the ESM regressions and the LOESS-residual z
are all identical. What changes is every statement in absolute units: the -1 reference line,
the heatmap colour range, "fraction of variants as bad as a stop", and comparability with
another dataset. With `missense_min` nothing can fall below -1 by construction, so the
stop-like fraction stops being a meaningful quantity.

`filters.keep_classes: [missense]` drops the controls from the figures *after* they have set
the scale, so the scores are unchanged and only the rows shown differ.

`esm` writes `data/external/<GENE>/<ID>_esm1v.csv` and sets `evolution.esm_scores` in the
config. If you already have ESM-1v scores, point `evolution.esm_scores` at a CSV with
columns `pos, mut` and one of `esm1v` / `llr` / `score`.

A15 fits a LOESS of ESM-1v against DMS abundance. A variant's residual is how far its
ESM-1v score falls below the curve, scaled by the robust spread (MAD) of all residuals. A
site is called functional when its variants' residuals are significantly below zero
(one-sided Wilcoxon test, BH q < 0.05) and its median z is ≤ −1.0 (`Z_SITE` in `a15_esm_functional.py`). The
`abundance_tolerant` column marks the functional sites whose abundance itself is near
wild type.

A15 also writes two coloured structures per protein to `outputs/<GENE>/structures/`:

| file | colouring |
|---|---|
| `a15_zscore.pdb/.pml/.cxc(.png)` | site median z: red (−4, constrained beyond abundance) → white (0) → blue (+4); untested residues grey |
| `a15_functional.pdb/.pml/.cxc(.png)` | functional sites red, everything else white |

The B-factor column holds the value (z-score, or 1/0 for functional), so any viewer can
colour by it. Easiest is the self-contained PyMOL session: double-click `a15_zscore.pse`.
To use the script instead, keep the `.pml` next to its `.pdb` and in PyMOL type
`cd <that folder>` then `@a15_zscore.pml`. Don't use "Run Script" on the `.pdb`: PyMOL will
try to execute it as Python. ChimeraX: `open a15_zscore.cxc`. If PyMOL is installed in the pipeline environment (`pip install pymol-open-source-whl`),
the `.pse` sessions and PNGs are written automatically during the run.

## Assay validation and model benchmarks

```bash
python -m mpdms validate configs/QDR2.yaml configs/AQR1.yaml ...   # is this biogenesis? does it generalise?
python -m mpdms bench    configs/*.yaml --split protein            # how well do models predict it?
```

`validate` runs a pre-registered battery of twelve directional predictions (A16) per protein
and as a protein × test matrix, then tests leave-one-protein-out transfer using only features
that exist for any membrane protein. `bench` evaluates models under four splits, with
pluggable zero-shot score files and embedding heads. See `docs/validation_and_benchmarks.md`.

## Transmembrane regions from TMHMM

```bash
bash scripts/install_tmhmm.sh                      # once (patches tmhmm.py for modern NumPy/Cython)
python -m mpdms tmhmm configs/AQR1.yaml configs/QDR2.yaml
# or, with DeepTMHMM output from https://dtu.biolib.com/DeepTMHMM :
python -m mpdms tmhmm configs/AQR1.yaml --from-file AQR1_deeptmhmm.gff3
```

Writes the TM segments into the `topology:` block of each config, so every downstream
analysis uses them. TMHMM labels residues inside / membrane / outside, so helix
**orientation** comes from the prediction rather than from the positive-inside heuristic
used by `init`. The per-residue annotation is cached at `data/external/<GENE>/<ID>.tmhmm`.
`--dry-run` prints the prediction without editing anything.

## Same-family comparison

```bash
python -m mpdms family configs/AQR1.yaml configs/QDR2.yaml
# -> outputs/_family_AQR1_QDR2/
```

**Aligned positions** (`a17a_aligned_correlation`). Aligns the two sequences (Biopython
global BLOSUM62 for a pair; MAFFT if installed, required for three or more) and correlates
the mean missense effect at aligned positions, split into all aligned / both TM / both TM
and α-helical / both loop. Each is tested against a permutation *within* the same topology
class, which preserves the fact that TM cores are sensitive and loops are not, and destroys
only the residue-level correspondence — so a surviving correlation is position-specific
agreement rather than shared topology. Panel (c) does the same for each position's full
19-substitution profile.

**Helix profiles** (`a17b_helix_profiles`, `a17c_helix_small_multiples`). For TM segments
that are at least `--helical-frac` α-helical in the AlphaFold model (default 0.7; the
per-helix fractions are printed so you can tune it), the effect of proline, glycine and
lysine/arginine at each position from the helix N-terminus to its C-terminus.

Helices alternate orientation, so the N-terminal end is cytosolic in half of them and
lumenal in the other half: pooling strictly N→C cancels any membrane-sided effect. Both
views are produced — N→C as a single pool, and the same data split by orientation.

## Per-protein report

```bash
# 1. define TM regions (DeepTMHMM .gff3 recommended)
python -m mpdms tmhmm configs/AQR1.yaml --from-file AQR1_TMRs.gff3
python -m mpdms tmhmm configs/QDR2.yaml --from-file QDR2_TMRs.gff3
# 2. build the report
python -m mpdms report configs/AQR1.yaml configs/QDR2.yaml
```

Produces `outputs/<ID>/report/<ID>_report.pdf` (all figures in one file) and
`reports/<ID>_report.md`:

- fitness distributions by variant class, replicate QC and the read-count filter;
- the DMS heatmap with **three** SSDraw tracks — mean missense, proline, and the mean
  effect of lysine/arginine — plus a **solvent-accessibility strip** under them
  (one hue, dark = buried, from the AlphaFold model);
- standalone SSDraw figures for each track;
- **A18**: for every TM helix, K/R substitutions against proline substitutions, with
  Welch's and Student's t-tests, Hedges' g, a bootstrap CI on the difference, and BH-FDR
  across helices. Below zero on the summary panel means K/R is more damaging than proline
  in that helix.

TM regions come from whatever is in the config's `topology:` block, so the heatmap tracks,
the RSA strip and A18 all follow the TMHMM definition once step 1 has run. `--skip-run`
rebuilds the PDF from figures already on disk.

