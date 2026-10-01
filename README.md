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

`esm` writes `data/external/<GENE>/<ID>_esm1v.csv` and sets `evolution.esm_scores` in the
config. If you already have ESM-1v scores, point `evolution.esm_scores` at a CSV with
columns `pos, mut` and one of `esm1v` / `llr` / `score`.

A15 fits a LOESS of ESM-1v against DMS abundance. A variant's residual is how far its
ESM-1v score falls below the curve, scaled by the robust spread (MAD) of all residuals. A
site is called functional when its variants' residuals are significantly below zero
(one-sided Wilcoxon test, BH q < 0.05) and its median z is ≤ −1. The
`abundance_tolerant` column marks the functional sites whose abundance itself is near
wild type.

