# mpdms — membrane-protein DMS lab-meeting pipeline

This pipeline takes one DMS folder per yeast membrane protein (a `fitness_estimation.tsv`
plus the three QC PDFs) through these steps:

1. **QC.** Read-count filter, replicate correlations, and fitness distributions by variant
   class (missense blue, synonymous green, stop dark red).
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

Column names are detected automatically. Any of them can be overridden under `columns:` in
the config.

| what | detected names |
|---|---|
| variant | `Pos`/`WT_AA`/`Mut`, or a `variant`/`hgvs` column (`M1A`, `p.Met1Ala`, `p.Gly7=`), or DiMSum `aa_seq` (+ `nt_seq`, `WT`) |
| replicate fitness | `rescaled_fitness_rep1..N`, `fitness1_uncorr`, `fitness_rep1`, `rep1`, … |
| mean fitness / error | `rescaled_fitness`/`fitness`, `rescaled_sigma`/`sigma` |
| counts | `input1..N`, `output1..N` (DiMSum `input1_e1_s0_bNA_count` works too) |

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
