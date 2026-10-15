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
outputs/<GENE>/tables/    tidy CSVs; structure_{mean_missense,proline,lys_arg}.pdb
                          (B-factor = the value, -99 = not measured) + .pml scripts that
                          colour with the heatmap palette and `save` a .pse session;
                          a10 PyMOL selection
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

### The smallest pooled model that still predicts

```bash
python -m mpdms ladder configs/*.yaml --split family
```

Features are added one at a time; at each step the one chosen is whichever most improves
prediction on a protein family the model has not seen. The curve flattens where extra
features stop buying anything, and that is the model worth writing down. The run prints the
standardised equation, and `--no-esm` asks what the biophysics carries without ESM-1v.

Ranking features by their separate correlations does not answer this question. The features
are badly collinear - "in a TM helix", "WT is hydrophobic" and "local hydrophobicity" are
largely one fact told three ways - so a strong marginal correlation says nothing about what
a feature *adds*. And when the per-protein correlations disagree (high I2 in
`mpdms univariate`), one pooled coefficient describes no protein in particular, so every
coefficient is drawn with its per-protein refits beside it and a sign-agreement warning
when they do not point the same way.

### The functional-site threshold

A site is called functional when its median residual from the ESM-1v-versus-abundance fit
sits at least `Z_SITE` MAD-scaled units below it. The default is 1.0; raise it to call
fewer, more confident sites:

```bash
MPDMS_Z_SITE=1.5 python -m mpdms run configs/HXT1.yaml --only a15,a19,a21 --style paper
```

It is read once at import, so A15, A19, A21 and the cross-protein panels all see the same
number and no figure can draw a line its own table did not use. Either sign works: 1.5 and
-1.5 both mean -1.5.

### What does the family predict for one unmeasured protein?

```bash
python -m mpdms predict \
  --family hexose=configs/HXT1.yaml,configs/HXT2.yaml,configs/HXT3.yaml,configs/HXT7.yaml \
  --human SLC2A1=P11166 --table variant_summary.txt.gz
```

One row per residue, and a figure along the sequence. The call is not `mu` against a
threshold: a column where the family agrees on a mild effect and one where it disagrees
wildly can share a mean and say very different things about an unmeasured member. Each
residue is scored by the posterior predictive for a new family member,

    effect ~ Normal(mu, tau^2 + var(mu))

so the number reported is P(effect < threshold) - the probability that a protein the family
has never seen loses abundance there. A column the family is unsure about lands near one
half, which is the honest answer rather than a confident one, and those positions are
called `uncertain` rather than forced either way.

With `--table`, the predicted-loss positions are cross-tabulated against ClinVar: how many
pathogenic and benign variants sit at them, with an odds ratio and a Fisher test.

### How related are the proteins?

```bash
python -m mpdms identity configs/*.yaml --human SLC2A1=P11166 --human SLC2A2=P11168
```

A pairwise identity heatmap, clustered so relatives sit together, with a second panel
placing each unmeasured query against the measured set.

This is the context for every pooled result. A hierarchical family model borrows strength
between its members, which helps while they share constraint and hurts once they do not:
a distant relative raises the between-protein spread at every column, the pooled mean is
shrunk toward the family average, and the contrast a predictor needs is flattened. The
`clinvar` and `bayes` runs print median τ against the spread of μ for exactly this reason
- when that ratio approaches 1, the family is disagreeing about as much as its columns
differ, and a tighter family will separate better.

### Does the yeast signal predict human clinical variants?

```bash
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz

python -m mpdms clinvar \
  --family hexose=configs/HXT1.yaml,configs/HXT2.yaml,configs/HXT3.yaml,configs/HXT7.yaml \
  --human SLC2A1=P11166 --human SLC2A2=P11168 --human SLC2A10=O95528 --human SLC2A9=Q9NRM0 \
  --table variant_summary.txt.gz
```

The human protein is aligned into the same frame as the measured family, each ClinVar
missense variant is mapped to its alignment column, and pathogenic and benign variants are
compared by the constraint the yeast data assigns to that column. Reported as an AUROC with
a bootstrap interval, overall and per gene.

Every run also writes `<family>_variant_ranking.csv` and the C02 figure: one row and one
point per variant, ordered by constraint, with `pairs_beaten` — for a pathogenic variant,
the fraction of benign variants it is correctly ranked above; for a benign variant, the
fraction of pathogenic it is correctly ranked below. Averaging that column over either
class returns the AUROC exactly, so a variant with a low value is literally one of the
ones costing the score, and the terminal names them worst-first. The threshold shown is
Youden's J on the same variants, so the `called` column describes the figure rather than
predicting anything.

### Against AlphaMissense and ESM-1v

Getting the two inputs. **AlphaMissense** is a download — `AlphaMissense_aa_substitutions.tsv.gz`
(1.2 GB, CC BY 4.0) from the DeepMind bucket, keyed by UniProt accession:

```bash
curl -L -C - -O https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz
# or keep ~10 KB instead of 1.2 GB:
curl -sL https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz \
  | zcat | awk -F'\t' 'NR==1 || /^#/ || $1=="O95528"' > SLC2A10_alphamissense.tsv
```

**ESM-1v is not a download** — there are no precomputed scores for an arbitrary protein,
so the model has to be run. A query protein has no config, so `esm` takes an accession
directly:

```bash
export TORCH_HOME=/lustre/.../torch        # weights are 2.6 GB per model, five models
python -m mpdms esm --human SLC2A10=O95528 --device cuda
```

It prints the `--esm GENE=FILE` string to paste into the clinvar command. `--models 1`
scores with one model instead of the ensemble of five, which is five times faster and
usually within a few hundredths of an AUROC.


```bash
python -m mpdms clinvar \
  --family hexose=configs/HXT1.yaml,configs/HXT2.yaml,configs/HXT3.yaml,configs/HXT7.yaml \
  --human SLC2A10=O95528 --table variant_summary.txt.gz \
  --alphamissense AlphaMissense_aa_substitutions.tsv.gz \
  --esm SLC2A10=SLC2A10_esm1v.csv \
  --mochi mochi_project/task_1/weights/weights_Folding.txt \
  --mochi-aln hxt1_hxt3_slc2a10.aln.fasta
```

**`--family` is optional.** With none, there is no alignment, no pooling and no μ:
the run compares only the predictors that score a human variant directly, which is the
right shape when the question is "how do ESM-1v, AlphaMissense and my energy model
rank against each other" rather than "does yeast abundance predict pathogenicity".
It also lifts the position-score ceiling this section otherwise warns about, because
every predictor left is per variant. Output is `c03_predictors.csv`,
`c03_predictor_auroc.csv` and the C03 figure.

```bash
python -m mpdms clinvar \
  --human SLC2A10=O95528 --table variant_summary.txt.gz \
  --alphamissense SLC2A10_alphamissense.tsv \
  --esm SLC2A10=SLC2A10_esm1v.csv \
  --mochi weights_Folding.txt --mochi-aln hxt1_hxt3_slc2a10.aln.fasta
```

**A masked-alignment energy model** joins the comparison with `--mochi`. A MoCHI
weights table holds one energy per (alignment column, mutant residue): the model is
fitted on a masked alignment of the yeast members, so a row is not a substitution of a
named wild type — the id reads `-3A`, a gap where the wild type would be — and the
merge is on (column, mutant) with no wild-type identity to check. That makes it the
only yeast-derived predictor here that is **per variant** rather than per position, so
unlike the family model it can separate two substitutions at one residue, and it is not
bounded by the position-score ceiling the rest of this section is subject to.

`Pos` in the weights is a column of the alignment the model was fitted on, which is
**not** the alignment this command builds — so `--mochi-aln` is required and must
contain the human query. The guard that matters: if the weights run past the end of
that alignment the run stops rather than mapping, because a scrambled column frame
produces a plausible AUROC from nonsense. Weights fitted in fewer than
`--mochi-min-folds` folds (default 8 of 10) are dropped and the count is printed; the
real tables carry rows whose 95% interval is several kcal/mol wide. Direction is
higher = destabilising = pathogenic, the same convention as AlphaMissense.

C03 puts the family model beside the predictors it has to beat: one violin per
predictor in its own units (benign grey, pathogenic blue), then one ROC with all of
them. Column conservation joins the ROC as a dashed reference.

The AlphaMissense release is 216 million rows, so it is streamed and filtered by
accession, and its columns are found by name rather than position. The ESM-1v table is
whatever `python -m mpdms esm` writes, or any csv with pos, mut and esm1v.

Two things the design turns on:

- **Direction.** Every predictor is flipped into one convention before scoring — higher
  AlphaMissense means pathogenic, lower ESM-1v and lower μ mean pathogenic. A sign error
  here gives 1 − AUROC and looks entirely plausible, so a test plants a separation in
  all four and asserts each comes back above 0.9, then flips one and asserts it inverts.
- **Which variants.** `scope` is reported twice per predictor. `own` is that predictor's
  AUROC on every variant it scores; `shared` is the AUROC on the variants **all** of
  them score. Only `shared` is a comparison — AlphaMissense scores every missense while
  the family model only reaches mapped alignment columns, so their `own` numbers answer
  different questions. The ROC panel draws the shared set and says how many variants
  that is.

### Variants of uncertain significance

```bash
python -m mpdms clinvar --family hexose=... --human SLC2A10=O95528 \
  --table variant_summary.txt.gz --include-vus
```

`--include-vus` keeps the variants ClinVar itself calls uncertain, scores them with the
same μ, and adds a panel placing them beside the benign and pathogenic sets. They never
enter the AUROC, the position-level AUROC, the conservation control or the per-variant
ranking: a VUS is not a negative, and counting it as one is the commonest way to make a
predictor look good on ClinVar. One split in `run_one` separates them before any number
is computed, and a test asserts every statistic is bit-identical with and without them.

Conflicting interpretations stay excluded even with the flag. A variant two submitters
disagree about is not the same as one nobody has called, and the model may well side
with one of them.

`<family>_uncertain.csv` gives each VUS its `beats_benign` and `beats_pathogenic` — the
fraction of each reference set it is more constrained than — and `sits_with`, which is
one of *with the benign set*, *between the two*, *with the pathogenic set*, or *beyond
every pathogenic*. The gap between the two reference ranges gets its own answer rather
than being rounded toward pathogenic. These are hypotheses about positions, not calls on
variants, and the terminal says so.

### Calling out particular residues on A15

```bash
MPDMS_HIGHLIGHT=335,336,370,473 MPDMS_BOX=1 MPDMS_SQUARE=1 \
  python -m mpdms run configs/HXT1.yaml --only a15 --style paper
```

or, to keep it with the protein, `plotting.highlight_positions: [335, 336, 370, 473]`
in the config; the environment variable overrides it for one run.

Highlighted residues are ringed and named in black on all three A15 panels — the
variants in (a), the site in (b), a line and a label along the sequence in (c) — and
bold in the A15b ranking where the statistics also called them functional. Black,
because red is reserved for what the analysis found on its own: the point of marking a
residue you already care about is to see where it falls among those, **including when
it is not one of them**. The terminal says how many of the highlighted positions were
called functional, and names any that have no site-level estimate at all because they
carry too few scored variants.

### Boxed, square figures for slides

```bash
MPDMS_BOX=1 MPDMS_SQUARE=1 python -m mpdms clinvar ...
```

`MPDMS_BOX=1` closes the frame — all four spines, drawn heavier (`MPDMS_BOX_WIDTH`
changes the weight, default 1.4). `MPDMS_SQUARE=1` makes each panel square, skipping panels with a fixed data aspect, a
frame switched off, or an explicit exemption — a full-width track along a sequence is
exempt, since squaring 600 residues into a box as tall as it is wide is a worse picture
rather than a tidier one. Both are
presentation choices rather than analysis ones, so they are applied at save time, at the
one place every figure passes through: any command picks them up on a rerun and no
analysis module knows about them. Square is skipped for a panel with a fixed data aspect
— a heatmap, a structure projection in angstroms — since squaring the box there would
stretch the data instead.

One gene is a perfectly good run, and the figure adapts: with a single `--human` the
per-gene panel (c) is dropped, since it would be panel (b)'s AUROC redrawn as one bar,
and the gene's name moves into the titles of (a) and (b).

```bash
python -m mpdms clinvar \
  --family hexose=configs/HXT1.yaml,configs/HXT2.yaml,configs/HXT3.yaml,configs/HXT7.yaml \
  --human SLC2A10=O95528 --table variant_summary.txt.gz
```

Only unambiguous calls are used: anything uncertain or conflicting is dropped rather than
counted as benign, which is the usual way a predictor is flattered on ClinVar. Variants
whose wild-type residue disagrees with the UniProt sequence (a different transcript) are
dropped and counted. The run also prints variants per position, because this is a position
score: two substitutions at one residue get the same prediction, and that ratio is the
ceiling on what the comparison can resolve.

### Bayesian pooling across a family

```bash
python -m mpdms bayes \
  --family PF07690=configs/AQR1_fitness.yaml,configs/QDR2_fitness.yaml,configs/TNA1_fitness.yaml,configs/TPO3.yaml,configs/FEN2.yaml \
  --family PF00083=configs/PHO84_fitness.yaml,configs/HXT1.yaml,configs/HXT2.yaml,configs/HXT3.yaml,configs/HXT7.yaml,configs/ITR1.yaml
```

Each family is aligned, and every alignment column gets a hierarchical normal model:

    y_j | theta_j ~ Normal(theta_j, s_j^2)      protein j's measured effect, with its error
    theta_j | mu, tau ~ Normal(mu, tau^2)       the family's spread around a shared value

For a family spanning more than one Pfam, two options matter. `--mafft-args
"--maxiterate 1000 --localpair"` uses L-INS-i rather than letting `--auto` fall back to a
fast progressive method, and `--min-topology-agreement 0.8` drops columns where the
proteins disagree about whether the aligned residues are transmembrane. Topology comes
from DeepTMHMM and is independent of the alignment, so it tests it: a column where one
protein's membrane residue is aligned against another's loop residue is an alignment
error, and pooling across it averages two different things.

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

### What does a residue's environment say about what it tolerates?

```bash
python -m mpdms environ \
  configs/AQR1_fitness.yaml configs/HXT2.yaml \
  configs/HXT1.yaml configs/HXT3.yaml
```

Ten analyses, each one panel per protein, **laid out in the order the configs are
given**: with four proteins and the default two columns that is top-left, top-right,
bottom-left, bottom-right. Nothing is pooled across proteins — these assays differ in
dynamic range, and a comparison between proteins belongs in a model that says so
(`bayes`, `famstruct`) rather than in a box plot. Colour is the project's abundance
scale throughout: dark red at the stop-codon level, white at no effect, blue above.

A membrane protein has several kinds of buried and several kinds of exposed, and RSA
cannot tell them apart. Each position is classified with the membrane frame instead:

| class | what it is |
| --- | --- |
| core | in a TM helix, RSA < 0.15: packed against other helices |
| cavity | in a TM helix, exposed, inside the median TM radius: facing the central pathway, which in a transporter is water |
| lipid | in a TM helix, exposed, outside it: facing acyl chains |
| cyto / lumen | outside the membrane, by side from the topology |

RSA merges cavity and lipid, and they are the two with opposite chemistry — a polar
residue is at home in the cavity and out of place against lipid. Separating them is what
the `radial` coordinate is for. The cut is each protein's own median TM radius, not a
fixed number of angstroms, and it is printed and stored. Positions with pLDDT < 70 are
left unclassified rather than placed on geometry the model is unsure of.

| figure | question |
| --- | --- |
| E01 | mean effect per position by class; each violin filled by its own median |
| E02a/b | burial against depth together: does burial cost more in the bilayer centre than at its edge? |
| E03 | Δ hydrophobicity against effect, in three bands of burial, with the quadratic term |
| E04 | polar→hydrophobic and back, by class, each against that class's own baseline |
| E05 | effect against contact number, with RSA partialled out |
| E06a/b/c | where the ten largest ESM-1v residuals sit, and whether they cluster |
| E07 | is a helix's intolerance on one face? the 3.6-residue moment, with a wheel |
| E08 | every residue placed in the membrane, coloured by what it costs |
| E09 | the 20 × 20 substitution matrix, ordered by hydrophobicity |
| E10 | what a charge costs, against how deep it is buried |

E04 states the hypothesis directly. Each cell carries two numbers: the mean effect, and
in brackets the same minus the average substitution at that kind of position. The second
is the one about chemistry — a class intolerant of everything is intolerant of each
direction in turn, which says nothing. E05 reports the partial Spearman given RSA as well
as the raw one, because contacts and RSA are two views of the same thing.

E07 is the classic membrane-protein question made quantitative. An α-helix turns 100° per
residue, so a helix with one face against other helices and one against lipid has its
intolerance repeating every 3.6 residues. The moment is 1 if every intolerant residue is
on one side and 0 if they are spread evenly; effects are centred first, so a uniformly
dead helix — which has no face at all — scores 0 rather than 1. The null shuffles effects
*within the same helix*, keeping its mean, spread and length, so the p-value is about the
arrangement and nothing else.

E08 is the most direct picture the data allows: horizontal is distance from the bundle
axis, vertical is depth, colour is the mean effect, the bilayer is shaded, and a strip
down the side gives the depth profile. E10 turns the textbook claim into a measurement —
a charge buried mid-bilayer should cost more than the same charge near the headgroups,
and a hydrophobic substitution should be flat with depth.

E06 scores clustering against **two** nulls: ten positions drawn from anywhere measured,
and ten drawn with the same TM/loop composition as the observed set. The second is the
one to believe — ten TM residues are close together because TM residues are close
together, and on the test fixture that alone takes p from 0.016 to 0.089. Each protein
also gets a PyMOL `.pml` and a ChimeraX `.cxc` script under `structures/<protein>/`, with
the ten residues as spheres and every pair under 10 Å as a real distance object.

### Ten structural questions, asked within one protein

```bash
python -m mpdms arch configs/AQR1_fitness.yaml configs/HXT1.yaml configs/HXT3.yaml
```

| | question | what the answer looks like |
| --- | --- | --- |
| S01 | which helix–helix interfaces are load-bearing? | a 12 × 12 matrix; the diagonal is each helix alone |
| S02 | is there room to make a small interface residue bigger? | two slopes of effect on Δvolume, interface vs not |
| S03 | proline, both directions | introducing one by position in the helix; removing a native kink |
| S04 | holding your own helix, or holding the fold together? | two coefficients, each at the other's fixed value |
| S05 | is too big worse than too small? | two slopes either side of zero, and their asymmetry |
| S06 | do salt-bridge partners matter, and is reversal worse than a swap? | paired vs unpaired charged residues; K→R vs K→D |
| S07 | does the fold treat its two bundles the same way? | effect at structurally equivalent positions, with the CA RMSD |
| S08 | how far does constraint travel through space? | a correlogram in ångströms, against a topology-matched null |
| S09 | which burial measure actually predicts tolerance? | |ρ| alone against |β| with the other four |
| S10 | how fast does sensitivity decay away from the pathway? | mean effect by shell outward from the cavity lining |

**S07 searches the pairing rather than assuming it.** The textbook MFS inverted repeat
pairs TM1 with TM7, TM2 with TM8 and so on; here every one-to-one pairing of the first
six helices against the second six is scored by a Kabsch fit on helix centroids, the
best is kept, and the canonical pairing's own RMSD is reported next to it. So the
figure can say whether the textbook answer is what the structure gives. Residues are
then matched inside each paired helix by rank along the helix, and the CA RMSD of that
correspondence is the number to judge it by.

**S08's null shuffles effects within transmembrane and non-transmembrane separately.**
TM residues are both closer together and less tolerant, so an unstratified shuffle
would manufacture a patch out of the topology alone.

**Nothing here is pooled across proteins**, and the run prints a warning for any
dataset normalised against a wild-type anchor rather than synonymous variants — HXT1 is
one, so its absolute effect sizes are not on the same scale as the others even though
ranks within its panel are fine.

### Does insertion energetics set a floor on what a helix tolerates?

```bash
python -m mpdms insertion \
  configs/AQR1_fitness.yaml configs/HXT2.yaml configs/HXT1.yaml configs/HXT3.yaml
```

A TM helix has to be recognised by the translocon and released into the bilayer. A
strongly hydrophobic helix inserts with margin to spare and a slightly more polar
substitution changes nothing it depends on; a marginal helix has no margin, and the same
substitution pushes it below threshold.

So the claim is an **interaction**, not a main effect: the same chemical change should
cost more in a marginal helix than in a comfortable one. The main effect of
hydrophobicity would be there whether or not insertion energetics had anything to do with
it, because a substitution that raises ΔΔG is usually also one that buries badly — which
is why RSA is a covariate in every model here and the interaction is what gets reported.

There are two scales, chosen with `--scale`. The default is **dsTβL** (Elazar et al.
2016, eLife 5:e12125): a fourth-order polynomial in
depth per residue, kcal/mol relative to alanine, taken from the authors' own
`polyval_Kfix_14Jan2016.txt`. A segment is laid across the membrane at `linspace(−15,
+15)` and its ΔG is the sum of the per-residue values at those depths, which is that
code's convention.

`--scale hessa` is the position-specific ΔG_app matrix of **Hessa et al. 2007**
(Nature 450:1026, supplementary table S2), measured on the translocon itself with the
Lep glycosylation assay rather than by mutational scanning — 20 residues × 19 positions
in a 19-residue H-segment, interpolated to the depth each residue sits at. Three
differences decide what it can be used for:

| | dsTβL | Hessa table S2 |
| --- | --- | --- |
| reference | relative to alanine (Ala ≡ 0) | absolute kcal/mol (Ala = +0.13 at the centre) |
| symmetry | asymmetric — the positive-inside rule is in the scale | symmetric about the centre, so orientation does nothing |
| completeness | the whole scale | the position term only; no length or flanking correction |

So **ΔΔG is comparable between the two and an absolute segment ΔG is not**: every
substitution effect is a difference at one position, where the reference state and the
missing constant terms cancel, which covers I02, I03 and I04. I01 and I05 use the
wild-type helix ΔG itself, so under Hessa they report the position term alone — and the
two scales need not even rank the helices the same way. The table is indexed by position
in a 19-residue segment rather than by ångström, so `--depth structure` is refused for it
instead of extrapolated, and the transcription is checked for symmetry at import because
a mistyped row is the realistic failure mode.

**Is the insertion scale earning its place?** I07 is the control: the same pooled
curve as I04 with the Kyte–Doolittle difference between mutant and wild type on x
instead of ΔΔG — one number per substitution, no scale, no depth, no orientation. If
I07 reproduces I04, that figure did not need an insertion scale and is about
hydrophobicity in general rather than about the translocon. `i07_hydrophobicity_slopes.csv`
makes the comparison directly: each predictor's slope and R² alone, then both in one
model, where `slope_ddg_given_kd` and `r2_added_by_ddg` are what the depth-dependent
scale adds once plain hydrophobicity has explained whatever it can. The two predictors
correlate around −0.75, so this is the only honest way to separate them.

I07C exists to be read for its **ordering and colour, not its heights**. Each helix
carries hundreds of variants, so a slope of almost any size clears q = 0.05 and the
bars run to tens of orders of magnitude; what separates the helices is the slope and
its interval, which `i07b_hydrophobicity_by_helix.csv` holds. The correction is within
each protein, because twelve helices are twelve chances and two proteins are not one
family of tests.

I09 is the published scales' own presentation turned on the measurements: the
twenty-panel grid where a charge is a bell peaked at the bilayer centre and a
hydrophobe is flat. It writes **twelve pages per protein** by default, from three independent choices:

| choice | pages |
| --- | --- |
| which prediction | `i09_profiles` = the scale's **ΔG**, the published quantity, what it costs to *have* that residue there; `i09_effect` = the **ΔΔG** of changing the wild types actually present into it, which is what blue measures |
| which sign | plain, and `_flip`, which negates the energy axis |
| which wild types | all of them; `_avli`, only substitutions **from** A, V, L or I; `_ala`, only **from** alanine. `--profile-wt` selects them |

The sign matters because energy and effect run opposite ways — a costly substitution
has a large *positive* ΔΔG and a large *negative* effect — so the two curves mirror
each other even when they agree perfectly. On the `_flip` pages agreement looks like
agreement.

The same `ala` restriction is available for the pooled curve, as `--wt-from ala`:
under dsTβL that makes I04's x axis the scale's own ΔG for the mutant residue, and it
removes the part of the ΔΔG spread that is really about which wild type happened to be
at each position rather than about the substitution.

**Alanine is the set that matters for dsTβL**, and not merely as one more filter.
That scale is quoted *relative to alanine*, so ΔG(A) is exactly zero at every depth
and an alanine wild type makes ΔΔG = ΔG(X) − ΔG(A) = ΔG(X). The measured A→X profile
is therefore the direct empirical counterpart of the published curve, and the ΔG and
ΔΔG pages of an `_ala` run under dsTβL are the same figure. A test asserts the
identity across all twenty residues at five depths, and asserts it does *not* hold
under Hessa, which is absolute.

The wild-type restriction matters more generally because pooling every wild type
mixes L→K with D→K, and those are not the same experiment: the second starts from a
residue the bilayer already disliked. On the `_avli` pages every panel is a change from a comfortable
hydrophobic baseline, so the profile is about the residue arriving rather than the one
leaving. `i09_profiles.csv` carries a row per (wild-type set, residue) with a rank
correlation against each prediction. x is the scale's own coordinate — position in a nineteen-residue
segment, so it means the same thing here as in table S2 — and positions are pooled
into bins `--profile-bin` wide (default 3, giving seven points across the segment)
because a (residue, depth) cell holds only about as many measurements as the protein
has helices. The run prints the median behind each point and suggests
`--profile-bin 1` — the published nineteen-column resolution — when the cells are full
enough to carry it, which twelve helices generally are. The red curves are drawn on their own dense grid rather than at the data's bin
centres, so the scale stays a smooth function while the measurement stays as bumpy as
it is. Each panel carries two dotted zero lines, grey for the measured axis and red
for the prediction axis: the two origins are in different places, and one line would
be read as belonging to both. They share **one** axis across all twenty panels, as the published figure does: per-panel autoscaling makes alanine,
which is identically zero under dsTβL since it is the reference, look as structured
as arginine. `i09_profiles.csv` carries each residue's rank correlation with the
scale's profile, its effect at the centre and its effect at the edges.

**I09B measures symmetry without needing the two curves to share units.** Any
profile is exactly its even plus its odd part, `even(i) = [y(i)+y(−i)]/2` and
`odd(i) = [y(i)−y(−i)]/2`, and the index is `Σodd² / (Σodd² + Σ(even−mean)²)` — 0
symmetric, 1 purely antisymmetric. It is a ratio of two sums of squares in the same
units, so multiplying the profile by any constant cancels and adding one is removed by
centring: a measured profile in fitness units and a predicted one in kcal/mol are
directly comparable, and so are two residues.

**A scale's per-residue value is a ΔG, not a ΔΔG**, and the two references differ
because of it. Table S2 gives the cost of *having* a residue at a depth; your
measurement is the cost of *changing* the wild type into it. Hessa's ΔG profile is
symmetric by construction (asserted at import) so its index is exactly zero — but the
matched ΔΔG, that residue against the wild types actually sitting at each depth, need
not be symmetric under either scale, because the two halves of the membrane hold
different wild-type residues. On the SYN1 fixture Hessa's ΔΔG reference is 0.05, not
0. So a tilt in your data is beyond a scale's reach only when it exceeds that scale's
**ΔΔG** reference, not merely when it exceeds zero. All four numbers are in the CSV
(`asymmetry_{dstbl,hessa}_{dg,ddg}`). And the null is exchangeability: if a profile really
is symmetric then a variant at +i and one at −i are interchangeable, so the depth sign
is permuted per variant and the index recomputed, which assumes nothing about the
distribution. The interval comes from resampling variants, and `q` is BH over the
twenty residues. A signed statistic travels alongside, since the index throws away
direction: `lumenal_minus_cytosolic`, positive when the substitution costs more
towards the lumen.

**Mind the sign when reading I07B against I03.** Both report ρ(helix slope, helix
ΔG) and the claim has the *opposite* sign in each, because the x axes run opposite
ways. A ΔΔG slope is negative — a more destabilising substitution lowers the effect —
so "the marginal helices are the steepest" reads as **ρ < 0** in I03. A Kyte–Doolittle
slope is positive — more hydrophobic is less damaging — and ΔG is *least* negative for
the helix with least margin, so the same claim reads as **ρ > 0** in I07B. A test
plants graded slopes across five helices and asserts the positive sign, because this
is precisely the error that would otherwise be invisible.

**Which membrane the scale came from.** dsTβL was measured in a bacterial
inner-membrane reporter and Hessa on Sec61 in dog pancreas rough microsomes, an ER
membrane; neither is a yeast plasma membrane. `--only i06` answers the question that
follows from that directly, by computing both scales on the same substitutions at the
same depths and asking how much the choice changes anything. On the SYN1 fixture the
two agree well on ordering (r = +0.80 over 1,430 variants, ρ = +0.80) and not at all on
magnitude — the regression slope of Hessa on dsTβL is +0.39, because dsTβL has 1.4 to
4.8 times the dynamic range in every residue class. So a correlation or a ranking is
robust to the choice of scale and an effect size in kcal/mol is not. The helix ordering
is the fragile one: with only four helices on the fixture the two scales rank them at
ρ = +0.40, which is why I01 and I05 should be read with the scale named.

Two details that matter and are easy to get wrong:

- **Negative z is cytoplasmic, and the profiles are not symmetric.** Lysine costs +0.24
  kcal/mol at z = −15 and +2.41 at z = +12; arginine is outright favourable on the inner
  side. That is the positive-inside rule sitting inside the scale as measured. Laying
  every helix N-to-C from −15 to +15 would score every second helix of a transporter
  backwards, so helix `orientation` from the topology is respected.
- **Standard errors are clustered by helix.** ΔG takes one value per helix, so the
  interaction has as many independent units as there are helices — twelve, not the
  thousands of variants inside them.

| figure | question |
| --- | --- |
| I01 | does a helix's ΔG predict its mean tolerance? (the weak version) |
| I02 | the interaction, two ways: slope-of-effect-on-ΔΔG per helix against that helix's ΔG, and a per-protein model with the interaction term |
| I03 | effect against ΔΔG for the **hardest and easiest helix to insert**, with the mean (± SEM) and median across all helices in bold (`--all-helices` draws all twelve, coloured by ΔG) |
| I04 | the same thing pooled: every variant in any TM helix, binned by ΔΔG and averaged — one curve per protein, no grouping. `--wt-from {all,avli,ala}` restricts it to substitutions **from** those wild types, which writes `i04_pooled_curve_<set>` beside the unrestricted one |
| I05 | the TM helices **ranked** by the insertion energy of their wild-type sequence, within each protein |
| I06 | the two scales against each other: the same substitutions at the same depths, with the helix ordering and each scale's correlation with the measured effect in `i06_scale_agreement.csv` |
| I06B | the twenty residues under both scales, at the bilayer centre and at the interface — where the two disagree, and which residues do it |
| I07 | **the control for I04**: the same pooled curve with plain Kyte–Doolittle change on x instead of ΔΔG — one number per substitution, identical for every residue at every depth |
| I07B | the same control helix by helix — **every** TM helix as its own curve, coloured by insertion ΔG with the hardest and easiest drawn over the top, plus the correlation between a helix's slope and its ΔG. The analogue of I03 with no scale in it |
| I07C | one bar per helix: −log₁₀ q that its hydrophobicity slope is not zero, BH-corrected within each protein and coloured by insertion ΔG |
| I09B | how tilted each residue's profile is about the bilayer centre, on a scale-free index, against a depth-sign permutation null — with dsTβL's value for the same residue as a reference and Hessa's zero as the floor |
| I08 | the two predictors against each other, one point per mutation: ΔΔG on y, Kyte–Doolittle change on x, coloured by depth. The vertical spread at one x is what the depth-dependent scale adds |
| I09 | **Hessa's figure, measured**: twenty panels, one per mutant residue, with position across the membrane on x — the mean effect in blue on the left axis, the scale's ΔΔG profile for that residue in red on one right axis shared by all twenty |
| I08B | the part of ΔΔG that hydrophobicity cannot explain, against the measured effect |

I05 is the premise of all the others and worth reading first: if a protein's helices
all sit within a kcal/mol of each other there is no contrast to test, and a bar past
zero marks a helix that would not insert on its own. Bars are ranked within the protein
and coloured by the helix's mean mutational effect, so length and colour are independent
readings. I03 then shows only the two ends of that ranking by default — the hardest and
the easiest helix to insert — because twelve lines show that helices differ while two
show by how much.

I04 is the plain version and the one to put on a slide: every missense variant that
sits in a TM helix, binned by ΔΔG and averaged, one curve per protein. Because it
averages over variants, a helix that is better covered in the library counts for more —
the right estimator for "what does a destabilising substitution cost in this membrane
domain", and the wrong one for "do helices differ", which is I03's job. Its slope is
still reported with the error clustered by helix.

I03 draws two bold summaries, and they answer different questions. Both weight each
helix once, so neither is dominated by a helix carrying more variants: the **median** is
the typical helix, the **mean** follows a helix that plunges. Where they separate, a few
helices are carrying the trend. `i03_median_curve.csv` also records
`pooled_over_variants`, the variant-weighted average — which is what the older
marginal/comfortable figure drew, and the reason that figure and this one do not look
alike.

I02 still uses a within-protein median split to label helices "marginal" in its model,
but I03 does not split at all: it draws every helix and lets ΔG be the colour. The split
was throwing away the quantity being tested — ΔG is continuous, every helix of these
transporters inserts comfortably in absolute terms (roughly −9 to −15 kcal/mol), and
cutting them at their own median produced two groups whose difference was a difference
in rank rather than in energetics. Read I03 first.

`--depth structure` evaluates each ΔΔG at the residue's real depth from the AlphaFold
membrane frame instead of its idealised position in the helix.

Two caveats the module states in its own docstring: the assay measured abundance, so a
variant that fails to insert and one that inserts and then misfolds look identical here;
and dsTβL was measured in a single-span reporter, while these are twelve-span
transporters whose helices insert with help from their neighbours.

### Burial in the centre of the protein, and hydrophobicity

```bash
python -m mpdms burial configs/AQR1_fitness.yaml configs/HXT1.yaml configs/HXT3.yaml
```

A membrane protein has three chemically different kinds of place to put a residue, and
the whole module turns on telling them apart: **core** (inside the bundle, packed
against other helices), **lipid** (inside the membrane, facing acyl chains) and
**aqueous** (outside the membrane, in water). The lumenal and cytosolic loops are
pooled into that one aqueous class on purpose — they differ in which side they are on,
which matters for insertion and for the positive-inside rule but not for a
hydrophobicity argument, since both are solvated. **Cavity-lining** is kept as a fourth
class and plotted alongside: in a transporter the central pathway is full of water, and
merging it with lipid would hide the most interesting positions in the protein.

| | question | what the answer looks like |
| --- | --- | --- |
| B01 | does hydrophobicity tolerance differ between lipid, core and aqueous? | two slopes per class, hinged at zero: a one-sided V against a symmetric tent. B01a draws lipid against aqueous, B01b all four classes |
| B11 | **the plain version of B01** | one ordinary straight line per class and one test: the difference between the two slopes, as the interaction in `effect ~ ΔKD * is_lipid`. One number, one standard error, one p-value |
| B02 | is that a boundary, or a gradient? | the same slope swept along the radial coordinate in sliding windows |
| B03 | is the intolerant face of a helix its hydrophobic face? | one arrow per helix on a wheel, and the resultant against a within-helix null |
| B04 | what does a charge cost, and does depth matter? | effect against \|z\| for buried and exposed separately; the interaction is the test |
| B05 | buried polar: packing defect, or substrate site? | six violins, core/cavity/lipid × polar/hydrophobic wild type |
| B06 | volume or hydrophobicity — which does each environment care about? | four standardised coefficients per class, signed and absolute |
| B07 | small residues at close helix–helix approaches | three violins, and the slope of effect on Δvolume where there is no room |
| B08 | the aromatic belt | losing the ring separated from losing hydrophobicity, by depth |
| B09 | is the cost a mismatch with the surroundings rather than an absolute? | mean effect over mutant hydrophobicity × local hydrophobicity |
| B10 | are ESM-1v's own errors structured by burial? | the residual against radius, and a model with burial, depth and buried-polar |

**B01 fits two slopes, not one.** A single straight line through Δhydrophobicity assumes
the cost of a change is the same in both directions, which is exactly the assumption the
analysis exists to test. Both arms are reported as d(effect)/d(ΔKD) on their own side of
zero, so a symmetric tent — bad in either direction — has one positive and one negative
arm that cancel, and their **sum is the asymmetry**. A straight line through a tent
reads as zero slope and would have missed it entirely; `tests/test_burial.py` plants one
and checks that it does. The asymmetry is a linear contrast of two correlated
estimates, so its interval comes from a `t_test` on `neg + pos_` rather than from adding
the two arms' standard errors — `asymmetry_se`, `asymmetry_p` and the interval are in
`b01_three_way_slopes.csv`, and the panel prints the estimate ± its error.

**Standard errors are clustered by position throughout.** Variants at one position all
inherit whatever that position is, so an ordinary variant-level error would treat a
protein with 500 positions as though it had 9,500 independent observations. `cluster_ols`
refuses outright when there are fewer than six positions behind a fit.

**B02 is B01's own diagnostic.** B01 cuts the bundle at its median radius and calls one
side cavity and the other lipid. If tolerance really is a property of facing lipid there
should be a step near that cut; if it drifts smoothly, the three-class picture is a
summary rather than a boundary, and B02 is where you find out which.

**B09 is the one that changes how you read a substitution.** A leucine is not
intrinsically a good substitution; it is a good substitution somewhere hydrophobic. The
local environment is the mean Kyte-Doolittle hydrophobicity of everything whose Cα is
within 8 Å, skipping the four residues either side in sequence — those are the same
helix turn and would mostly measure the residue against itself. If mismatch is what
matters, the pale diagonal runs bottom-left to top-right and the cost lives in the
off-diagonal corners; the R² of the mismatch model is printed against the R² of
\|ΔKD\| from the wild type, which is the conventional predictor it has to beat.

**B10 is a criticism of the language model, stated as a measurement.** ESM-1v knows
about evolution, not about the membrane. If its residuals were noise they would have
nothing to do with where a position sits; if they line up with burial, depth or the
polarity of what is being introduced, that is a specific account of what it is missing
and an argument for putting structure into the prior rather than hoping the model has
learnt it. The residual is defined exactly as A15 defines it, so a position A15 calls
functional is the same position here.

Nothing is pooled across proteins. Every panel is one protein, in the order the configs
are given on the command line.

### What solvent accessibility does, and does not, say

```bash
python -m mpdms access configs/AQR1_fitness.yaml configs/HXT1.yaml configs/HXT3.yaml
```

RSA is the strongest structural predictor of tolerance in a soluble protein, and in a
membrane protein it is a **broken measurement**: it rolls a *water* probe over an
isolated monomer, so a residue against acyl chains and a residue in the cytosol both
score "exposed" while being in opposite chemical worlds, and in a transporter a residue
lining the water-filled cavity scores exposed as well. That is not a reason to drop it.
It is the reason these ten exist — most of them are about what RSA is confusing with
what.

| | question | what the answer looks like |
| --- | --- | --- |
| R01 | the dose–response | effect against RSA, fitted separately in TM and in loops; the slope ratio is the number |
| R02 | water-facing against lipid-facing area | per-atom SASA split by each atom's own depth: one predictor becomes two with opposite meanings |
| R03 | where is the knee? | a free breakpoint on RSA with a bootstrap interval, against the 0.15 cut used elsewhere |
| R04 | your burial, or your neighbourhood's? | own RSA and mean RSA within 8 Å in one model, over a 2-D map |
| R05 | unique and shared variance | Shapley (LMG) shares of R² across RSA, contacts, depth and radius, stacked |
| R06 | the shape, not the mean | SD and skew of each position's own substitutions against RSA |
| R07 | which residues care about being buried? | the RSA slope for each of the 20 mutant residues, ranked |
| R08 | per helix | the within-helix slope against that helix's RSA range: no range, no prediction |
| R09 | the fraction at stop level | fraction of substitutions ≤ −0.8 against RSA, beside the rescaled mean |
| R10 | burial that moves | ΔRSA between two conformations, with mean RSA held fixed |
| R11 | what burial costs a charge, a proline and a glycine | **TM helices only**: RSA on x, the mean effect of K/R, of proline and of glycine at each position on y, three series in one panel |

**R02 is the point of the module.** Shrake–Rupley is run at the atom level and each
exposed atom's area is assigned to lipid or water by its own depth in the membrane
frame, with the slab half-thickness taken from the protein's own TM residues rather than
assumed. A residue then has two accessibilities instead of one, and both go in the same
model. Exposure to water should be protective; exposure to lipid should not be. If the
two coefficients come out equal, the membrane adds nothing to the burial story.

**R03 audits the rest of the pipeline.** The RSA < 0.15 "buried" cut used throughout
`environ`, `arch` and `burial` is inherited from soluble-protein work and had never been
checked against these data. The breakpoint is found by fitting a hinge at every
candidate quantile and keeping the smallest residual sum of squares, with a bootstrap
interval over positions. If the interval excludes 0.15, several existing figures are
cutting in the wrong place and say so.

**R05 answers "is RSA doing work, or is it a proxy?"** S09 compares burial measures;
this one splits the overlap between them. The Shapley (LMG) share averages each
predictor's marginal R² over every ordering, so the shares are non-negative and add to
exactly the full model's R²; the tick inside each block is that predictor's *unique*
contribution, what it adds entering last. A tall block with a tiny tick is a predictor
doing nothing the others could not.

**R07 is the sanity check as much as the figure.** Proline and the charged residues
should be ruinous when buried and harmless when exposed — a steep positive slope —
while alanine and serine should be nearly flat. If that ordering does not appear,
something upstream is wrong.

**R10 needs a second structure** and skips cleanly when there is none: set
`structure.alt_path` in the config or pass `--alt GENE=FILE`. Mean RSA across the two
states is in the model, so the ΔRSA coefficient is about *moving* rather than about
being exposed on average. Handing it the same file twice is refused rather than fitted.

### Two homologues and two symmetries, asked together

```bash
python -m mpdms homolog configs/HXT1.yaml configs/HXT3.yaml
```

An MFS transporter carries its own internal replicate: two six-helix bundles related by
a pseudo-twofold and joined by a cytosolic loop, so **TM1 and TM7 are the same helix
built twice**. Two paralogues give a second replicate of a different kind — the same
fold under a different sequence lineage. Having both at once is what makes this module
possible, and H06 is the analysis that needs both.

| | question | what the answer looks like |
| --- | --- | --- |
| H01 | the whole protein | one scatter inside the membrane, one outside it, and both against the split-half ceiling |
| H02 | per helix: the *pattern* | one ρ per TM helix over the aligned positions inside it, each with its own ceiling |
| H03 | per helix: the *level* | twelve points, helix mean against helix mean, coloured by bundle |
| H04 | the symmetry mates | per-pair ρ against per-pair Cα RMSD, six pairs per protein |
| H05 | the bundle, seen | twelve discs where the helices really are, mates joined; written twice, once filled by effect and once as a plain black-and-white diagram |
| H06 | the mate, and the paralogue | two scatters on the same positions: mean effect against its symmetry mate, and against the paralogue |
| H07 | where they disagree | the residual mapped into both structures, with S08's correlogram |
| H08 | does identity predict agreement? | ρ refitted inside identical / similar / dissimilar columns |
| H09 | substitution profiles | the 19-vector correlated at each column, against a mismatched-position null |
| H10 | transfer | RMSE and AUROC for the worst tenth, both directions |
| H11 | the two proteins as pretend replicates | the same substitution to the same conserved residue, in both, against the attenuation ceiling |
| H12 | what the context actually is | the disagreement at those columns regressed on how different the surroundings are |

**Everything is quoted against a split-half ceiling.** Two position means measured with
noise cannot correlate at 1 even if the biology is identical, so each protein's
variants are split at random, the two sets of means correlated, and the result
Spearman-Brown corrected; the attainable maximum is the geometric mean of the two
reliabilities. Without it, "TM5 is divergent" and "TM5 is poorly measured" look the
same. H01 reports **ρ as a percentage of attainable** and H02 draws each helix's own
ceiling as a tick on its bar.

**Correlations are scale-free**, so unlike the violin figures this module is comparable
across proteins even though HXT1 was normalised against a wild-type anchor rather than
synonymous variants. Worth saying out loud, since the opposite caveat applies
everywhere else.

**H02 and H03 are different questions and the tests enforce it.** A helix can be the
most sensitive one in both proteins while the arrangement of sensitivity inside it
agrees not at all. `tests/test_homology.py` plants exactly that case — matched helix
levels, independent within-helix patterns — and checks that H03 sees ρ > 0.7 while H02
sees a median near zero.

**H04 breaks out what S07 pools.** S07 superposes the two bundles and reports one
correlation. H04 superposes each of the six mate pairs on its own and plots its ρ
against its own Cα RMSD, so the figure can say whether the pairs that superpose best
are the pairs whose constraint agrees. The pairing is **searched, not assumed** —
every one-to-one correspondence of the six against the six is scored by a Kabsch fit on
helix centroids — and whether the answer is the textbook TM1–TM7 is reported alongside;
`--canonical` forces it. A helix pair carries only a dozen or so measured positions, so
one pair in six reaching |ρ| = 0.7 by chance is ordinary: every pair gets a bootstrap
interval and a single striking pair is not a finding.

**H05 is the panel that shows the twofold instead of claiming it.** The membrane normal
is recovered by regressing each residue's depth on its coordinates, and the twelve
helix centroids are projected into the plane across it. Each disc is filled by its
helix's mean effect and outlined by which bundle it belongs to; the line joining a pair
is as wide as the pair agrees.

It writes **two versions**. `h05_bundle_map` is the figure about the data. `h05b_bundle_map_plain`
is the same geometry with nothing encoded in colour — no fill, no correlations on the
links, the two bundles told apart by a solid against a dashed outline, which survives
greyscale printing and a projector. That one is a diagram of the fold, for a slide that
explains the architecture before any result is shown. Both are drawn from one
`bundle_geometry` call, so the plain version cannot drift from the real one. PyMOL and
ChimeraX scripts go to `structures/<protein>/h05_bundles.*`.

**H11 treats the two proteins as replicates of one experiment, and measures where
that breaks.** At an alignment column where both carry the *same* wild-type residue,
HXT1 L→A and HXT3 L→A are the same chemical change to the same residue; a replicate
would reproduce it. What they do not share is everything around the residue, so the gap
between how well a protein agrees with *itself* and how well it agrees with its
homologue at these columns is a measurement of context with the chemistry held fixed
rather than merely controlled for. Four bars, left to right: the attenuation ceiling
(one minus the mean squared standard error over the observed variance — the variant-level
counterpart of H01's split-half), the identical-wild-type columns, the
different-wild-type columns, and a null that keeps the mutant residue but pairs it with
the wrong column. A third panel breaks the agreement down **by wild-type residue**:
which residues carry their behaviour with them wherever they sit, and which do not.

The assay's own wild type has to match the alignment's at every matched row, or the
numbering is off by something and every "identical" column is quietly a different
residue. Rows that disagree are dropped and counted, and the count is printed.

**H12 asks what the context is.** With the chemistry held fixed, the remaining
disagreement is regressed on how different the two positions' surroundings are — |Δ RSA|,
|Δ depth|, |Δ contacts| — and on how much of the immediate neighbourhood is itself
conserved (the fraction of residues within 8 Å whose own alignment column is identical).
A flat result says the disagreement is noise; a slope says the context is doing
something specific, and the third panel puts the worst-disagreeing columns in the
membrane frame so you can see where.

**H06 is two scatters and nothing else.** The left panel correlates each position's
mean effect with the effect at its structurally equivalent position in the other bundle
of the same protein; the right panel correlates the same positions with their aligned
counterpart in the paralogue. Both are computed on exactly the same set of positions,
so the only difference between the panels is which partner the effect is compared with
— the fold's own copy, or the other lineage. If the mate wins, constraint follows the
architecture; if the paralogue wins, it follows the sequence. ρ with its bootstrap
interval is printed inside each panel rather than in the title, because the paper style
moves panel titles into the caption file and a number put there never appears on the
figure. `h06_mate_vs_paralog.csv` also carries the calibrated RMSE and each
comparison's own ceiling.

**H01 splits the scatter by topology.** Both proteins have sensitive TM cores and
tolerant loops, so a correlation over all aligned positions is partly guaranteed before
any biology. One panel takes the positions that are in a TM helix in both, the other
those outside the membrane in both, each with its own fit and its own permutation
p-value, and the other class stays visible in grey so the two share a scale. A
correlation that survives inside each class separately is not the TM/loop contrast in
disguise.

**H03 colours by bundle rather than labelling every point.** Twelve text labels on a
square panel either collide or push the points into a corner. The N-bundle runs through
a blue ramp and the C-bundle through a red one, circles against squares, with the key
outside the axes — so the colour carries which half of the fold a helix is in and where
in that half, and it matches the bundle colours H05 uses.

### Is the cavity lining more constrained than the rest of the protein?

```bash
python -m mpdms cavity configs/AQR1_fitness.yaml configs/HXT1.yaml
```

The quantity is the **site median ESM-1v residual z**, defined exactly as A15 defines
it: how much more constrained a position looks to a language model than its measured
abundance explains. Strongly negative is this project's signature for functional —
conserved for a reason the abundance assay cannot see. The question is whether the
residues lining the central pathway carry more of it than the rest of the protein.

Three things would manufacture that answer, and each is handled rather than hoped away:

| confound | what is done |
| --- | --- |
| **topology** | cavity residues are transmembrane and TM is more constrained than loops anyway, so the test is also run with loops excluded — cavity against the rest of the *membrane domain* |
| **spatial autocorrelation** | S08 measured constraint travelling ~20 Å, so neighbours are not independent. Standard errors are clustered on k-means spatial blocks of about that size, and the naive interval is drawn behind the corrected one so the gap is visible |
| **the threshold** | "cavity-lining" is a median split on distance from the bundle axis, so the claim could be an artefact of where the line fell. The radial coordinate is therefore also used continuously, with no threshold at all |

**The three classes partition the protein.** Inside the membrane a residue is cavity
or lipid by the radial coordinate alone — which side of the bundle it faces — so a
buried helix–helix residue is counted with whichever face it sits on rather than
forming a class of its own. Outside the membrane the cytosolic and lumenal loops are
pooled: they differ in which side they are on, which matters for insertion and not for
this question, and both are solvated. Every scored residue is therefore in exactly one
violin and in exactly one side of the test, which the earlier five-class split did not
manage — buried residues were in the test but absent from the figure. The run prints
how many formerly-core residues fall on the cavity side, so the redefinition is visible
rather than silent.

**The spatial correction is the part that matters, and it is load-bearing.**
`tests/test_cavity.py` plants a nuisance shared within each 20 Å patch and *no* cavity
effect: the naive test calls the difference real at p < 0.05, while the block-clustered
error comes out more than twice as wide and the test correctly finds nothing. Without
it a figure like this reports a patch of the fold as a discovery.

**The units are the residual's own, not the z.** A15 plots the residual divided by
1.4826 x MAD; that is one constant shared by every residue in the protein, so a rank
test, a KS test and the clustered p-values are numerically identical in either column —
only the effect sizes change units, and a shift quoted in ESM-1v log-likelihood ratio is
one a reader can interpret. (Dividing by a constant is not what makes a z-score awkward
to test; standardising *within* each group is, and nothing here does that.) The signed
residual is used, never its absolute value: the sign carries the direction of the claim.

**Three distributions mean three comparisons**, so V01 reports all three pairs —
cavity against each grey class, and the two grey classes against each other, which is
the comparison that says whether "cavity" does anything "inside the membrane at all"
does not — with Holm correction across them. Beside the rank test sit two tests the
rank test cannot do: **Kolmogorov–Smirnov**, which sees a heavier negative tail at the
same median, and **Brown–Forsythe**, which asks whether the spreads differ at all. A
third p-value, the **within-block permutation**, centres the values inside each 20 Å
patch and shuffles the labels only within patches, so every comparison the null makes
is between residues sitting in the same patch; a large-scale gradient that happens to
line up with the bundle axis fails it.

Effect size is **Cliff's delta** — how often a cavity residue outranks a non-cavity one
— with a confidence interval from a bootstrap that resamples *blocks*, not residues.
Resampling residues would treat one patch as many independent draws and return an
interval far too narrow.

| figure | what it shows |
| --- | --- |
| V01 | (a) the three distributions as rainclouds — half violin, box, and every scored residue — with Cliff's delta and the Holm-corrected rank p above each pair; (b) the same three as cumulative curves, where the KS statistic is a vertical distance you can point at |
| V02 | cavity − rest per protein and per comparison; thin bar = block-clustered interval, thick grey = naive |
| V03 | residual against distance from the bundle axis, TM sites only — the same claim with no threshold in it |

"Consistently" is the word in the question, so the run ends by saying whether the
proteins actually agree: same direction and significant in both, same direction only,
or contradicting each other.

### Can the structure make the family model better - and does it beat ESM-1v?

```bash
python -m mpdms famstruct \
  --family hexose=configs/HXT1.yaml,configs/HXT2.yaml,configs/HXT3.yaml,configs/HXT7.yaml,configs/PHO84_fitness.yaml \
  --human SLC2A1=P11166 --human SLC2A2=P11168 --human SLC2A9=Q9NRM0 --human SLC2A10=O95528 \
  --mafft-args "--maxiterate 1000 --localpair"
```

The plain family model shrinks every alignment column toward one number. That number is a
property of the column's measurements, not of the position, so the model has nothing to
say where the family is thin and nothing to say at all about geometry. `famstruct` gives
each column a prior mean predicted from the structures instead:

    theta_jc ~ Normal(m_c, tau^2),    m_c = beta0 + beta . x_c

The features are the ones that mean something in a membrane protein, which is what makes
the fitted `beta` worth reading (F03):

| feature | what it distinguishes |
| --- | --- |
| distance from the bundle axis | lining the central cavity against facing lipid - RSA calls both "exposed" |
| depth across the membrane | the middle of the bilayer against the interface |
| contact number | packed against other helices against free to move |
| solvent accessibility | the usual burial measure |
| pLDDT | whether the model is confident enough for the rest to mean anything |
| in a TM helix | the topology baseline, so the others are read on top of it |

The prior has to be **proper**. Under the flat prior on `mu` used by `bayes`, a
column-specific prior mean does exactly nothing: shifting the data by `m_c` and shifting
the answer back returns the original posterior. So the prior's width is the residual
spread of the structural regression - a regression that explains little gives a wide prior
and the model reverts to the flat answer by itself. That is asserted in the tests, in both
directions.

Four models are then compared leave-one-protein-out, with the structural regression *and*
the ESM calibration refitted without the held-out protein each time (F04, F06):

| model | what it knows |
| --- | --- |
| intercept only | the plain family model |
| structure | plus where the column sits in the membrane |
| ESM-1v | no family data at all: ESM, put on the assay scale by a fit to the training proteins |
| structure + ESM-1v | the two combined by precision, no new parameters |

ESM is on a log-likelihood-ratio scale, so it is calibrated before anything but rank is
compared, and the residual spread of that calibration is the uncertainty it is given. The
combination is then just inverse-variance weighting of two predictions of the same
quantity. Panel (c) of F06 is the one to look at first: the less the family model and ESM
agree with each other, the more there is to gain by using both.

Other outputs: F01 is the protein-by-protein grid of column effects (how much the family
agrees at aligned positions at all), F02 is effect against depth, radial distance and RSA,
and F05 is what each unmeasured human protein is predicted to be, under the structural
prior and without it, with the number of residues whose call changes.

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

