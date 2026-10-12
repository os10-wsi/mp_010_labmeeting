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
  --esm SLC2A10=SLC2A10_esm1v.csv
```

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

### Boxed, square figures for slides

```bash
MPDMS_BOX=1 MPDMS_SQUARE=1 python -m mpdms clinvar ...
```

`MPDMS_BOX=1` closes the frame — all four spines, drawn heavier (`MPDMS_BOX_WIDTH`
changes the weight, default 1.4). `MPDMS_SQUARE=1` makes each panel square. Both are
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

The scale is **dsTβL** (Elazar et al. 2016, eLife 5:e12125): a fourth-order polynomial in
depth per residue, kcal/mol relative to alanine, taken from the authors' own
`polyval_Kfix_14Jan2016.txt`. A segment is laid across the membrane at `linspace(−15,
+15)` and its ΔG is the sum of the per-residue values at those depths, which is that
code's convention.

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
| I03 | effect against ΔΔG, **one line per helix** coloured by that helix's ΔG, with the mean (± SEM) and median across helices in bold |
| I04 | the same thing pooled: every variant in any TM helix, binned by ΔΔG and averaged — one curve per protein, no grouping |

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

