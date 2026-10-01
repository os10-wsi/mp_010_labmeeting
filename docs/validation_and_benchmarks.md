# Validating the assay, and benchmarking models on it

Two separate claims, with separate evidence:

1. **The assay measures membrane-protein biogenesis**, and the determinants it measures are
   general enough to say something about other yeast membrane proteins.
2. **These measurements are useful training data** — fine-tuning a protein language model on
   them improves prediction on data the model has not seen.

Claim 1 has to be settled before claim 2 means anything, so they are separate commands.

---

## Part 1 — is this biogenesis? (`python -m mpdms validate`)

```bash
python -m mpdms validate configs/QDR2.yaml configs/AQR1.yaml configs/FEN2.yaml configs/TPO3.yaml
# -> outputs/_validation/{figures,tables,summary.json}
```

### 1a. A pre-registered battery (A16)

Twelve directional predictions, each derived from biophysics established independently of
these data. Every one is stated as a direction with a pass criterion *before* looking, so
the outcome is a count rather than a narrative. Three kinds:

| | test | why it matters |
|---|---|---|
| sanity | **K01** stops separate from synonymous (AUC ≥ 0.90) | without this nothing else is interpretable |
| sanity | **K02** replicates agree (mean r ≥ 0.6) | |
| mechanism | **K03** TM positions more sensitive than non-TM | tested by circular shift, which preserves positional autocorrelation; a per-residue test here is anticonservative |
| mechanism | **K04** the von Heijne biological scale explains TM substitutions better than Kyte–Doolittle | the biological scale is calibrated on Sec61 insertion specifically. If a *transfer-free-energy* scale fit equally well, the readout would not be distinguishing insertion from generic hydrophobicity |
| mechanism | **K05** hydrophobic→polar worse in TM than loop | |
| mechanism | **K06** proline worse in TM helices than loops | helix integrity, not transfer energy |
| mechanism | **K07** introduced charge worse at the TMD centre than its ends | the depth dependence of charge burial |
| mechanism | **K08** positive-inside: K/R favoured over D/E at the cytosolic end | also a check on the topology annotation — an inversion means one or the other is wrong |
| independent | **K09** buried TM residues more sensitive than lipid-facing | uses AlphaFold RSA, which the assay never saw |
| independent | **K10** ESM-1v constraint correlates with measured sensitivity | uses evolution, which the assay never saw |
| independent | **K11** truncation is deleterious throughout the protein (≥80% of 20-residue windows) | catches a readout that only reports on a fragment — e.g. a terminal tag, or a transcript that is not full length |
| independent | **K12** the two pseudo-symmetric TM bundles give correlated profiles | **the strongest internal control available for an MFS transporter** |

**On K12.** MFS transporters are built from two structurally equivalent 6-TM bundles related
by a pseudo-twofold. Helices are paired TM1↔TM7, TM2↔TM8 and so on; each helix's profile is
interpolated onto a common relative-position grid and the two halves are correlated. The null
is a random re-pairing of the bundles, which preserves every per-helix profile and only
destroys the structural correspondence. A readout of protein structure must show the
correlation; a library-construction or sequencing artefact has no reason to. Nothing about
this test can be satisfied by a protein-level confound.

**How to read a failure.** A failed test is informative, not fatal — K08 failing alongside an
inverted asymmetry in A06 points at the topology annotation rather than the assay. The thing
that would undermine the assay is the *mechanism* block failing as a group while the sanity
block passes: that is the signature of a readout that is reproducible but is not reporting on
membrane insertion.

### 1b. Does it extrapolate? Leave-one-protein-out

A model is trained on the biophysical feature table (`mpdms/features.py`) for all proteins but
one and asked to predict the protein it has never seen. The features are deliberately
restricted to quantities that exist for *any* yeast membrane protein — topology class, position
within a TM helix, distance from the helix end, RSA, depth in the bilayer, substitution
chemistry. Nothing protein-specific (absolute position, protein identity) is included, so
transfer cannot happen through memorisation.

Two reference points are plotted beside it:

- **held-out positions, same protein** — the practical ceiling, given assay noise;
- **`site_mean`** — predicts each variant from the other variants at its position. It scores
  well on a random split and collapses to zero when positions or proteins are held out, which
  is exactly the point: it shows how much of an easy-looking benchmark is just site identity.

Transfer that holds across held-out proteins is what licenses a claim about membrane proteins
in general. Transfer that collapses means the determinants are protein-specific, and the
honest conclusion is that these eight proteins are eight case studies.

**A caution about pooling.** All of these transporters share a fold. Leave-one-protein-out
across eight MFS transporters demonstrates generalisation *within the MFS family*; it does not
by itself establish generalisation to, say, a polytopic protein with a different architecture
or a tail-anchored protein. Say "MFS transporters" unless an out-of-family test is added.

---

## Part 2 — benchmarking models (`python -m mpdms bench`)

```bash
python -m mpdms bench configs/*.yaml --split protein
python -m mpdms bench configs/*.yaml --split position \
    --scores saprot_dg=preds/saprot.csv --scores esm3_dg=preds/esm3.csv \
    --embeddings emb/variant_embeddings.npz \
    --external configs/EXT_CFTR.yaml=data/external/cftr_dms.csv
```

### Splits — the split *is* the question

| split | question |
|---|---|
| `random` | interpolation inside the measured set. Easiest, least informative; report it only to show the gap to the others |
| `position` | prediction at unseen sites (all variants at a position held out together, so there is no leakage through the site mean) |
| `protein` | extrapolation to an unseen protein. The only split that supports a claim about membrane proteins generally |
| `segment` | train on loops, test on TM (and the reverse). Tests whether a model has learnt membrane-specific physics or generic constraint |

### Metrics

Spearman (overall and per protein), Pearson, RMSE, AUROC and average precision for
"clearly lost biogenesis" (`score_z < −0.5`), and precision in the top-k most-damaging
predictions. A model whose prediction has no variance scores 0, not NaN, so an uninformative
model stays visible in the table instead of silently dropping out.

### Plugging in a model

Anything with `name`, optional `fit(train_df)` and `predict(test_df)`:

- **`ScoreColumn`** — scores computed elsewhere, supplied as a CSV with
  `uniprot|gene, pos, mut, score`. This is how to evaluate a ΔG-style predictor
  (SaProt/ESM3 variants) zero-shot. `--scores-sign auto` orients the sign on the *training
  fold only*, so an inverted convention (more positive = more destabilising) is handled
  without leaking the test labels.
- **`EmbeddingHead`** — a ridge or MLP head fitted on precomputed per-variant embeddings
  (`.npz` keyed `DATASET:POS:MUT`). This is a genuine, runnable fine-tune of the readout layer
  and does not need back-propagation through the large model.
- **`Baseline`** — ridge or gradient boosting on the biophysical features. The number to beat:
  if a pLM cannot exceed a ridge regression on hydrophobicity and topology, it has not learnt
  anything membrane-specific.

Full parameter-efficient fine-tuning (LoRA) of SaProt or ESM3 is **not implemented here** — it
needs the model weights and a GPU, and the published training code is the right thing to use.
The harness is the evaluation half: export with `features.build`, fine-tune externally, write
predictions to a CSV, feed them back through `--scores`.

### Transfer to data the model never saw

`--external CONFIG=CSV` evaluates models trained on the yeast data against another DMS. It
needs a config for that protein so the same features can be computed (run `mpdms init` for
its UniProt accession). If the external CSV has synonymous and stop controls, its scores are
rescaled to the same syn = 0 / stop = −1 convention; otherwise scores are used as given and
only the rank metrics are comparable.

Useful transfer targets, in increasing order of difficulty: another yeast MFS transporter;
a yeast membrane protein of a different fold; a human membrane protein DMS. The third is the
real test and the one most likely to fail — assay, organism, lipid environment and phenotype
all differ at once, so attribute a drop carefully.

### Benchmarks worth reporting

1. **Zero-shot**, no training: how well do existing models already do on membrane proteins?
   Report per protein, since a mean over proteins hides a model that works on one fold.
2. **Fine-tuned, held-out positions**: does training on membrane DMS help at unseen sites?
3. **Fine-tuned, held-out protein**: does it help on a protein never seen in training? This is
   where a fine-tune most often turns out to have memorised protein-level offsets.
4. **Fine-tuned, external DMS**: does it transfer off this dataset entirely?
5. **Ablation**: fine-tune on an equal number of *soluble* protein variants. If membrane data
   and soluble data help equally, the gain is from fine-tuning on anything, not from membrane
   physics. Without this control, a "membrane proteins improve the model" claim is unsupported.

Report (2)–(4) as a *gain over the zero-shot model on the same split*, not as a bare number —
the splits differ in difficulty and the absolute values are not comparable across them.

---

## Reproducibility notes

- Every figure and table is stamped with the git commit.
- `outputs/_validation/tables/variant_features.parquet` is the exact table the models saw.
- Pin versions with `requirements.lock`.
- The battery's thresholds (AUC ≥ 0.90, r ≥ 0.6, ≥80% of windows) are in
  `mpdms/analyses/a16_assay_validation.py`. Changing one after seeing the result is no longer a
  pre-registered test, so change it in the code and re-run everything rather than reinterpreting
  an existing output.
