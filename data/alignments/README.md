# AQR1 ↔ SLC22A1 column map

`aqr1_slc22a1.aln.fasta` is MAFFT L-INS-i over the two full-length sequences,
`aqr1_slc22a1_columns.csv` is the 480 columns where both proteins have a residue.

```bash
mafft --maxiterate 1000 --localpair AQR1_SLC22A1.fasta > aqr1_slc22a1.aln.fasta
```

| | |
| --- | --- |
| SLC22A1 | O15245, 554 aa, 12 helical TM segments (UniProt) |
| AQR1 | P53943, 586 aa, 11 helical TM segments (UniProt) |
| identity | 101 / 480 aligned columns = **21.0%** |

## Read this before using it position by position

Topology is annotated independently of the alignment, so TM-against-TM agreement
tests the alignment rather than restating it. It does not pass.

| | columns |
| --- | --- |
| transmembrane in both | 142 |
| loop in both | 230 |
| **transmembrane in one, loop in the other** | **108** |

That is 77.5% agreement overall, and only **56.8%** among the 250 columns where
either protein is in the membrane — against the 80% that `bayes
--min-topology-agreement` would demand. At 21% identity a pairwise alignment of two
MFS transporters places the helices and not the residues inside them, which is the
regime this number is describing.

Of the 28 SLC22A1 positions D02 calls functional, 20 have an AQR1 partner column at
all and **14** have one whose topology agrees. A position-by-position AQR1 → SLC22A1
correlation therefore rests on roughly half the sites of interest, each mapped by a
column that is wrong about the membrane 43% of the time.

**Use the family route instead.** `bayes` aligns the whole PF07690 set at once and
drops columns that fail the topology check, so the column a human residue is scored
against is supported by several yeast proteins rather than by one pairwise guess:

```bash
python -m mpdms bayes --family PF07690=configs/AQR1_fitness.yaml,... \
    --mafft-args "--maxiterate 1000 --localpair" --min-topology-agreement 0.8
```

The pairwise map is kept here because the single-donor comparison is still worth
running as a baseline for the pooled one — it is the thing pooling has to beat.

## The join table

`slc22a1_sites_with_aqr1_column.csv` is one row per SLC22A1 position tested by D02
(540 of them), carrying the measured abundance, function and residual, the D02 call,
the distance to the nearest bound ligand, and the AQR1 position it aligns to. 467 have
an AQR1 partner and 361 of those pass the topology check.

To run the AQR1 comparison, join your AQR1 site medians onto `aqr1_pos`, and filter on
`topology_ok` before reading anything position by position:

```python
aqr1 = ...                                    # columns: pos, site_median
t = (pd.read_csv("data/alignments/slc22a1_sites_with_aqr1_column.csv")
       .merge(aqr1.rename(columns={"pos": "aqr1_pos"}), on="aqr1_pos")
       .query("topology_ok == 1"))
```

The same table, with `site_median` renamed to `score` and `slc22a1_pos` to `pos`, is
the `--donor AQR1=...` file `mpdms transfer` wants.
