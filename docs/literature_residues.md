# Published functional residues for AQR1 and QDR2

## Status of this file

Every residue below was obtained from **web-search snippets, not from the
primary text**. The container running this pipeline has outbound access to
journals, PMC, Europe PMC and SGD blocked by the network egress proxy, so no
PDF or full-text page could be opened. Two independent searches returned the
same three residues with consistent descriptions, which is reassuring but is
not verification.

Everything is therefore marked `confidence: unverified` in
`configs/_literature.yaml`. **Open the PDF and confirm the three numbers before
any of this goes in a talk.** A19 will keep printing an "unverified" caveat
until the flags are changed.

The one paper to read:

> Overlapping Roles of Yeast Transporters Aqr1, Qdr2, and Qdr3 in Amino Acid
> Excretion and Cross-Feeding of Lactic Acid Bacteria.
> *Frontiers in Microbiology* 12:752742 (2021).
> doi:10.3389/fmicb.2021.752742 — PMC8649695

## AQR1 (YNL065W)

Three residues in the modelled substrate-binding cavity, identified by docking
threonine and homoserine onto an Aqr1 model built on the *E. coli* MdfA
template, then mutated and assayed for amino acid excretion.

| Residue | Allele | Reported effect on excretion | Reported abundance/localisation |
|---|---|---|---|
| Q149 | Q149A | reduced | GFP fusion **more** abundant at the cell surface than internal membranes |
| Q238 | Q238A | reduced | not reported in what could be retrieved |
| R504 | R504A | reduced (homoserine) | as Q149A |

Q149 and Q238 were reported as hydrogen-bonding or salt-bridging **both** Thr
and Hom across docking poses; R504 only Hom.

The pocket overall is described as shaped by **TM1, 2, 4, 5, 6, 7 and 11**.

### Why Q149 and R504 are the most valuable rows here

The localisation result is the important part, and it is easy to skim past.
Those mutants were reported as *well expressed and correctly localised* — if
anything more surface-abundant than wild type — while losing transport
activity. That is precisely the dissociation this project is trying to
demonstrate:

- an **abundance** screen should score them as tolerated,
- an **evolutionary** model (ESM-1v) should score them as constrained.

So the prediction is specific and falsifiable: Q149 and R504 should land in the
bottom-right quadrant of the A19 figure — functional by A15, abundance-tolerant.
If instead the screen calls them dead, either the assay is picking up something
other than biogenesis, or the published localisation result does not hold in
this strain background.

Three residues cannot carry a statistical claim. The Fisher test A19 prints is
descriptive; the per-residue table is the result.

## QDR2 (YIL121W)

**No residue-level mutagenesis was found.** The published work on Qdr2 is at
gene level: deletion and overexpression phenotypes for quinidine, barban,
bleomycin and cisplatin resistance; K+ (Rb+) uptake independent of Trk1/Trk2;
copper and oxidative-stress phenotypes. Searches for point mutants, alanine
scans or binding-pocket residues returned nothing for this protein.

`configs/_literature.yaml` records QDR2 with an empty `measured` list, and a
test (`test_a19_curated_file_parses_and_qdr2_claims_nothing`) fails if anyone
later fills it in without updating the test — a deliberate guard against
residues drifting in from a paralogue.

**Do not transfer the Aqr1 numbers onto Qdr2.** They are paralogues, not the
same protein, and 149/238/504 in Aqr1 are different residues in Qdr2. If you
want the equivalent positions, map them through the AQR1/QDR2 alignment the
`family` module already builds — and treat the result as a hypothesis to test
against the DMS data, not as published evidence.

## Adding a residue

```yaml
GENE:
  measured:
    - pos: 149
      wt: Q               # A19 checks this against the FASTA and drops the
                          # entry if it disagrees
      substitution: Q149A
      function: loss | partial | none | gain | unknown
      abundance: normal | reduced | increased | unknown
      assay: what was measured
      doi: 10.xxxx/yyyy
      confidence: verified      # only after reading the primary text
```

Residues named only in a docking pose or a family motif go in `inferred:`,
never `measured:` — A19 scores `measured` as a positive-control set, and a
predicted residue in there would quietly corrupt that.
