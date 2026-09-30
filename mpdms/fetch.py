"""Retrieve UniProt entry + AlphaFold model for a S. cerevisiae gene, with a local cache.

Everything lands in data/external/<GENE>/. If the network is unavailable, drop the files
there by hand (<ACC>.fasta, <ACC>.json from UniProt, AF-<ACC>-F1-model_v*.pdb) and
re-run `python -m mpdms init`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

from .config import REPO_ROOT

YEAST_TAXID = 559292  # S. cerevisiae S288C
UNIPROT = "https://rest.uniprot.org/uniprotkb"
AFDB = "https://alphafold.ebi.ac.uk"
TIMEOUT = 30

GENE_RE = re.compile(r"^[A-Za-z]{3}\d{1,3}[A-Za-z]?$")          # e.g. SEC61, PMA1, ERG11
NOT_GENES = {"rep", "run", "lib", "exp", "tmp", "set", "day", "plt", "bin", "seq", "out", "fig"}
ORF_RE = re.compile(r"^Y[A-P][LR]\d{3}[CW](?:-[A-Z])?$", re.I)   # e.g. YAL001C


def external_dir(gene: str) -> Path:
    d = REPO_ROOT / "data" / "external" / gene.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d


def gene_from_path(path: Path) -> str:
    """Nearest path component that looks like a yeast gene name / ORF id."""
    for part in reversed(Path(path).resolve().parts):
        stem = Path(part).stem
        if stem[:3].lower() in NOT_GENES:
            continue
        if GENE_RE.match(stem) or ORF_RE.match(stem):
            return stem.upper()
    raise ValueError(f"no yeast gene-like folder name in {path}; pass --gene")


def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    return r


def uniprot_entry(gene: str) -> dict:
    """Return the UniProt JSON entry for a yeast gene (cached)."""
    d = external_dir(gene)
    cached = sorted(d.glob("*.uniprot.json"))
    if cached:
        return json.loads(cached[0].read_text())
    queries = [
        f"(gene_exact:{gene}) AND (organism_id:{YEAST_TAXID}) AND (reviewed:true)",
        f"(gene:{gene}) AND (organism_id:{YEAST_TAXID})",
    ]
    acc = None
    for q in queries:
        r = _get(f"{UNIPROT}/search", params={"query": q, "fields": "accession,gene_names", "format": "json", "size": 5})
        res = r.json().get("results", [])
        for hit in res:
            names = {n.get("value", "").upper() for g in hit.get("genes", [])
                     for k in ("geneName", "orderedLocusNames", "synonyms", "orfNames")
                     for n in (g.get(k) if isinstance(g.get(k), list) else [g.get(k)] if g.get(k) else [])}
            if gene.upper() in names:
                acc = hit["primaryAccession"]
                break
        if acc is None and res:
            acc = res[0]["primaryAccession"]
        if acc:
            break
    if acc is None:
        raise LookupError(f"no UniProt entry for {gene} in taxon {YEAST_TAXID}")
    entry = _get(f"{UNIPROT}/{acc}.json").json()
    (d / f"{acc}.uniprot.json").write_text(json.dumps(entry))
    return entry


def write_fasta(entry: dict, gene: str) -> Path:
    acc = entry["primaryAccession"]
    p = external_dir(gene) / f"{acc}.fasta"
    if not p.exists():
        seq = entry["sequence"]["value"]
        p.write_text(f">sp|{acc}|{gene} S. cerevisiae\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    return p


def alphafold_model(acc: str, gene: str) -> Path:
    d = external_dir(gene)
    have = sorted(d.glob(f"AF-{acc}-F1-model_v*.pdb"))
    if have:
        return have[-1]
    url = None
    try:
        meta = _get(f"{AFDB}/api/prediction/{acc}").json()
        if meta:
            url = meta[0].get("pdbUrl")
    except Exception:
        pass
    candidates = [url] if url else [f"{AFDB}/files/AF-{acc}-F1-model_v{v}.pdb" for v in (6, 5, 4)]
    for u in candidates:
        try:
            r = _get(u)
            p = d / Path(u).name
            p.write_bytes(r.content)
            return p
        except Exception:
            continue
    raise LookupError(f"no AlphaFold model for {acc}")


def local_files(gene: str) -> dict:
    d = external_dir(gene)
    out = {}
    fa = sorted(d.glob("*.fasta"))
    pdb = sorted(d.glob("*.pdb")) + sorted(d.glob("*.cif"))
    uj = sorted(d.glob("*.uniprot.json"))
    if fa:
        out["fasta"] = fa[0]
    if pdb:
        out["pdb"] = pdb[-1]
    if uj:
        out["uniprot"] = json.loads(uj[0].read_text())
    return out
