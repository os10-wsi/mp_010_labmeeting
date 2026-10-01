"""`python -m mpdms init <folder> [<folder> ...]`

Finds fitness_estimation.tsv under each folder, works out the gene name from the path,
retrieves UniProt sequence/topology and the AlphaFold model, and writes configs/<GENE>.yaml.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from . import fetch
from .annot import build_segments, positive_inside_n_terminus, predict_tm_hydropathy, segments_from_features
from .config import REPO_ROOT, defaults

SOURCE_NAMES = ("fitness_estimation.tsv", "fitness_estimation.txt", "fitness_estimation.csv")


def find_sources(root: Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    hits = []
    for name in SOURCE_NAMES:
        hits += [p for p in root.rglob(name)]
    return sorted(set(hits))


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def wt_sequence_from_data(src: Path) -> str | None:
    """Last-resort WT sequence reconstructed from the DMS table (gaps = X)."""
    from .config import Config
    from .io import detect_columns, parse_variants, read_source
    cfg = Config.wrap(defaults())
    raw = read_source(src)
    try:
        v = parse_variants(raw, detect_columns(raw, cfg), cfg, None).dropna(subset=["pos"])
    except Exception:
        return None
    v = v[v["pos"] > 0]
    wt = v.groupby("pos")["wt"].agg(lambda s: s.mode().iloc[0])
    L = int(wt.index.max())
    return "".join(wt.get(i, "X") for i in range(1, L + 1))


def make_config(src: Path, gene: str | None = None, offline: bool = False, dataset_id: str | None = None) -> dict:
    gene = (gene or fetch.gene_from_path(src.parent if src.is_file() else src)).upper()
    cfg = defaults()
    notes = []
    entry, fasta, pdb = None, None, None
    local = fetch.local_files(gene)
    if not offline:
        try:
            entry = fetch.uniprot_entry(gene, fetch.accession_from_path(src))
            fasta = fetch.write_fasta(entry, gene)
        except Exception as e:
            notes.append(f"UniProt lookup failed ({e.__class__.__name__}: {e})")
        if entry:
            try:
                pdb = fetch.alphafold_model(entry["primaryAccession"], gene)
            except Exception as e:
                notes.append(f"AlphaFold download failed ({e})")
    entry = entry or local.get("uniprot")
    fasta = fasta or local.get("fasta")
    pdb = pdb or local.get("pdb")

    seq = None
    if fasta:
        from .config import read_fasta
        seq = read_fasta(fasta)
    elif entry:
        seq = entry["sequence"]["value"]
    if seq is None:
        seq = wt_sequence_from_data(src)
        if seq:
            p = fetch.external_dir(gene) / f"{gene}_from_dms.fasta"
            p.write_text(f">{gene} reconstructed from DMS WT residues\n{seq}\n")
            fasta = p
            notes.append("sequence reconstructed from DMS table - verify against UniProt")

    # topology
    segs, n_term, topo_src = [], "cytosolic", "none"
    if entry:
        segs, n_term = segments_from_features(entry.get("features", []), len(seq))
        topo_src = f"UniProt {entry['primaryAccession']} features"
        if not any(s["type"] == "TM" for s in segs):
            segs = []
    if not segs and seq:
        tms = predict_tm_hydropathy(seq)
        n_term = positive_inside_n_terminus(seq, tms)
        segs, n_term = build_segments(tms, len(seq), None, n_term)
        topo_src = "Kyte-Doolittle 19-aa window prediction + positive-inside rule (NOT curated)"
        notes.append("topology predicted from hydropathy - curate before interpreting A04-A08")

    acc = entry["primaryAccession"] if entry else None
    pname = None
    if entry:
        pname = (entry.get("proteinDescription", {}).get("recommendedName", {})
                 .get("fullName", {}).get("value"))
    did = dataset_id or gene
    label = did if did == gene else f"{gene} [{did[len(gene) + 1:]}]"
    cfg.update(id=did, display_name=label if not pname else f"{label} ({pname})")
    cfg["source"].update(path=_rel(src), citation="")
    cfg["protein"].update(gene=gene, uniprot=acc, sequence=_rel(fasta) if fasta else None, region=None)
    cfg["topology"].update(n_terminus=n_term, source=topo_src, segments=segs)
    cfg["structure"].update(path=_rel(pdb) if pdb else None)
    cfg["_init_notes"] = notes
    return cfg


class _Dumper(yaml.SafeDumper):
    pass


def _flow_dicts(dumper, data):
    flow = all(not isinstance(v, (dict, list)) for v in data.values()) and "start" in data
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=flow)


_Dumper.add_representer(dict, _flow_dicts)


def write_config(cfg: dict, force: bool = False) -> Path:
    out = REPO_ROOT / "configs" / f"{cfg['id']}.yaml"
    if out.exists() and not force:
        print(f"  {out.name} exists - keeping it (use --force to overwrite)")
        return out
    notes = cfg.pop("_init_notes", [])
    header = "# generated by `python -m mpdms init`\n" + "".join(f"# NOTE: {n}\n" for n in notes)
    out.write_text(header + yaml.dump(cfg, Dumper=_Dumper, sort_keys=False, width=110))
    return out


def _tag(root: Path, src: Path) -> str:
    """Short label for the results set a source came from: the nearest folder named like
    '008_default_results' gives '008'; otherwise the name of the folder passed to init."""
    for part in reversed(src.resolve().parent.parts):
        m = re.match(r"^(\d+)[_\-]", part)
        if m:
            return m.group(1)
    m = re.match(r"^(\d+)", root.resolve().name)
    return m.group(1) if m else re.sub(r"\W+", "_", root.resolve().name)


def dedupe(found: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    """Drop byte-identical copies of the same gene's table (e.g. default_results/ and
    fitness/default_results/ holding the same file); keep the shortest path."""
    import hashlib
    seen, out = {}, []
    for root, src in sorted(found, key=lambda x: (len(x[1].parts), str(x[1]))):
        try:
            g = fetch.gene_from_path(src.parent)
        except ValueError:
            g = str(src)
        h = (g, hashlib.md5(src.read_bytes()).hexdigest())
        if h in seen:
            print(f"same table as {seen[h]} - ignoring {src}")
            continue
        seen[h] = src
        out.append((root, src))
    return out


def dataset_ids(found: list[tuple[Path, Path]], gene: str | None = None) -> dict:
    """Dataset id per source: the gene name, suffixed with the results-set tag when the same
    gene occurs more than once (e.g. SEC61_008 and SEC61_010)."""
    genes = {}
    for root, src in found:
        try:
            genes[src] = (gene or fetch.gene_from_path(src.parent)).upper()
        except ValueError as e:
            print(f"skip {src}: {e}")
    counts = {}
    for g in genes.values():
        counts[g] = counts.get(g, 0) + 1
    tags = {src: _tag(root, src) for root, src in found if src in genes}
    sets = {}
    for src, g in genes.items():
        sets.setdefault(g, set()).add(tags[src])
    ids = {}
    for root, src in found:
        if src not in genes:
            continue
        g = genes[src]
        # suffix with the results set only when the gene occurs in more than one set
        ids[src] = g if counts[g] == 1 or len(sets[g]) == 1 else f"{g}_{tags[src]}"
    seen: dict[str, int] = {}
    for s in ids:  # same gene twice within one results set (different tables): _2, _3, ...
        base = ids[s]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            ids[s] = f"{base}_{seen[base]}"
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mpdms init")
    ap.add_argument("paths", nargs="+", help="DMS folders (searched recursively) or fitness_estimation.tsv files")
    ap.add_argument("--gene", help="override gene name (only with a single dataset)")
    ap.add_argument("--offline", action="store_true", help="use only files already in data/external/<GENE>/")
    ap.add_argument("--force", action="store_true", help="overwrite existing configs")
    a = ap.parse_args(argv)
    found = [(Path(p), s) for p in a.paths for s in find_sources(Path(p))]
    if not found:
        sys.exit(f"no {SOURCE_NAMES[0]} found under {a.paths}")
    found = dedupe(found)
    srcs = [s for _, s in found]
    ids = dataset_ids(found, a.gene if len(srcs) == 1 else None)
    for src in srcs:
        if src not in ids:
            continue
        try:
            cfg = make_config(src, a.gene if len(srcs) == 1 else None, a.offline, ids[src])
        except ValueError as e:
            print(f"skip {src}: {e}")
            continue
        notes = cfg.get("_init_notes", [])
        out = write_config(cfg, a.force)
        print(f"{cfg['id']:>8}  {src}\n          -> {_rel(out)}  "
              f"[{len([s for s in cfg['topology']['segments'] if s['type'] == 'TM'])} TM, "
              f"structure: {'yes' if cfg['structure']['path'] else 'NO'}]")
        for n in notes:
            print(f"          ! {n}")


if __name__ == "__main__":
    main()
