"""Config loading, defaults and validation.

Every analysis reads only from a Config plus the canonical table; nothing
dataset-specific is hard-coded anywhere else.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "configs" / "_template.yaml"


class ConfigError(ValueError):
    pass


class Config(dict):
    """dict with attribute access; nested dicts are Configs too."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict) and not isinstance(obj, Config):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur or cur[part] is None:
                return default
            cur = cur[part]
        return cur

    def resolve(self, p: str | None) -> Path | None:
        """Resolve a path from the config relative to the repo root."""
        if not p:
            return None
        path = Path(p).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    def to_plain(self) -> dict:
        return yaml.safe_load(yaml.safe_dump(_plain(self)))


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def defaults() -> dict:
    with open(TEMPLATE) as fh:
        d = yaml.safe_load(fh)
    # template placeholders that must not leak into real datasets
    d["id"] = None
    d["display_name"] = None
    d["source"]["path"] = None
    d["protein"].update(gene=None, uniprot=None, sequence=None)
    d["structure"]["path"] = None
    d["topology"]["segments"] = []
    return d


def load_config(path: str | Path) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config.wrap(_deep_merge(defaults(), raw))
    cfg["_config_path"] = str(path)
    if not cfg.get("id"):
        raise ConfigError(f"{path}: 'id' is required")
    if not cfg.get("display_name"):
        cfg["display_name"] = cfg["id"]
    return cfg


def read_fasta(path: Path) -> str:
    seq = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if seq:
                    break
                continue
            seq.append(line.strip())
    return "".join(seq).upper()


def protein_sequence(cfg: Config) -> str | None:
    p = cfg.resolve(cfg.get_path("protein.sequence"))
    if p and p.exists():
        return read_fasta(p)
    return None


# ----------------------------------------------------------------- validation
def validate_topology(cfg: Config, region: tuple[int, int] | None) -> list[str]:
    """Return a list of problems (empty = OK)."""
    errs: list[str] = []
    segs = sorted(cfg.get_path("topology.segments", []) or [], key=lambda s: s["start"])
    if not segs:
        return errs
    for s in segs:
        if s["start"] > s["end"]:
            errs.append(f"segment {s['name']}: start > end")
        if s.get("type") not in ("TM", "loop", "soluble", "intramembrane", "signal"):
            errs.append(f"segment {s['name']}: unknown type {s.get('type')!r}")
    for a, b in zip(segs, segs[1:]):
        if b["start"] <= a["end"]:
            errs.append(f"segments {a['name']} and {b['name']} overlap")
        elif b["start"] > a["end"] + 1:
            errs.append(f"gap between {a['name']} (ends {a['end']}) and {b['name']} (starts {b['start']})")
    if region:
        lo, hi = region
        if segs[0]["start"] > lo or segs[-1]["end"] < hi:
            errs.append(f"segments cover {segs[0]['start']}-{segs[-1]['end']}, region is {lo}-{hi}")
    # alternating orientation of consecutive TMs, starting from n_terminus
    side = "cytosolic" if cfg.get_path("topology.n_terminus", "cytosolic") == "cytosolic" else "lumenal"
    for s in segs:
        if s.get("type") == "TM":
            want = "in_out" if side == "cytosolic" else "out_in"
            if s.get("orientation") and s["orientation"] != want:
                errs.append(
                    f"{s['name']}: orientation {s['orientation']} but preceding side is {side} "
                    f"(expected {want}) - check n_terminus / segment list"
                )
            side = "lumenal" if (s.get("orientation") or want) == "in_out" else "cytosolic"
        elif s.get("type") in ("loop", "soluble") and s.get("side") and s["side"] != side:
            errs.append(f"{s['name']}: side {s['side']} but alternation implies {side}")
    return errs


def validate(cfg: Config, df=None, strict: bool = True) -> list[str]:
    """Validate config (and the loaded source table, if given).

    Hard errors raise ConfigError when strict; the returned list holds warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    src = cfg.resolve(cfg.get_path("source.path"))
    if src is None or not src.exists():
        errors.append(f"source.path not found: {src}")

    seq = protein_sequence(cfg)
    if df is not None:
        if df["score_raw"].isna().all():
            errors.append("score column is entirely NaN")
        reps = [c for c in df.columns if c.startswith("rep") and c[3:].isdigit()]
        if len(reps) < 1:
            warnings.append("no replicate columns detected")
        # Every topology-dependent analysis silently inherits whatever is in the config,
        # so say out loud when that is still the hydropathy guess `init` wrote.
        topo_src = str(cfg.get_path("topology.source", "") or "")
        n_tm = len([x for x in (cfg.get_path("topology.segments") or []) if x.get("type") == "TM"])
        if n_tm and "tmhmm" not in topo_src.lower() and "uniprot" not in topo_src.lower():
            warnings.append(f"topology source is {topo_src or 'unset'!r}, not TMHMM/UniProt - the "
                            f"{n_tm} TM helices used by A04-A08, A13, A14, A18 and the heatmap tracks "
                            "are a hydropathy guess; run `python -m mpdms tmhmm <config> --from-file <gff3>`")
        if seq:
            wt = df.drop_duplicates("pos").set_index("pos")["wt"]
            bad = [(p, w, seq[p - 1] if 0 < p <= len(seq) else "?")
                   for p, w in wt.items() if not (0 < p <= len(seq)) or seq[p - 1] != w]
            if bad:
                frac = len(bad) / max(len(wt), 1)
                msg = (f"WT residue disagrees with FASTA at {len(bad)} positions "
                       f"({frac:.0%}); first: {bad[:5]} - check source.numbering_offset")
                (errors if frac > 0.02 else warnings).append(msg)
        region = (int(df["pos"].min()), int(df["pos"].max()))
    else:
        region = cfg.get_path("protein.region")
    errors += validate_topology(cfg, tuple(cfg.get_path("protein.region") or region) if region else None)

    if errors and strict:
        raise ConfigError(f"[{cfg.id}] invalid config:\n  - " + "\n  - ".join(errors))
    return warnings + errors
