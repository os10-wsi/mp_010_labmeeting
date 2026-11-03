"""Everything the test suite needs that is generated rather than committed.

    python tests/bootstrap_fixtures.py

`data/` and `configs/SYN*.yaml` are git-ignored, so a fresh checkout - a new CI runner,
a rebuilt cloud container - has none of them, and the tests that reach for a fixture
fail on the missing file rather than on the thing they exist to check. Three steps, in
this order, because each needs the one before it:

    1. the synthetic datasets and their fake structures   (tests/make_synthetic.py)
    2. a config per dataset                               (mpdms init --offline)
    3. synthetic ESM-1v scores for SYN1, and the config key pointing at them

Step 3 is separate because `mpdms init` never writes `evolution.esm_scores` - only
`mpdms esm` does, and that needs the real ESM-1v weights, a GPU's worth of patience and
the network. The scores written here are not ESM's: they are a plausible stand-in with
the right shape and column names, so the code paths that consume ESM scores (a15, e06,
b10) are exercised rather than skipped. No claim is made about them, and nothing that
reports a number about a real protein ever reads this file.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SETS = ["SYN1", "SYN2", "SYN3"]
ESM_FOR = "SYN1"          # the fixture the ESM-dependent tests expect to find scores for


def run(*cmd: str) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def synthetic_esm(gene: str, seed: int = 11) -> Path:
    """An ESM-1v-shaped score table for one synthetic protein.

    Correlated with the simulated fitness, because a predictor uncorrelated with the
    measurement would make every analysis that uses it test nothing, and noisy enough
    that the residual analyses have residuals to find. Log-likelihood ratios are
    negative and roughly in [-20, 0], which is the range the readers expect.
    """
    src = ROOT / "data/raw" / gene / "fitness_estimation/fitness_estimation.tsv"
    raw = pd.read_csv(src, sep="\t")
    # the lab format already has its own `wt` column - a 0/1 flag - so the residue
    # columns are picked out by name rather than renamed in place, which would leave
    # two columns called `wt` and turn every later `d.wt` into a DataFrame
    fit = next((c for c in ("rescaled_fitness_rep1", "raw_fitness_rep1", "fitness")
                if c in raw.columns), None)
    if fit is None:
        raise SystemExit(f"{src}: no fitness column to build a stand-in predictor from")
    d = pd.DataFrame({"pos": raw["pos"], "wt": raw["wt aa"], "mut": raw["mut aa"],
                      "fit": pd.to_numeric(raw[fit], errors="coerce")})
    d = d[d.pos.notna() & d.wt.notna() & d.mut.notna()].reset_index(drop=True)
    d["pos"] = d.pos.astype(int)
    y = d.fit.to_numpy(float)
    y = np.where(np.isfinite(y), y, np.nanmedian(y))
    rng = np.random.default_rng(seed)
    z = (y - y.mean()) / (y.std(ddof=1) or 1.0)
    llr = -6.0 + 3.0 * z + rng.normal(0, 2.5, len(d))
    out = ROOT / "data/external" / gene / f"{gene}_esm1v.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pos": d.pos.to_numpy(), "wt": d.wt.to_numpy(),
                  "mut": d.mut.to_numpy(),
                  "esm1v": np.clip(llr, -20.0, 0.0).round(4)}).to_csv(out, index=False)
    return out


def point_config_at(gene: str, path: Path) -> None:
    """Set `evolution.esm_scores` in the generated config, after init has written it.

    Edited as text rather than round-tripped through yaml, so the comments `init` puts
    in the file - including the one warning that the topology is predicted - survive.
    """
    cfg = ROOT / "configs" / f"{gene}.yaml"
    rel = path.relative_to(ROOT).as_posix()
    text = cfg.read_text()
    if "  esm_scores:" not in text:
        raise SystemExit(f"{cfg}: no evolution.esm_scores key to set")
    lines = [f"  esm_scores: {rel}" if l.startswith("  esm_scores:") else l
             for l in text.splitlines()]
    cfg.write_text("\n".join(lines) + "\n")


def main() -> int:
    run(sys.executable, "tests/make_synthetic.py")
    run(sys.executable, "-m", "mpdms", "init",
        *[f"data/raw/{g}" for g in SETS], "--offline", "--force")
    p = synthetic_esm(ESM_FOR)
    point_config_at(ESM_FOR, p)
    print(f"  wrote {p.relative_to(ROOT)} and pointed configs/{ESM_FOR}.yaml at it")
    print("\n  fixtures ready: " + ", ".join(f"configs/{g}.yaml" for g in SETS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
