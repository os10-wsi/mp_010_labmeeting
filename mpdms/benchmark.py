"""Benchmark harness: splits, baselines, pluggable models, metrics.

The unit of evaluation is a variant with a measured `score_z`. Models are anything with
`name`, optional `fit(train_df)` and `predict(test_df) -> array`. That covers three cases
without the harness caring which is which:

  * feature baselines fitted here (ridge / gradient boosting on biophysical features);
  * zero-shot scores computed elsewhere and supplied as a CSV (ESM-1v, SaProt, ESM3,
    any ΔG predictor) via `ScoreColumn`;
  * a head fitted on precomputed per-variant embeddings (`EmbeddingHead`) - a real,
    runnable fine-tune of the readout layer when full back-propagation through a large
    model is not practical.

Splits decide what question is being asked:
  random    - interpolation within the measured set; the easiest, and the least informative
  position  - all variants at a position are held out together (no leakage through the
              site mean); tests prediction at unseen sites
  protein   - leave one protein out; the only split that tests extrapolation to a protein
              the model has never seen, and therefore the one that supports any claim
              about other membrane proteins
  family    - leave one family of paralogs out; `protein` leaks when close paralogs are in
              the set, because the held-out protein is nearly present in training
  segment   - train on loops/soluble, test on TM (and the reverse); tests whether the
              model has learnt membrane-specific physics or just generic constraint
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

from .features import FEATURES

DELETERIOUS = -0.5  # score_z below this counts as a clear loss of biogenesis


# ------------------------------------------------------------------- splits
def split_folds(df: pd.DataFrame, how: str = "protein", n_folds: int = 5, seed: int = 0):
    """Yield (name, train_index, test_index)."""
    rng = np.random.default_rng(seed)
    if how == "protein":
        for g in sorted(df.dataset_id.unique()):
            m = df.dataset_id == g
            if m.sum() and (~m).sum():
                yield g, df.index[~m], df.index[m]
    elif how == "position":
        key = df.dataset_id + ":" + df.pos.astype(str)
        groups = np.array(sorted(key.unique()))
        assign = dict(zip(groups, rng.integers(0, n_folds, len(groups))))
        f = key.map(assign)
        for k in range(n_folds):
            yield f"fold{k + 1}", df.index[f != k], df.index[f == k]
    elif how == "random":
        f = rng.integers(0, n_folds, len(df))
        for k in range(n_folds):
            yield f"fold{k + 1}", df.index[f != k], df.index[f == k]
    elif how == "family":
        # Close paralogs are not independent: training on HXT1/2/3 and testing HXT7 asks the
        # model to recognise a protein it has nearly seen. Holding out the whole family is
        # the split that supports a claim about a protein the model has not met.
        if "family" not in df.columns:
            raise ValueError("split 'family' needs a 'family' column (see assign_families)")
        for g in sorted(df.family.unique()):
            m = df.family == g
            if m.sum() and (~m).sum():
                yield str(g), df.index[~m], df.index[m]
    elif how == "segment":
        tm = df.seg_type == "TM"
        yield "train_nonTM_test_TM", df.index[~tm], df.index[tm]
        yield "train_TM_test_nonTM", df.index[tm], df.index[~tm]
    else:
        raise ValueError(f"unknown split {how!r}")


# ------------------------------------------------------------------- models
@dataclass
class Baseline:
    """Ridge or gradient boosting on the biophysical feature table."""
    kind: str = "ridge"
    features: list = field(default_factory=lambda: list(FEATURES))
    name: str = ""

    def __post_init__(self):
        self.name = self.name or f"features_{self.kind}"

    def _X(self, d):
        X = d.reindex(columns=self.features).astype(float)
        return X.fillna(self._med if hasattr(self, "_med") else X.median(numeric_only=True)).to_numpy()

    def fit(self, train):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self._med = train.reindex(columns=self.features).astype(float).median(numeric_only=True)
        y = train.score_z.to_numpy()
        self.m = (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
                  if self.kind == "ridge" else
                  HistGradientBoostingRegressor(max_depth=3, learning_rate=0.06, max_iter=400,
                                                l2_regularization=1.0, random_state=0))
        self.m.fit(self._X(train), y)
        return self

    def predict(self, test):
        return self.m.predict(self._X(test))


@dataclass
class SiteMean:
    """Mean of the other variants at the same position. Not usable across proteins or
    positions - included to show how much of a random split is explained by site identity."""
    name: str = "site_mean"

    def fit(self, train):
        self.mu = train.groupby(["dataset_id", "pos"]).score_z.mean()
        self.g = train.score_z.mean()
        return self

    def predict(self, test):
        k = pd.MultiIndex.from_arrays([test.dataset_id, test.pos])
        return self.mu.reindex(k).fillna(self.g).to_numpy()


@dataclass
class ScoreColumn:
    """A precomputed score already merged into the table (e.g. `esm1v`), or supplied as a CSV
    with columns uniprot/gene, pos, wt, mut, score. `sign`: +1 if higher means more tolerated,
    -1 if higher means more damaging, 'auto' to orient on the training fold."""
    column: str = "esm1v"
    sign: object = 1
    name: str = ""

    def __post_init__(self):
        self.name = self.name or f"{self.column}_zeroshot"
        self._s = None

    def fit(self, train):
        if self.sign == "auto":
            ok = train[self.column].notna()
            r = ss.spearmanr(train.loc[ok, self.column], train.loc[ok, "score_z"])[0] if ok.sum() > 20 else 1
            self._s = float(np.sign(r)) or 1.0
        return self

    def predict(self, test):
        if self.column not in test:
            raise KeyError(f"no '{self.column}' scores for this evaluation set")
        s = self._s if self._s is not None else (1.0 if self.sign == "auto" else float(self.sign))
        v = test[self.column].to_numpy(float)
        cov = np.isfinite(v).mean()
        if cov < 0.2:
            raise ValueError(f"'{self.column}' covers only {cov:.0%} of these variants")
        return s * np.where(np.isfinite(v), v, np.nanmedian(v))


@dataclass
class EmbeddingHead:
    """Ridge (or small MLP) head on precomputed per-variant embeddings: `embeddings` maps
    (dataset_id, pos, mut) -> vector, loaded from an .npz written by the user's model."""
    embeddings: dict = field(default_factory=dict)
    kind: str = "ridge"
    name: str = "embedding_head"

    def _X(self, d):
        dim = len(next(iter(self.embeddings.values())))
        keys = [(r.dataset_id, int(r.pos), r.mut) for r in d.itertuples()]
        cov = np.mean([k in self.embeddings for k in keys])
        if cov < 0.2:
            raise ValueError(f"embeddings cover only {cov:.0%} of these variants")
        return np.array([self.embeddings.get(k, np.zeros(dim)) for k in keys])

    def fit(self, train):
        from sklearn.linear_model import RidgeCV
        from sklearn.neural_network import MLPRegressor
        self.m = (RidgeCV(alphas=np.logspace(-2, 4, 13)) if self.kind == "ridge" else
                  MLPRegressor((128,), max_iter=400, early_stopping=True, random_state=0))
        self.m.fit(self._X(train), train.score_z.to_numpy())
        return self

    def predict(self, test):
        return self.m.predict(self._X(test))


def load_score_csv(path: Path, df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Merge an external per-variant score file onto the benchmark table."""
    e = pd.read_csv(path)
    e.columns = [c.strip().lower() for c in e.columns]
    sc = next((c for c in e.columns if c in (column, "score", "pred", "ddg", "dg", "value")), None)
    key = "uniprot" if "uniprot" in e.columns and df.uniprot.notna().any() else "gene"
    if key not in e.columns:
        e[key] = df[key].dropna().iloc[0]  # single-protein file
    if sc is None:
        raise ValueError(f"{path}: no score column found in {list(e.columns)}")
    e = e[[key, "pos", "mut", sc]].rename(columns={sc: column})
    e["pos"] = e.pos.astype(int)
    return df.merge(e, on=[key, "pos", "mut"], how="left", suffixes=("", "_new"))


# ------------------------------------------------------------------ metrics
def metrics(y, p, deleterious: float = DELETERIOUS) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    out = {"n": len(y)}
    if len(y) < 10:
        return out
    # a prediction with no variance carries no ranking information: score it 0, not NaN,
    # so an uninformative model appears in the summary instead of silently dropping out
    if np.ptp(p) == 0 or np.ptp(y) == 0:
        out.update(spearman=0.0, pearson=0.0, constant_prediction=bool(np.ptp(p) == 0))
    else:
        out["spearman"] = float(ss.spearmanr(y, p)[0])
        out["pearson"] = float(ss.pearsonr(y, p)[0])
        out["constant_prediction"] = False
    out["rmse"] = float(np.sqrt(np.mean((y - p) ** 2)))
    lab = y < deleterious
    if 0 < lab.sum() < len(lab):
        from sklearn.metrics import average_precision_score, roc_auc_score
        out["auroc_deleterious"] = float(roc_auc_score(lab, -p))
        out["auprc_deleterious"] = float(average_precision_score(lab, -p))
        out["base_rate_deleterious"] = float(lab.mean())
    k = min(50, max(10, len(y) // 20))
    out[f"precision_at_{k}"] = float(lab[np.argsort(p)[:k]].mean())
    return out


def evaluate(df: pd.DataFrame, models, split: str = "protein", n_folds: int = 5,
             seed: int = 0, deleterious: float = DELETERIOUS) -> pd.DataFrame:
    """Fit and score every model on every fold. Returns one row per (model, fold)."""
    rows = []
    folds = list(split_folds(df.reset_index(drop=True), split, n_folds, seed))
    d = df.reset_index(drop=True)
    for mk in models:
        for fname, tr, te in folds:
            train, test = d.loc[tr], d.loc[te]
            m = mk() if callable(mk) and not hasattr(mk, "predict") else mk
            try:
                if hasattr(m, "fit"):
                    m.fit(train)
                pred = np.asarray(m.predict(test), float)
                r = metrics(test.score_z, pred, deleterious)
                r.update(model=m.name, split=split, fold=fname, n_train=len(train),
                         proteins_test=",".join(sorted(test.dataset_id.unique())))
            except Exception as e:
                r = {"model": getattr(m, "name", str(m)), "split": split, "fold": fname,
                     "error": f"{e.__class__.__name__}: {e}"}
            rows.append(r)
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in res.columns if res[c].dtype.kind == "f" and c != "n"]
    g = res.groupby(["model", "split"])[num].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index().sort_values("spearman_mean", ascending=False)


def default_models(df: pd.DataFrame) -> list:
    """Baselines worth beating. ESM-1v is included only if scores are present."""
    ms = [Baseline("ridge"), Baseline("gbm"), SiteMean()]
    if "esm1v" in df and df.esm1v.notna().mean() > 0.5:
        ms.append(ScoreColumn("esm1v", sign="auto"))
        ms.append(Baseline("gbm", features=list(FEATURES) + ["esm1v"], name="features+esm1v_gbm"))
    return ms


# -------------------------------------------------------------- paralog groups
def assign_families(seqs: dict[str, str], threshold: float = 0.40) -> dict[str, str]:
    """Group datasets whose sequences are at least `threshold` identical.

    Leave-one-protein-out quietly becomes leave-nothing-out when the set contains close
    paralogs, so the groups are derived from the sequences rather than from the names:
    a name can be misleading, and 40% identity over a global alignment is comfortably
    above what unrelated membrane transporters reach by chance.
    """
    from Bio import Align
    ids = sorted(seqs)
    al = Align.PairwiseAligner(scoring="blastp", mode="global")
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ident = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sa, sb = seqs[a], seqs[b]
            if not sa or not sb:
                continue
            aln = al.align(sa, sb)[0]
            same = sum(x == y for x, y in zip(*[str(r) for r in (aln[0], aln[1])]) if x != "-")
            frac = same / max(1, min(len(sa), len(sb)))
            ident[(a, b)] = frac
            if frac >= threshold:
                parent[find(a)] = find(b)
    groups: dict[str, list] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    out = {}
    for members in groups.values():
        label = "+".join(sorted(members)) if len(members) > 1 else members[0]
        for m in members:
            out[m] = label
    return out, ident


def learning_curve(df: pd.DataFrame, model_factory, group: str = "family",
                   repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """How much does another training protein buy? Held-out group, growing training set.

    For each held-out group and each training size k, k other groups are drawn at random,
    the model is fitted on them and scored on the held-out one. This is the number that
    says whether measuring a twelfth protein would help or whether the curve has flattened.
    """
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True)
    if group not in d.columns:
        raise ValueError(f"learning_curve needs a {group!r} column")
    groups = sorted(d[group].unique())
    rows = []
    for held in groups:
        others = [g for g in groups if g != held]
        test = d[d[group] == held]
        if len(test) < 50 or not others:
            continue
        for k in range(1, len(others) + 1):
            draws = 1 if k == len(others) else repeats
            for r in range(draws):
                pick = list(rng.choice(others, size=k, replace=False))
                train = d[d[group].isin(pick)]
                m = model_factory()
                try:
                    m.fit(train)
                    met = metrics(test.score_z, np.asarray(m.predict(test), float))
                except Exception as e:
                    met = {"error": f"{type(e).__name__}: {e}"[:80]}
                met.update(held_out=held, k_train_groups=k, repeat=r,
                           n_train=len(train), model=getattr(m, "name", "model"))
                rows.append(met)
    return pd.DataFrame(rows)
