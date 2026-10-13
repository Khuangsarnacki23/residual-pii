"""How many users are singled out by the quasi-identifiers they revealed.

For a combination of profile dimensions, users who have a usable value for every dimension in it
fall into equivalence classes of size k (users sharing the same values). k = 1 means the
combination is unique to one user *within this sample*.

Within-sample uniqueness overstates population uniqueness: someone unique among 2,000 sampled users
is often not unique among all users of the service. The tables therefore also report uniqueness at
several sample sizes, so the trend is visible; see Rocher, Hendrickx & de Montjoye (2019) for the
population-level argument. `population_estimate` adds a model-based estimate of uniqueness among
all eligible users. Nothing here links a profile to a real identity.

Every output is a share or a count of users over dimension *names*; no profile value is written.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .profiles import DIMENSIONS, known_values
from .sampling import ACTIVITY_LABELS

GROUP = ["dataset", "variant", "level"]


def combos(max_size: int = 4, dims: tuple[str, ...] = DIMENSIONS) -> list[tuple[str, ...]]:
    return [c for r in range(1, max_size + 1) for c in combinations(dims, r)]


def class_sizes(known: pd.DataFrame, combo: tuple[str, ...]) -> np.ndarray:
    """Equivalence-class size k per user for one combination; NaN where the user lacks any of it."""
    has = known[list(combo)].notna().all(axis=1)
    k = pd.Series(np.nan, index=known.index)
    if has.any():
        k[has] = known[has].groupby(list(combo))[combo[0]].transform("size")
    return k.to_numpy()


def k_matrix(known: pd.DataFrame, all_combos: list[tuple[str, ...]]) -> np.ndarray:
    """users x combinations matrix of class sizes."""
    return np.column_stack([class_sizes(known, c) for c in all_combos]) if len(known) else np.empty((0, len(all_combos)))


def unique_upto(K: np.ndarray, all_combos: list[tuple[str, ...]], max_size: int) -> np.ndarray:
    """users x max_size 0/1 matrix: unique on at least one combination of at most n dimensions."""
    sizes = np.array([len(c) for c in all_combos])
    return np.column_stack([(K[:, sizes <= n] == 1).any(axis=1) for n in range(1, max_size + 1)]).astype(float)


def _boot_weights(n: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap over users as multinomial resampling counts (reps x users). Class sizes are kept as
    observed, so the interval reflects which users were drawn, not re-formed classes."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=reps).astype(float)


def _ci(num: np.ndarray, den: np.ndarray, level: float) -> tuple[float, float]:
    ok = den > 0
    if not ok.any():
        return np.nan, np.nan
    a = (1 - level) / 2
    lo, hi = np.quantile(num[ok] / den[ok], [a, 1 - a])
    return float(lo), float(hi)


def _blank(row: dict, keep: tuple[str, ...]) -> dict:
    """Small-cell suppression: keep only the labels, blank every count and statistic."""
    return {k: (v if k in keep else np.nan) for k, v in row.items()} | {"suppressed": True}


def analyze(profiles: pd.DataFrame, ucfg: dict, acfg: dict, seed: int,
            min_cell: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (by_combination, by_activity, summary); all three are aggregate and suppressed."""
    max_size = int(ucfg.get("max_combination_size", 4))
    reps, level_ci = int(acfg["bootstrap_reps"]), float(acfg["ci_level"])
    by_combo, by_act, summary = [], [], []
    for (ds, variant, level), g in profiles.groupby(GROUP):
        all_combos = combos(max_size)
        rng = np.random.default_rng(seed)
        g = g.reset_index(drop=True)
        n = len(g)
        labels = {"dataset": ds, "variant": variant, "level": level}
        known = known_values(g)
        K = k_matrix(known, all_combos)
        W = _boot_weights(n, reps, rng)
        has = ~np.isnan(K)

        for j, combo in enumerate(all_combos):
            m = int(has[:, j].sum())
            row = {**labels, "combination": "+".join(combo), "n_qi": len(combo), "n_users": m,
                   "share_k1": np.nan, "share_k1_ci_low": np.nan, "share_k1_ci_high": np.nan,
                   "share_k_lt5": np.nan, "share_k_lt5_ci_low": np.nan, "share_k_lt5_ci_high": np.nan,
                   "median_k": np.nan, "suppressed": False}
            if m < min_cell:
                by_combo.append(_blank(row, (*GROUP, "combination", "n_qi")))
                continue
            k = K[has[:, j], j]
            den = W @ has[:, j].astype(float)
            for name, hit in (("share_k1", K[:, j] == 1), ("share_k_lt5", K[:, j] < 5)):
                row[name] = float(hit.sum() / m)
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = _ci(W @ hit.astype(float), den, level_ci)
            row["median_k"] = float(np.median(k))
            by_combo.append(row)

        U = unique_upto(K, all_combos, max_size)
        n_known = known.notna().sum(axis=1).to_numpy()

        def share_rows(mask: np.ndarray, extra: dict, keep: tuple[str, ...]) -> list[dict]:
            m = int(mask.sum())
            out = []
            for q in range(max_size):
                row = {**labels, **extra, "n_qi": q + 1, "n_users": m, "share_unique": np.nan,
                       "ci_low": np.nan, "ci_high": np.nan,
                       "mean_known_attributes": float(n_known[mask].mean()) if m else np.nan, "suppressed": False}
                if m < min_cell:
                    out.append(_blank(row, (*GROUP, *keep, "n_qi")))
                    continue
                row["share_unique"] = float(U[mask, q].mean())
                row["ci_low"], row["ci_high"] = _ci(W[:, mask] @ U[mask, q], W[:, mask].sum(axis=1), level_ci)
                out.append(row)
            return out

        # Headline: share of all sampled users unique on >= 1 combination of at most n_qi dimensions.
        everyone = np.ones(n, dtype=bool)
        summary += share_rows(everyone, {"sample_users": n, "ci_kind": "bootstrap_users"}, ("ci_kind",))
        for b in ACTIVITY_LABELS:
            mask = (g["activity_bucket"] == b).to_numpy()
            if mask.any():
                by_act += share_rows(mask, {"activity_bucket": b}, ("activity_bucket",))

        # Trend with sample size: recompute classes inside random subsamples of users.
        draws = int(ucfg.get("subsample_draws", 20))
        for size in sorted(ucfg.get("sample_sizes") or []):
            if not min_cell <= size < n:
                continue
            shares = np.empty((draws, max_size))
            for d in range(draws):
                idx = rng.choice(n, size=size, replace=False)
                shares[d] = unique_upto(k_matrix(known.iloc[idx], all_combos), all_combos, max_size).mean(axis=0)
            a = (1 - level_ci) / 2
            for q in range(max_size):
                summary.append({**labels, "sample_users": size, "ci_kind": "subsample_draws", "n_qi": q + 1,
                                "n_users": size, "share_unique": float(shares[:, q].mean()),
                                "ci_low": float(np.quantile(shares[:, q], a)),
                                "ci_high": float(np.quantile(shares[:, q], 1 - a)),
                                "mean_known_attributes": np.nan, "suppressed": False})

    def frame(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        for c in ("n_users", "sample_users"):
            if c in df:
                df[c] = df[c].astype("Int64")
        return df

    return frame(by_combo), frame(by_act), frame(summary)


# ---------------------------------------------------------------- population estimate

# Dimensions that largely determine each other (a region implies its country, a school an education
# level) are modelled as one joint factor; treating them as independent would count the same fact twice.
JOINT = (("region", "country", "nationality", "language_native"),
         ("occupation_group", "employer_type"), ("education_level", "school"))
POPULATION_COLUMNS = [*GROUP, "subset", "n_qi", "sample_users", "target_users", "share_unique_model",
                      "ci_low", "ci_high", "share_unique_observed"]


def _factors(combo: tuple[str, ...]) -> list[tuple[str, ...]]:
    joint = [tuple(d for d in combo if d in group) for group in JOINT]
    return [j for j in joint if j] + [(d,) for d in combo if not any(d in group for group in JOINT)]


def _cell_codes(known: pd.DataFrame, dims: tuple[str, ...]) -> np.ndarray:
    """Cell index per user for one factor; -1 where the user lacks a usable value for it."""
    has = known[list(dims)].notna().all(axis=1).to_numpy()
    codes = np.full(len(known), -1)
    if has.any():
        codes[has] = known[has].groupby(list(dims)).ngroup().to_numpy()
    return codes


def _model_shares(codes: dict, all_combos: list, w: np.ndarray, targets: list[int], max_size: int,
                  prior: float) -> np.ndarray:
    """targets x max_size: modelled share of users unique among `target` users.

    The chance that another user matches a user on a combination is the product, over the
    combination's factors, of the share of *other* users in the same cell (leave-one-out, with
    `prior` pseudo-users so a cell seen once is not given probability zero). A user is unique among
    M users when none of the other M - 1 match; over several combinations the most identifying one
    is taken."""
    total, n = w.sum(), len(w)
    log_pi = {}
    for f, c in codes.items():
        ok = c >= 0
        size = np.bincount(c[ok], weights=w[ok])[c[ok]] if ok.any() else np.empty(0)
        lp = np.full(n, np.nan)
        # A resampled user's own copies are not other people, so all of them are left out.
        lp[ok] = np.log(np.clip(size - np.maximum(w[ok], 1), 0, None) + prior) - np.log(total)
        log_pi[f] = lp
    L = np.column_stack([sum(log_pi[f] for f in _factors(c)) for c in all_combos])
    miss = np.log1p(-np.exp(np.minimum(L, -1e-12)))          # log P(one other user does not match)
    sizes = np.array([len(c) for c in all_combos])
    out = np.empty((len(targets), max_size))
    for t, m in enumerate(targets):
        P = np.nan_to_num(np.exp((m - 1) * miss), nan=0.0)   # no usable value: never unique on it
        for q in range(max_size):
            out[t, q] = (w * P[:, sizes <= q + 1].max(axis=1)).sum() / total
    return out


def population_estimate(profiles: pd.DataFrame, summary: pd.DataFrame, population: dict, ucfg: dict,
                        acfg: dict, seed: int, min_cell: int = 10) -> pd.DataFrame:
    """Modelled share of users unique among all eligible users of a dataset (`population[dataset]`),
    and, as a check on the model, the share it predicts at the sample sizes where uniqueness was
    observed. Follows the logic of Rocher et al. (2019) with a simpler model: dimensions are
    independent of each other, except the groups in JOINT, which are modelled jointly."""
    max_size = int(ucfg.get("max_combination_size", 4))
    prior = float(ucfg.get("population_prior", 0.5))
    reps, a = int(ucfg.get("population_bootstrap_reps", 200)), (1 - float(acfg["ci_level"])) / 2
    rows = []
    for (ds, variant, level), g in profiles.groupby(GROUP):
        n = len(g)
        if n < min_cell:
            continue
        rng = np.random.default_rng(seed)
        all_combos = combos(max_size)
        known = known_values(g.reset_index(drop=True))
        codes = {f: _cell_codes(known, f) for c in all_combos for f in _factors(c)}
        sizes = sorted({int(x) for x in ucfg.get("sample_sizes") or [] if min_cell <= x < n} | {n})
        big = int(population.get(ds) or 0)
        targets = sizes + ([big] if big > n else [])
        point = _model_shares(codes, all_combos, np.ones(n), targets, max_size, prior)
        boot = np.stack([_model_shares(codes, all_combos, w, targets, max_size, prior)
                         for w in _boot_weights(n, reps, rng)])
        lo, hi = np.quantile(boot, [a, 1 - a], axis=0)
        for t, m in enumerate(targets):
            for q in range(max_size):
                rows.append({"dataset": ds, "variant": variant, "level": level,
                             "subset": "population" if m > n else "model_at_sample_size", "n_qi": q + 1,
                             "sample_users": n, "target_users": m, "share_unique_model": float(point[t, q]),
                             "ci_low": float(lo[t, q]), "ci_high": float(hi[t, q])})
    out = pd.DataFrame(rows, columns=POPULATION_COLUMNS[:-1])
    obs = (summary[summary["share_unique"].notna()]
           .rename(columns={"sample_users": "target_users", "share_unique": "share_unique_observed"})
           [[*GROUP, "n_qi", "target_users", "share_unique_observed"]])
    out = out.merge(obs.astype({"target_users": "int64"}), on=[*GROUP, "n_qi", "target_users"], how="left")
    return out.astype({"sample_users": "Int64", "target_users": "Int64"})
