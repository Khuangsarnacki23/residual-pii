"""Stratified random sampling with at most one conversation per user proxy.

Estimand note: where a user proxy exists, one conversation is drawn per user *before*
stratification, so weighted prevalence is closer to "share of users" than "share of
conversations". lmsys-chat-1m has no proxy, so its sample is per-conversation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def turn_bucket(n_turns: pd.Series, edges: list[int]) -> pd.Series:
    edges = sorted(edges)
    labels = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] - 1 if i + 1 < len(edges) else None
        labels.append(str(lo) if hi == lo else (f"{lo}-{hi}" if hi else f"{lo}+"))
    bins = [*edges, np.inf]
    return pd.cut(n_turns.clip(lower=edges[0]), bins=bins, right=False, labels=labels).astype(str)


def eligible(meta: pd.DataFrame, languages: list[str] | None) -> pd.DataFrame:
    df = meta[meta["n_turns"] > 0]
    if languages:
        df = df[df["language"].isin(languages)]
    return df.reset_index(drop=True)


def stratified_sample(meta: pd.DataFrame, n: int, cfg: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    """Returns (sample with stratum + design weight, summary counts)."""
    rng = np.random.default_rng(seed)
    scfg = cfg["sampling"]
    df = eligible(meta, cfg.get("languages"))
    summary = {"eligible_conversations": len(df)}

    has_users = df["user_key"].notna().any()
    if scfg.get("one_per_user") and has_users:
        # Shuffle, then keep the first conversation per user: a uniformly random pick per user.
        df = df.iloc[rng.permutation(len(df))]
        no_user = df[df["user_key"].isna()]
        df = pd.concat([df[df["user_key"].notna()].drop_duplicates("user_key"), no_user])
        summary["one_per_user_applied"] = True
        summary["eligible_users"] = int(df["user_key"].nunique())
        summary["conversations_without_user_proxy"] = len(no_user)
    else:
        summary["one_per_user_applied"] = False
    summary["sampling_frame"] = len(df)

    df = df.copy()
    df["month"] = (df["timestamp"].dt.strftime("%Y-%m") if df["timestamp"].notna().any() else "na")
    df["month"] = df["month"].fillna("na")
    df["turn_bucket"] = turn_bucket(df["n_turns"], scfg["turn_bucket_edges"])
    strata_cols = [c for c in scfg["strata"] if c in df and not (df[c] == "na").all()]
    df["stratum"] = df[strata_cols].astype(str).agg("|".join, axis=1) if strata_cols else "all"
    summary["strata_columns"] = strata_cols

    n = min(n, len(df))
    alloc = _proportional_allocation(df["stratum"].value_counts().sort_index(), n)
    parts = []
    for stratum, k in alloc.items():
        if k == 0:
            continue
        pool = df[df["stratum"] == stratum]
        take = pool.iloc[rng.choice(len(pool), size=k, replace=False)].copy()
        take["weight"] = len(pool) / k
        parts.append(take)
    sample = pd.concat(parts).sort_values(["_file", "_rg", "_row"]).reset_index(drop=True)
    summary["sample_size"] = len(sample)
    summary["n_strata"] = int((alloc > 0).sum())
    return sample, summary


def _proportional_allocation(sizes: pd.Series, n: int) -> pd.Series:
    """Largest-remainder proportional allocation, guaranteeing k <= stratum size."""
    exact = sizes / sizes.sum() * n
    alloc = np.floor(exact).astype(int).clip(upper=sizes)
    remainder = (exact - np.floor(exact)).sort_values(ascending=False)
    for stratum in remainder.index:
        if alloc.sum() >= n:
            break
        if alloc[stratum] < sizes[stratum]:
            alloc[stratum] += 1
    return alloc
