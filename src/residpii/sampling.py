"""Sampling. Two modes (`sampling.mode`):

per_user_one_conv  Stratified random sample with at most one conversation per user proxy.
    Estimand note: where a user proxy exists, one conversation is drawn per user *before*
    stratification, so weighted prevalence is closer to "share of users" than "share of
    conversations". lmsys-chat-1m has no proxy, so its sample is per-conversation.

user_level  A uniform sample of *users* with enough conversations, keeping their conversations
    (up to a cap) together, for per-user reporting. The only link between
    conversations is the dataset's own user proxy, already replaced by a keyed hash (`user_key`).
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
    summary = {"mode": "per_user_one_conv", "eligible_conversations": len(df)}

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


# Conversations-per-user buckets, counted before the per-user cap is applied.
ACTIVITY_EDGES = [1, 2, 3, 4, 6, 11, 26, 51]
ACTIVITY_LABELS = ["1", "2", "3", "4-5", "6-10", "11-25", "26-50", ">50"]


class NoUserProxy(ValueError):
    """Raised when user-level sampling is asked of a dataset that has no user proxy."""


def activity_bucket(n_conversations: pd.Series) -> pd.Series:
    return pd.cut(n_conversations, bins=[*ACTIVITY_EDGES, np.inf], right=False,
                  labels=ACTIVITY_LABELS).astype(str)


def _user_counts(meta: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """(eligible conversations that have a user proxy, conversations per user sorted by user_key)."""
    df = eligible(meta, cfg.get("languages"))
    df = df[df["user_key"].notna()]
    if df.empty:
        raise NoUserProxy("no conversation has a user proxy")
    return df, df.groupby("user_key", sort=True).size()


def user_activity(meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Users by conversations-per-user bucket. Bucketed counts only; no user keys."""
    _, counts = _user_counts(meta, cfg)
    n = activity_bucket(counts).value_counts().reindex(ACTIVITY_LABELS, fill_value=0)
    min_conv = cfg["user_level"]["min_conversations"]
    return pd.DataFrame({"activity_bucket": ACTIVITY_LABELS, "n_users": n.to_numpy(),
                         "share_users": n.to_numpy() / max(len(counts), 1),
                         "meets_min_conversations": [lo >= min_conv for lo in ACTIVITY_EDGES]})


def user_level_sample(meta: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    """Uniform sample of users with >= min_conversations, each with up to
    max_conversations_per_user of their conversations (a seeded random subset above the cap).

    The conversation weight is (eligible users / sampled users) * (user's conversations / kept), so
    conversation-level tables still estimate totals over the conversations of eligible users."""
    rng = np.random.default_rng(seed)
    ucfg = cfg["user_level"]
    min_conv, cap = int(ucfg["min_conversations"]), int(ucfg["max_conversations_per_user"])
    df, counts = _user_counts(meta, cfg)
    elig = counts[counts >= min_conv]
    n_users = min(int(ucfg["n_users"]), len(elig))
    chosen = elig.index.to_numpy()[np.sort(rng.choice(len(elig), size=n_users, replace=False))]

    sub = df[df["user_key"].isin(chosen)]
    sub = sub.iloc[rng.permutation(len(sub))]              # random order, so the cap keeps a random subset
    sub = sub[sub.groupby("user_key").cumcount() < cap].copy()
    sub["user_n_conversations"] = sub["user_key"].map(counts).astype("int64")
    sub["activity_bucket"] = activity_bucket(sub["user_n_conversations"])
    sub["month"] = (sub["timestamp"].dt.strftime("%Y-%m") if sub["timestamp"].notna().any() else "na")
    sub["month"] = sub["month"].fillna("na")
    sub["turn_bucket"] = turn_bucket(sub["n_turns"], cfg["sampling"]["turn_bucket_edges"])
    sub["stratum"] = "user_level"
    kept = sub["user_n_conversations"].clip(upper=cap)
    sub["weight"] = (len(elig) / max(n_users, 1)) * (sub["user_n_conversations"] / kept)
    sample = sub.sort_values(["_file", "_rg", "_row"]).reset_index(drop=True)

    summary = {
        "mode": "user_level",
        "eligible_conversations": len(df),
        "users_with_proxy": len(counts),
        "min_conversations": min_conv,
        "max_conversations_per_user": cap,
        "eligible_users": len(elig),
        "sampling_frame": int(elig.sum()),
        "sampled_users": n_users,
        "sample_size": len(sample),
        # Users above the cap are likely shared or NAT addresses rather than one person.
        "users_over_cap": int((elig > cap).sum()),
        "sampled_users_over_cap": int((counts[chosen] > cap).sum()),
        "user_activity": user_activity(meta, cfg).to_dict("records"),
    }
    return sample, summary
