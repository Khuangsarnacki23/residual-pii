"""Aggregate metrics with design weights and stratified bootstrap CIs. Outputs are aggregate-only."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANY = "any"   # pseudo-detector/category meaning "union over detectors/categories"


def residual(spans: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Spans that count as residual PII: not a pre-existing placeholder, not a dummy value, and (for
    Presidio) at or above the analysis score threshold."""
    keep = ~spans["in_placeholder"] & ~spans["is_dummy"]
    keep &= (spans["detector"] != "presidio") | (spans["score"] >= threshold)
    return spans[keep]


def conversation_indicators(sample: pd.DataFrame, spans: pd.DataFrame, taxonomy: dict,
                            threshold: float = 0.0) -> pd.DataFrame:
    """Wide 0/1 matrix: one row per sampled conversation, one column per (detector, category),
    plus per-tier and union columns. Column names are 'detector::category'."""
    res = residual(spans, threshold)
    key = ["dataset", "conversation_id"]
    base = sample[key].copy()
    cats = list(taxonomy["categories"])
    tiers = sorted({v["tier"] for v in taxonomy["categories"].values()})
    detectors = sorted(spans["detector"].unique()) if len(spans) else []

    def hits(df: pd.DataFrame, col: str) -> pd.Series:
        flagged = df[key].drop_duplicates().assign(**{col: 1})
        return base.merge(flagged, on=key, how="left")[col].fillna(0).astype(np.int8)

    out = base.copy()
    for det in [*detectors, ANY]:
        d = res if det == ANY else res[res["detector"] == det]
        for cat in cats:
            out[f"{det}::{cat}"] = hits(d[d["category"] == cat], f"{det}::{cat}")
        for tier in tiers:
            out[f"{det}::tier_{tier}"] = hits(d[d["tier"] == tier], f"{det}::tier_{tier}")
    # Drop category columns a detector cannot produce (e.g. secrets::name) to keep tables readable.
    produced = set(zip(spans["detector"], spans["category"])) if len(spans) else set()
    keep = [c for c in out.columns if "::" not in c or c.split("::")[0] == ANY
            or c.split("::")[1].startswith("tier_") or tuple(c.split("::")) in produced]
    return out[keep]


def _boot_indices(strata: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty((reps, len(strata)), dtype=np.int64)
    col = 0
    for h in np.unique(strata):
        members = np.flatnonzero(strata == h)
        draws = rng.integers(0, len(members), size=(reps, len(members)))
        idx[:, col:col + len(members)] = members[draws]
        col += len(members)
    return idx


def weighted_prevalence(X: np.ndarray, w: np.ndarray, strata: np.ndarray, reps: int, level: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = (w @ X) / w.sum()
    if len(w) < 2:
        return point, np.full_like(point, np.nan), np.full_like(point, np.nan)
    boots = np.empty((reps, X.shape[1]))
    for b, idx in enumerate(_boot_indices(strata, reps, rng)):
        wb = w[idx]
        boots[b] = (wb @ X[idx]) / wb.sum()
    a = (1 - level) / 2
    return point, np.quantile(boots, a, axis=0), np.quantile(boots, 1 - a, axis=0)


def variants(conv: pd.DataFrame) -> dict[str, pd.Series]:
    v = {"all": pd.Series(True, index=conv.index),
         "excl_story_roleplay": ~conv["is_story_roleplay"]}
    if "dedup_status" in conv:
        v["excl_arena_overlap"] = conv["dedup_status"].fillna("unique").eq("unique")
    return v


def prevalence_table(conv: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    acfg = cfg["analysis"]
    ind_cols = [c for c in conv.columns if "::" in c]
    rows = []
    for ds, g in conv.groupby("dataset"):
        for vname, mask in variants(g).items():
            if vname == "excl_arena_overlap" and g["dedup_status"].isna().all():
                continue
            sub = g[mask]
            if sub.empty:
                continue
            rng = np.random.default_rng(seed)
            X = sub[ind_cols].to_numpy(dtype=float)
            w = sub["weight"].to_numpy(dtype=float)
            point, lo, hi = weighted_prevalence(X, w, sub["stratum"].to_numpy(), acfg["bootstrap_reps"],
                                                acfg["ci_level"], rng)
            for j, col in enumerate(ind_cols):
                det, cat = col.split("::")
                rows.append({"dataset": ds, "variant": vname, "detector": det, "category": cat,
                             "n_conversations": len(sub), "n_positive": int(X[:, j].sum()),
                             "prevalence": point[j], "ci_low": lo[j], "ci_high": hi[j]})
    return pd.DataFrame(rows)


def by_task_table(conv: pd.DataFrame, cfg: dict, seed: int, columns: list[str]) -> pd.DataFrame:
    acfg = cfg["analysis"]
    rows = []
    for (ds, task), sub in conv.groupby(["dataset", "task_type"]):
        rng = np.random.default_rng(seed)
        X = sub[columns].to_numpy(dtype=float)
        point, lo, hi = weighted_prevalence(X, sub["weight"].to_numpy(dtype=float), sub["stratum"].to_numpy(),
                                            acfg["bootstrap_reps"], acfg["ci_level"], rng)
        share = sub["weight"].sum() / conv.loc[conv["dataset"] == ds, "weight"].sum()
        for j, col in enumerate(columns):
            det, cat = col.split("::")
            rows.append({"dataset": ds, "task_type": task, "task_share": share, "detector": det,
                         "category": cat, "n_conversations": len(sub), "n_positive": int(X[:, j].sum()),
                         "prevalence": point[j], "ci_low": lo[j], "ci_high": hi[j]})
    return pd.DataFrame(rows)


def threshold_sensitivity(sample: pd.DataFrame, spans: pd.DataFrame, taxonomy: dict,
                          thresholds: list[float]) -> pd.DataFrame:
    """Point prevalence of each Presidio category as the score threshold varies."""
    rows = []
    for thr in thresholds:
        ind = conversation_indicators(sample, spans, taxonomy, thr).merge(
            sample[["dataset", "conversation_id", "weight"]], on=["dataset", "conversation_id"])
        cols = [c for c in ind.columns if c.startswith("presidio::")]
        for ds, g in ind.groupby("dataset"):
            w = g["weight"].to_numpy(dtype=float)
            prev = (w @ g[cols].to_numpy(dtype=float)) / w.sum()
            rows += [{"dataset": ds, "threshold": thr, "category": c.split("::")[1], "prevalence": p}
                     for c, p in zip(cols, prev)]
    return pd.DataFrame(rows)


def rates_per_1k_tokens(spans: pd.DataFrame, stats: pd.DataFrame, sample: pd.DataFrame,
                        threshold: float = 0.0) -> pd.DataFrame:
    """Design-weighted residual spans per 1,000 whitespace tokens of user text."""
    w = sample.set_index(["dataset", "conversation_id"])["weight"]
    tok = stats.groupby(["dataset", "conversation_id"])["n_tokens"].sum()
    tok_w = (tok * w.reindex(tok.index)).groupby(level="dataset").sum()
    res = residual(spans, threshold)
    cnt = res.groupby(["dataset", "conversation_id", "detector", "category"]).size().rename("n").reset_index()
    cnt["wn"] = cnt["n"] * w.reindex(pd.MultiIndex.from_frame(cnt[["dataset", "conversation_id"]])).to_numpy()
    out = cnt.groupby(["dataset", "detector", "category"])["wn"].sum().reset_index()
    out["spans_per_1k_tokens"] = 1000 * out["wn"] / out["dataset"].map(tok_w)
    return out.drop(columns="wn")


def exclusion_table(spans: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """How many stored detections were set aside, and why."""
    below = (spans["detector"] == "presidio") & (spans["score"] < threshold)
    s = spans.assign(reason=np.select([spans["in_placeholder"], spans["is_dummy"], below],
                                      ["placeholder_overlap", "dummy_value", "below_threshold"], "residual"))
    t = s.groupby(["dataset", "detector", "category", "reason"]).size().unstack("reason", fill_value=0)
    return t.reset_index()


def placeholder_table(stats: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    per = stats.groupby(["dataset", "conversation_id"])[["n_placeholders", "n_tokens"]].sum().reset_index()
    per = per.merge(sample[["dataset", "conversation_id", "weight", "flag_redacted"]], on=["dataset", "conversation_id"])
    rows = []
    for ds, g in per.groupby("dataset"):
        w = g["weight"]
        flag = g["flag_redacted"]
        rows.append({
            "dataset": ds,
            "n_conversations": len(g),
            "share_with_placeholder": float((w * (g["n_placeholders"] > 0)).sum() / w.sum()),
            "placeholders_per_1k_tokens": float(1000 * (w * g["n_placeholders"]).sum() / (w * g["n_tokens"]).sum()),
            "share_flagged_redacted": (float((w * flag.fillna(False).astype(bool)).sum() / w.sum())
                                       if flag.notna().any() else np.nan),
        })
    return pd.DataFrame(rows)


def coverage_table(stats: pd.DataFrame) -> pd.DataFrame:
    g = stats.groupby("dataset")
    return pd.DataFrame({
        "user_turns": g.size(),
        "chars_total": g["n_chars"].sum(),
        "chars_scanned": g["n_chars_scanned"].sum(),
    }).assign(share_unscanned=lambda d: 1 - d["chars_scanned"] / d["chars_total"]).reset_index()
