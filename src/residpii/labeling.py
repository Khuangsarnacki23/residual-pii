"""Human-labelled benchmark: label-set sampling, the label store, inter-annotator agreement, and
precision/recall of each detector against adjudicated labels.

Design
  * The unit is one user turn from the main sample. Turns are drawn in strata defined by what the
    stored detectors flagged (rare direct identifier / name / quasi-only / nothing), oversampling rare
    strata. Each item carries an inclusion weight (conversation design weight x stratum pool / draws),
    so weighted precision and recall estimate the rates over all eligible turns, not over the label set.
  * Annotators are blind: the labelling server never sends detector output to them.
  * Labels hold character offsets (Python code points) plus category, whose information it is, and
    per-user attributes marked stated or inferred. They live under the restricted directory, because
    attribute values and notes are text taken from the data. Only aggregate tables leave it.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import _boot_indices, residual
from .config import Paths
from .sampling import _proportional_allocation

RARE_DIRECT = {"email", "phone", "account_id", "ip_address", "api_key_token"}
ANY = "any"


# ---------------------------------------------------------------- paths


def label_dir(paths: Paths, ds: str) -> Path:
    d = paths.restricted / "labels" / ds
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d.parent, 0o700)
    os.chmod(d, 0o700)
    return d


def items_path(paths: Paths, ds: str) -> Path:
    d = paths.work / "labels" / ds
    d.mkdir(parents=True, exist_ok=True)
    return d / "items.parquet"


def labelled_datasets(paths: Paths) -> list[str]:
    root = paths.work / "labels"
    return sorted(p.parent.name for p in root.glob("*/items.parquet")) if root.exists() else []


# ---------------------------------------------------------------- sampling the label set


def turn_strata(turns: pd.DataFrame, spans: pd.DataFrame, threshold: float) -> pd.Series:
    """Label stratum of each turn, from residual detector spans at the analysis threshold."""
    key = ["dataset", "conversation_id", "turn_index"]
    res = residual(spans, threshold)
    cats = res.groupby(key)["category"].agg(set).reindex(pd.MultiIndex.from_frame(turns[key]))

    def one(c):
        c = c if isinstance(c, set) else set()
        if c & RARE_DIRECT:
            return "rare_direct"
        if "name" in c:
            return "name"
        if c:
            return "quasi_only"
        return "none"

    return pd.Series([one(c) for c in cats], index=turns.index)


def build_items(sample: pd.DataFrame, turns: pd.DataFrame, spans: pd.DataFrame, cfg: dict,
                seed: int) -> tuple[pd.DataFrame, dict]:
    """Draws the label set. Returns (items without text, summary counts)."""
    lcfg, thr = cfg["labeling"], cfg["presidio"]["score_threshold"]
    rng = np.random.default_rng(seed)
    t = turns[turns["text"].str.strip().str.len() > 0].copy()
    t["n_chars"] = t["text"].str.len()
    n_nonempty = len(t)
    t = t[t["n_chars"] <= lcfg["max_chars"]]
    summary = {"nonempty_turns": n_nonempty, "eligible_turns": len(t),
               "share_excluded_long": round(1 - len(t) / max(n_nonempty, 1), 4),
               "max_chars": lcfg["max_chars"]}
    t = t.merge(sample[["dataset", "conversation_id", "weight"]], on=["dataset", "conversation_id"])
    t = t.rename(columns={"weight": "frame_weight"})
    t["label_stratum"] = turn_strata(t, spans, thr).to_numpy()

    shares = {h: s["share"] for h, s in lcfg["strata"].items()}
    pools = t["label_stratum"].value_counts().reindex(list(shares), fill_value=0)
    n = min(lcfg["n_items"], len(t))
    # Allocate by configured share, capped at pool size; leftover draws go to strata with room.
    target = pd.Series({h: shares[h] * n for h in shares})
    alloc = np.minimum(np.floor(target), pools).astype(int)
    while alloc.sum() < n and (alloc < pools).any():
        room = (pools - alloc) > 0
        gap = (target - alloc).where(room, -np.inf)
        alloc[gap.idxmax()] += 1

    parts = []
    for h, k in alloc.items():
        if k == 0:
            continue
        pool = t[t["label_stratum"] == h]
        take = pool.iloc[rng.choice(len(pool), size=k, replace=False)].copy()
        take["label_weight"] = take["frame_weight"] * len(pool) / k
        parts.append(take)
    items = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    items.insert(0, "item_id", [f"i{i:04d}" for i in range(1, len(items) + 1)])
    summary["strata"] = {h: {"pool": int(pools[h]), "drawn": int(alloc[h])} for h in shares}
    summary["n_items"] = len(items)
    cols = ["item_id", "dataset", "conversation_id", "turn_index", "label_stratum", "frame_weight",
            "label_weight", "n_chars"]
    return items[cols], summary


# ---------------------------------------------------------------- label store


def annotator_file(paths: Paths, ds: str, annotator: str) -> Path:
    if not annotator.replace("_", "").replace("-", "").isalnum():
        raise ValueError("annotator names may contain only letters, digits, '-' and '_'")
    return label_dir(paths, ds) / f"{annotator}.jsonl"


def save_label(paths: Paths, ds: str, annotator: str, record: dict) -> None:
    """Appends one label record. The latest record per item wins, so history is kept."""
    rec = {**record, "annotator": annotator, "saved_at": datetime.now(timezone.utc).isoformat()}
    path = annotator_file(paths, ds, annotator)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_labels(paths: Paths, ds: str, annotator: str) -> dict[str, dict]:
    path = annotator_file(paths, ds, annotator)
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["item_id"]] = rec
    return out


def validate_record(rec: dict, lcfg: dict, text_len: int) -> dict:
    """Checks a record posted by the labelling page and keeps only known fields."""
    cats = set(lcfg["span_categories"]["direct"]) | set(lcfg["span_categories"]["quasi"])
    whose, attrs = set(lcfg["whose"]), set(lcfg["attributes"])
    spans = []
    for s in rec.get("spans", []):
        start, end = int(s["start"]), int(s["end"])
        if not (0 <= start < end <= text_len):
            raise ValueError(f"span offsets out of range: {start}-{end}")
        if s["category"] not in cats or s["whose"] not in whose:
            raise ValueError(f"unknown category/whose: {s['category']}/{s['whose']}")
        spans.append({"start": start, "end": end, "category": s["category"], "whose": s["whose"]})
    attributes = []
    for a in rec.get("attributes", []):
        if a["attribute"] not in attrs or a["kind"] not in ("stated", "inferred"):
            raise ValueError(f"unknown attribute/kind: {a['attribute']}/{a['kind']}")
        attributes.append({"attribute": a["attribute"], "kind": a["kind"], "value": str(a.get("value", ""))[:200]})
    return {"item_id": str(rec["item_id"]), "spans": sorted(spans, key=lambda s: (s["start"], s["end"])),
            "attributes": attributes, "skip": bool(rec.get("skip", False)),
            "note": str(rec.get("note", ""))[:1000]}


def tier_of(category: str, lcfg: dict) -> str:
    return "direct" if category in lcfg["span_categories"]["direct"] else "quasi"


# ---------------------------------------------------------------- span matching


def overlaps(a: tuple[int, int], b: tuple[int, int], min_overlap: float) -> bool:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return False
    shorter = min(a[1] - a[0], b[1] - b[0])
    return inter / shorter >= min_overlap if min_overlap > 0 else True


def greedy_match(a: list[dict], b: list[dict], min_overlap: float, same_category: bool = True) -> list[tuple[int, int]]:
    """One-to-one matching of spans by largest overlap. Returns index pairs (i in a, j in b)."""
    cands = []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if same_category and x["category"] != y["category"]:
                continue
            if overlaps((x["start"], x["end"]), (y["start"], y["end"]), min_overlap):
                inter = min(x["end"], y["end"]) - max(x["start"], y["start"])
                cands.append((-inter, i, j))
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in sorted(cands):
        if i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j), pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------- agreement


def cohen_kappa(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x), np.asarray(y)
    if len(x) == 0:
        return float("nan")
    labels = np.union1d(x, y)
    po = float((x == y).mean())
    pe = float(sum((x == lab).mean() * (y == lab).mean() for lab in labels))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def agreement(items: pd.DataFrame, a: dict[str, dict], b: dict[str, dict], lcfg: dict) -> pd.DataFrame:
    """Unweighted agreement between two annotators on the items both labelled (and neither skipped)."""
    ids = [i for i in items["item_id"] if i in a and i in b and not a[i]["skip"] and not b[i]["skip"]]
    mo = lcfg["min_overlap"]
    private = set(lcfg["private_whose"])
    cats = lcfg["span_categories"]["direct"] + lcfg["span_categories"]["quasi"]
    rows = []

    def add(metric, category, value, n):
        rows.append({"metric": metric, "category": category, "value": value, "n": n})

    # Span F1 per category (and overall), plus agreement on `whose` for matched spans.
    matched_whose_a, matched_whose_b = [], []
    for cat in [*cats, ANY]:
        m = na = nb = 0
        for i in ids:
            sa = [s for s in a[i]["spans"] if cat == ANY or s["category"] == cat]
            sb = [s for s in b[i]["spans"] if cat == ANY or s["category"] == cat]
            pairs = greedy_match(sa, sb, mo, same_category=True)
            m, na, nb = m + len(pairs), na + len(sa), nb + len(sb)
            if cat == ANY:
                matched_whose_a += [sa[p]["whose"] for p, _ in pairs]
                matched_whose_b += [sb[q]["whose"] for _, q in pairs]
        if na + nb:
            add("span_f1", cat, 2 * m / (na + nb), na + nb)
    if matched_whose_a:
        wa, wb = np.array(matched_whose_a), np.array(matched_whose_b)
        add("whose_agreement", ANY, float((wa == wb).mean()), len(wa))
        add("whose_kappa", ANY, cohen_kappa(wa, wb), len(wa))
        add("private_kappa", ANY, cohen_kappa(np.isin(wa, list(private)), np.isin(wb, list(private))), len(wa))

    # Item-level: does the turn contain private PII of this category / tier?
    def has(rec, pred):
        return any(pred(s) and s["whose"] in private for s in rec["spans"])

    groups = {**{c: (lambda s, c=c: s["category"] == c) for c in cats},
              "tier_direct": lambda s: tier_of(s["category"], lcfg) == "direct",
              "tier_quasi": lambda s: tier_of(s["category"], lcfg) == "quasi"}
    for name, pred in groups.items():
        x = np.array([has(a[i], pred) for i in ids])
        y = np.array([has(b[i], pred) for i in ids])
        if len(ids) and (x.any() or y.any()):
            add("item_kappa_private", name, cohen_kappa(x, y), len(ids))
            add("item_agreement_private", name, float((x == y).mean()), len(ids))

    # Attributes: presence agreement per attribute (stated or inferred), and value match when both mark it.
    for attr in lcfg["attributes"]:
        pa = np.array([any(t["attribute"] == attr for t in a[i]["attributes"]) for i in ids])
        pb = np.array([any(t["attribute"] == attr for t in b[i]["attributes"]) for i in ids])
        if len(ids) and (pa.any() or pb.any()):
            add("attribute_kappa", attr, cohen_kappa(pa, pb), len(ids))
            both = [i for i, x, y in zip(ids, pa, pb) if x and y]
            if both:
                def vals(rec):
                    return {t["value"].strip().casefold() for t in rec["attributes"] if t["attribute"] == attr}
                add("attribute_value_match", attr, float(np.mean([bool(vals(a[i]) & vals(b[i])) for i in both])), len(both))
    add("items_compared", ANY, float(len(ids)), len(ids))
    return pd.DataFrame(rows)


def prefill_adjudication(a: dict, b: dict, lcfg: dict) -> dict:
    """Starting point for the adjudicator: spans and attributes both annotators agree on exactly."""
    key = lambda s: (s["start"], s["end"], s["category"], s["whose"])  # noqa: E731
    kb = {key(s) for s in b["spans"]}
    akey = lambda t: (t["attribute"], t["kind"], t["value"].strip().casefold())  # noqa: E731
    ab = {akey(t) for t in b["attributes"]}
    return {"item_id": a["item_id"], "spans": [s for s in a["spans"] if key(s) in kb],
            "attributes": [t for t in a["attributes"] if akey(t) in ab],
            "skip": a["skip"] and b["skip"], "note": ""}


# ---------------------------------------------------------------- scoring detectors against gold


def _ratio_ci(num: np.ndarray, den: np.ndarray, w: np.ndarray, strata: np.ndarray, reps: int, level: float,
              rng: np.random.Generator) -> tuple[float, float, float]:
    """Weighted ratio sum(w*num)/sum(w*den) with a stratified item bootstrap."""
    d = (w * den).sum()
    if d == 0:
        return float("nan"), float("nan"), float("nan")
    point = float((w * num).sum() / d)
    boots = []
    for idx in _boot_indices(strata, reps, rng):
        db = (w[idx] * den[idx]).sum()
        if db > 0:
            boots.append((w[idx] * num[idx]).sum() / db)
    a = (1 - level) / 2
    return point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def detector_spans_for_items(spans: pd.DataFrame, items: pd.DataFrame, lcfg: dict, threshold: float) -> pd.DataFrame:
    key = ["dataset", "conversation_id", "turn_index"]
    res = residual(spans, threshold)
    res = res.merge(items[[*key, "item_id"]], on=key)
    res = res.assign(category=res["category"].map(lcfg["detector_map"])).dropna(subset=["category"])
    return res[["item_id", "detector", "category", "start", "end", "score"]]


def score(items: pd.DataFrame, gold: dict[str, dict], det: pd.DataFrame, lcfg: dict, acfg: dict,
          seed: int) -> dict[str, pd.DataFrame]:
    """Precision/recall per detector and gold category, gold prevalence, and false-positive breakdown."""
    items = items[items["item_id"].map(lambda i: i in gold and not gold[i]["skip"])].reset_index(drop=True)
    mo, private = lcfg["min_overlap"], set(lcfg["private_whose"])
    w = items["label_weight"].to_numpy(float)
    strata = items["label_stratum"].to_numpy()
    reps, level = acfg["bootstrap_reps"], acfg["ci_level"]
    by_item = {k: g.to_dict("records") for k, g in det.groupby("item_id")} if len(det) else {}
    detectors = sorted(det["detector"].unique()) + [ANY] if len(det) else [ANY]
    cats = sorted(set(det["category"]) if len(det) else set())

    pr_rows, fp_rows = [], []
    for d in detectors:
        for cat in [*cats, ANY]:
            # Per item: detector spans, the ones matching private gold (TP for precision), gold private
            # spans, and the ones found (TP for recall).
            n_det, tp_det, tp_det_anycat, n_gold, tp_gold = (np.zeros(len(items)) for _ in range(5))
            whose_counts: dict[str, np.ndarray] = {}
            for k, iid in enumerate(items["item_id"]):
                ds = [s for s in by_item.get(iid, []) if (d == ANY or s["detector"] == d)
                      and (cat == ANY or s["category"] == cat)]
                gs_all = gold[iid]["spans"]
                gp = [s for s in gs_all if s["whose"] in private and (cat == ANY or s["category"] == cat)]
                gp_any = [s for s in gs_all if s["whose"] in private]
                same = cat != ANY   # for the ANY row, a match need not agree on category
                n_det[k], n_gold[k] = len(ds), len(gp)
                # Precision: is each detector span real PII of the same category? Many-to-one allowed
                # (Presidio may split one gold name into two spans); recall counts gold spans covered.
                for s in ds:
                    if any(overlaps((s["start"], s["end"]), (g["start"], g["end"]), mo)
                           and (not same or g["category"] == s["category"]) for g in gp):
                        tp_det[k] += 1
                    if any(overlaps((s["start"], s["end"]), (g["start"], g["end"]), mo) for g in gp_any):
                        tp_det_anycat[k] += 1
                    # What the gold says about this detector span (for the false-alarm breakdown).
                    hit = [g for g in gs_all if overlaps((s["start"], s["end"]), (g["start"], g["end"]), mo)]
                    label = hit[0]["whose"] if hit else "unlabelled"
                    whose_counts.setdefault(label, np.zeros(len(items)))[k] += 1
                for g in gp:
                    if any(overlaps((s["start"], s["end"]), (g["start"], g["end"]), mo)
                           and (not same or s["category"] == g["category"]) for s in ds):
                        tp_gold[k] += 1
            rng = np.random.default_rng(seed)
            for metric, num, den in (("precision", tp_det, n_det), ("precision_any_category", tp_det_anycat, n_det),
                                     ("recall", tp_gold, n_gold)):
                p, lo, hi = _ratio_ci(num, den, w, strata, reps, level, rng)
                count = int(den.sum())
                if count:
                    pr_rows.append({"detector": d, "category": cat, "metric": metric, "value": p,
                                    "ci_low": lo, "ci_high": hi, "n": count})
            total = (w * n_det).sum()
            if total > 0:
                for label, cnt in sorted(whose_counts.items()):
                    fp_rows.append({"detector": d, "category": cat, "whose": label,
                                    "share_of_flags": float((w * cnt).sum() / total), "n": int(cnt.sum())})

    # Gold prevalence: weighted share of eligible turns with private PII, per category and tier.
    gcats = lcfg["span_categories"]["direct"] + lcfg["span_categories"]["quasi"]
    groups = {**{c: (lambda s, c=c: s["category"] == c) for c in gcats},
              "tier_direct": lambda s: tier_of(s["category"], lcfg) == "direct",
              "tier_quasi": lambda s: tier_of(s["category"], lcfg) == "quasi"}
    prev_rows = []
    ones = np.ones(len(items))
    for name, pred in groups.items():
        x = np.array([float(any(pred(s) and s["whose"] in private for s in gold[i]["spans"])) for i in items["item_id"]])
        p, lo, hi = _ratio_ci(x, ones, w, strata, reps, level, np.random.default_rng(seed))
        prev_rows.append({"category": name, "prevalence": p, "ci_low": lo, "ci_high": hi,
                          "n_items": len(items), "n_positive": int(x.sum())})
    return {"labels_precision_recall": pd.DataFrame(pr_rows),
            "labels_flag_breakdown": pd.DataFrame(fp_rows),
            "labels_gold_prevalence": pd.DataFrame(prev_rows)}


def progress(items: pd.DataFrame, labels: dict[str, dict[str, dict]]) -> pd.DataFrame:
    rows = []
    for who, recs in labels.items():
        done = items["item_id"].map(lambda i: i in recs)
        skipped = items["item_id"].map(lambda i: i in recs and recs[i]["skip"])
        rows.append({"annotator": who, "labelled": int(done.sum()), "skipped": int(skipped.sum()),
                     "total": len(items)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- commands


def _seed(cfg: dict) -> int:
    return cfg["seed"] + 7919


def cmd_build(cfg: dict, ds: str, force: bool = False) -> dict:
    """Draws the label set from an existing `residpii run` (sample, turns and span store)."""
    paths = Paths(cfg).ensure()
    out = items_path(paths, ds)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; labels may already refer to it. Pass --force to redraw.")
    sample = pd.read_parquet(paths.work / f"sample_{ds}.parquet")
    turns = pd.read_parquet(paths.restricted / f"turns_{ds}.parquet")
    spans = pd.read_parquet(paths.restricted / f"spans_{ds}.parquet")
    items, summary = build_items(sample, turns, spans, cfg, _seed(cfg))
    items.to_parquet(out, index=False)
    summary = {"dataset": ds, "seed": _seed(cfg), **summary}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _load_all(cfg: dict, ds: str, extra: tuple[str, ...] = ()):
    paths = Paths(cfg).ensure()
    items = pd.read_parquet(items_path(paths, ds))
    lcfg = cfg["labeling"]
    who = dict.fromkeys([*lcfg["annotators"], lcfg["adjudicator"], *extra])
    return paths, items, {a: load_labels(paths, ds, a) for a in who}


def cmd_agreement(cfg: dict, ds: str, pair: tuple[str, str] | None = None) -> pd.DataFrame:
    from .report import write_table

    a, b = pair or cfg["labeling"]["annotators"][:2]
    paths, items, labels = _load_all(cfg, ds, (a, b))
    out = agreement(items, labels[a], labels[b], cfg["labeling"]).assign(dataset=ds, annotator=f"{a}~{b}")
    write_table(out, paths.tables, f"labels_agreement_{ds}")
    write_table(progress(items, labels).assign(dataset=ds), paths.tables, f"labels_progress_{ds}")
    return out


def cmd_score(cfg: dict, datasets: list[str] | None = None, gold_from: str | None = None) -> dict[str, pd.DataFrame]:
    """Scores detectors against gold labels for each dataset; writes combined tables with a dataset column."""
    from .report import write_table

    lcfg = cfg["labeling"]
    gold_from = gold_from or lcfg["adjudicator"]
    paths = Paths(cfg).ensure()
    datasets = datasets or labelled_datasets(paths)
    out: dict[str, list[pd.DataFrame]] = {}
    scored = []
    for ds in datasets:
        _, items, labels = _load_all(cfg, ds, (gold_from,))
        gold = labels[gold_from]
        if not gold:
            print(f"[label] {ds}: no labels from '{gold_from}', skipped", file=sys.stderr)
            continue
        spans = pd.read_parquet(paths.restricted / f"spans_{ds}.parquet")
        for thr in sorted({cfg["presidio"]["score_threshold"], *cfg["presidio"]["sensitivity_thresholds"]}):
            det = detector_spans_for_items(spans, items, lcfg, thr)
            res = score(items, gold, det, lcfg, cfg["analysis"], _seed(cfg))
            for name, df in res.items():
                if name == "labels_gold_prevalence" and thr != cfg["presidio"]["score_threshold"]:
                    continue   # does not depend on the detector threshold
                df = df.assign(dataset=ds, gold=gold_from)
                if name != "labels_gold_prevalence":
                    df = df.assign(threshold=thr)
                out.setdefault(name, []).append(df)
        scored.append(ds)
    if not scored:
        raise RuntimeError(f"No labels from '{gold_from}' in any dataset. Adjudicate first, or pass --gold <annotator>.")
    tables = {name: pd.concat(dfs, ignore_index=True) for name, dfs in out.items()}
    for name, df in tables.items():
        cols = ["dataset", *[c for c in df.columns if c != "dataset"]]
        write_table(df[cols], paths.tables, name)
    return tables
