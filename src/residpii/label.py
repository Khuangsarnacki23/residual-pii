"""Hand-labeling: export a batch of user turns for annotation, then score detectors against it.

The export contains raw text, so it is written only under the restricted directory. Scoring reads
the annotated file back and produces aggregate precision/recall/F1 per detector x category, plus
agreement between the LLM and the human on whom a piece of information is about.

See docs/annotation_guide.md for the labels.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import residual

ID_COLUMNS = ["item_id", "dataset", "conversation_id", "turn_index", "text"]
ANNOTATION_COLUMNS = ["span_text", "start", "end", "pii_type", "about", "sensitive_topic", "notes"]
LABEL_COLUMNS = [*ID_COLUMNS, *ANNOTATION_COLUMNS]
ABOUT_LABELS = ("user", "third_party", "public", "fictional")
NONE = "none"
TURN = ["dataset", "conversation_id", "turn_index"]


# ---------------------------------------------------------------- export


def _even_allocation(sizes: pd.Series, n: int) -> pd.Series:
    """Spread n as evenly as possible over strata, never asking a stratum for more than it has."""
    alloc = pd.Series(0, index=sizes.index)
    remaining = min(n, int(sizes.sum()))
    while remaining > 0:
        open_ = alloc.index[alloc < sizes]
        share = max(remaining // len(open_), 1)
        for s in open_:
            take = min(share, int(sizes[s] - alloc[s]), remaining)
            alloc[s] += take
            remaining -= take
            if remaining == 0:
                break
    return alloc


def export_sample(turns: pd.DataFrame, spans: pd.DataFrame, tasks: pd.DataFrame, n: int, seed: int,
                  threshold: float, max_text_chars: int) -> pd.DataFrame:
    """Stratified sample of user turns by task type x has-any-detection, so the batch holds both
    turns the detectors flagged and turns they did not. One row per turn, annotation columns empty."""
    rng = np.random.default_rng(seed)
    t = turns[(turns["role"] == "user") & (turns["text"].str.strip() != "")].copy()
    hits = residual(spans[spans["role"] == "user"], threshold)[TURN].drop_duplicates().assign(has_any_detection=True)
    t = t.merge(hits, on=TURN, how="left").merge(tasks[["dataset", "conversation_id", "task_type"]],
                                                 on=["dataset", "conversation_id"], how="left")
    t["has_any_detection"] = t["has_any_detection"].fillna(False).astype(bool)
    t["stratum"] = t["task_type"].fillna("unknown") + "|" + t["has_any_detection"].map({True: "det", False: "nodet"})
    alloc = _even_allocation(t["stratum"].value_counts().sort_index(), n)
    picked = [pool.iloc[np.sort(rng.choice(len(pool), size=int(alloc[s]), replace=False))]
              for s, pool in t.groupby("stratum", sort=True) if alloc[s] > 0]
    out = pd.concat(picked).iloc[rng.permutation(sum(len(p) for p in picked))].reset_index(drop=True)
    out["item_id"] = [f"it{i + 1:04d}" for i in range(len(out))]
    out["text"] = out["text"].str[:max_text_chars]     # spreadsheet cells have a length limit
    for c in ANNOTATION_COLUMNS:
        out[c] = ""
    return out[LABEL_COLUMNS]


def write_batch(items: pd.DataFrame, restricted: Path, fmt: str = "csv", today: date | None = None) -> Path:
    """Writes restricted/labels/batch_<date>.<fmt> (0600). The destination is fixed: raw text has
    no other place to go."""
    if fmt not in ("csv", "xlsx"):
        raise ValueError("format must be csv or xlsx")
    out_dir = Path(restricted) / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    stem = f"batch_{(today or date.today()).isoformat()}"
    path, i = out_dir / f"{stem}.{fmt}", 1
    while path.exists():                               # never overwrite a batch someone may be annotating
        i += 1
        path = out_dir / f"{stem}_{i}.{fmt}"
    if fmt == "xlsx":
        try:
            items.to_excel(path, index=False)
        except ImportError:
            raise RuntimeError("xlsx export needs openpyxl (pip install openpyxl); or use --format csv") from None
    else:
        items.to_csv(path, index=False)
    os.chmod(path, 0o600)
    return path


# ---------------------------------------------------------------- reading labels


def read_labels(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".xlsx":
        df = pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"labels file is missing columns: {missing}")
    df = df[df["item_id"].str.strip() != ""].copy()
    # Annotators add rows for extra spans and may fill in only item_id: carry the turn down from
    # the item's first row.
    first = df[df["text"] != ""].drop_duplicates("item_id").set_index("item_id")
    for c in ID_COLUMNS[1:]:
        df[c] = df["item_id"].map(first[c])
    df = df[df["text"].notna()]
    df["turn_index"] = df["turn_index"].astype(int)
    for c in ("pii_type", "about"):
        df[c] = df[c].str.strip().str.casefold()
    return df.reset_index(drop=True)


def gold_spans(labels: pd.DataFrame, categories: list[str]) -> tuple[pd.DataFrame, dict]:
    """Annotated rows -> gold spans with offsets. Offsets come from start/end when given; otherwise
    every occurrence of span_text in the turn is a gold span. Returns (spans, counts of skipped rows)."""
    rows, skipped = [], {"unlabeled": 0, "unknown_pii_type": 0, "span_not_found": 0}
    for r in labels.itertuples(index=False):
        if r.pii_type in ("", NONE):
            skipped["unlabeled"] += r.pii_type == ""
            continue
        if r.pii_type not in categories:
            skipped["unknown_pii_type"] += 1
            continue
        about = r.about if r.about in ABOUT_LABELS else ""
        if str(r.start).strip().isdigit() and str(r.end).strip().isdigit() and int(r.start) < int(r.end):
            found = [(int(r.start), int(r.end))]
        else:
            found, needle, pos = [], r.span_text.strip(), 0
            while needle and (pos := r.text.find(needle, pos)) != -1:
                found.append((pos, pos + len(needle)))
                pos += len(needle)
        if not found:
            skipped["span_not_found"] += 1
        rows += [(r.dataset, r.conversation_id, r.turn_index, s, e, r.pii_type, about) for s, e in found]
    cols = [*TURN, "start", "end", "category", "about"]
    return pd.DataFrame(rows, columns=cols).drop_duplicates().reset_index(drop=True), skipped


# ---------------------------------------------------------------- scoring


def spans_match(s1: int, e1: int, s2: int, e2: int, min_overlap: float = 0.5) -> bool:
    """True when the overlap covers at least `min_overlap` of the shorter span."""
    overlap = min(e1, e2) - max(s1, s2)
    return overlap > 0 and overlap >= min_overlap * min(e1 - s1, e2 - s2)


def _matched(a: pd.DataFrame, b: pd.DataFrame, min_overlap: float) -> np.ndarray:
    """For each span in `a`: does a span in `b` in the same turn and category match it?"""
    index = {k: g[["start", "end"]].to_numpy() for k, g in b.groupby([*TURN, "category"])}
    out = np.zeros(len(a), dtype=bool)
    for i, r in enumerate(a[[*TURN, "category", "start", "end"]].itertuples(index=False)):
        out[i] = any(spans_match(r.start, r.end, s, e, min_overlap) for s, e in index.get(tuple(r[:4]), ()))
    return out


def _prf(n_gold: int, n_pred: int, tp_pred: int, tp_gold: int) -> dict:
    p = tp_pred / n_pred if n_pred else np.nan
    r = tp_gold / n_gold if n_gold else np.nan
    f1 = 2 * p * r / (p + r) if n_pred and n_gold and (p + r) > 0 else (0.0 if n_pred and n_gold else np.nan)
    return {"n_gold": n_gold, "n_pred": n_pred, "tp_pred": tp_pred, "tp_gold": tp_gold,
            "precision": p, "recall": r, "f1": f1}


def score_detectors(pred: pd.DataFrame, gold: pd.DataFrame, min_overlap: float = 0.5) -> pd.DataFrame:
    """Precision (share of detector spans that match a gold span), recall (share of gold spans some
    detector span matches) and F1, per detector x category, plus the union of detectors ("any") and
    all categories together ("all"). `pred` and `gold` must cover the same turns."""
    rows = []
    for det in [*sorted(pred["detector"].unique()), "any"]:
        p = pred if det == "any" else pred[pred["detector"] == det]
        p = p.drop_duplicates([*TURN, "category", "start", "end"])
        p_hit, g_hit = _matched(p, gold, min_overlap), _matched(gold, p, min_overlap)
        for cat in [*sorted(set(p["category"]) | set(gold["category"])), "all"]:
            pm = np.ones(len(p), bool) if cat == "all" else (p["category"] == cat).to_numpy()
            gm = np.ones(len(gold), bool) if cat == "all" else (gold["category"] == cat).to_numpy()
            rows.append({"detector": det, "category": cat,
                         **_prf(int(gm.sum()), int(pm.sum()), int(p_hit[pm].sum()), int(g_hit[gm].sum()))})
    return pd.DataFrame(rows)


def cohen_kappa(a, b) -> float:
    """Cohen's kappa for two raters' labels over the same items. NaN when it is undefined."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    if len(a) == 0:
        return np.nan
    cats = sorted(set(a) | set(b), key=str)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


LLM_ABOUT_TO_HUMAN = {"user": "user", "third_party": "third_party", "public_figure": "public",
                      "fictional": "fictional"}


def about_agreement(labels: pd.DataFrame, gold: pd.DataFrame, llm_attrs: pd.DataFrame,
                    llm_turns: pd.DataFrame, min_overlap: float = 0.5) -> pd.DataFrame:
    """LLM-vs-human agreement on `about`, over labeled turns the LLM processed.

    turn level:  does the turn hold anything about the user? (human: a gold span with about=user;
                 LLM: an attribute with about=user). Binary kappa over turns.
    span level:  where an LLM attribute's value appears in the text and overlaps a gold span, do the
                 two `about` labels agree? Kappa over those pairs (LLM `unclear` is left out)."""
    done = llm_turns[llm_turns["status"] == "ok"][TURN].drop_duplicates()
    items = labels.drop_duplicates(TURN)[[*TURN, "text"]].merge(done, on=TURN)
    key = lambda df: set(map(tuple, df[TURN].to_numpy()))  # noqa: E731
    human_user = key(gold[gold["about"] == "user"])
    llm_user = key(llm_attrs[llm_attrs["about"] == "user"])
    turn_keys = list(map(tuple, items[TURN].to_numpy()))
    rows = [{"detector": "llm_vs_human", "category": "about_user_turn_level", "n_items": len(turn_keys),
             "kappa": cohen_kappa([k in human_user for k in turn_keys], [k in llm_user for k in turn_keys])}]

    texts = {tuple(r[:3]): r[3] for r in items.itertuples(index=False)}
    gold_by_turn = {k: g for k, g in gold[gold["about"] != ""].groupby(TURN)}
    pairs = []
    for r in llm_attrs.itertuples(index=False):
        k = (r.dataset, r.conversation_id, r.turn_index)
        if r.about not in LLM_ABOUT_TO_HUMAN or k not in texts or k not in gold_by_turn:
            continue
        pos = texts[k].casefold().find(r.value.casefold())
        if pos == -1:
            continue                                   # inferred or reworded: no span to compare
        for g in gold_by_turn[k].itertuples(index=False):
            if spans_match(pos, pos + len(r.value), g.start, g.end, min_overlap):
                pairs.append((g.about, LLM_ABOUT_TO_HUMAN[r.about]))
                break
    rows.append({"detector": "llm_vs_human", "category": "about_matched_spans", "n_items": len(pairs),
                 "kappa": cohen_kappa([h for h, _ in pairs], [m for _, m in pairs])})
    return pd.DataFrame(rows)
