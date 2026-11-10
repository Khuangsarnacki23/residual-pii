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
# Quasi-identifying attributes of a message, labeled next to its spans (review page). An inferred
# attribute has no span to select, so these live in their own file: attrs_<batch file name>.
ATTR_LABEL_COLUMNS = ["item_id", "dataset", "conversation_id", "turn_index", "attribute_type", "evidence", "about",
                      "value"]
ATTR_EVIDENCE = ("stated", "inferred")
AFTER_REVEAL = "after_reveal"   # "yes" on a label added after the annotator looked at the detectors' output


def attrs_path(labels_path: str | Path) -> Path:
    p = Path(labels_path)
    return p.with_name(f"attrs_{p.stem}.csv")


def read_attribute_labels(labels_path: str | Path) -> pd.DataFrame:
    """The attribute labels that go with a batch file; empty when none were made."""
    path = attrs_path(labels_path)
    if not path.exists():
        return pd.DataFrame(columns=[*ATTR_LABEL_COLUMNS, AFTER_REVEAL])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if AFTER_REVEAL not in df:
        df[AFTER_REVEAL] = ""
    df["turn_index"] = df["turn_index"].astype(int)
    return df
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
        blind = str(getattr(r, AFTER_REVEAL, "")).strip().casefold() != "yes"
        rows += [(r.dataset, r.conversation_id, r.turn_index, s, e, r.pii_type, about, blind) for s, e in found]
    cols = [*TURN, "start", "end", "category", "about", "blind"]
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


# ---------------------------------------------------------------- attributes and second annotator

HUMAN_ABOUT_FROM_LLM = {**LLM_ABOUT_TO_HUMAN, "unclear": ""}


def score_attributes(gold: pd.DataFrame, llm_attrs: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """LLM attributes against hand-labeled ones, on labeled turns the LLM answered.

    The unit is (turn, attribute type, about): the LLM is right when the annotator recorded the same
    attribute type about the same kind of person in that turn. Values are not compared. One row per
    attribute type x evidence, where evidence is the annotator's for recall (how much of what was
    stated, or only inferable, the model found) and the model's for precision."""
    unit = [*TURN, "attribute_type", "about"]
    turns = turns[TURN].drop_duplicates()
    g = gold.merge(turns, on=TURN)[[*unit, "evidence"]].drop_duplicates(unit)
    m = llm_attrs.rename(columns={"type": "attribute_type"}).merge(turns, on=TURN)
    m = m.assign(about=m["about"].map(HUMAN_ABOUT_FROM_LLM))[[*unit, "evidence"]].drop_duplicates(unit)
    g_hit = g.merge(m[unit].assign(hit=True), on=unit, how="left")["hit"].fillna(False).to_numpy(dtype=bool)
    m_hit = m.merge(g[unit].assign(hit=True), on=unit, how="left")["hit"].fillna(False).to_numpy(dtype=bool)
    rows = []
    for typ in [*sorted(set(g["attribute_type"]) | set(m["attribute_type"])), "all"]:
        for ev in (*ATTR_EVIDENCE, "any"):
            gm = ((g["attribute_type"] == typ) | (typ == "all")) & ((g["evidence"] == ev) | (ev == "any"))
            mm = ((m["attribute_type"] == typ) | (typ == "all")) & ((m["evidence"] == ev) | (ev == "any"))
            rows.append({"attribute_type": typ, "evidence": ev,
                         **_prf(int(gm.sum()), int(mm.sum()), int(m_hit[mm.to_numpy()].sum()), int(g_hit[gm.to_numpy()].sum()))})
    return pd.DataFrame(rows)


def annotator_agreement(a: pd.DataFrame, b: pd.DataFrame, a_attrs: pd.DataFrame, b_attrs: pd.DataFrame,
                        categories: list[str], min_overlap: float = 0.5) -> pd.DataFrame:
    """Agreement between two annotators of the same batch, on the turns both labeled.

    Spans: F1 per category, counting a span as agreed when the other annotator has a matching span of
    the same category (F1 is the usual measure for spans; kappa needs a fixed set of items). Turns:
    kappa on whether the turn holds any personal information. Attributes: F1 on (turn, type, about),
    for stated and inferred labels separately, by each annotator's own evidence label."""
    def labeled(df):
        return df[df.groupby("item_id")["pii_type"].transform(lambda s: (s != "").any())]
    a, b = labeled(a), labeled(b)
    both = a[TURN].drop_duplicates().merge(b[TURN].drop_duplicates(), on=TURN)
    ga, gb = (gold_spans(x.merge(both, on=TURN), categories)[0] for x in (a, b))
    rows = []
    hit_a, hit_b = _matched(ga, gb, min_overlap), _matched(gb, ga, min_overlap)
    for cat in [*sorted(set(ga["category"]) | set(gb["category"])), "all"]:
        ma = np.ones(len(ga), bool) if cat == "all" else (ga["category"] == cat).to_numpy()
        mb = np.ones(len(gb), bool) if cat == "all" else (gb["category"] == cat).to_numpy()
        s = _prf(int(mb.sum()), int(ma.sum()), int(hit_a[ma].sum()), int(hit_b[mb].sum()))
        rows.append({"measure": "span_f1", "category": cat, "n_a": s["n_pred"], "n_b": s["n_gold"], "agreement": s["f1"]})
    key = lambda df: set(map(tuple, df[TURN].to_numpy()))  # noqa: E731
    turns = list(map(tuple, both.to_numpy()))
    xa = key(ga) | key(a_attrs.merge(both, on=TURN)) if len(a_attrs) else key(ga)
    xb = key(gb) | key(b_attrs.merge(both, on=TURN)) if len(b_attrs) else key(gb)
    rows.append({"measure": "turn_kappa", "category": "has_personal_information", "n_a": len(xa), "n_b": len(xb),
                 "agreement": cohen_kappa([t in xa for t in turns], [t in xb for t in turns])})
    unit = [*TURN, "attribute_type", "about"]
    ua, ub = (x.merge(both, on=TURN)[[*unit, "evidence"]].drop_duplicates(unit) for x in (a_attrs, b_attrs))
    in_b = ua.merge(ub[unit].assign(hit=True), on=unit, how="left")["hit"].fillna(False).to_numpy(dtype=bool)
    in_a = ub.merge(ua[unit].assign(hit=True), on=unit, how="left")["hit"].fillna(False).to_numpy(dtype=bool)
    for ev in (*ATTR_EVIDENCE, "any"):
        ma = ((ua["evidence"] == ev) | (ev == "any")).to_numpy()
        mb = ((ub["evidence"] == ev) | (ev == "any")).to_numpy()
        s = _prf(int(mb.sum()), int(ma.sum()), int(in_b[ma].sum()), int(in_a[mb].sum()))
        rows.append({"measure": "attribute_f1", "category": ev, "n_a": s["n_pred"], "n_b": s["n_gold"], "agreement": s["f1"]})
    out = pd.DataFrame(rows)
    out.insert(0, "n_turns_both", len(turns))
    return out
