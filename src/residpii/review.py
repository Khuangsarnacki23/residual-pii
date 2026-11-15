"""Blind, pooled human review of the sanitizers (RQ4).

For each review turn, every offline sanitizer stage is run and their edits are pooled: one row per
distinct span, with no indication of which tool made it. The reviewer marks each span as personal
(private information about the writer or a real person: should change), not personal, or unsure,
and selects any personal information that no tool touched. Scoring then credits each stage with its
own edits:

  precision  share of a stage's edits the reviewer judged personal
  recall     share of the real items (pooled spans judged personal + spans the reviewer added) a
             stage covers. "Pooled recall": items that every tool and the reviewer missed are not counted.
  clean      share of turns with no real item that the stage edited anyway

Review turns come from the same pinned WildChat sample as everything else. Source `fresh` (default)
draws turns outside the 300-turn label set, so rules tuned on the label set are judged on unseen text;
source `label_set` reviews the 260 non-templated label items instead. Fresh turns are drawn from a
random pool, oversampling turns that some tool edited; inclusion weights correct for that.

Candidate edits are frozen when the set is built, so later rule changes do not shift what was reviewed
(rebuild with --force to review new rules; earlier judgments of changed spans are then dropped).
Everything stays in the restricted directory except the aggregate scores.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import labeling, sanitize
from .config import Paths

JUDGMENTS = ("personal", "not_personal", "unsure")
KEY = ["dataset", "conversation_id", "turn_index"]


def review_dir(paths: Paths, ds: str, source: str) -> Path:
    d = paths.restricted / "review" / ds / source
    d.mkdir(parents=True, exist_ok=True)
    for p in (d, d.parent, d.parent.parent):
        os.chmod(p, 0o700)
    return d


def items_file(paths: Paths, ds: str, source: str) -> Path:
    return review_dir(paths, ds, source) / "items.parquet"


def pool_edits(stage_spans: dict[str, list[dict]]) -> list[dict]:
    """One candidate per distinct (start, end), listing the stages that edited it."""
    pooled: dict[tuple[int, int], dict] = {}
    for stage, spans in stage_spans.items():
        for s in spans:
            k = (int(s["start"]), int(s["end"]))
            c = pooled.setdefault(k, {"start": k[0], "end": k[1], "category": s["category"], "stages": []})
            if stage not in c["stages"]:
                c["stages"].append(stage)
    out = sorted(pooled.values(), key=lambda c: (c["start"], c["end"]))
    for i, c in enumerate(out):
        c["edit_id"] = f"e{i}"
    return out


def _eligible(cfg: dict, paths: Paths, ds: str) -> pd.DataFrame:
    turns = pd.read_parquet(paths.restricted / f"turns_{ds}.parquet")
    turns = turns[turns["text"].str.strip().str.len() > 0].reset_index(drop=True)
    templ = labeling.templated_conversations(turns, labeling.template_config(cfg))
    turns = turns[~pd.Series([(d, c) in templ for d, c in zip(turns["dataset"], turns["conversation_id"])],
                             index=turns.index)]
    turns = turns[turns["text"].str.len() <= cfg["labeling"]["max_chars"]]
    label_items = pd.read_parquet(labeling.items_path(paths, ds))[KEY]
    turns = turns.merge(label_items.assign(_in=1), on=KEY, how="left")
    return turns[turns["_in"].isna()].drop(columns="_in").reset_index(drop=True)


def cmd_build(cfg: dict, ds: str, source: str | None = None, n: int | None = None, force: bool = False) -> dict:
    rcfg = cfg["review"]
    source = source or rcfg["source"]
    n = n or rcfg["n_items"]
    paths = Paths(cfg).ensure()
    out = items_file(paths, ds, source)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass --force to redraw (earlier judgments of changed spans are dropped)")
    rng = np.random.default_rng(cfg["seed"] + 104729)
    if source == "label_set":
        items, _, _ = sanitize.eval_set(cfg, ds, cfg["sanitizer"]["gold"])
        items = items.rename(columns={"label_weight": "weight", "label_stratum": "stratum"})
        items = items[["item_id", *KEY, "text", "weight", "stratum"]]
    elif source == "fresh":
        elig = _eligible(cfg, paths, ds)
        pool = elig.iloc[rng.permutation(len(elig))[: rcfg["pool_size"]]].reset_index(drop=True)
        pool["item_id"] = [f"r{i:04d}" for i in range(len(pool))]
        items = pool
    else:
        raise ValueError("source must be 'fresh' or 'label_set'")
    _, spans = sanitize.offline_texts(cfg, ds, items, paths)
    stages = rcfg["stages"]
    items["edits"] = [json.dumps(pool_edits({s: spans[s][i] for s in stages}))
                      for i in items["item_id"]]
    items["n_edits"] = items["edits"].map(lambda e: len(json.loads(e)))
    summary = {"dataset": ds, "source": source, "stages": stages}
    if source == "fresh":
        # Oversample turns some tool edited; weights scale each stratum back to the eligible population.
        items["stratum"] = np.where(items["n_edits"] > 0, "edited", "unedited")
        sizes = items["stratum"].value_counts()
        want_edited = min(int(round(n * rcfg["edited_share"])), int(sizes.get("edited", 0)))
        take = {"edited": want_edited, "unedited": min(n - want_edited, int(sizes.get("unedited", 0)))}
        parts = []
        for h, k in take.items():
            g = items[items["stratum"] == h]
            pick = g.iloc[rng.choice(len(g), size=k, replace=False)].copy()
            pick["weight"] = (len(g) / k) * (len(elig) / len(items)) if k else 0.0
            parts.append(pick)
        items = pd.concat(parts).sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
        summary.update(eligible_turns=len(elig), pool=int(sizes.sum()),
                       pool_edited=int(sizes.get("edited", 0)), drawn=take)
    summary["n_items"] = len(items)
    summary["edits"] = int(items["n_edits"].sum())
    items.drop(columns=["text"]).to_parquet(out, index=False)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def cmd_refresh(cfg: dict, ds: str, source: str | None = None) -> dict:
    """Re-runs the current sanitizer rules on the same review turns (same draw, same weights) and carries
    every reviewer's judgments over to spans at the same offsets. Turns with new spans reopen on the page.
    Spans no stage makes any more are dropped with their judgments."""
    source = source or cfg["review"]["source"]
    paths = Paths(cfg).ensure()
    f = items_file(paths, ds, source)
    items = pd.read_parquet(f)
    turns = pd.read_parquet(paths.restricted / f"turns_{ds}.parquet")[[*KEY, "text"]]
    work = items.merge(turns, on=KEY, how="left")
    _, spans = sanitize.offline_texts(cfg, ds, work, paths)
    stages = cfg["review"]["stages"]
    old = {i: json.loads(e) for i, e in zip(items["item_id"], items["edits"])}
    new = {i: pool_edits({s: spans[s][i] for s in stages}) for i in items["item_id"]}
    summary = {"dataset": ds, "source": source, "edits_before": sum(map(len, old.values())),
               "edits_after": sum(map(len, new.values())), "annotators": {}}
    for lf in sorted(review_dir(paths, ds, source).glob("*.jsonl")):
        who = lf.stem
        labels = load_labels(paths, ds, source, who)
        carried = new_spans = dropped = 0
        for iid, rec in labels.items():
            by_pos = {(e["start"], e["end"]): rec["judgments"].get(e["edit_id"]) for e in old[iid]}
            judg = {}
            for e in new[iid]:
                v = by_pos.get((e["start"], e["end"]))
                if v:
                    judg[e["edit_id"]] = v
                    carried += 1
                else:
                    new_spans += 1
            dropped += sum(1 for k, v in by_pos.items() if v and k not in {(e["start"], e["end"]) for e in new[iid]})
            rec["judgments"] = judg
        lf.rename(lf.with_suffix(".jsonl.before_refresh"))   # keep the old file
        for rec in labels.values():
            save_label(paths, ds, source, who, rec)
        summary["annotators"][who] = {"judgments_carried": carried, "spans_to_judge": new_spans,
                                      "judgments_dropped": dropped}
    items["edits"] = [json.dumps(new[i]) for i in items["item_id"]]
    items["n_edits"] = [len(new[i]) for i in items["item_id"]]
    items.to_parquet(f, index=False)
    return summary


# ---------------------------------------------------------------- labels

def label_file(paths: Paths, ds: str, source: str, annotator: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", annotator):
        raise ValueError("annotator names may use letters, digits, '-' and '_' only")
    return review_dir(paths, ds, source) / f"{annotator}.jsonl"


def load_labels(paths: Paths, ds: str, source: str, annotator: str) -> dict[str, dict]:
    p = label_file(paths, ds, source, annotator)
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            out[r["item_id"]] = r
    return out


def validate(rec: dict, edits: list[dict], text_len: int, lcfg: dict) -> dict:
    ids = {e["edit_id"] for e in edits}
    judg = {k: v for k, v in (rec.get("judgments") or {}).items()}
    if set(judg) - ids or set(judg.values()) - set(JUDGMENTS):
        raise ValueError("bad judgments")
    cats = set(lcfg["span_categories"]["direct"]) | set(lcfg["span_categories"]["quasi"])
    missed = []
    for s in rec.get("missed") or []:
        a, b = int(s["start"]), int(s["end"])
        if not (0 <= a < b <= text_len) or s.get("category") not in cats or s.get("whose") not in ("self", "third_party"):
            raise ValueError("bad missed span")
        missed.append({"start": a, "end": b, "category": s["category"], "whose": s["whose"]})
    return {"item_id": rec["item_id"], "judgments": judg, "missed": sorted(missed, key=lambda s: (s["start"], s["end"])),
            "skip": bool(rec.get("skip")), "note": str(rec.get("note") or "")[:2000]}


def save_label(paths: Paths, ds: str, source: str, annotator: str, rec: dict) -> None:
    fd = os.open(label_file(paths, ds, source, annotator), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- server

class ReviewApp:
    """Plugs into annotate.make_handler: same host check, token and offset conversion."""

    def __init__(self, cfg: dict, ds: str, annotator: str, source: str | None = None):
        import secrets

        self.cfg, self.lcfg = cfg, cfg["labeling"]
        self.source = source or cfg["review"]["source"]
        self.paths = Paths(cfg).ensure()
        self.ds, self.annotator = ds, annotator
        label_file(self.paths, ds, self.source, annotator)   # validates the name
        self.items = pd.read_parquet(items_file(self.paths, ds, self.source))
        turns = pd.read_parquet(self.paths.restricted / f"turns_{ds}.parquet")
        self.text = (self.items[["item_id", *KEY]].merge(turns[[*KEY, "text"]], on=KEY)
                     .set_index("item_id")["text"].to_dict())
        self.edits = {i: json.loads(e) for i, e in zip(self.items["item_id"], self.items["edits"])}
        self.token = secrets.token_urlsafe(24)

    def config(self) -> dict:
        return {"annotator": f"{self.annotator} · {self.ds} · {self.source}", "mode": "review",
                "direct": self.lcfg["span_categories"]["direct"], "quasi": self.lcfg["span_categories"]["quasi"]}

    def item_list(self) -> list[dict]:
        own = load_labels(self.paths, self.ds, self.source, self.annotator)

        def done(i):   # every span judged (a refresh can add new ones), or skipped
            return i in own and (own[i]["skip"] or all(e["edit_id"] in own[i]["judgments"] for e in self.edits[i]))
        return [{"item_id": i, "done": done(i), "skip": i in own and own[i]["skip"]} for i in self.items["item_id"]]

    def item(self, iid: str) -> dict:
        from .annotate import _to_u16, _u16_offsets

        text = self.text[iid]
        u16 = _u16_offsets(text)
        own = load_labels(self.paths, self.ds, self.source, self.annotator).get(iid)
        # Blind: the page gets offsets and ids only, never categories or which tool made an edit.
        edits = _to_u16([{"edit_id": e["edit_id"], "start": e["start"], "end": e["end"]} for e in self.edits[iid]], u16)
        if own:
            own = {**own, "missed": _to_u16(own["missed"], u16)}
        return {"item_id": iid, "text": text, "edits": edits, "label": own}

    def save(self, iid: str, body: dict) -> None:
        from .annotate import _from_u16, _u16_offsets

        text = self.text[iid]
        body = {**body, "item_id": iid, "missed": _from_u16(body.get("missed") or [], _u16_offsets(text))}
        save_label(self.paths, self.ds, self.source, self.annotator, validate(body, self.edits[iid], len(text), self.lcfg))


def serve(cfg: dict, ds: str, annotator: str, source: str | None = None, port: int | None = None) -> None:
    import sys
    from http.server import ThreadingHTTPServer

    from .annotate import make_handler

    app = ReviewApp(cfg, ds, annotator, source)
    host, port = cfg["labeling"]["host"], port or cfg["review"]["port"]
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("the review server only binds to 127.0.0.1")
    httpd = ThreadingHTTPServer((host, port), make_handler(app, port, page="review.html"))
    print(f"[review] {ds} ({app.source}) as '{annotator}': {len(app.items)} turns.\n"
          f"         Open http://127.0.0.1:{port}/#t={app.token}\n         Ctrl+C to stop.", file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# ---------------------------------------------------------------- scoring

def score(cfg: dict, ds: str, annotator: str, source: str | None = None) -> pd.DataFrame:
    from .labeling import _ratio_ci, overlaps, tier_of

    lcfg, acfg = cfg["labeling"], cfg["analysis"]
    source = source or cfg["review"]["source"]
    paths = Paths(cfg).ensure()
    items = pd.read_parquet(items_file(paths, ds, source))
    labels = load_labels(paths, ds, source, annotator)
    items = items[items["item_id"].map(lambda i: i in labels and not labels[i]["skip"])].reset_index(drop=True)
    if items.empty:
        raise RuntimeError(f"No review labels from '{annotator}' ({ds}, {source}).")
    w, strata = items["weight"].to_numpy(float), items["stratum"].astype(str).to_numpy()
    mo = lcfg["min_overlap"]
    tier = lambda c: tier_of(c, lcfg) if c in lcfg["span_categories"]["direct"] + lcfg["span_categories"]["quasi"] else "direct"  # noqa: E731
    rows = []
    stages = cfg["review"]["stages"]
    per_item = []
    for iid, ej in zip(items["item_id"], items["edits"]):
        edits, lab = json.loads(ej), labels[iid]
        j = lab["judgments"]
        real = [{**e, "kind": "edit"} for e in edits if j.get(e["edit_id"]) == "personal"]
        real += [{**m, "kind": "missed", "stages": []} for m in lab["missed"]]
        per_item.append((edits, j, real))
    for stage in stages:
        for t in ("direct", "quasi", "any"):
            n_ed, n_ok, n_real, n_cov, clean, clean_ed = (np.zeros(len(items)) for _ in range(6))
            for k, (edits, j, real) in enumerate(per_item):
                mine = [e for e in edits if stage in e["stages"]]
                judged = [e for e in mine if j.get(e["edit_id"]) in ("personal", "not_personal")
                          and (t == "any" or tier(e["category"]) == t)]
                n_ed[k] = len(judged)
                n_ok[k] = sum(j[e["edit_id"]] == "personal" for e in judged)
                rr = [r for r in real if t == "any" or tier(r["category"]) == t]
                n_real[k] = len(rr)
                n_cov[k] = sum(any(overlaps((e["start"], e["end"]), (r["start"], r["end"]), mo) for e in mine) for r in rr)
                if t == "any":
                    clean[k] = not real
                    clean_ed[k] = (not real) and bool(mine)
            for metric, num, den in (("precision", n_ok, n_ed), ("pooled_recall", n_cov, n_real)) + \
                    ((("clean_turns_edited", clean_ed, clean),) if t == "any" else ()):
                p, lo, hi = _ratio_ci(num, den, w, strata, acfg["bootstrap_reps"], acfg["ci_level"],
                                      np.random.default_rng(labeling._seed(cfg)))
                if den.sum():
                    rows.append({"stage": stage, "tier": t, "metric": metric, "value": p, "ci_low": lo,
                                 "ci_high": hi, "n": int(den.sum())})
    unsure = sum(sum(v == "unsure" for v in j.values()) for _, j, _ in per_item)
    rows.append({"stage": "all", "tier": "any", "metric": "turns_reviewed", "value": float(len(items)),
                 "ci_low": np.nan, "ci_high": np.nan, "n": len(items)})
    rows.append({"stage": "all", "tier": "any", "metric": "unsure_judgments", "value": float(unsure),
                 "ci_low": np.nan, "ci_high": np.nan, "n": int(sum(len(j) for _, j, _ in per_item))})
    return pd.DataFrame(rows).assign(dataset=ds, source=source, annotator=annotator)


def cmd_score(cfg: dict, ds: str, annotator: str, source: str | None = None) -> pd.DataFrame:
    from .report import write_table

    t = score(cfg, ds, annotator, source)
    cols = ["dataset", "source", "annotator", *[c for c in t.columns if c not in ("dataset", "source", "annotator")]]
    write_table(t[cols], Paths(cfg).ensure().tables, f"review_scores_{t['source'].iloc[0]}")
    return t
