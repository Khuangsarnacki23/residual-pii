"""Local review interface for the two manual checks: judging a sample of LLM extractions, and
hand-labeling a batch of user turns (docs/annotation_guide.md).

`residpii review` serves one page on 127.0.0.1. The page shows raw text, so the server never
listens on another interface, needs a per-session token on every request, loads nothing from the
network and logs no request. Everything the reviewer enters is written under restricted/labels/;
the only output outside it is aggregate tables that pass `assert_safe_table`.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from . import label, llm, report
from .safety import keyed_hash, suppress_small_cells

ATTR_KEY = ["dataset", "conversation_id", "turn_index", "attribute_type", "value_hmac"]
ANSWER_COLUMNS = ["value_ok", "about_human", "evidence_human", "missed", "notes"]
REVIEW_COLUMNS = ["item_id", *ATTR_KEY, "evidence", "about", "confidence", *ANSWER_COLUMNS]
ANSWERS = {
    "value_ok": ("correct", "wrong", "unsure"),
    "about_human": tuple(llm.ABOUT),
    "evidence_human": ("stated", "inferred", "unsupported"),
    "missed": ("no", "yes"),             # did the model miss something about the user in this turn?
}
TOPICS = ("health", "sexual_orientation", "religion", "politics", "finances", "criminal_record", "immigration")
MAX_NOTES = 2000
MAX_CONTEXT_CHARS = 1500
MAX_BODY = 2 * 2**20
UI = Path(__file__).with_name("review_ui.html")


def _write_private(df: pd.DataFrame, path: Path) -> None:
    """Atomic 0600 write inside a 0700 directory, so a crash never leaves half a labels file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", newline="") as f:
        df.to_csv(f, index=False)
    os.replace(tmp, path)


def review_summary(rev: pd.DataFrame, min_cell: int) -> pd.DataFrame:
    """Aggregate agreement between the reviewer and the LLM. Counts and shares only."""
    judged = rev["value_ok"].isin(("correct", "wrong"))
    about = rev["about_human"] != ""
    evid = rev["evidence_human"] != ""
    missed = rev["missed"] != ""
    rows = [("value_correct", judged.sum(), (rev["value_ok"] == "correct").sum()),
            ("about_agrees", about.sum(), (about & (rev["about_human"] == rev["about"])).sum()),
            ("evidence_agrees", evid.sum(), (evid & (rev["evidence_human"] == rev["evidence"])).sum()),
            ("turn_has_missed_attribute", missed.sum(), (rev["missed"] == "yes").sum())]
    out = pd.DataFrame(rows, columns=["metric", "n_reviewed", "n_yes"]).astype({"n_reviewed": int, "n_yes": int})
    out["share"] = out["n_yes"] / out["n_reviewed"].where(out["n_reviewed"] > 0)
    return suppress_small_cells(out, "n_reviewed", ["n_yes", "share"], min_cell)


class Review:
    """State behind the page. Every method returns plain JSON-ready data."""

    def __init__(self, pipe, n_attrs: int = 20):
        self.pipe, self.n_attrs = pipe, n_attrs
        self.restricted = Path(pipe.paths.restricted)
        self.dir = self.restricted / "labels"
        self.categories = list(pipe.taxonomy["categories"])
        self._rev = self._attr_view = None
        self._batch: tuple[str, list[dict]] | None = None

    # ------------------------------------------------------------------ shared
    def _turn_texts(self, keys: pd.DataFrame, with_previous: bool = False) -> dict:
        """(dataset, conversation_id, turn_index) -> text, read from the restricted turn store."""
        out = {}
        for ds, g in keys.groupby("dataset"):
            path = self.restricted / f"turns_{ds}.parquet"
            if not path.exists():
                continue
            t = pd.read_parquet(path, columns=["conversation_id", "turn_index", "text"],
                                filters=[("conversation_id", "in", sorted(set(g["conversation_id"])))])
            for cid, idx, text in t.itertuples(index=False):
                out[(ds, cid, int(idx))] = text
        return out

    def overview(self) -> dict:
        rev = self._review()
        return {"run": self.pipe.cfg["run_name"], "categories": self.categories,
                "about_labels": list(label.ABOUT_LABELS), "llm_about": list(llm.ABOUT), "topics": list(TOPICS),
                "attrs": None if rev is None else {"n": len(rev), "done": int(self._attr_done(rev).sum())},
                "batches": self.batches()}

    # ------------------------------------------------------------------ check 1: LLM extractions
    @property
    def attr_path(self) -> Path:
        return self.dir / "attr_review.csv"

    def _llm_attrs(self) -> pd.DataFrame | None:
        frames = [pd.read_parquet(p) for p in sorted(self.restricted.glob("llm_attrs_*.parquet"))]
        frames = [f for f in frames if len(f)]
        if not frames:
            return None
        a = pd.concat(frames, ignore_index=True).rename(columns={"type": "attribute_type"})
        a["turn_index"] = a["turn_index"].astype(int)
        a["value_hmac"] = [keyed_hash(self.pipe.key, str(v)) for v in a["value"]]
        return a.drop_duplicates(ATTR_KEY).reset_index(drop=True)

    def _sample_attrs(self, attrs: pd.DataFrame) -> pd.DataFrame:
        """Half the sample is about the user (what profiles are built from), half is everything else."""
        rng = np.random.default_rng(self.pipe.cfg["seed"])
        n = min(self.n_attrs, len(attrs))
        user, other = np.flatnonzero(attrs["about"] == "user"), np.flatnonzero(attrs["about"] != "user")
        k = min(len(user), max((n + 1) // 2, n - len(other)))
        pick = np.concatenate([rng.choice(user, k, replace=False), rng.choice(other, n - k, replace=False)])
        out = attrs.iloc[rng.permutation(pick)][[*ATTR_KEY, "evidence", "about", "confidence"]].copy()
        out.insert(0, "item_id", [f"a{i + 1:03d}" for i in range(len(out))])
        for c in ANSWER_COLUMNS:
            out[c] = ""
        return out.reset_index(drop=True)

    def _review(self) -> pd.DataFrame | None:
        if self._rev is not None:
            return self._rev
        attrs = self._llm_attrs()
        if attrs is None:
            return None
        if self.attr_path.exists():
            rev = pd.read_csv(self.attr_path, dtype=str, keep_default_na=False)
            rev["turn_index"] = rev["turn_index"].astype(int)
        else:
            rev = self._sample_attrs(attrs)
            _write_private(rev[REVIEW_COLUMNS], self.attr_path)
        # Values and text are looked up fresh each session; the review file holds neither.
        view = rev[["item_id", *ATTR_KEY]].merge(attrs[[*ATTR_KEY, "value", "occupation_group"]], on=ATTR_KEY)
        texts = self._turn_texts(pd.concat([view, view.assign(turn_index=view["turn_index"] - 1)]))
        view["text"] = [texts.get((r.dataset, r.conversation_id, r.turn_index), "")
                        for r in view.itertuples(index=False)]
        view["previous"] = [texts.get((r.dataset, r.conversation_id, r.turn_index - 1), "")[:MAX_CONTEXT_CHARS]
                            for r in view.itertuples(index=False)]
        self._rev, self._attr_view = rev, view.set_index("item_id")
        return rev

    @staticmethod
    def _attr_done(rev: pd.DataFrame) -> pd.Series:
        return (rev["value_ok"] != "") & (rev["missed"] != "")

    def _tally(self, rev: pd.DataFrame) -> list[dict]:
        s = review_summary(rev, min_cell=1)        # shown to the reviewer only; the table is suppressed
        return [{"metric": r.metric, "n": int(r.n_reviewed or 0), "yes": int(r.n_yes or 0)}
                for r in s.fillna(0).itertuples(index=False)]

    def attr_items(self) -> dict:
        rev = self._review()
        if rev is None:
            return {"items": [], "tally": []}
        max_chars = int(self.pipe.cfg["label"]["max_text_chars"])
        items = []
        for r in rev.to_dict("records"):
            if r["item_id"] not in self._attr_view.index:
                continue                             # the attribute store changed since the sample was drawn
            v = self._attr_view.loc[r["item_id"]]
            group = v["occupation_group"]
            items.append({"item_id": r["item_id"], "type": r["attribute_type"], "value": str(v["value"]),
                          "evidence": r["evidence"], "about": r["about"], "confidence": float(r["confidence"]),
                          "occupation_group": group if isinstance(group, str) else "",
                          "text": v["text"][:max_chars], "truncated": len(v["text"]) > max_chars,
                          "previous": v["previous"], "answers": {c: r[c] for c in ANSWER_COLUMNS}})
        return {"items": items, "tally": self._tally(rev)}

    def save_attr(self, item_id: str, answers: dict) -> dict:
        rev = self._review()
        if rev is None or item_id not in set(rev["item_id"]):
            raise ValueError("unknown item")
        row = rev["item_id"] == item_id
        for col, allowed in ANSWERS.items():
            val = str(answers.get(col, ""))
            if val not in ("", *allowed):
                raise ValueError(f"invalid answer for {col}")
            rev.loc[row, col] = val
        rev.loc[row, "notes"] = str(answers.get("notes", ""))[:MAX_NOTES]
        _write_private(rev[REVIEW_COLUMNS], self.attr_path)
        report.write_table(review_summary(rev, self.pipe.min_cell), self.pipe.paths.tables, "llm_attribute_review")
        return {"done": int(self._attr_done(rev).sum()), "tally": self._tally(rev)}

    # ------------------------------------------------------------------ check 2: span labels
    def _batch_path(self, name: str) -> Path:
        paths = {p.name: p for p in self.dir.glob("batch_*.csv")}
        if name not in paths:                        # only files this tool wrote, never a caller's path
            raise ValueError("unknown batch")
        return paths[name]

    @staticmethod
    def _parse_batch(df: pd.DataFrame) -> list[dict]:
        items = []
        for item_id, g in df[df["item_id"].str.strip() != ""].groupby("item_id", sort=False):
            with_text = g[g["text"] != ""]
            if with_text.empty:
                continue
            first, spans = with_text.iloc[0], []
            text = first["text"]
            for r in g.itertuples(index=False):
                typ = r.pii_type.strip().casefold()
                if typ in ("", label.NONE):
                    continue
                if str(r.start).strip().isdigit() and str(r.end).strip().isdigit():
                    found = [(int(r.start), int(r.end))]
                else:                                # a row filled in by hand in a spreadsheet
                    found, needle, pos = [], r.span_text.strip(), 0
                    while needle and (pos := text.find(needle, pos)) != -1:
                        found.append((pos, pos + len(needle)))
                        pos += len(needle)
                spans += [{"start": s, "end": e, "pii_type": typ, "about": r.about.strip().casefold()}
                          for s, e in found if 0 <= s < e <= len(text)]
            none = not spans and (g["pii_type"].str.strip().str.casefold() == label.NONE).any()
            items.append({"item_id": item_id, "dataset": first["dataset"], "conversation_id": first["conversation_id"],
                          "turn_index": first["turn_index"], "text": text, "spans": spans, "none": bool(none),
                          "sensitive_topic": max(g["sensitive_topic"], key=len), "notes": max(g["notes"], key=len)})
        return items

    @staticmethod
    def _batch_frame(items: list[dict]) -> pd.DataFrame:
        rows = []
        for it in items:
            ids = [it[c] for c in label.ID_COLUMNS]
            tail = [it["sensitive_topic"], it["notes"]]
            if it["spans"]:
                rows += [[*ids, it["text"][s["start"]:s["end"]], s["start"], s["end"], s["pii_type"], s["about"], *tail]
                         for s in it["spans"]]
            else:
                rows.append([*ids, "", "", "", label.NONE if it["none"] else "", "", *tail])
        return pd.DataFrame(rows, columns=label.LABEL_COLUMNS)

    def _load_batch(self, name: str) -> list[dict]:
        if self._batch is None or self._batch[0] != name:
            df = pd.read_csv(self._batch_path(name), dtype=str, keep_default_na=False)
            missing = [c for c in label.LABEL_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError("batch file is missing label columns")
            self._batch = (name, self._parse_batch(df))
        return self._batch[1]

    def batches(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("batch_*.csv")):
            try:
                items = self._load_batch(p.name) if self._batch and self._batch[0] == p.name else \
                    self._parse_batch(pd.read_csv(p, dtype=str, keep_default_na=False))
            except (ValueError, KeyError, AttributeError):
                continue
            out.append({"name": p.name, "n": len(items), "done": sum(bool(i["spans"]) or i["none"] for i in items)})
        return out

    def new_batch(self, n: int) -> dict:
        if not 1 <= n <= 2000:
            raise ValueError("batch size must be between 1 and 2000")
        return {"name": self.pipe.label_export(n).name}

    def batch_items(self, name: str) -> dict:
        return {"name": name, "items": self._load_batch(name)}

    def save_item(self, name: str, item_id: str, data: dict) -> dict:
        items = self._load_batch(name)
        item = next((i for i in items if i["item_id"] == item_id), None)
        if item is None:
            raise ValueError("unknown item")
        spans = set()
        for s in data.get("spans") or []:
            start, end, typ, about = int(s["start"]), int(s["end"]), str(s["pii_type"]), str(s.get("about", ""))
            if not 0 <= start < end <= len(item["text"]) or typ not in self.categories \
                    or about not in ("", *label.ABOUT_LABELS):
                raise ValueError("invalid span")
            spans.add((start, end, typ, about))
        topic = str(data.get("sensitive_topic", ""))
        if topic not in ("", *TOPICS):
            raise ValueError("invalid sensitive topic")
        item.update(spans=[dict(zip(("start", "end", "pii_type", "about"), s)) for s in sorted(spans)],
                    none=bool(data.get("none")) and not spans, sensitive_topic=topic,
                    notes=str(data.get("notes", ""))[:MAX_NOTES])
        _write_private(self._batch_frame(items), self._batch_path(name))
        return {"done": sum(bool(i["spans"]) or i["none"] for i in items)}

    def score(self, name: str) -> dict:
        """Detector precision/recall against the batch. The result is the aggregate table itself."""
        out = self.pipe.label_score(self._batch_path(name))
        return {"columns": list(out.columns), "rows": json.loads(out.to_json(orient="values"))}


# ---------------------------------------------------------------------- HTTP


def make_server(review: Review, port: int = 0) -> tuple[HTTPServer, str]:
    token = secrets.token_urlsafe(24)
    page = UI.read_text(encoding="utf-8").replace("__TOKEN__", token).encode("utf-8")
    get = {"/api/overview": lambda q: review.overview(),
           "/api/attrs": lambda q: review.attr_items(),
           "/api/batch": lambda q: review.batch_items(q.get("name", [""])[0])}
    post = {"/api/attr": lambda b: review.save_attr(str(b.get("item_id")), b.get("answers") or {}),
            "/api/item": lambda b: review.save_item(str(b.get("name")), str(b.get("item_id")), b),
            "/api/new-batch": lambda b: review.new_batch(int(b.get("n", 0))),
            "/api/score": lambda b: review.score(str(b.get("name")))}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):               # request lines stay out of the terminal
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                             "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"))

        def _local(self) -> bool:
            # A page on another origin cannot guess the token, and a rebound hostname fails here.
            return self.headers.get("Host", "") in (f"127.0.0.1:{self.server.server_port}",
                                                    f"localhost:{self.server.server_port}")

        def _authorized(self) -> bool:
            return secrets.compare_digest(self.headers.get("X-Review-Token", ""), token)

        def _run(self, fn, arg) -> None:
            try:
                self._json(200, fn(arg))
            except ValueError as e:                 # messages raised above; they never hold text
                self._json(400, {"error": str(e)})
            except Exception as e:                  # the message may quote data, so only its type is sent
                self._json(500, {"error": type(e).__name__})

        def do_GET(self):
            url = urlparse(self.path)
            if not self._local():
                return self._json(403, {"error": "forbidden"})
            if url.path == "/":
                if not secrets.compare_digest(parse_qs(url.query).get("t", [""])[0], token):
                    return self._json(403, {"error": "forbidden"})
                return self._send(200, page, "text/html")
            if url.path not in get:
                return self._json(404, {"error": "not found"})
            if not self._authorized():
                return self._json(403, {"error": "forbidden"})
            self._run(get[url.path], parse_qs(url.query))

        def do_POST(self):
            url = urlparse(self.path)
            if not self._local() or not self._authorized():
                return self._json(403, {"error": "forbidden"})
            if url.path not in post:
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._json(413, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": "invalid JSON"})
            self._run(post[url.path], body if isinstance(body, dict) else {})

    server = HTTPServer(("127.0.0.1", port), Handler)
    return server, f"http://127.0.0.1:{server.server_port}/?t={token}"


def serve(pipe, port: int = 0, n_attrs: int = 20, open_browser: bool = True) -> None:
    server, url = make_server(Review(pipe, n_attrs), port)
    print(f"Review page (this computer only):\n  {url}\nIt shows raw text; do not share your screen. "
          "Press Ctrl-C here when you are done. Your answers are saved as you go.", file=sys.stderr)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
