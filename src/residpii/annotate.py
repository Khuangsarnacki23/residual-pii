"""Local labelling server (standard library only).

Binds to 127.0.0.1, checks the Host header (DNS-rebinding guard) and requires a random per-session
token on every API call (so other local web pages cannot read or write labels). In `annotate` mode it
never sends detector output, so annotators stay blind. In `adjudicate` mode it sends every
annotator's labels for the item, prefilled with what they agree on.

Offsets: browsers index strings in UTF-16 code units, Python in code points. All conversion happens
here, so stored labels are always code-point offsets that line up with the detector span store.
"""
from __future__ import annotations

import json
import secrets
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import labeling
from .config import Paths


def _u16_offsets(text: str) -> np.ndarray:
    """u16[i] = UTF-16 offset of code point i (length len(text)+1)."""
    widths = np.fromiter((2 if ord(c) > 0xFFFF else 1 for c in text), dtype=np.int64, count=len(text))
    return np.concatenate([[0], np.cumsum(widths)])


def _to_u16(spans: list[dict], u16: np.ndarray) -> list[dict]:
    return [{**s, "start": int(u16[s["start"]]), "end": int(u16[s["end"]])} for s in spans]


def _from_u16(spans: list[dict], u16: np.ndarray) -> list[dict]:
    out = []
    for s in spans:
        a, b = np.searchsorted(u16, [int(s["start"]), int(s["end"])])
        out.append({**s, "start": int(a), "end": int(b)})
    return out


class LabelApp:
    def __init__(self, cfg: dict, ds: str, annotator: str, mode: str = "annotate"):
        self.cfg, self.lcfg = cfg, cfg["labeling"]
        self.paths = Paths(cfg).ensure()
        self.ds, self.annotator, self.mode = ds, annotator, mode
        labeling.annotator_file(self.paths, ds, annotator)   # validates the name
        self.items = pd.read_parquet(labeling.items_path(self.paths, ds))
        turns = pd.read_parquet(self.paths.restricted / f"turns_{ds}.parquet")
        key = ["dataset", "conversation_id", "turn_index"]
        self.text = (self.items[["item_id", *key]].merge(turns[[*key, "text"]], on=key)
                     .set_index("item_id")["text"].to_dict())
        self.token = secrets.token_urlsafe(24)
        if mode == "adjudicate":
            self.refs = {a: labeling.load_labels(self.paths, ds, a) for a in self.lcfg["annotators"]}

    def config(self) -> dict:
        return {"annotator": f"{self.annotator} · {self.ds}", "mode": self.mode,
                "direct": self.lcfg["span_categories"]["direct"], "quasi": self.lcfg["span_categories"]["quasi"],
                "whose": self.lcfg["whose"], "attributes": self.lcfg["attributes"]}

    def item_list(self) -> list[dict]:
        own = labeling.load_labels(self.paths, self.ds, self.annotator)
        return [{"item_id": i, "done": i in own, "skip": i in own and own[i]["skip"]} for i in self.items["item_id"]]

    def item(self, iid: str) -> dict:
        text = self.text[iid]
        u16 = _u16_offsets(text)
        own = labeling.load_labels(self.paths, self.ds, self.annotator).get(iid)
        refs = {}
        if self.mode == "adjudicate":
            refs = {a: r[iid] for a, r in self.refs.items() if iid in r}
            if own is None and len(refs) == 2:
                own = labeling.prefill_adjudication(*refs.values(), self.lcfg)
        conv = lambda r: r and {**r, "spans": _to_u16(r["spans"], u16)}  # noqa: E731
        return {"item_id": iid, "text": text, "label": conv(own),
                "refs": {a: conv(r) for a, r in refs.items()}}

    def save(self, iid: str, body: dict) -> None:
        text = self.text[iid]
        body = {**body, "item_id": iid, "spans": _from_u16(body.get("spans", []), _u16_offsets(text))}
        labeling.save_label(self.paths, self.ds, self.annotator, labeling.validate_record(body, self.lcfg, len(text)))


def make_handler(app: LabelApp, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    page = files("residpii").joinpath("annotate.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):   # request lines can contain item ids only, but keep the console quiet
            pass

        def _send(self, status, body: bytes, ctype="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode())

        def _guard(self) -> bool:
            if self.headers.get("Host") not in allowed_hosts:
                self._send(HTTPStatus.FORBIDDEN, b"bad host", "text/plain")
                return False
            if urlparse(self.path).path.startswith("/api/") and not secrets.compare_digest(
                    self.headers.get("X-Token", ""), app.token):
                self._send(HTTPStatus.FORBIDDEN, b"bad token", "text/plain")
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            path = urlparse(self.path).path
            if path == "/":
                return self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            if path == "/api/config":
                return self._json(app.config())
            if path == "/api/items":
                return self._json(app.item_list())
            if path.startswith("/api/item/"):
                iid = path.rsplit("/", 1)[-1]
                if iid not in app.text:
                    return self._json({"error": "unknown item"}, HTTPStatus.NOT_FOUND)
                return self._json(app.item(iid))
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if not self._guard():
                return
            path = urlparse(self.path).path
            if not path.startswith("/api/label/"):
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            iid = path.rsplit("/", 1)[-1]
            if iid not in app.text:
                return self._json({"error": "unknown item"}, HTTPStatus.NOT_FOUND)
            try:
                n = int(self.headers.get("Content-Length", 0))
                app.save(iid, json.loads(self.rfile.read(n)))
            except (ValueError, KeyError, TypeError) as e:
                return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            self._json({"ok": True})

    return Handler


def serve(cfg: dict, ds: str, annotator: str, mode: str = "annotate", port: int | None = None) -> None:
    app = LabelApp(cfg, ds, annotator, mode)
    host, port = cfg["labeling"]["host"], port or cfg["labeling"]["port"]
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("the labelling server only binds to 127.0.0.1")
    httpd = ThreadingHTTPServer((host, port), make_handler(app, port))
    print(f"[label] {mode} {ds} as '{annotator}': {len(app.items)} items.\n"
          f"        Open http://127.0.0.1:{port}/#t={app.token}\n        Ctrl+C to stop.",
          file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
