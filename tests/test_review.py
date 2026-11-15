"""Blind pooled review of the sanitizers: set building, blindness, validation, scoring, server guard,
and the canary check (no text in the score table)."""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pandas as pd
import pytest

from residpii import annotate, review

from test_labeling import labelled_run  # noqa: F401  (fixture)
from test_pipeline import CANARIES
from test_sanitize import _label


def test_pool_edits_dedupes_and_records_stages():
    pooled = review.pool_edits({"a": [{"start": 0, "end": 4, "category": "name"}],
                                "b": [{"start": 0, "end": 4, "category": "name"}, {"start": 6, "end": 9, "category": "phone"}]})
    assert [(p["start"], p["end"], p["stages"]) for p in pooled] == [(0, 4, ["a", "b"]), (6, 9, ["b"])]
    assert [p["edit_id"] for p in pooled] == ["e0", "e1"]


@pytest.mark.parametrize("source", ["fresh", "label_set"])
def test_build_review_score_and_blindness(labelled_run, source):  # noqa: F811
    cfg, pipe, _, _ = labelled_run
    _label(cfg, pipe)
    cfg["review"].update(n_items=8, pool_size=20)
    summary = review.cmd_build(cfg, "wildchat", source)
    items = pd.read_parquet(review.items_file(pipe.paths, "wildchat", source))
    assert "text" not in items.columns and len(items) == summary["n_items"] > 0
    if source == "fresh":   # outside the label set
        lab = pd.read_parquet(__import__("residpii.labeling", fromlist=["x"]).items_path(pipe.paths, "wildchat"))
        assert items.merge(lab, on=review.KEY).empty
    with pytest.raises(FileExistsError):
        review.cmd_build(cfg, "wildchat", source)

    app = review.ReviewApp(cfg, "wildchat", "kai", source)
    # Judge every edit "personal" and add one missed span per turn.
    for iid in items["item_id"]:
        it = app.item(iid)
        assert set(it["edits"][0]) == {"edit_id", "start", "end"} if it["edits"] else True   # blind: no tool, no category
        body = {"judgments": {e["edit_id"]: "personal" for e in it["edits"]},
                "missed": [{"start": 0, "end": 1, "category": "other_quasi", "whose": "self"}], "skip": False}
        app.save(iid, body)
    with pytest.raises(ValueError):
        app.save(items["item_id"].iloc[0], {"judgments": {"e999": "personal"}})
    t = review.cmd_score(cfg, "wildchat", "kai", source)
    prec = t[(t.metric == "precision") & (t.tier == "any")]
    assert (prec["value"] == 1).all()                       # everything judged personal
    blob = (pipe.paths.tables / f"review_scores_{source}.csv").read_text().casefold()
    assert not any(c.casefold() in blob for c in CANARIES)
    assert review.label_file(pipe.paths, "wildchat", source, "kai").stat().st_mode & 0o077 == 0


def test_review_server_requires_token(labelled_run):  # noqa: F811
    cfg, pipe, _, _ = labelled_run
    _label(cfg, pipe)
    cfg["review"].update(n_items=4, pool_size=10)
    review.cmd_build(cfg, "wildchat", "fresh")
    app = review.ReviewApp(cfg, "wildchat", "kai", "fresh")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), annotate.make_handler(app, 0, page="review.html"))
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = annotate.make_handler(app, port, page="review.html")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        def get(path, token):
            c = http.client.HTTPConnection("127.0.0.1", port)
            c.request("GET", path, headers={"Host": f"127.0.0.1:{port}", "X-Token": token})
            r = c.getresponse()
            return r.status, r.read()
        assert get("/api/items", "wrong")[0] == 403
        status, body = get("/api/items", app.token)
        assert status == 200 and len(json.loads(body)) == 4
        assert b"Sanitizer review" in get("/", "")[1]
    finally:
        httpd.shutdown()
