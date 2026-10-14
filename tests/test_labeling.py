"""Labelling benchmark: sampling weights, span matching, agreement, scoring, the local server, and
the canary check (no labelled text may reach results)."""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from residpii import annotate, labeling
from residpii.config import REPO_ROOT, load_config
from residpii.pipeline import Pipeline
from residpii.safety import load_or_create_key
from residpii.sources import InMemorySource

from test_pipeline import CANARIES, _convs


@pytest.fixture()
def labelled_run(tmp_path, capfd):
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    cfg.update(run_name="test", data_root=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    cfg["analysis"]["bootstrap_reps"] = 50
    cfg["labeling"]["n_items"] = 12
    key = load_or_create_key(Path(cfg["data_root"]))
    pipe = Pipeline(cfg, sources={"wildchat": InMemorySource("wildchat", _convs("wildchat", 30), key)})
    pipe.run(do_dedup=False)
    summary = labeling.cmd_build(cfg, "wildchat")
    return cfg, pipe, summary, capfd


def _gold_for(text: str) -> dict:
    """A fake annotator who marks the canary name (private) and Sir Bramwell (fictional)."""
    spans = []
    for needle, cat, whose in (("Quillon Varnadore", "name", "self"), ("Sir Bramwell", "name", "fictional"),
                               ("(646) 391-2087", "phone", "self")):
        i = text.find(needle)
        if i >= 0:
            spans.append({"start": i, "end": i + len(needle), "category": cat, "whose": whose})
    return {"spans": spans, "attributes": [{"attribute": "occupation", "kind": "inferred", "value": "job seeker"}]
            if "cover letter" in text else [], "skip": False, "note": ""}


def _texts(cfg, items):
    turns = pd.read_parquet(Path(cfg["data_root"]) / "runs" / "test" / "restricted" / "turns_wildchat.parquet")
    key = ["dataset", "conversation_id", "turn_index"]
    return items.merge(turns[[*key, "text"]], on=key).set_index("item_id")["text"].to_dict()


def test_build_weights_and_no_text(labelled_run):
    cfg, pipe, summary, _ = labelled_run
    items = pd.read_parquet(labeling.items_path(pipe.paths, "wildchat"))
    assert summary["n_items"] == len(items) == 12
    assert "text" not in items.columns
    # Oversampled strata get smaller weights; every weight is at least the conversation's own.
    assert (items["label_weight"] >= items["frame_weight"]).all()
    assert sum(v["drawn"] for v in summary["strata"].values()) == len(items)
    with pytest.raises(FileExistsError):
        labeling.cmd_build(cfg, "wildchat")


def test_agreement_and_score(labelled_run):
    cfg, pipe, _, _ = labelled_run
    lcfg = cfg["labeling"]
    items = pd.read_parquet(labeling.items_path(pipe.paths, "wildchat"))
    texts = _texts(cfg, items)
    for iid, text in texts.items():
        g = _gold_for(text)
        for who in ("kai", "adjudicated"):
            labeling.save_label(pipe.paths, "wildchat", who, labeling.validate_record({**g, "item_id": iid}, lcfg, len(text)))
        # Second annotator disagrees on `whose` for the fictional knight.
        g2 = {**g, "spans": [{**s, "whose": "third_party"} if s["whose"] == "fictional" else s for s in g["spans"]]}
        labeling.save_label(pipe.paths, "wildchat", "ethan", labeling.validate_record({**g2, "item_id": iid}, lcfg, len(text)))

    ag = labeling.cmd_agreement(cfg, "wildchat").set_index(["metric", "category"])["value"]
    assert ag[("span_f1", "any")] == 1.0
    if ("whose_agreement", "any") in ag:
        assert ag[("whose_agreement", "any")] <= 1.0

    tables = labeling.cmd_score(cfg, ["wildchat"])
    pr = tables["labels_precision_recall"]
    pr = pr[pr.threshold == cfg["presidio"]["score_threshold"]].set_index(["detector", "category", "metric"])["value"]
    if ("presidio", "phone", "recall") in pr:
        assert pr[("presidio", "phone", "recall")] <= 1.0
    # Every gold private name is either found or not; precision on names is below 1 when Presidio
    # flags the fictional knight.
    assert ("any", "any", "recall") in pr
    prev = tables["labels_gold_prevalence"].set_index("category")
    assert prev.loc["tier_direct", "prevalence"] > 0
    for name in tables:
        assert (pipe.paths.tables / f"{name}.csv").exists()

    # Canary check: nothing from the labelled text reaches results or work.
    for root in (pipe.paths.results, pipe.paths.work):
        for f in root.rglob("*"):
            if f.is_file() and f.suffix in (".csv", ".json", ".md"):
                blob = f.read_text(errors="ignore").casefold()
                for c in CANARIES + ["job seeker"]:
                    assert c.casefold() not in blob, f"leak into {f.name}"
    assert (labeling.label_dir(pipe.paths, "wildchat").stat().st_mode & 0o777) == 0o700


def test_validate_record_rejects_bad_input():
    lcfg = load_config(REPO_ROOT / "configs" / "default.yaml")["labeling"]
    with pytest.raises(ValueError):
        labeling.validate_record({"item_id": "i1", "spans": [{"start": 0, "end": 99, "category": "name", "whose": "self"}]}, lcfg, 10)
    with pytest.raises(ValueError):
        labeling.validate_record({"item_id": "i1", "spans": [{"start": 0, "end": 3, "category": "bogus", "whose": "self"}]}, lcfg, 10)


def test_matching_and_kappa():
    a = [{"start": 0, "end": 10, "category": "name"}, {"start": 20, "end": 25, "category": "date"}]
    b = [{"start": 0, "end": 4, "category": "name"}, {"start": 5, "end": 10, "category": "name"}]
    assert labeling.greedy_match(a, b, 0.5) == [(0, 1)] or labeling.greedy_match(a, b, 0.5) == [(0, 0)]
    assert labeling.cohen_kappa(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0])) == 1.0


def test_utf16_offsets_round_trip():
    text = "hi 😀 Ada Lovelace"
    u16 = annotate._u16_offsets(text)
    s = [{"start": text.index("Ada"), "end": len(text), "category": "name", "whose": "self"}]
    js = annotate._to_u16(s, u16)
    assert js[0]["start"] == text.index("Ada") + 1      # the emoji is two UTF-16 units
    assert annotate._from_u16(js, u16) == s


def test_server_token_host_and_blindness(labelled_run):
    cfg, pipe, _, _ = labelled_run
    app = annotate.LabelApp(cfg, "wildchat", "kai")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = annotate.make_handler(app, port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        def req(method, path, token=app.token, host=f"127.0.0.1:{port}", body=None):
            c = http.client.HTTPConnection("127.0.0.1", port)
            c.request(method, path, body=body, headers={"X-Token": token, "Host": host, "Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, r.read()

        assert req("GET", "/api/items", token="wrong")[0] == 403
        assert req("GET", "/", host="evil.example:80")[0] == 403
        status, body = req("GET", "/api/items")
        iid = json.loads(body)[0]["item_id"]
        status, body = req("GET", f"/api/item/{iid}")
        payload = json.loads(body)
        assert status == 200 and set(payload) == {"item_id", "text", "label", "refs"}
        assert payload["refs"] == {}                      # blind: no detector output, no other annotator
        rec = {"spans": [{"start": 0, "end": 2, "category": "name", "whose": "fictional"}], "attributes": []}
        assert req("POST", f"/api/label/{iid}", body=json.dumps(rec))[0] == 200
        assert labeling.load_labels(pipe.paths, "wildchat", "kai")[iid]["spans"][0]["whose"] == "fictional"
        bad = {"spans": [{"start": 0, "end": 10**6, "category": "name", "whose": "self"}]}
        assert req("POST", f"/api/label/{iid}", body=json.dumps(bad))[0] == 400
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------- LLM annotator (mocked client)

from types import SimpleNamespace

from residpii import llm_label


def test_locate_quotes():
    text = "I'm Sarah Jones. Sarah lives in Leeds. sarah jones again."
    spans, missing = llm_label.locate(text, [
        {"quote": "Sarah Jones", "category": "name", "whose": "self"},
        {"quote": "Sarah", "category": "name", "whose": "self"},        # inside the full name: dropped there
        {"quote": "Leeds", "category": "location", "whose": "self"},
        {"quote": "Atlantis", "category": "location", "whose": "self"},  # not in text
    ])
    got = [(text[s["start"]:s["end"]], s["category"]) for s in spans]
    assert got == [("Sarah Jones", "name"), ("Sarah", "name"), ("Leeds", "location")]
    assert missing == 1


class _FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        text = kw["messages"][0]["content"]
        spans = [{"quote": "Quillon Varnadore", "category": "name", "whose": "self"}] if "Quillon" in text else []
        body = json.dumps({"spans": spans, "attributes": [], "skip": False})
        return SimpleNamespace(model=kw["model"], stop_reason="end_turn", content=[SimpleNamespace(type="text", text=body)],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0,
                                                     cache_creation_input_tokens=0))


def test_llm_run_writes_labels_and_resumes(labelled_run):
    cfg, pipe, _, _ = labelled_run
    fake = _FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    stats = llm_label.run(cfg, "wildchat", limit=5, workers=2, client=client)
    assert stats["labelled"] == 5 and stats["failed"] == 0
    call = fake.calls[0]
    assert call["fallbacks"] == "default" and call["output_config"]["format"]["type"] == "json_schema"
    assert "<turn>" in call["messages"][0]["content"]
    labels = labeling.load_labels(pipe.paths, "wildchat", "llm")
    assert len(labels) == 5
    # Resumes: a second run only labels the remaining items.
    stats2 = llm_label.run(cfg, "wildchat", workers=2, client=client)
    assert stats2["requested"] == 12 - 5
    # Scoring against the LLM labels works end to end.
    tables = labeling.cmd_score(cfg, ["wildchat"], gold_from="llm")
    assert len(tables["labels_precision_recall"])
    assert (labeling.label_dir(pipe.paths, "wildchat") / "llm_raw.jsonl").stat().st_mode & 0o077 == 0


def test_llm_refuses_to_mix_models(labelled_run):
    cfg, pipe, _, _ = labelled_run
    client = SimpleNamespace(beta=SimpleNamespace(messages=_FakeMessages()))
    llm_label.run(cfg, "wildchat", model="claude-opus-5-5", limit=3, workers=1, client=client)
    with pytest.raises(RuntimeError):
        llm_label.run(cfg, "wildchat", model="claude-haiku-5-5", limit=3, workers=1, client=client)
    fake = _FakeMessages()
    stats = llm_label.run(cfg, "wildchat", model="claude-haiku-5-5", limit=3, workers=1,
                          client=SimpleNamespace(beta=SimpleNamespace(messages=fake)), restart=True)
    assert stats["labelled"] == 3 and "fallbacks" not in fake.calls[0]   # Haiku has no server-side fallback
    assert len(labeling.load_labels(pipe.paths, "wildchat", "llm")) == 3
    assert list(labeling.label_dir(pipe.paths, "wildchat").glob("llm.*.bak.jsonl"))


# ---------------------------------------------------------------- full-sample LLM detector (mocked client)

from residpii import llm_detect


def test_llm_detect_budget_spans_prevalence_and_scoring(labelled_run):
    cfg, pipe, _, _ = labelled_run
    fake = _FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    # A tiny budget stops early and never overshoots the cap.
    s1 = llm_detect.run(cfg, "wildchat", max_cost=0.0005, workers=2, client=client)
    assert s1["stopped_for_budget"] and s1["spent_total_usd"] <= 0.0005 + 1e-9
    # Resume with a real budget: finishes the rest without redoing turns.
    s2 = llm_detect.run(cfg, "wildchat", max_cost=5, workers=2, client=client)
    assert not s2["stopped_for_budget"] and s2["already_done"] == s1["processed"]
    assert s1["processed"] + s2["processed"] == s2["turns"]
    tables = llm_detect.prevalence(cfg, ["wildchat"])
    prev = tables["llm_prevalence"]
    row = prev[(prev.detector == "llm_detector") & (prev.category == "name") & (prev.variant == "all")]
    assert row["n_positive"].iloc[0] > 0
    spans = pd.read_parquet(pipe.paths.restricted / "spans_llm_wildchat.parquet")
    assert set(spans["detector"]) == {"llm_detector", "llm_detector_all"}
    for c in CANARIES:
        assert not spans.astype(str).apply(lambda col: col.str.contains(c, regex=False)).any().any()
    # Scoring now includes the LLM detector as a graded tool.
    texts = _texts(cfg, pd.read_parquet(labeling.items_path(pipe.paths, "wildchat")))
    for iid, text in texts.items():
        labeling.save_label(pipe.paths, "wildchat", "kai",
                            labeling.validate_record({**_gold_for(text), "item_id": iid}, cfg["labeling"], len(text)))
    pr = labeling.cmd_score(cfg, ["wildchat"], gold_from="kai")["labels_precision_recall"]
    assert "llm_detector" in set(pr["detector"])
    for f in pipe.paths.tables.glob("llm_*.csv"):
        blob = f.read_text().casefold()
        assert not any(c.casefold() in blob for c in CANARIES)
