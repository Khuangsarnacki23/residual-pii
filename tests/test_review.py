"""The local review page: what it writes, where, and who may talk to it. All text is fabricated."""
import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from residpii import label, review
from residpii.safety import assert_safe_table, keyed_hash

from test_label import labeled  # noqa: F401  (fixture)
from test_pipeline import CANARIES, _assert_no_canaries, llm_run  # noqa: F401  (fixture)


def test_review_summary_counts_and_suppression():
    rev = pd.DataFrame({"about": ["user", "user", "fictional"], "evidence": ["stated", "inferred", "stated"],
                        "value_ok": ["correct", "wrong", "unsure"], "about_human": ["user", "third_party", ""],
                        "evidence_human": ["stated", "stated", ""], "missed": ["no", "yes", ""]})
    s = review.review_summary(rev, min_cell=1).set_index("metric")
    assert (s.loc["value_correct", "n_reviewed"], s.loc["value_correct", "n_yes"]) == (2, 1)      # unsure is left out
    assert s.loc["about_agrees", "share"] == 0.5 and s.loc["evidence_agrees", "n_yes"] == 1
    assert s.loc["turn_has_missed_attribute", "n_yes"] == 1
    hidden = review.review_summary(rev, min_cell=10)
    assert hidden["suppressed"].all() and hidden[["n_reviewed", "n_yes", "share"]].isna().all().all()


def test_attribute_check_keeps_values_restricted(llm_run, capfd):
    pipe, _, _ = llm_run
    rv = review.Review(pipe, n_attrs=6)
    items = rv.attr_items()["items"]
    assert len(items) == 6 and all(i["text"] for i in items)
    abouts = [i["about"] for i in items]
    assert abouts.count("user") >= 3 and "fictional" in abouts          # both halves of the sample
    assert any(c in i["value"] + i["text"] for i in items for c in CANARIES)   # raw text and values, on the page only
    # A turn with several extractions lists the rest beside the one under review, without repeating it.
    multi = [i for i in items if i["others"]]
    assert multi and all(i["value"] not in [o["value"] for o in i["others"]] for i in multi)
    first = items[0]
    out = rv.save_attr(first["item_id"], {"value_ok": "correct", "about_human": first["about"],
                                          "evidence_human": "stated", "missed": "no", "notes": "fine"})
    assert out["done"] == 1 and {"metric": "value_correct", "n": 1, "yes": 1} in out["tally"]
    with pytest.raises(ValueError):
        rv.save_attr(first["item_id"], {"value_ok": first["value"]})    # free text is not an answer
    # The review file is restricted, holds a keyed hash rather than the value, and resumes.
    assert rv.attr_path.parent == pipe.paths.restricted / "labels" and (rv.attr_path.stat().st_mode & 0o777) == 0o600
    saved = pd.read_csv(rv.attr_path, dtype=str, keep_default_na=False)
    assert list(saved.columns) == review.REVIEW_COLUMNS
    assert not any(i["value"] in rv.attr_path.read_text() for i in items if len(i["value"]) > 6)
    assert saved["value_hmac"][0] == keyed_hash(pipe.key, first["value"])
    again = review.Review(pipe, n_attrs=6).attr_items()["items"]
    assert [i["item_id"] for i in again] == [i["item_id"] for i in items] and again[0]["answers"]["value_ok"] == "correct"
    # One review is below the publication threshold, so the table shows nothing but the metric names.
    table = pd.read_csv(pipe.paths.tables / "llm_attribute_review.csv")
    assert table["suppressed"].all() and table["n_reviewed"].isna().all()
    _assert_no_canaries(pipe, "".join(capfd.readouterr()))


def test_labeling_round_trip_and_score(labeled):
    pipe, path, before, capfd = labeled
    rv = review.Review(pipe)
    assert rv.batches() == [{"name": path.name, "n": 12, "done": 0}]
    items = rv.batch_items(path.name)["items"]
    for it in items:
        spans = [{"start": it["text"].index(needle), "end": it["text"].index(needle) + len(needle),
                  "pii_type": typ, "about": about}
                 for needle, typ, about in (("Quillon Varnadore", "name", "user"), ("Denver", "location", "user"))
                 if needle in it["text"]]
        rv.save_item(path.name, it["item_id"], {"spans": spans, "none": not spans, "notes": "n"})
    assert rv.batches()[0]["done"] == 12 and (path.stat().st_mode & 0o777) == 0o600
    # The file is what `label-score` already reads, and a fresh session sees the same labels.
    gold, skipped = label.gold_spans(label.read_labels(path), rv.categories)
    assert len(gold) and set(gold["category"]) <= {"name", "location"} and skipped["span_not_found"] == 0
    assert review.Review(pipe).batch_items(path.name)["items"] == items
    with pytest.raises(ValueError):
        rv.save_item(path.name, items[0]["item_id"], {"spans": [{"start": 0, "end": 10**6, "pii_type": "name"}]})
    with pytest.raises(ValueError):
        rv.batch_items("../turns_wildchat.parquet")                      # only batches in restricted/labels
    scores = rv.score(path.name)
    assert "precision" in scores["columns"] and scores["rows"]
    # Clearing one turn's labels takes it out of the scored set instead of making it all false positives.
    rv.save_item(path.name, items[0]["item_id"], {"spans": [], "none": False})
    assert rv.batches()[0]["done"] == 11 and rv.score(path.name)["rows"][0][0] == 11
    new = sorted(p.name for p in pipe.paths.results.rglob("*") if p.is_file() and p not in before)
    assert new == ["detector_validation.csv"]
    _assert_no_canaries(pipe, "".join(capfd.readouterr()))


def test_offsets_are_code_points(labeled):
    pipe, path, _, _ = labeled
    rv = review.Review(pipe)
    item = rv.batch_items(path.name)["items"][0]
    item["text"] = "😀 Ada 𝒳"                                           # astral characters before the span
    rv.save_item(path.name, item["item_id"], {"spans": [{"start": 2, "end": 5, "pii_type": "name", "about": "user"}]})
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert rows.loc[rows["item_id"] == item["item_id"], "span_text"].tolist() == ["Ada"]


def _get(url, token=None, host=None):
    req = urllib.request.Request(url, headers={k: v for k, v in (("X-Review-Token", token), ("Host", host)) if v})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), e.headers


def test_server_is_local_and_token_gated(labeled):
    pipe, path, _, capfd = labeled
    server, url = review.make_server(review.Review(pipe))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base, token = url.split("/?t=")
        assert server.server_address[0] == "127.0.0.1"
        status, page, headers = _get(url)
        assert status == 200 and token in page and "default-src 'none'" in headers["Content-Security-Policy"]
        assert "http://" not in page.replace("http://127.0.0.1", "") and "https://" not in page   # no outside resources
        assert _get(base + "/?t=wrong")[0] == 403 and _get(base + "/api/overview")[0] == 403
        assert _get(base + "/api/overview", token, host="evil.example:80")[0] == 403           # DNS rebinding
        status, body, _ = _get(base + "/api/overview", token)
        assert status == 200 and json.loads(body)["batches"][0]["name"] == path.name
        status, body, _ = _get(base + "/api/batch?name=nope.csv", token)
        assert status == 400 and not any(c in body for c in CANARIES)
        req = urllib.request.Request(base + "/api/item", data=b'{"name": "x"}', headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError):                       # a cross-site form post has no token
            urllib.request.urlopen(req, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
    _assert_no_canaries(pipe, "".join(capfd.readouterr()))


def test_attributes_annotators_and_agreement(labeled):
    pipe, path, _, capfd = labeled
    before = path.read_bytes()
    ann, ben = review.Review(pipe, annotator="Ann"), review.Review(pipe, annotator="ben")
    with pytest.raises(ValueError):
        review.Review(pipe, annotator="../x")
    items = ann.batch_items(path.name)["items"]
    first, second, third = items[0], items[1], items[2]
    name_span = {"start": 0, "end": 4, "pii_type": "name", "about": "user"}
    job = {"type": "occupation", "evidence": "stated", "about": "user", "value": "nurse"}
    student = {"type": "education_level", "evidence": "inferred", "about": "user", "value": ""}
    ann.save_item(path.name, first["item_id"], {"spans": [name_span], "attrs": [job, student]})
    ann.save_item(path.name, second["item_id"], {"spans": [], "attrs": [student]})      # attributes only
    ann.save_item(path.name, third["item_id"], {"spans": [], "none": True})
    ben.save_item(path.name, first["item_id"], {"spans": [{**name_span, "end": 5, "after_reveal": True}], "attrs": [job]})
    ben.save_item(path.name, second["item_id"], {"spans": [], "none": True})
    ben.save_item(path.name, third["item_id"], {"spans": [], "none": True})
    with pytest.raises(ValueError):
        ann.save_item(path.name, first["item_id"], {"attrs": [{**job, "type": "shoe_size"}]})

    # Each annotator has their own copy; the exported batch is untouched and still the only one listed.
    a_path, b_path = (path.with_name(f"{path.stem}__{n}.csv") for n in ("ann", "ben"))
    assert path.read_bytes() == before and a_path.exists() and b_path.exists()
    assert [b["name"] for b in ann.batches()] == [path.name] and ann.batches()[0]["done"] == 3
    assert review.Review(pipe).batches()[0]["done"] == 0
    for f in (a_path, label.attrs_path(a_path)):
        assert (f.stat().st_mode & 0o777) == 0o600
    # A fresh session reads the same labels back, and the attribute-only turn counts as labeled.
    again = {i["item_id"]: i for i in review.Review(pipe, annotator="ann").batch_items(path.name)["items"]}
    assert [a["type"] for a in again[first["item_id"]]["attrs"]] == ["occupation", "education_level"]
    assert again[second["item_id"]]["attrs"] and not again[second["item_id"]]["none"]

    out = pipe.label_agreement(a_path, b_path).set_index(["measure", "category"])
    assert out["n_turns_both"].iloc[0] == 3
    assert out.loc[("span_f1", "name"), "agreement"] == 1.0                       # overlapping spans, same type
    assert out.loc[("turn_kappa", "has_personal_information"), ["n_a", "n_b"]].tolist() == [2, 1]
    assert out.loc[("attribute_f1", "stated"), "agreement"] == 1.0
    assert out.loc[("attribute_f1", "inferred"), ["n_a", "n_b"]].tolist() == [2, 0]
    assert (pipe.paths.tables / "annotator_agreement.csv").exists()

    # Scoring: ben's only span was added after the reveal, so the blind-only gold has no spans.
    scores = pipe.label_score(b_path)
    assert set(scores["gold"].dropna()) == {"all", "blind_only"}
    blind = scores[(scores["gold"] == "blind_only") & (scores["detector"] == "any") & (scores["category"] == "all")]
    assert blind["n_gold"].iloc[0] == 0
    _assert_no_canaries(pipe, "".join(capfd.readouterr()))


def test_llm_attributes_scored_against_labels():
    turn = lambda i: {"dataset": "d", "conversation_id": f"c{i}", "turn_index": 0}  # noqa: E731
    gold = pd.DataFrame([{**turn(1), "attribute_type": "occupation", "evidence": "stated", "about": "user"},
                         {**turn(1), "attribute_type": "age", "evidence": "inferred", "about": "user"},
                         {**turn(2), "attribute_type": "city", "evidence": "stated", "about": "third_party"},
                         {**turn(9), "attribute_type": "city", "evidence": "stated", "about": "user"}])   # not scored
    llm_attrs = pd.DataFrame([{**turn(1), "type": "occupation", "evidence": "stated", "about": "user"},
                              {**turn(2), "type": "city", "evidence": "stated", "about": "user"},         # wrong person
                              {**turn(3), "type": "gender", "evidence": "inferred", "about": "fictional"}])
    turns = pd.DataFrame([turn(1), turn(2), turn(3)])
    out = label.score_attributes(gold, llm_attrs, turns).set_index(["attribute_type", "evidence"])
    assert out.loc[("all", "any"), ["n_gold", "n_pred", "tp_pred", "tp_gold"]].tolist() == [3, 3, 1, 1]
    assert out.loc[("all", "inferred"), "recall"] == 0 and out.loc[("occupation", "stated"), "f1"] == 1
    assert_safe_table(out.reset_index(), "attribute_validation")
