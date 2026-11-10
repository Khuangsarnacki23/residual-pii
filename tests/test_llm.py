"""LLM detector, exercised without a model.

A fake backend returns canned JSON, so the parse -> store -> aggregate path runs offline. All
"personal" values are fabricated.
"""
import json

import pandas as pd
import pytest

from residpii import llm
from residpii.config import REPO_ROOT, load_config
from residpii.safety import assert_safe_table
from residpii.pipeline import Pipeline


def _cfg() -> dict:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


def _attr(**kw) -> dict:
    return {"type": "city", "value": "Denver", "evidence": "stated", "about": "user", "confidence": 0.9, **kw}


def _resp(*attrs: dict) -> str:
    return json.dumps({"attributes": list(attrs)})


# ---------------------------------------------------------------- parser


def test_parse_response_accepts_schema():
    got = llm.parse_response(_resp(_attr(), _attr(type="age", value=34, evidence="inferred", confidence=1),
                                   _attr(type="occupation", value="nurse", occupation_group="29",
                                         occupation_minor_group="29-1141")))
    assert [a.type for a in got] == ["city", "age", "occupation"]
    assert got[1].value == "34" and got[1].confidence == 1.0
    assert (got[2].occupation_group, got[2].occupation_minor_group) == ("29", "29-1")
    assert llm.parse_response('{"attributes": []}') == []


@pytest.mark.parametrize("raw", [
    "not json", "[]", '{"attrs": []}', '{"attributes": "none"}', '{"attributes": ["x"]}',
    _resp(_attr(type="ssn")), _resp(_attr(evidence="guessed")), _resp(_attr(about="me")),
    _resp(_attr(confidence=1.5)), _resp(_attr(confidence="high")), _resp(_attr(value="")),
    _resp(_attr(value="x" * 500)), _resp({"type": "city"}), '```json\n{"attributes": []}\n```',
])
def test_parse_response_is_strict(raw):
    with pytest.raises(llm.ParseError) as e:
        llm.parse_response(raw)
    assert "Denver" not in str(e.value)       # reason codes only, never response text


def test_parse_response_drops_unusable_occupation_codes():
    a, b = llm.parse_response(_resp(
        _attr(type="occupation", value="nurse", occupation_group="99", occupation_minor_group="15-1"),
        _attr(type="city", occupation_group="15")))       # a code on a non-occupation attribute
    assert (a.occupation_group, a.occupation_minor_group) == ("", "")
    assert b.occupation_group == ""


# ---------------------------------------------------------------- hosted opt-in


@pytest.mark.parametrize("backend", ["fireworks", "anthropic", "openai"])
def test_hosted_backend_needs_opt_in(backend):
    lcfg = {**_cfg()["llm"], "backend": backend}
    with pytest.raises(llm.HostedBackendNotAllowed):
        llm.check_backend_allowed(lcfg)
    with pytest.raises(llm.HostedBackendNotAllowed):
        llm.make_backend(lcfg)
    llm.check_backend_allowed({**lcfg, "allow_hosted": True})


def test_defaults_are_off_and_not_allowed(tmp_path):
    cfg = _cfg()
    assert cfg["llm"]["enabled"] is False and cfg["llm"]["allow_hosted"] is False
    with pytest.raises(llm.HostedBackendNotAllowed):
        llm.check_backend_allowed(cfg["llm"])              # the default backend is hosted too
    with pytest.raises(ValueError):
        llm.check_backend_allowed({**cfg["llm"], "backend": "ollama"})   # there is no local backend
    # An enabled run without the opt-in aborts at start-up, before any data is touched.
    cfg.update(run_name="t", data_root=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    cfg["llm"]["enabled"] = True
    with pytest.raises(llm.HostedBackendNotAllowed):
        Pipeline(cfg, sources={})
    restricted = tmp_path / "data" / "runs" / "t" / "restricted"
    assert not restricted.exists() or not list(restricted.iterdir())
    # Only the pilot config turns the detector on; the user-level configs inherit the defaults.
    for name in ("user_level.yaml", "user_level_smoke.yaml", "smoke.yaml"):
        inherited = load_config(REPO_ROOT / "configs" / name)["llm"]
        assert inherited["enabled"] is False and inherited["allow_hosted"] is False


# ---------------------------------------------------------------- fireworks backend


class _FakeHTTP:
    """Stands in for urllib.request.urlopen: records requests, replays canned outcomes."""

    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    @staticmethod
    def ok(content: str, tokens=(1200, 40)):
        import io
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]}}).encode())

    @staticmethod
    def error(code: int):
        import urllib.error
        return urllib.error.HTTPError("https://api.fireworks.ai", code, "x", {}, None)


def _fireworks(monkeypatch, outcomes, **override):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test-key-000")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    http = _FakeHTTP(outcomes)
    monkeypatch.setattr(llm.urllib.request, "urlopen", http)
    lcfg = {**_cfg()["llm"], "allow_hosted": True, **override}
    return llm.make_backend(lcfg), http, lcfg


def test_fireworks_request_and_usage(monkeypatch, tmp_path):
    backend, http, lcfg = _fireworks(monkeypatch, [_FakeHTTP.ok(_resp(_attr()))])
    assert backend.name == "fireworks" and backend.model == lcfg["model"]
    prompt = llm.load_prompt(_cfg())
    attrs, tlog = llm.extract(_selected(["I live in Denver"]), lcfg, prompt, backend, b"k", tmp_path / "c.jsonl")
    req = http.requests[0]
    assert req.full_url == llm.FIREWORKS_URL and req.get_header("Authorization") == "Bearer fw-test-key-000"
    body = json.loads(req.data)
    assert body["model"] == lcfg["model"] and body["temperature"] == 0
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == llm.response_schema(lcfg)
    assert [m["role"] for m in body["messages"]] == ["system", "user"] and "I live in Denver" in body["messages"][1]["content"]
    assert len(attrs) == 1 and tlog["input_tokens"].iloc[0] == 1200 and tlog["output_tokens"].iloc[0] == 40
    stats = llm.run_stats(tlog, backend, lcfg).iloc[0]
    assert stats["input_tokens_new"] == 1200 and stats["output_tokens_new"] == 40
    # Served from cache next time: no request, nothing newly billed.
    _, tlog2 = llm.extract(_selected(["I live in Denver"]), lcfg, prompt, backend, b"k", tmp_path / "c.jsonl")
    assert len(http.requests) == 1
    assert llm.run_stats(tlog2, backend, lcfg).iloc[0]["input_tokens_new"] == 0


def test_fireworks_retries_then_counts_both_attempts(monkeypatch, tmp_path):
    # A 429 is retried at the HTTP level; a malformed answer is retried once and both are billed.
    backend, http, lcfg = _fireworks(monkeypatch, [_FakeHTTP.error(429), _FakeHTTP.ok("oops", (1000, 5)),
                                                   _FakeHTTP.ok(_resp(), (1000, 8))])
    _, tlog = llm.extract(_selected(["x"]), lcfg, llm.load_prompt(_cfg()), backend, b"k", tmp_path / "c.jsonl")
    assert len(http.requests) == 3 and tlog["status"].iloc[0] == "ok"
    assert (tlog["input_tokens"].iloc[0], tlog["output_tokens"].iloc[0]) == (2000, 13)


@pytest.mark.parametrize("code,hint", [(401, "FIREWORKS_API_KEY"), (402, "out of credit"), (404, "llm.model")])
def test_fireworks_hard_errors_stop_the_run(monkeypatch, tmp_path, code, hint):
    backend, _, lcfg = _fireworks(monkeypatch, [_FakeHTTP.error(code)])
    with pytest.raises(llm.BackendUnavailable) as e:
        llm.extract(_selected(["x"]), lcfg, llm.load_prompt(_cfg()), backend, b"k", tmp_path / "c.jsonl")
    assert hint in str(e.value) and "fw-test-key-000" not in str(e.value)


def test_fireworks_needs_key_and_json_object_fallback(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(llm.BackendUnavailable):
        llm.make_backend({**_cfg()["llm"], "allow_hosted": True})
    backend, _, _ = _fireworks(monkeypatch, [], fireworks_response_format="json_object")
    assert backend.response_format == {"type": "json_object"}


# ---------------------------------------------------------------- extraction + cache


def _selected(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"dataset": "d", "conversation_id": [f"c{i}" for i in range(len(texts))],
                         "turn_index": 0, "role": "user", "text": texts, "context": ""})


def test_extract_caches_and_counts_failures(tmp_path):
    cfg = _cfg()
    prompt, calls = llm.load_prompt(cfg), []

    def fake(system, user):
        calls.append(user)
        if "garbage" in user:
            return "I cannot answer in JSON"
        return _resp(_attr(value="Denver"))

    backend = llm.CallableBackend(fake)
    sel = _selected(["I live in Denver", "garbage please", "I live in Denver"])
    cache = tmp_path / "cache.jsonl"
    attrs, tlog = llm.extract(sel, cfg["llm"], prompt, backend, b"key", cache)
    # Identical turns are asked once; the malformed answer is retried exactly once, then recorded.
    assert len(calls) == 1 + 2
    assert list(tlog["status"]) == ["ok", "parse_failure", "ok"]
    assert len(attrs) == 2 and set(attrs["value"]) == {"Denver"}
    stats = llm.run_stats(tlog, backend, cfg["llm"]).iloc[0]
    assert stats["turns_processed"] == 3 and stats["failures"] == 1

    # A rerun is free: nothing is sent to the backend, results are identical.
    attrs2, tlog2 = llm.extract(sel, cfg["llm"], prompt, backend, b"key", cache)
    assert len(calls) == 3 and tlog2["cached"].all()
    assert attrs2.equals(attrs)
    # The cache is keyed by HMAC; the turn text itself is not in it.
    assert "I live in Denver" not in cache.read_text()
    # A different model or prompt is a different cache entry.
    llm.extract(sel, cfg["llm"], prompt, llm.CallableBackend(fake, model="other"), b"key", cache)
    assert len(calls) == 6


def test_extract_truncates_and_passes_context(tmp_path):
    cfg = _cfg()
    cfg["llm"].update(max_turn_chars=10, max_context_chars=5)
    seen = []
    backend = llm.CallableBackend(lambda s, u: (seen.append(u), '{"attributes": []}')[1])
    sel = _selected(["0123456789ABCDEF"]).assign(context="previous turn text")
    _, tlog = llm.extract(sel, cfg["llm"], llm.load_prompt(cfg), backend, b"k", tmp_path / "c.jsonl")
    assert "0123456789" in seen[0] and "ABCDEF" not in seen[0]
    assert "previ" in seen[0] and "previous" not in seen[0]
    assert bool(tlog["truncated"].iloc[0])


def test_refusal_is_a_failure_without_retry(tmp_path):
    cfg, n = _cfg(), []

    def refuse(system, user):
        n.append(1)
        raise llm.BackendRefusal()

    _, tlog = llm.extract(_selected(["x"]), cfg["llm"], llm.load_prompt(cfg), llm.CallableBackend(refuse), b"k",
                          tmp_path / "c.jsonl")
    assert list(tlog["status"]) == ["refusal"] and len(n) == 1


def test_select_turns_keeps_whole_users():
    turns = pd.DataFrame([("d", f"c{u}{i}", t, role, f"text {u}{i}{t}")
                          for u in "abcdef" for i in range(2) for t, role in enumerate(["user", "assistant", "user"])],
                         columns=["dataset", "conversation_id", "turn_index", "role", "text"])
    sample = pd.DataFrame({"conversation_id": turns["conversation_id"].unique()})
    sample["user_key"] = sample["conversation_id"].str[1]
    sel = llm.select_turns(turns, sample, {"sample_turns": 10}, seed=1, whole_users=True)
    per_user = sel["conversation_id"].str[1].value_counts()
    assert len(sel) <= 10 and (per_user == 4).all() and len(per_user) == 2     # 4 user turns per user
    assert set(sel["role"]) == {"user"}
    # The assistant's reply is the context of the user turn that follows it; a first turn has none.
    assert (sel.loc[sel["turn_index"] == 2, "context"].str.startswith("text")).all()
    assert (sel.loc[sel["turn_index"] == 0, "context"] == "").all()
    again = llm.select_turns(turns, sample, {"sample_turns": 10}, seed=1, whole_users=True)
    assert list(again["conversation_id"]) == list(sel["conversation_id"])
    assert len(llm.select_turns(turns, sample, {"sample_turns": None}, 1, True)) == 24


# ---------------------------------------------------------------- direct identifiers as spans


def _span_resp(*spans: tuple, attrs: tuple = ()) -> str:
    return json.dumps({"spans": [{"text": t, "type": ty, "about": ab} for t, ty, ab in spans],
                       "attributes": list(attrs)})


def test_parse_spans_is_strict():
    assert llm.parse_spans(_resp(_attr())) == []                    # a v1-shaped answer has no spans
    assert llm.parse_spans(_span_resp(("Ada", "name", "user"))) == [{"text": "Ada", "type": "name", "about": "user"}]
    for bad in ('{"spans": "Ada"}', _span_resp(("Ada", "pet_name", "user")), _span_resp(("Ada", "name", "me")),
                _span_resp(("", "name", "user")), _span_resp(("x" * 500, "name", "user"))):
        with pytest.raises(llm.ParseError):
            llm.parse_spans(bad)


def test_prompt_and_extract_spans_must_agree():
    cfg = _cfg()
    assert cfg["llm"]["extract_spans"] and cfg["llm"]["prompt_version"] == "v2"
    system = llm.load_prompt(cfg).system
    assert "street_address" in system and "{{" not in system
    assert llm.response_schema(cfg["llm"])["required"] == ["task_type", "spans", "attributes"]
    prompt = llm.load_prompt(cfg)
    assert prompt.task_types[0] == "story_roleplay" and prompt.task_types[-1] == "info_or_other"
    assert all(f"{t}:" in system for t in prompt.task_types)
    for version, spans in (("v1", True), ("v2", False)):
        cfg["llm"].update(prompt_version=version, extract_spans=spans)
        with pytest.raises(ValueError):
            llm.load_prompt(cfg)
    for name in ("fireworks_pilot.yaml", "fireworks_users.yaml", "fireworks_users_2000.yaml"):
        old = load_config(REPO_ROOT / "configs" / name)
        assert old["llm"]["prompt_version"] == "v1" and llm.load_prompt(old)      # earlier runs keep their prompt


def test_quotes_become_offsets_and_hashes_only(tmp_path):
    cfg = _cfg()
    cfg["placeholders"] = {"common": [r"NAME_\d+"]}
    turn = "Hi, I'm Quillon Varnadore. Mail quillon@zephyrmailtest.org, not a@example.com. QUILLON VARNADORE and NAME_1."

    def fake(system, user):
        return _span_resp(("Quillon Varnadore", "name", "user"), ("quillon@zephyrmailtest.org", "email", "user"),
                          ("a@example.com", "email", "unclear"), ("NAME_1", "name", "third_party"),
                          ("Quillon V. Varnadore", "name", "user"),              # not in the text
                          attrs=(_attr(),))

    sel = _selected([turn])
    attrs, tlog, quotes = llm.extract_with_spans(sel, cfg["llm"], llm.load_prompt(cfg), llm.CallableBackend(fake),
                                                 b"key", tmp_path / "c.jsonl")
    assert len(attrs) == 1 and tlog["n_spans"].tolist() == [5] and len(quotes) == 5
    spans, unlocated = llm.locate_spans(quotes, sel, cfg, {"categories": {"name": {"tier": "direct"},
                                                                         "email": {"tier": "direct"}}}, b"key")
    assert unlocated == 1
    assert_safe_table(spans.drop(columns=["span_hmac"]), "llm spans")           # no column holds text
    assert set(spans["detector"]) == {"llm"} and set(spans["tier"]) == {"direct"}
    found = {(turn[s:e], cat) for s, e, cat in zip(spans["start"], spans["end"], spans["category"])}
    # Exact occurrences only: the upper-case repeat is a different string ("Will" must not find "will").
    assert found == {("Quillon Varnadore", "name"), ("NAME_1", "name"),
                     ("quillon@zephyrmailtest.org", "email"), ("a@example.com", "email")}
    assert llm._occurrences("ask WILL or Will", "will") == [4, 12]              # no exact match: ignore case
    by_text = {turn[s:e]: r for s, e, r in zip(spans["start"], spans["end"], spans.to_dict("records"))}
    assert by_text["a@example.com"]["is_dummy"] and by_text["NAME_1"]["in_placeholder"]
    assert by_text["Quillon Varnadore"]["about"] == "user" and by_text["Quillon Varnadore"]["about_user"]
    # extract() keeps its two-frame shape for callers that only want attributes.
    assert len(llm.extract(sel, cfg["llm"], llm.load_prompt(cfg), llm.CallableBackend(fake), b"key",
                           tmp_path / "c.jsonl")) == 2


def test_detector_agreement_counts_overlaps():
    def row(det, cat, start, end, about=None, turn=0):
        return {"dataset": "d", "conversation_id": "c", "turn_index": turn, "role": "user", "start": start,
                "end": end, "detector": det, "category": cat, "about": about, "in_placeholder": False,
                "is_dummy": False}

    llm_spans = pd.DataFrame([row("llm", "name", 0, 10, "user"), row("llm", "name", 50, 60, "fictional"),
                              row("llm", "email", 80, 100, "user")])
    other = pd.DataFrame([row("presidio", "name", 2, 8), row("presidio", "name", 50, 60),
                          row("presidio", "name", 200, 210),                    # the LLM did not quote this one
                          row("presidio", "name", 0, 10, turn=1),               # a turn the LLM failed on
                          row("presidio", "location", 0, 10)])                  # not a direct-identifier type
    tlog = pd.DataFrame({"dataset": "d", "conversation_id": "c", "turn_index": [0, 1], "status": ["ok", "parse_failure"]})
    out = llm.detector_agreement(llm_spans, other, tlog, 6000).set_index(["category", "detector"])
    assert_safe_table(out.reset_index(), "llm_detector_agreement")
    p = out.loc[("name", "presidio")]
    assert (p["n_spans"], p["n_matched"], p["n_about_user"], p["n_about_fictional"]) == (3, 2, 1, 1)
    assert out.loc[("name", "llm"), ["n_spans", "n_matched"]].tolist() == [2, 2]
    assert out.loc[("email", "llm"), ["n_spans", "n_matched", "n_about_user"]].tolist() == [1, 0, 1]
    assert ("location", "presidio") not in out.index


def test_task_label_is_parsed_and_rolled_up(tmp_path):
    allowed = ("story_roleplay", "code", "info_or_other")
    assert llm.parse_task('{"task_type": "code", "attributes": []}', allowed) == "code"
    assert llm.parse_task('{"attributes": []}', allowed) is None
    with pytest.raises(llm.ParseError):
        llm.parse_task('{"task_type": "gardening", "attributes": []}', allowed)

    cfg = _cfg()
    replies = {"fix my loop": "code", "write a tale": "story_roleplay", "and a sequel": "story_roleplay"}

    def fake(system, user):
        turn = user.split("USER TURN:", 1)[1]
        return json.dumps({"task_type": next(v for k, v in replies.items() if k in turn), "spans": [], "attributes": []})

    sel = _selected(list(replies))
    sel["conversation_id"], sel["turn_index"] = ["a", "b", "b"], [0, 0, 1]
    _, tlog = llm.extract(sel, cfg["llm"], llm.load_prompt(cfg), llm.CallableBackend(fake), b"k", tmp_path / "c.jsonl")
    assert tlog["task_type"].tolist() == ["code", "story_roleplay", "story_roleplay"]
    tied = pd.concat([tlog, tlog.iloc[[0]].assign(turn_index=1, task_type="info_or_other")])
    tied.loc[tied["task_type"].eq("story_roleplay") & tied["turn_index"].eq(1), "task_type"] = None
    got = llm.conversation_tasks(tied).set_index("conversation_id")["task_type"]
    assert got.to_dict() == {"a": "code", "b": "story_roleplay"}            # tie -> earliest turn; unlabeled ignored
