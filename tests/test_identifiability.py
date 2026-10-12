"""LLM detector, per-user profiles and uniqueness, exercised without a model.

A fake backend returns canned JSON, so the parse -> store -> aggregate path runs offline. All
"personal" values are fabricated.
"""
import json

import numpy as np
import pandas as pd
import pytest

from residpii import llm, uniqueness
from residpii.config import REPO_ROOT, load_config
from residpii.pipeline import Pipeline
from residpii.profiles import (AMBIGUOUS, DIMENSIONS, UNKNOWN, build_profiles, load_coarsener, profile_summary,
                               resolve)


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
    # The pilot config ships with the opt-in still off.
    pilot = load_config(REPO_ROOT / "configs" / "fireworks_pilot.yaml")
    assert pilot["llm"]["enabled"] is True and pilot["llm"]["allow_hosted"] is False


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
    assert body["response_format"]["json_schema"]["schema"] == llm.RESPONSE_SCHEMA
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


# ---------------------------------------------------------------- coarsening + profiles


def test_coarsening_is_deterministic_and_closed():
    c1, c2 = load_coarsener(_cfg()), load_coarsener(_cfg())
    cases = [("city", "NYC"), ("city", "Smallville"), ("state_or_region", "CA"), ("country", "U.S."),
             ("age", "27"), ("age_range", "mid 30s"), ("gender", "Woman"), ("school", "Ohio State University"),
             ("education_level", "PhD student"), ("occupation", "software engineer"), ("occupation", "wizard")]
    for typ, val in cases:
        for level in ("fine", "coarse"):
            assert c1.candidates(typ, val, level) == c2.candidates(typ, val, level) == c1.candidates(typ, val, level)
    assert dict(c1.candidates("city", "NYC", "fine")) == {"country": "united states", "region": "new york"}
    assert dict(c1.candidates("city", "NYC", "coarse")) == {"country": "united states", "region": "us_northeast"}
    assert c1.candidates("city", "Smallville", "fine") == [("region", UNKNOWN)]   # unmatched -> unknown
    assert c1.candidates("age", "27", "fine") == [("age_band", "20-29")]
    assert c1.candidates("age", "27", "coarse") == [("age_band", "20-39")]
    assert c1.candidates("school", "Ohio State University", "coarse") == [("education_level", "college")]
    assert c1.candidates("occupation", "software engineer", "fine") == [("occupation_group", "15-1")]
    assert c1.candidates("occupation", "software engineer", "coarse") == [("occupation_group", "15")]
    assert c1.candidates("occupation", "anything", "coarse", "29", "29-1") == [("occupation_group", "29")]
    assert c1.candidates("country", "Germany", "fine") == [("country", "germany")]
    assert c1.candidates("nationality", "German", "fine") == []                   # not a profile dimension


def test_resolve_majority_and_ties():
    assert resolve(pd.Series({"texas": 3, "ohio": 1})) == ("texas", True)
    assert resolve(pd.Series({"ohio": 1, "texas": 1})) == (AMBIGUOUS, True)
    assert resolve(pd.Series({"texas": 2})) == ("texas", False)
    assert resolve(pd.Series({UNKNOWN: 5, "texas": 1})) == ("texas", False)      # unknown never outvotes a value
    assert resolve(pd.Series({UNKNOWN: 2})) == (UNKNOWN, False)


def _profile_inputs():
    """Three users. u1: stated Texas in two conversations, Ohio in one; inferred age. u2: a tie.
    u3: only a low-confidence attribute and one about a third party."""
    rows = [("c1", "state_or_region", "Texas", "stated", "user", 0.9),
            ("c2", "state_or_region", "TX", "stated", "user", 0.9),
            ("c3", "state_or_region", "Ohio", "stated", "user", 0.9),
            ("c1", "age", "27", "inferred", "user", 0.8),
            ("c4", "gender", "woman", "stated", "user", 0.9),
            ("c5", "gender", "man", "stated", "user", 0.9),
            ("c6", "occupation", "nurse", "stated", "user", 0.3),          # below min_confidence
            ("c6", "city", "Chicago", "stated", "third_party", 0.95)]      # not about the user
    attrs = pd.DataFrame(rows, columns=["conversation_id", "type", "value", "evidence", "about", "confidence"])
    attrs = attrs.assign(dataset="d", turn_index=0, occupation_group="", occupation_minor_group="")
    convs = {"c1": "u1", "c2": "u1", "c3": "u1", "c4": "u2", "c5": "u2", "c6": "u3"}
    sample = pd.DataFrame({"dataset": "d", "conversation_id": list(convs), "user_key": list(convs.values()),
                           "user_n_conversations": [3, 3, 3, 2, 2, 1]})
    tlog = pd.DataFrame({"dataset": "d", "conversation_id": list(convs), "turn_index": 0, "status": "ok"})
    return attrs, tlog, sample


def test_profiles_conflicts_and_variants():
    attrs, tlog, sample = _profile_inputs()
    prof = build_profiles(attrs, tlog, sample, load_coarsener(_cfg()), min_confidence=0.6)
    assert len(prof) == 3 * 2 * 2                                    # users x variants x levels
    p = prof.set_index(["user", "variant", "level"])
    # Majority vote across conversations, with the conflict recorded.
    assert p.loc[("u1", "stated_only", "fine"), "region"] == "texas"
    assert p.loc[("u1", "stated_only", "fine"), "conflict_region"]
    assert p.loc[("u1", "stated_only", "coarse"), "region"] == "us_south"
    assert p.loc[("u1", "stated_only", "fine"), "country"] == "united states"
    assert not p.loc[("u1", "stated_only", "fine"), "conflict_country"]
    # Inferred attributes are left out of stated_only and present in stated_plus_inferred.
    assert pd.isna(p.loc[("u1", "stated_only", "fine"), "age_band"])
    assert p.loc[("u1", "stated_plus_inferred", "fine"), "age_band"] == "20-29"
    # A tie is ambiguous.
    assert p.loc[("u2", "stated_only", "fine"), "gender"] == AMBIGUOUS
    # Low-confidence and third-party attributes never reach a profile, but the user is still counted.
    assert p.loc[("u3", "stated_plus_inferred", "fine"), list(DIMENSIONS)].isna().all()
    assert p.loc[("u1", "stated_only", "fine"), "activity_bucket"] == "3"

    summ = profile_summary(prof, min_cell=1).set_index(["variant", "level", "attribute"])
    assert summ.loc[("stated_only", "fine", "region"), "n_conflict"] == 1
    assert summ.loc[("stated_only", "fine", "gender"), "n_ambiguous"] == 1
    assert summ.loc[("stated_only", "fine", "gender"), "n_with_value"] == 0
    assert summ.loc[("stated_only", "fine", "any"), "share_conflict"] == pytest.approx(2 / 3)
    # With the real threshold every cell here is a small cell.
    assert profile_summary(prof, min_cell=10)["suppressed"].all()


# ---------------------------------------------------------------- uniqueness


def _profiles_with_known_k() -> pd.DataFrame:
    """20 users: 12 Texas + 8 Ohio; genders split so (region, gender) classes are 11, 1, 4, 4."""
    region = ["texas"] * 12 + ["ohio"] * 8
    gender = ["male"] * 11 + ["female"] + ["male"] * 4 + ["female"] * 4
    df = pd.DataFrame({"dataset": "d", "variant": "stated_only", "level": "fine",
                       "user": [f"u{i}" for i in range(20)], "activity_bucket": ["3"] * 10 + ["4-5"] * 10,
                       "region": region, "gender": gender})
    for d in DIMENSIONS:
        if d not in df:
            df[d] = None
    df.loc[19, "gender"] = AMBIGUOUS         # not a usable value: this user drops out of gender combos
    df.loc[18, "age_band"] = UNKNOWN
    return df


def test_uniqueness_known_class_sizes():
    df = _profiles_with_known_k()
    by_combo, by_act, summary = uniqueness.analyze(df, {"sample_sizes": [10]}, {"bootstrap_reps": 100,
                                                                              "ci_level": 0.95}, seed=1, min_cell=1)
    c = by_combo.set_index("combination")
    assert c.loc["region", "n_users"] == 20 and c.loc["region", "share_k1"] == 0
    assert c.loc["region", "median_k"] == 12                 # 12 users in a class of 12, 8 in a class of 8
    assert c.loc["region", "share_k_lt5"] == 0
    # (region, gender): classes of 11, 1 (the lone Texas woman), 4, 3 over the 19 users with both.
    rg = c.loc["region+gender"]
    assert rg["n_users"] == 19 and rg["share_k1"] == pytest.approx(1 / 19)
    assert rg["share_k_lt5"] == pytest.approx(8 / 19) and rg["median_k"] == 11
    assert rg["share_k1_ci_low"] <= rg["share_k1"] <= rg["share_k1_ci_high"]
    assert c.loc["age_band", "suppressed"]                   # `unknown` is not a value: nobody has age_band
    assert len(by_combo) == 6 + 15 + 20 + 15

    s = summary[summary["ci_kind"] == "bootstrap_users"].set_index("n_qi")
    assert s.loc[1, "share_unique"] == 0 and s.loc[2, "share_unique"] == pytest.approx(1 / 20)
    assert s.loc[4, "share_unique"] == pytest.approx(1 / 20)      # cumulative in n_qi
    a = by_act.set_index(["activity_bucket", "n_qi"])
    assert a.loc[("4-5", 2), "share_unique"] == pytest.approx(1 / 10) and a.loc[("3", 2), "share_unique"] == 0
    # Sample-size trend rows exist for sizes below the sample.
    assert set(summary.loc[summary["ci_kind"] == "subsample_draws", "sample_users"]) == {10}


def test_uniqueness_suppression_and_no_values():
    df = _profiles_with_known_k()
    by_combo, by_act, summary = uniqueness.analyze(df, {}, {"bootstrap_reps": 50, "ci_level": 0.95}, seed=1,
                                                   min_cell=10)
    c = by_combo.set_index("combination")
    assert not c.loc["region", "suppressed"] and c.loc["region", "n_users"] == 20
    small = c.loc["age_band"]                                # fewer than 10 users have it
    assert small["suppressed"] and pd.isna(small["n_users"]) and pd.isna(small["share_k1"])
    assert not by_act["suppressed"].any()                    # both buckets hold exactly 10 users
    df.loc[0, "activity_bucket"] = "6-10"                    # now buckets hold 9, 10 and 1 users
    _, by_act, _ = uniqueness.analyze(df, {}, {"bootstrap_reps": 50, "ci_level": 0.95}, seed=1, min_cell=10)
    sup = by_act.groupby("activity_bucket")["suppressed"].all()
    assert sup["3"] and sup["6-10"] and not sup["4-5"]
    assert by_act.loc[by_act["suppressed"], ["n_users", "share_unique", "mean_known_attributes"]].isna().all().all()
    # No profile value appears in any output cell; only dimension names do.
    for table in (by_combo, by_act, summary):
        text = table.astype(str).to_string().casefold()
        for value in ("texas", "ohio", "female"):
            assert value not in text
