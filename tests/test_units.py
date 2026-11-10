import numpy as np
import pandas as pd

from residpii import dedup
from residpii.analysis import weighted_prevalence
from residpii.config import REPO_ROOT, load_config, load_taxonomy
from residpii.analysis import exclusion_table, residual
from residpii.detect import is_dummy, rule_matches, secret_matches, selfid_matches, tag_tasks
from residpii.redactions import compile_patterns, discover, matches_configured
from residpii.safety import assert_safe_table, keyed_hash
from residpii.sampling import (NoUserProxy, _proportional_allocation, turn_bucket, user_activity,
                               user_level_sample)
from residpii.safety import suppress_small_cells
from residpii.sources import InMemorySource


def test_keyed_hash_normalizes_and_depends_on_key():
    assert keyed_hash(b"k1", "Ada  LOVELACE") == keyed_hash(b"k1", "ada lovelace")
    assert keyed_hash(b"k1", "x") != keyed_hash(b"k2", "x")


def test_assert_safe_table_blocks_text_columns():
    assert_safe_table(pd.DataFrame({"dataset": ["a"], "n": [1]}), "ok")
    try:
        assert_safe_table(pd.DataFrame({"text": ["leak"]}), "bad")
    except ValueError:
        return
    raise AssertionError("text column was not blocked")


def test_dummy_values():
    assert is_dummy("EMAIL_ADDRESS", "a@example.com")
    assert is_dummy("PHONE_NUMBER", "555-0100")
    assert is_dummy("IP_ADDRESS", "192.168.1.10")
    assert is_dummy("CREDIT_CARD", "4111 1111 1111 1111")
    assert is_dummy("PERSON", "DAN")
    assert not is_dummy("PERSON", "Dan")
    assert not is_dummy("EMAIL_ADDRESS", "a@zephyrmailtest.org")


def test_secret_patterns():
    hits = list(secret_matches("key = 'sk-abcdefghijklmnopqrstuvwx1234' and AKIAABCDEFGHIJKLMNOP"))
    assert len(hits) == 2
    s, e, _ = next(iter(secret_matches('password = "hunter2hunter2"')))
    assert 'password = "hunter2hunter2"'[s:e] == "hunter2hunter2"


def test_turn_buckets():
    assert list(turn_bucket(pd.Series([1, 2, 3, 4, 9]), [1, 2, 4])) == ["1", "2-3", "2-3", "4+", "4+"]


def test_allocation_sums_and_caps():
    sizes = pd.Series({"a": 5, "b": 1000, "c": 3})
    alloc = _proportional_allocation(sizes, 100)
    assert alloc.sum() == 100 and (alloc <= sizes).all()


def test_weighted_prevalence_ci_brackets_point():
    rng = np.random.default_rng(0)
    X = (rng.random((500, 2)) < [0.1, 0.5]).astype(float)
    p, lo, hi = weighted_prevalence(X, np.ones(500), np.zeros(500), 300, 0.95, rng)
    assert (lo <= p).all() and (p <= hi).all()
    assert abs(p[0] - 0.1) < 0.05


def test_dedup_exact_and_near():
    ref = pd.Series({"a": "write a poem about the sea and the sky at night in summer",
                     "b": "completely different text about databases and indexes"})
    q = pd.Series({"x": "write a poem about the sea and the sky at night in summer",
                   "y": "write a poem about the sea and the sky at night in summer please",
                   "z": "nothing in common whatsoever here"})
    cfg = {"num_perm": 128, "shingle_words": 3, "bands": 32, "jaccard_threshold": 0.7}
    res = dedup.match(ref, q, cfg, seed=1).set_index("conversation_id")["dedup_status"]
    assert res["x"] == "exact" and res["y"] == "near" and res["z"] == "unique"


def test_placeholder_discovery_is_k_thresholded():
    turns = pd.DataFrame({
        "dataset": ["d"] * 6, "conversation_id": [f"c{i}" for i in range(6)],
        "text": ["hi NAME_1"] * 5 + ["ONE_OFF_SECRETNAME"],
    })
    out = discover(turns, min_conversations=5)
    assert list(out["placeholder_shape"]) == ["NAME_#"]


def test_dummy_filter_needs_whole_value_placeholder():
    # An ellipsis or tag inside a longer value does not make it a dummy.
    assert not is_dummy("PERSON", "Dr. Smith...")
    assert not is_dummy("PERSON", "<b>Ada Lovelace</b>")
    assert not is_dummy("LOCATION", "Paris… ")
    assert is_dummy("PERSON", "...")
    assert is_dummy("PERSON", "<your name>")
    assert is_dummy("EMAIL_ADDRESS", "<email>")
    assert is_dummy("SECRET_TOKEN", "sk-xxxxxxxxxxxxxxxxxxxx")


def _task(text: str) -> str:
    taxonomy = load_taxonomy(load_config(REPO_ROOT / "configs" / "default.yaml"))
    turns = pd.DataFrame({"dataset": ["d"], "conversation_id": ["c"], "turn_index": [0], "text": [text]})
    return tag_tasks(turns, taxonomy)["task_type"].iloc[0]


def test_task_type_code_patterns_next_to_punctuation():
    assert _task("I get error: list index out of range") == "code"
    assert _task("can you help with c++ homework") == "code"
    assert _task("#include <stdio.h>") == "code"
    assert _task("why does this fail\n```\nx\n```") == "code"


def test_task_type_code_keywords_need_context():
    # Lone English words that happen to be code keywords are not enough.
    assert _task("There is a bug in my garden") == "info_or_other"
    assert _task("When should I return the library book?") == "info_or_other"
    assert _task("What is the function of the liver?") == "info_or_other"
    # ...but they count with code-like context or a second distinct keyword.
    assert _task("what does return x = 5; do") == "code"
    assert _task("the function f(x) is slow") == "code"
    assert _task("should the class import that module") == "code"


def test_placeholder_config_keys():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert cfg["placeholders"]["discovery_min_conversations"] == 5
    pattern = compile_patterns(cfg, "wildchat")    # the scalar key must not break pattern compilation
    assert matches_configured("NAME_#", pattern)
    assert not matches_configured("SOME_OTHER_#", pattern)


# ---------------------------------------------------------------- user-level sampling


def _user_meta(counts: dict[str, int]) -> pd.DataFrame:
    """Metadata for synthetic users with a known number of conversations each."""
    convs = [{"conversation_id": f"{u}-{i}", "user": u, "turns": [{"role": "user", "content": "hi"}]}
             for u, n in counts.items() for i in range(n)]
    return InMemorySource("d", convs, b"k").scan_meta()


def _ul_cfg(n_users=100, min_conversations=3, cap=5) -> dict:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    cfg["user_level"] = {"n_users": n_users, "min_conversations": min_conversations,
                         "max_conversations_per_user": cap}
    return cfg


def test_user_level_sample_respects_minimum_and_cap():
    counts = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 9, "g": 60}
    meta = _user_meta(counts)
    smp, summary = user_level_sample(meta, _ul_cfg(cap=5), seed=1)
    kept = smp.groupby("user_key").size()
    full = smp.groupby("user_key")["user_n_conversations"].first()
    assert summary["eligible_users"] == summary["sampled_users"] == 5        # c, d, e, f, g
    assert (full >= 3).all() and (kept <= 5).all()
    # Every conversation is kept for users at or under the cap; heavy users are cut to the cap.
    assert (kept[full <= 5] == full[full <= 5]).all()
    assert sorted(kept[full > 5]) == [5, 5]
    assert summary["users_over_cap"] == 2 and summary["sampled_users_over_cap"] == 2
    assert summary["sampling_frame"] == 3 + 4 + 5 + 9 + 60
    # Weights expand capped users back to their full conversation count.
    assert np.isclose(smp["weight"].sum(), summary["sampling_frame"])
    assert set(smp["activity_bucket"]) == {"3", "4-5", "6-10", ">50"}


def test_user_level_sample_is_seeded():
    meta = _user_meta({f"u{i}": 3 + i % 7 for i in range(40)})
    cfg = _ul_cfg(n_users=10, cap=4)
    a, _ = user_level_sample(meta, cfg, seed=7)
    b, _ = user_level_sample(meta, cfg, seed=7)
    c, _ = user_level_sample(meta, cfg, seed=8)
    assert list(a["conversation_id"]) == list(b["conversation_id"])
    assert a["user_key"].nunique() == 10
    assert set(a["conversation_id"]) != set(c["conversation_id"])


def test_user_activity_buckets_and_no_proxy():
    meta = _user_meta({"a": 1, "b": 1, "c": 3, "d": 4, "e": 5, "f": 30, "g": 51})
    act = user_activity(meta, _ul_cfg()).set_index("activity_bucket")
    assert act["n_users"].to_dict() == {"1": 2, "2": 0, "3": 1, "4-5": 2, "6-10": 0, "11-25": 0,
                                        "26-50": 1, ">50": 1}
    assert not act.loc["2", "meets_min_conversations"] and act.loc["3", "meets_min_conversations"]
    try:
        user_level_sample(meta.assign(user_key=None), _ul_cfg(), seed=1)
    except NoUserProxy:
        return
    raise AssertionError("a dataset without a user proxy was sampled at user level")


def test_small_cell_suppression():
    df = pd.DataFrame({"group": ["a", "b", "c"], "n_users": [3, 10, 0], "share": [0.3, 0.5, 0.0]})
    out = suppress_small_cells(df, "n_users", ["share"], min_n=10)
    assert list(out["suppressed"]) == [True, False, True]
    assert out["n_users"].isna().tolist() == [True, False, True]
    assert out["share"].isna().tolist() == [True, False, True]
    assert out.loc[1, "n_users"] == 10 and out.loc[1, "share"] == 0.5


def test_roles_select_turns():
    convs = [{"conversation_id": "c", "user": "u", "turns": [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "human", "content": "q2"}, {"role": "system", "content": "s"}]}]
    src = InMemorySource("d", convs, b"k")
    meta = src.scan_meta()
    assert list(src.fetch_turns(meta)["text"]) == ["q1", "q2"]
    both = src.fetch_turns(meta, ("user", "assistant"))
    assert list(both["role"]) == ["user", "assistant", "user"]
    assert list(both["turn_index"]) == [0, 1, 2]


# ---------------------------------------------------------------- self-introductions


def _selfid(text: str) -> list[str]:
    return [text[s:e] for s, e, _ in selfid_matches(text)]


def test_selfid_positives():
    assert _selfid("Hi, my name is Ada Lovelace and I need help.") == ["Ada Lovelace"]
    assert _selfid("my name's Grace") == ["Grace"]
    assert _selfid("You can call me Tariq.") == ["Tariq"]
    assert _selfid("Hello, this is Mei Ling speaking, I called earlier") == ["Mei Ling"]
    assert _selfid("I'm Priya, a nurse from Leeds.") == ["Priya"]
    assert _selfid("i am Juan Carlos Ortega") == ["Juan Carlos Ortega"]
    # At most three tokens, and a stoplisted word ends the name.
    assert _selfid("My name is Anna Maria Della Rovere") == ["Anna Maria Della"]
    assert _selfid("My name is Omar From Cairo") == ["Omar"]


def test_selfid_negatives():
    for text in ["I'm Sorry to bother you", "I'm Happy with that", "I'm Not sure", "I am American.",
                 "I'm Muslim and I fast", "i'm tired", "my name is unknown", "Call me Maybe",
                 "call me back later", "I'm Working On A Project", "I'm Batman fighting crime",
                 "This is Sparta", "this is John's book"]:
        assert _selfid(text) == [], text


def test_selfid_dummy_names():
    assert is_dummy("SELF_NAME", "John Doe") and is_dummy("SELF_NAME", "DAN")
    assert not is_dummy("SELF_NAME", "Ada Lovelace")


# ---------------------------------------------------------------- rules detector


def _rules(text: str) -> list[tuple[str, str]]:
    return [(text[s:e], ent) for s, e, ent in rule_matches(text)]


def test_rules_positives():
    assert _rules("my page: https://www.linkedin.com/in/ada-lovelace-1815/ thanks") == [
        ("https://www.linkedin.com/in/ada-lovelace-1815", "PERSONAL_URL")]
    assert _rules("see github.com/adalove.") == [("github.com/adalove", "PERSONAL_URL")]
    assert [v for v, _ in _rules("x.com/adalove t.me/adalovelace tiktok.com/@ada.l reddit.com/u/ada_l")] == [
        "x.com/adalove", "t.me/adalovelace", "tiktok.com/@ada.l", "reddit.com/u/ada_l"]
    assert _rules("follow me on instagram @ada.lovelace or twitter: @ada_l") == [
        ("@ada.lovelace", "SOCIAL_HANDLE"), ("@ada_l", "SOCIAL_HANDLE")]
    assert _rules("my discord is adal#4521") == [("adal#4521", "SOCIAL_HANDLE")]
    assert _rules("I live at 221B Baker Street, Flat 2.") == [("221B Baker Street, Flat 2", "STREET_ADDRESS")]
    assert _rules("ship to 1600 N. Pennsylvania Ave.") == [("1600 N. Pennsylvania Ave.", "STREET_ADDRESS")]
    assert [v for v, _ in _rules("DOB: 03/14/1990; born on March 5, 1990; my birthday is 5th of May 1991")] == [
        "03/14/1990", "March 5, 1990", "5th of May 1991"]
    assert _rules("student ID: B7720491, passport no. X98765432") == [
        ("B7720491", "CONTEXT_ID"), ("X98765432", "CONTEXT_ID")]
    assert _rules("call +61 491 570 006") == [("+61 491 570 006", "INTL_PHONE")]


def test_rules_negatives():
    for text in ["github.com/adalove/repo and https://github.com/features",      # a repo and a site page
                 "max.com/foo and facebook.com/groups",
                 "email ig ada@gmail.com",                                       # an email is not a handle
                 "username\n@property\ndef f(self): ...",                         # decorator on the next line
                 "@media (max-width: 600px)",                                    # no platform word nearby
                 "a 2 TB Hard Drive, 3 Easy Way and 1 Google Drive",
                 "born in 1990", "the report is due 03/14/2024",
                 "student id is required", "employee number of staff",
                 "1+2, +5, C++17 and +1"]:
        assert _rules(text) == [], text


def test_rules_dummy_values():
    assert is_dummy("PERSONAL_URL", "linkedin.com/in/username") and is_dummy("SOCIAL_HANDLE", "@your_handle")
    assert not is_dummy("PERSONAL_URL", "linkedin.com/in/ada-lovelace") and not is_dummy("SOCIAL_HANDLE", "@ada_l")


def _span(detector, entity, start, end, score=1.0, turn=0):
    return {"dataset": "d", "conversation_id": "c", "turn_index": turn, "role": "user", "start": start, "end": end,
            "detector": detector, "entity": entity, "category": entity.lower(), "score": score,
            "in_placeholder": False, "is_dummy": False}


def test_overlapping_detections_are_counted_once():
    spans = pd.DataFrame([
        _span("rules", "PERSONAL_URL", 10, 40), _span("presidio", "URL", 10, 45, 0.6),       # rule refines the URL
        _span("rules", "CONTEXT_ID", 60, 69), _span("presidio", "US_PASSPORT", 60, 69, 0.6),   # Presidio already has it
        _span("rules", "INTL_PHONE", 80, 95), _span("presidio", "PHONE_NUMBER", 80, 95, 0.4),  # below threshold
        _span("rules", "STREET_ADDRESS", 10, 30, turn=1), _span("presidio", "LOCATION", 15, 22, 0.85, turn=1),
        _span("presidio", "LOCATION", 50, 56, 0.85, turn=1),                                   # elsewhere: untouched
    ])
    kept = residual(spans, 0.5)
    assert sorted(zip(kept["detector"], kept["entity"])) == [
        ("presidio", "LOCATION"), ("presidio", "US_PASSPORT"), ("rules", "INTL_PHONE"), ("rules", "PERSONAL_URL"),
        ("rules", "STREET_ADDRESS")]
    assert kept.loc[kept["entity"] == "LOCATION", "start"].tolist() == [50]
    # No two kept spans share text, and every stored span has exactly one reason.
    for _, g in kept.groupby("turn_index"):
        iv = sorted(zip(g["start"], g["end"]))
        assert all(a[1] <= b[0] for a, b in zip(iv, iv[1:]))
    reasons = exclusion_table(spans, 0.5).set_index(["detector", "category"])
    assert reasons.loc[("rules", "context_id"), "covered_by_other_detector"] == 1
    assert reasons.loc[("presidio", "url"), "refined_by_rule"] == 1
    assert reasons.loc[("presidio", "location"), ["refined_by_rule", "residual"]].tolist() == [1, 1]
    assert reasons.loc[("presidio", "phone_number"), "below_threshold"] == 1
    # At a threshold that admits Presidio's phone hit, Presidio gets the count instead of the rule.
    low = residual(spans, 0.4)
    assert ("presidio", "PHONE_NUMBER") in set(zip(low["detector"], low["entity"]))
    assert "INTL_PHONE" not in set(low["entity"])
