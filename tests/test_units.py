import numpy as np
import pandas as pd

from residpii import dedup
from residpii.analysis import weighted_prevalence
from residpii.config import REPO_ROOT, load_config, load_taxonomy
from residpii.detect import is_dummy, secret_matches, tag_tasks
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
