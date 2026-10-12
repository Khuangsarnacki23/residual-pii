import numpy as np
import pandas as pd

from residpii import dedup
from residpii.analysis import weighted_prevalence
from residpii.config import REPO_ROOT, load_config, load_taxonomy
from residpii.detect import is_dummy, secret_matches, tag_tasks
from residpii.redactions import compile_patterns, discover, matches_configured
from residpii.safety import assert_safe_table, keyed_hash
from residpii.sampling import _proportional_allocation, turn_bucket


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
