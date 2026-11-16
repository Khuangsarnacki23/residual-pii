"""User-level adversarial loop: risk against the reference sample, target choice, both loops with a
fake model client, the utility and edit gates, the call cache and the budget cap. All values fabricated."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from residpii import user_adversarial as ua
from residpii.config import REPO_ROOT, load_config
from residpii.profiles import DIMENSIONS, load_coarsener


def _cfg():
    return load_config(REPO_ROOT / "configs" / "user_level.yaml")


def _ref(rows):
    """rows: user -> {dim: value}; everything else unknown."""
    recs = [{"dataset": "d", "user": u, "variant": "stated_plus_inferred", **{d: None for d in DIMENSIONS}, **vals}
            for u, vals in rows.items()]
    return ua.Reference(pd.DataFrame(recs))


# Reference: occupation 29 and country UK are each common, but only "me" has both.
REF = {"me": {"occupation_group": "29", "country": "united kingdom"},
       **{f"o{i}": {"occupation_group": "29"} for i in range(12)},
       **{f"c{i}": {"country": "united kingdom"} for i in range(12)}}


def test_min_k_and_target():
    ref = _ref(REF)
    me = {"occupation_group": "29", "country": "united kingdom"}
    k, combo = ref.min_k(me, "me")
    assert k == 1 and combo == ("country", "occupation_group")
    assert ref.min_k({"country": "united kingdom"}, "me")[0] == 13      # 12 others + the user
    assert ref.min_k({}, "me")[0] == float("inf")
    # Dropping either leaves 13; ties go to the first name, and blocked attributes are skipped.
    assert ua.choose_target(me, ref, "me", set()) == "country"
    assert ua.choose_target(me, ref, "me", {"country"}) == "occupation_group"
    assert ua.choose_target(me, ref, "me", {"country", "occupation_group"}) is None


def test_profile_votes_per_conversation():
    c = load_coarsener(_cfg())
    inf = {"a#0": [{"attribute": "occupation", "value": "nurse", "confidence": 5, "cue": "x"}],
           "a#1": [{"attribute": "occupation", "value": "doctor", "confidence": 5, "cue": "x"}],
           "b#0": [{"attribute": "occupation", "value": "teacher", "confidence": 5, "cue": "x"}],
           "c#0": [{"attribute": "location_country", "value": "UK", "confidence": 3, "cue": "x"}]}   # not confident
    conv = {"a#0": "a", "a#1": "a", "b#0": "b", "c#0": "c"}
    # Conversation a counts once for group 29 (nurse and doctor), b once for 25: a tie, so dropped.
    assert ua.build_profile(inf, conv, c, 4) == {}
    inf["c#0"] = [{"attribute": "occupation", "value": "pharmacist", "confidence": 4, "cue": "x"}]
    assert ua.build_profile(inf, conv, c, 4) == {"occupation_group": "29"}


def test_changed_share():
    assert ua.changed_share("abc", "abc") == 0
    assert 0 < ua.changed_share("I am a nurse", "I am a worker") < 0.5
    assert ua.changed_share("abc", "") == 1


class _Fake:
    """Attacker: 'nurse' -> occupation, 'Leeds' -> UK. Rewrites replace the cue words.
    The judge rates a rewrite 2 if it contains 'BROKEN', else 4."""

    def __init__(self, break_rewrites=False, break_targeted=False):
        self.calls, self.break_rewrites, self.break_targeted = [], break_rewrites, break_targeted

    def create(self, **kw):
        self.calls.append(kw)
        props = kw["output_config"]["format"]["schema"]["properties"]
        user = kw["messages"][0]["content"]
        turns = [t.split("\n</turn>")[0] for t in user.split("<turn>\n")[1:]]
        if "text" in props:
            new = turns[-1].replace("nurse", "worker").replace("Leeds", "a city")
            broken = self.break_rewrites or (self.break_targeted and user.startswith("Someone who reads"))
            body = {"text": new + (" BROKEN" if broken else "")}
        elif "inferences" in props:
            joined = " ".join(turns)
            body = {"inferences": ([{"attribute": "occupation", "value": "nurse", "confidence": 5, "cue": "nurse"}]
                                   if "nurse" in joined else [])
                    + ([{"attribute": "location_country", "value": "UK", "confidence": 5, "cue": "Leeds"}]
                       if "Leeds" in joined else [])}
        else:
            body = {"score": 2 if "BROKEN" in user.split("Sanitized:")[-1] else 4}
        return SimpleNamespace(model=kw["model"], stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(body))],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                                     cache_read_input_tokens=0, cache_creation_input_tokens=0))


def _loops(tmp_path, fake, max_cost=5.0):
    cfg = _cfg()
    call = ua.Caller(SimpleNamespace(restricted=tmp_path), "d", cfg["sanitizer"], max_cost,
                     SimpleNamespace(beta=SimpleNamespace(messages=fake)))
    return ua.Loops(cfg, _ref(REF), load_coarsener(cfg), call), call


BASE = {"a#0": "I am a nurse and need a rota", "b#0": "Weekend plans near Leeds?", "c#0": "Fix my code"}
CONV = {"a#0": "a", "b#0": "b", "c#0": "c"}


def test_user_level_rewrites_only_what_identifies(tmp_path):
    loops, _ = _loops(tmp_path, _Fake())
    start = {k: loops.attack(v) for k, v in BASE.items()}
    texts, inf, log = loops.user_level("me", BASE, start, CONV)
    assert log["k_by_round"][0] == 1 and log["stop_reason"] == "safe"
    # One attribute removed is enough (k = 13 >= 5): exactly one message changed, the other cue kept.
    changed = [k for k in BASE if texts[k] != BASE[k]]
    assert len(changed) == 1 and log["rewrites"] == 1
    assert texts["c#0"] == BASE["c#0"]
    # The per-message loop rewrites every message with any confident inference.
    pm, _, pm_log = loops.per_message(BASE, start)
    assert sum(pm[k] != BASE[k] for k in BASE) == 2


def test_utility_gate_undoes_bad_rewrites(tmp_path):
    loops, _ = _loops(tmp_path, _Fake(break_rewrites=True))
    start = {k: loops.attack(v) for k, v in BASE.items()}
    texts, _, log = loops.user_level("me", BASE, start, CONV)
    assert texts == BASE                                   # every rewrite judged 2 < 4: all undone
    assert log["rejected_utility"] > 0 and log["stop_reason"] == "no_target"


def test_cache_and_budget(tmp_path):
    fake = _Fake()
    loops, call = _loops(tmp_path, fake)
    loops.attack(BASE["a#0"])
    n = len(fake.calls)
    loops2, call2 = _loops(tmp_path, fake)                 # a new run reads the cache from disk
    loops2.attack(BASE["a#0"])
    assert len(fake.calls) == n and call2.spent == pytest.approx(call.spent)
    tight, _ = _loops(tmp_path, fake, max_cost=call.spent)  # nothing left under the cap
    with pytest.raises(ua.BudgetStop):
        tight.attack("a message never seen before")
    assert (tmp_path / "adversarial" / "calls_d.jsonl").stat().st_mode & 0o077 == 0


def test_history_attacker_sees_all_messages(tmp_path):
    fake = _Fake()
    loops, _ = _loops(tmp_path, fake)
    prof = loops.held_out(BASE, list(BASE))
    assert prof == {"occupation_group": "29", "country": "united kingdom"}
    assert fake.calls[-1]["model"] == loops.s["eval_attacker_model"] != loops.s["attacker_model"]


def test_history_loop_reads_whole_history_and_retries_gently(tmp_path):
    fake = _Fake(break_targeted=True)                       # the strict rewrite breaks every request
    loops, _ = _loops(tmp_path, fake)
    start = {k: loops.attack(v) for k, v in BASE.items()}
    texts, _, log = loops.user_level("me", BASE, start, CONV, list(BASE), history=True)
    assert log["stop_reason"] == "safe" and log["gentle_kept"] == 1
    assert sum(texts[k] != BASE[k] for k in BASE) == 1 and "BROKEN" not in "".join(texts.values())
    hist_calls = [c for c in fake.calls if c["messages"][0]["content"].count("<turn>") == len(BASE)]
    assert hist_calls and all(c["model"] == loops.s["attacker_model"] for c in hist_calls)
    # Without history mode there is no gentle retry: the same rewrites are all undone.
    texts2, _, log2 = loops.user_level("me", BASE, start, CONV)
    assert texts2 == BASE and log2["gentle_kept"] == 0
