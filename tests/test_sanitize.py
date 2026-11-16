"""Layered sanitizer (RQ4): span layers, metrics, the hosted loop with a fake client, budget cap,
and the canary check (no sanitized or original text may reach results)."""
import json
from types import SimpleNamespace

import pandas as pd

from residpii import labeling, sanitize

from test_labeling import _gold_for, _texts, labelled_run  # noqa: F401  (fixture)
from test_pipeline import CANARIES


def test_redact_merges_overlaps():
    text = "Call Ann Lee at 555-0101 now"
    out, merged = sanitize.redact(text, [{"start": 5, "end": 12, "category": "name"},
                                         {"start": 9, "end": 12, "category": "name"},
                                         {"start": 16, "end": 24, "category": "phone"}])
    assert out == "Call [NAME] at [PHONE] now"
    assert [(m["start"], m["end"]) for m in merged] == [(5, 12), (16, 24)]
    assert sanitize.redact(text, []) == (text, [])


def test_leak_and_retention():
    assert sanitize.leaked("Sarah  Jones", "Hi, I'm sarah jones from Leeds")
    assert not sanitize.leaked("Sarah Jones", "Hi, I'm [NAME] from Leeds")
    assert sanitize.leaked("teach", "I am a teacher")          # gold span may end mid-word
    assert not sanitize.leaked("IT", "I submit forms")          # short values match whole words only
    assert sanitize.leaked("IT", "I work in IT.")
    orig = "I am a nurse in Leeds and I need help"
    kept, total = sanitize.retention(orig, "I am a healthcare worker in the UK and I need help",
                                     [(7, 12), (16, 21)])
    assert total == 8 and kept == 8                            # all non-PII words survive
    kept, total = sanitize.retention(orig, "I need help", [(7, 12), (16, 21)])
    assert kept == 3


class _FakeMessages:
    """Rewrites drop the canary name; the attacker is confident until the word 'nurse' is gone."""

    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        props = kw["output_config"]["format"]["schema"]["properties"]
        user = kw["messages"][0]["content"]
        turn = user.split("<turn>\n")[-1].split("\n</turn>")[0] if "<turn>" in user else ""
        if "text" in props:
            body = {"text": turn.replace("Quillon Varnadore", "[NAME]").replace("nurse", "healthcare worker")}
        elif "inferences" in props:
            inf = [{"attribute": "occupation", "value": "nurse", "confidence": 5, "cue": "nurse"}] if "nurse" in turn else []
            body = {"inferences": inf}
        else:
            body = {"score": 4}
        return SimpleNamespace(model=kw["model"], stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(body))],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0,
                                                     cache_creation_input_tokens=0))


def _label(cfg, pipe):
    items = pd.read_parquet(labeling.items_path(pipe.paths, "wildchat"))
    for iid, text in _texts(cfg, items).items():
        labeling.save_label(pipe.paths, "wildchat", "kai",
                            labeling.validate_record({**_gold_for(text), "item_id": iid}, cfg["labeling"], len(text)))


def test_offline_layers_evaluate_without_api(labelled_run):  # noqa: F811
    cfg, pipe, _, _ = labelled_run
    _label(cfg, pipe)
    tables = sanitize.cmd_sanitize(cfg, "wildchat", "kai")
    leak = tables["sanitizer_leakage"]
    assert set(leak["stage"]) == {"shipped", "presidio", "patterns", "llm_identifiers",
                                  "rules_identifiers", "rules_generalize"}
    shipped = leak[(leak.stage == "shipped") & (leak.metric == "span_leak_rate")]
    assert (shipped["value"] == 1).all()
    pres = leak[(leak.stage == "presidio") & (leak.tier == "direct") & (leak.metric == "span_leak_rate")]
    assert pres["value"].iloc[0] < 1                           # Presidio removes the canary name and phone
    assert "sanitizer_exposure" not in tables


def test_hosted_loop_budget_and_no_text_in_results(labelled_run):  # noqa: F811
    cfg, pipe, _, _ = labelled_run
    _label(cfg, pipe)
    fake = _FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    # A tiny cap stops early without overshooting by more than one item's in-flight calls.
    s1 = sanitize.run_hosted(cfg, "wildchat", "kai", max_cost=0.0003, workers=1, client=client)
    assert s1["stopped_for_budget"]
    s2 = sanitize.run_hosted(cfg, "wildchat", "kai", max_cost=5, workers=2, client=client)
    assert not s2["stopped_for_budget"] and s2["failed"] == 0
    store = sanitize.load_store(pipe.paths, "wildchat")
    items, _, _ = sanitize.eval_set(cfg, "wildchat", "kai")
    assert not sanitize.plan_steps(items, store)               # everything done; a rerun calls nothing
    n = len(fake.calls)
    sanitize.run_hosted(cfg, "wildchat", "kai", max_cost=5, workers=2, client=client)
    assert len(fake.calls) == n
    scfg = cfg["sanitizer"]
    assert scfg["eval_attacker_model"] != scfg["attacker_model"]          # held-out attacker is a different model
    attackers = {c["model"] for c in fake.calls if "inferences" in c["output_config"]["format"]["schema"]["properties"]}
    assert attackers == {scfg["attacker_model"], scfg["eval_attacker_model"]}
    for r in store.values():
        if r["step"] in ("generalize", "adversarial"):
            assert "Quillon Varnadore" not in r["text"]
    tables = sanitize.cmd_sanitize(cfg, "wildchat", "kai")
    assert {"sanitizer_exposure", "sanitizer_utility"} <= set(tables)
    leak = tables["sanitizer_leakage"]
    adv = leak[(leak.stage == "adversarial") & (leak.tier == "direct") & (leak.metric == "span_leak_rate")]
    assert adv["value"].iloc[0] < 1
    for f in pipe.paths.tables.glob("sanitizer_*.csv"):
        blob = f.read_text().casefold()
        assert not any(c.casefold() in blob for c in CANARIES)
    assert sanitize.store_path(pipe.paths, "wildchat").stat().st_mode & 0o077 == 0


# ---------------------------------------------------------------- rule-based layers (no LLM)

from residpii import rules_sanitize as rs


def _run(text, task="info_or_other", options=None):
    doc = rs.nlp()(text)
    ident = rs.identifier_edits(text, doc, task, options)
    return rs.apply(text, ident + rs.generalize_edits(text, doc, task, options))[0]


def test_rules_redact_real_names_but_not_characters():
    letter = "Dear Margaret Okafor,\nThank you for the offer.\n\nBest regards,\nTomasz Wilk"
    out = _run(letter, task="communication")
    assert "Okafor" not in out and "Tomasz" not in out and "[NAME]" in out
    assert "Elara" in _run("Write a story where Elara Moonwhisper finds a sword.", task="story_roleplay")
    assert "Hemingway" in _run("Summarize the style of Ernest Hemingway in two lines.")
    assert "Priya" not in _run("Please help me reply to my manager Priya about the deadline.")


def test_rules_identifiers_patterns():
    out = _run("Follow me @quill_dev or https://github.com/quilldev, server 192.168.1.20, I was born on 12/03/1991.")
    assert "@quill_dev" not in out and "github.com/quilldev" not in out
    assert "192.168.1.20" not in out and "1991" not in out


def test_rules_generalize_personal_sentences_only():
    out = _run("I am a nurse in Leeds and I am 34 years old. I work at St Mary's Hospital.")
    assert "Leeds" not in out and "the UK" in out
    assert "34" not in out and "30-39" in out
    assert "nurse" not in out and "healthcare professional" in out
    assert "St Mary's Hospital" not in out
    # General knowledge is left alone: no first person, not a CV or letter.
    assert "Leeds" in _run("Leeds is a city in West Yorkshire with a large university.")
    # A listed country stays as it is.
    assert "Canada" in _run("I moved to Canada last year.")
    # Code and stories are skipped by the generalisation rules.
    assert "Leeds" in _run("I am writing a story set in Leeds.", task="story_roleplay")


def test_rules_options_switch_organizations():
    text = "My favourite tool from Acme Robotics is broken."
    assert "Acme Robotics" in _run(text)
    assert "Acme Robotics" not in _run(text, options={"organizations_in_first_person": True})


def test_rules_paths_flag_personal_directories_only():
    out = _run("df = pd.read_csv('/Users/jsmith/Documents/thesis/data.csv')", task="code")
    assert "jsmith" not in out and "[PATH]" in out
    assert "Lopez" not in _run(r"open(r'C:\Users\Maria Lopez\Desktop\taxes.xlsx')", task="code")
    assert "akash" not in _run("python /home/akash/projects/acme/run.py", task="code")
    assert "tom" not in _run("cd /mnt/c/Users/tom/repos", task="code")
    assert "secrets" not in _run("cat ~/secrets/keys.txt", task="code")
    # System paths, web routes, URLs and relative source paths stay.
    for keep in ("export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu", "@app.route('/api/v1/users')",
                 "See https://example.com/a/b/c for docs", "cat ./src/app/main.py",
                 r"C:\Windows\System32\cmd.exe", r"C:\Program Files\Python311\python.exe"):
        assert "[PATH]" not in _run(keep, task="code"), keep


def test_rules_leave_code_alone():
    code = "import numpy as np\n@property\ndef name(self):\n    return np.array(Jordan_data)\n@app.route('/x')"
    out = _run(code, task="code")
    assert "@property" in out and "@app.route" in out and "np.array" in out and "[USERNAME_HANDLE]" not in out
    assert "[USERNAME_HANDLE]" in _run("my instagram is @sunny_days_88", task="code")    # cued handle in a code turn
    assert "Leeds" not in _run("I live in Leeds.")                                     # full stop is not code
    assert rs.code_context("x = np.array", 4, 6) and not rs.code_context("I live in Leeds.", 10, 15)


def test_public_fictional_and_common_names_are_kept():
    assert rs.is_known_name("Mark Zuckerberg") and rs.is_known_name("Zuckerberg") and rs.is_known_name("Luke Skywalker's")
    assert rs.is_known_name("John Doe") and not rs.is_known_name("Priya Raman") and not rs.is_known_name("Smith")
    wide = {"names_outside_story": True}
    # Celebrities and characters stay even where names are otherwise redacted.
    out = _run("What did Elon Musk say about Tesla, and would Harry Potter agree?", options=wide)
    assert "Elon Musk" in out and "Harry Potter" in out
    # A lone common first name stays without a strong cue, but a strong cue still redacts it.
    assert "Emma" in _run("Write two lines where Emma says hello.", options=wide)
    assert "Emma" not in _run("Please reply to my sister Emma about Sunday.", options=wide)
    # An unlisted full name is redacted under the wide option; with lists off, celebrities are too.
    assert "Priya Raman" not in _run("Summarise the email from Priya Raman about the budget.", options=wide)
    assert "Elon Musk" not in _run("What did Elon Musk say?", options={**wide, "public_name_lists": False})
