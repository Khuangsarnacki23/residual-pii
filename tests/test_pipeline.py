"""End-to-end test on synthetic conversations, including the canary leak check.

All "PII" below is fabricated: reserved 555 numbers, the long-invalidated Woolworth SSN, payment
test cards, made-up names. The canary check asserts that none of it reaches any file outside the
restricted directory, nor stdout/stderr.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from residpii.config import REPO_ROOT, load_config
from residpii import llm
from residpii.pipeline import Pipeline
from residpii.safety import load_or_create_key
from residpii.sources import InMemorySource

CANARIES = [
    "Quillon Varnadore",
    "quillon.varnadore@zephyrmailtest.org",
    "(646) 391-2087",
    "078-05-1120",
    "sk-canaryZQ7xLmNoPqRsTuVwXyZ8f3k2j9d0q4w",
    "Brisvale Unity Hospital",
]


def _convs(prefix: str, n: int) -> list[dict]:
    texts = [
        "Hi, my name is Quillon Varnadore and my email is quillon.varnadore@zephyrmailtest.org. "
        "Please write a cover letter. My phone number is (646) 391-2087.",
        "Write a story where a knight named Sir Bramwell fights a dragon in a chapter.",
        "Fix this python code: import os\nopenai.api_key = 'sk-canaryZQ7xLmNoPqRsTuVwXyZ8f3k2j9d0q4w'",
        "My SSN is 078-05-1120, I was treated at Brisvale Unity Hospital in Denver on March 3 2023.",
        "Translate into French: hello, NAME_1 told me to contact NAME_2.",
        "Email me at someone@example.com or call 555-0100.",
    ]
    return [{"conversation_id": f"{prefix}-{i}", "user": f"u{i % (n - 2)}", "language": "English",
             "timestamp": f"2023-0{4 + i % 3}-01",
             "turns": [{"role": "user", "content": texts[i % len(texts)]}, {"role": "assistant", "content": "ok"}]}
            for i in range(n)]


@pytest.fixture()
def run(tmp_path, capfd):
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    cfg.update(run_name="test", data_root=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    cfg["analysis"]["bootstrap_reps"] = 50
    # Treat "Denver" as a pre-existing placeholder so the overlap-exclusion path is exercised
    # (Presidio itself never tags NAME_1-style tokens).
    cfg["placeholders"]["wildchat"] = [r"\bDenver\b"]
    cfg["placeholders"]["discovery_min_conversations"] = 2
    key = load_or_create_key(Path(cfg["data_root"]))
    sources = {name: InMemorySource(name, _convs(name, 30), key) for name in ("wildchat", "arena")}
    pipe = Pipeline(cfg, sources=sources)
    manifest = pipe.run(do_dedup=False)
    out, err = capfd.readouterr()
    return pipe, manifest, out + err


def test_outputs_written(run):
    pipe, manifest, _ = run
    prev = pd.read_csv(pipe.paths.tables / "prevalence.csv")
    assert {"wildchat", "arena"} <= set(prev["dataset"])
    assert (pipe.paths.results / "methods.md").exists()
    assert (pipe.paths.figures / "prevalence_by_category.png").exists()
    # The fabricated email/phone/SSN/key must have been detected as residual.
    any_all = prev[(prev.detector == "any") & (prev.variant == "all")].set_index(["dataset", "category"])
    for cat in ("email", "phone", "api_key_token", "name"):
        assert any_all.loc[("wildchat", cat), "n_positive"] > 0, cat


def test_placeholders_and_dummies_excluded(run):
    pipe, _, _ = run
    spans = pd.read_parquet(pipe.paths.restricted / "spans_wildchat.parquet")
    loc = spans[spans["entity"] == "LOCATION"]
    assert loc["in_placeholder"].any()            # "Denver", configured as a placeholder above
    assert spans["is_dummy"].any()                # example.com, 555-0100
    red = pd.read_csv(pipe.paths.tables / "existing_redactions.csv").set_index("dataset")
    assert red.loc["wildchat", "share_with_placeholder"] > 0


def test_placeholder_discovery_command(run):
    pipe, _, _ = run
    disc = pipe.discover_placeholders().set_index(["dataset", "placeholder_shape"])
    assert disc.loc[("wildchat", "NAME_#"), "configured"]
    assert disc.loc[("wildchat", "NAME_#"), "conversations"] >= 2
    assert (pipe.paths.tables / "placeholder_shapes.csv").exists()


def test_one_conversation_per_user(run):
    pipe, manifest, _ = run
    smp = pd.read_parquet(pipe.paths.work / "sample_wildchat.parquet")
    assert smp["user_key"].is_unique
    assert manifest["datasets"]["wildchat"]["sampling"]["one_per_user_applied"]


def _assert_no_canaries(pipe, console: str) -> None:
    for c in CANARIES:
        assert c not in console, f"canary leaked to console: {c[:4]}…"
    roots = [pipe.paths.results, pipe.paths.work]
    for root in roots:
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix == ".parquet":
                df = pd.read_parquet(f)
                blob = "\n".join(df[c].astype(str).str.cat(sep="\n") for c in df.columns)
            elif f.suffix in (".png", ".pdf"):
                continue  # figures contain only category labels; checked via their source tables
            else:
                blob = f.read_text(errors="ignore")
            for c in CANARIES:
                assert c.casefold() not in blob.casefold(), f"canary leaked into {f.name}"


def test_no_canary_leaks_outside_restricted(run):
    pipe, _, console = run
    _assert_no_canaries(pipe, console)


def test_selfid_detector_reported_separately(run):
    pipe, _, _ = run
    spans = pd.read_parquet(pipe.paths.restricted / "spans_wildchat.parquet")
    own = spans[spans["detector"] == "selfid"]
    # "my name is <canary name>" is tagged as the user's own name; other detectors leave about_user unset.
    assert len(own) and (own["entity"] == "SELF_NAME").all() and own["about_user"].eq(True).all()
    assert (own["category"] == "name").all() and (own["tier"] == "direct").all()
    assert spans.loc[spans["detector"] != "selfid", "about_user"].isna().all()
    prev = pd.read_csv(pipe.paths.tables / "prevalence.csv")
    row = prev[(prev.dataset == "wildchat") & (prev.variant == "all") & (prev.detector == "selfid")
               & (prev.category == "name")]
    assert row["n_positive"].iloc[0] > 0


def test_span_store_has_no_text(run):
    pipe, _, _ = run
    spans = pd.read_parquet(pipe.paths.restricted / "spans_wildchat.parquet")
    blob = "\n".join(spans[c].astype(str).str.cat(sep="\n") for c in spans.columns)
    for c in CANARIES:
        assert c.casefold() not in blob.casefold()


def test_restricted_dir_permissions(run):
    pipe, _, _ = run
    assert (pipe.paths.restricted.stat().st_mode & 0o777) == 0o700
    assert ((pipe.paths.data_root / "hmac.key").stat().st_mode & 0o777) == 0o600


# ---------------------------------------------------------------- user-level mode

USER_CONV_COUNTS = {f"u{i}": n for i, n in enumerate([1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8, 12, 60])}


def _user_convs(prefix: str, with_users: bool = True) -> list[dict]:
    base = _convs(prefix, 6)
    out = []
    for user, n in USER_CONV_COUNTS.items():
        for i in range(n):
            turns = [dict(t) for t in base[(i + len(user)) % len(base)]["turns"]]
            turns[1]["content"] = "Sure. You could also ask Marguerite Oyelaran at the Lisbon office."
            out.append({"conversation_id": f"{prefix}-{user}-{i}", "user": user if with_users else None,
                        "language": "English", "timestamp": "2023-05-01", "turns": turns})
    return out


@pytest.fixture()
def user_run(tmp_path, capfd):
    cfg = load_config(REPO_ROOT / "configs" / "user_level_smoke.yaml")
    cfg.update(run_name="utest", data_root=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    cfg["analysis"]["bootstrap_reps"] = 50
    cfg["user_level"]["max_conversations_per_user"] = 10
    key = load_or_create_key(Path(cfg["data_root"]))
    sources = {"wildchat": InMemorySource("wildchat", _user_convs("wildchat"), key),
               "lmsys": InMemorySource("lmsys", _user_convs("lmsys", with_users=False), key)}
    pipe = Pipeline(cfg, sources=sources)
    manifest = pipe.run(do_dedup=False)
    out, err = capfd.readouterr()
    return pipe, manifest, out + err


def test_user_level_run(user_run):
    pipe, manifest, console = user_run
    # No user proxy -> skipped with a logged reason, not sampled per conversation.
    assert "no user proxy" in manifest["datasets"]["lmsys"]["skipped"]
    assert "lmsys" in console and "user proxy" in console
    smp = pd.read_parquet(pipe.paths.work / "sample_wildchat.parquet")
    per_user = smp.groupby("user_key").agg(kept=("conversation_id", "size"), full=("user_n_conversations", "first"))
    assert len(per_user) == 11                                   # users with >= 3 conversations
    assert (per_user["full"] >= 3).all() and (per_user["kept"] <= 10).all()
    assert (per_user.loc[per_user["full"] <= 10, "kept"] == per_user.loc[per_user["full"] <= 10, "full"]).all()
    sampling = manifest["datasets"]["wildchat"]["sampling"]
    assert sampling["mode"] == "user_level" and sampling["sampled_users"] == 11
    assert sampling["users_over_cap"] == "<10"                   # small counts are never published exactly


def test_user_level_tables(user_run):
    pipe, _, console = user_run
    act = pd.read_csv(pipe.paths.tables / "user_activity.csv")
    assert set(act.columns) == {"dataset", "activity_bucket", "meets_min_conversations", "n_users",
                                "share_users", "suppressed"}
    assert act["suppressed"].all() and act["n_users"].isna().all()   # every bucket has < 10 users here
    turns = pd.read_parquet(pipe.paths.restricted / "turns_wildchat.parquet")
    assert set(turns["role"]) == {"user", "assistant"}
    by_role = pd.read_csv(pipe.paths.tables / "prevalence_by_role.csv")
    assert set(by_role["role"]) == {"user", "assistant"}
    # The assistant-side name counts for the assistant role only; prevalence.csv stays user-only.
    spans = pd.read_parquet(pipe.paths.restricted / "spans_wildchat.parquet")
    assert (spans.loc[spans["role"] == "assistant", "entity"] == "PERSON").any()
    _assert_no_canaries(pipe, console)


# ---------------------------------------------------------------- LLM stage


def _fake_llm(system: str, user: str) -> str:
    """Canned extraction keyed on the turn text. Values deliberately carry canaries: they may exist
    in the restricted directory and nowhere else."""
    turn = user.split("USER TURN:", 1)[1]

    def attr(typ, value, evidence="stated", about="user", confidence=0.9, **extra):
        return {"type": typ, "value": value, "evidence": evidence, "about": about, "confidence": confidence,
                "occupation_group": "", "occupation_minor_group": "", **extra}

    if "Translate" in turn:
        return "Sure! Here is the JSON you asked for"            # malformed -> retried -> failure
    attrs = []
    if "cover letter" in turn:
        attrs += [attr("occupation", "nurse at Brisvale Unity Hospital", occupation_group="29",
                       occupation_minor_group="29-1"),
                  attr("other", "is called Quillon Varnadore"), attr("gender", "man", evidence="inferred")]
    if "SSN" in turn:
        attrs += [attr("city", "Denver"), attr("age", "34"), attr("school", "Brisvale Unity Hospital college")]
    if "knight" in turn:
        attrs += [attr("occupation", "knight", about="fictional")]
    if "python" in turn:
        attrs += [attr("occupation", "software developer", evidence="inferred", confidence=0.7)]
    spans = [{"text": c, "type": typ, "about": "user"} for c, typ in (("Quillon Varnadore", "name"),
             ("quillon.varnadore@zephyrmailtest.org", "email")) if c in turn]
    task = "job_application" if "cover letter" in turn else "story_roleplay" if "knight" in turn else "info_or_other"
    return json.dumps({"task_type": task, "spans": spans, "attributes": attrs})


@pytest.fixture()
def llm_run(tmp_path, capfd):
    cfg = load_config(REPO_ROOT / "configs" / "user_level_smoke.yaml")
    cfg.update(run_name="ltest", data_root=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    cfg["analysis"]["bootstrap_reps"] = 50
    cfg["llm"].update(enabled=True, sample_turns=None)
    cfg["user_level"]["min_conversations"] = 2
    key = load_or_create_key(Path(cfg["data_root"]))
    sources = {"wildchat": InMemorySource("wildchat", _user_convs("wildchat"), key)}
    pipe = Pipeline(cfg, sources=sources, llm_backend=llm.CallableBackend(_fake_llm))
    manifest = pipe.run(do_dedup=False)
    out, err = capfd.readouterr()
    return pipe, manifest, out + err


def test_llm_stage_outputs(llm_run):
    pipe, manifest, _ = llm_run
    t = pipe.paths.tables
    stats = pd.read_csv(t / "llm_run_stats.csv").iloc[0]
    assert stats["backend"] == "fake" and stats["failures"] > 0 and stats["turns_processed"] > stats["failures"]
    assert manifest["llm"]["hosted"] is False and manifest["llm"]["failures"] == stats["failures"]

    prev = pd.read_csv(t / "llm_attribute_prevalence.csv")
    assert set(prev["attribute_type"]) == {*llm.ATTRIBUTE_TYPES, "any"}
    assert set(prev["evidence"]) == {*llm.EVIDENCE, "any"} and set(prev["about"]) == {*llm.ABOUT, "any"}
    row = prev[(prev.attribute_type == "occupation") & (prev.evidence == "any") & (prev.about == "user")].iloc[0]
    assert row["n_users"] >= 10 and 0 < row["share_users"] <= 1
    assert prev.loc[prev["suppressed"], "n_users"].isna().any()

    # Quoted identifiers are stored as located spans, and compared with the other detectors.
    located = pd.read_parquet(pipe.paths.restricted / "llm_spans_wildchat.parquet")
    assert len(located) and set(located["detector"]) == {"llm"} and "text" not in located
    assert stats["spans_quoted"] > 0 and stats["quotes_not_found"] == 0 and manifest["llm"]["extract_spans"]
    span_prev = pd.read_csv(t / "llm_span_prevalence.csv")
    assert set(span_prev["category"]) == {*llm.SPAN_TYPES, "any"} and "private" in set(span_prev["about"])
    agree = pd.read_csv(t / "llm_detector_agreement.csv").set_index(["category", "detector"])
    assert agree.loc[("name", "llm"), "n_spans"] > 0 and agree.loc[("name", "presidio"), "n_matched"] > 0

    # The model's task label replaces the keyword one where it exists.
    agree = pd.read_csv(t / "task_type_agreement.csv")
    assert {"keyword_task_type", "task_type", "n_conversations"} <= set(agree.columns)
    assert manifest["task_types"]["from_llm"] > 0 and 0 <= manifest["task_types"]["keyword_llm_agreement"] <= 1
    by_task = pd.read_csv(t / "prevalence_by_task.csv")
    assert "job_application" in set(by_task["task_type"])

    # Raw values exist, and only under restricted/.
    attrs = pd.read_parquet(pipe.paths.restricted / "llm_attrs_wildchat.parquet")
    assert attrs["value"].str.contains("Brisvale").any()
    for root in (pipe.paths.results, pipe.paths.work):
        assert not [f.name for f in root.rglob("*") if "llm_attrs" in f.name or "llm_cache" in f.name
                    or "llm_spans" in f.name]


def test_llm_values_never_leave_restricted(llm_run):
    pipe, _, console = llm_run
    _assert_no_canaries(pipe, console)
    # Extracted values are personal data: none may appear in a published table.
    for f in pipe.paths.tables.glob("*.csv"):
        text = f.read_text().casefold()
        for value in ("colorado", "us_west", "30-39", "nurse", "software developer"):
            assert value not in text, f"profile value leaked into {f.name}"
    assert "Denver" not in console and "nurse" not in console
