"""End-to-end test on synthetic conversations, including the canary leak check.

All "PII" below is fabricated: reserved 555 numbers, the long-invalidated Woolworth SSN, payment
test cards, made-up names. The canary check asserts that none of it reaches any file outside the
restricted directory, nor stdout/stderr.
"""
from pathlib import Path

import pandas as pd
import pytest

from residpii.config import REPO_ROOT, load_config
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


def test_one_conversation_per_user(run):
    pipe, manifest, _ = run
    smp = pd.read_parquet(pipe.paths.work / "sample_wildchat.parquet")
    assert smp["user_key"].is_unique
    assert manifest["datasets"]["wildchat"]["sampling"]["one_per_user_applied"]


def test_no_canary_leaks_outside_restricted(run):
    pipe, _, console = run
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
