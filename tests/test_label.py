"""Hand-labeling export and scoring. All text is fabricated."""
import numpy as np
import pandas as pd
import pytest

from residpii import label, llm
from residpii.analysis import residual
from residpii.config import REPO_ROOT, load_config
from residpii.pipeline import Pipeline
from residpii.safety import load_or_create_key
from residpii.sources import InMemorySource

from test_pipeline import CANARIES, _assert_no_canaries, _convs


def test_spans_match_needs_half_of_the_shorter_span():
    assert label.spans_match(0, 10, 0, 10)
    assert label.spans_match(0, 10, 5, 15)            # 5 of 10
    assert not label.spans_match(0, 10, 6, 16)        # 4 of 10
    assert label.spans_match(0, 100, 40, 44)          # short span inside a long one
    assert label.spans_match(0, 4, 2, 100)            # 2 of 4
    assert not label.spans_match(0, 10, 10, 20)       # touching is not overlapping


def _spans(rows, with_detector=True):
    cols = ["dataset", "conversation_id", "turn_index", "start", "end", "category"]
    df = pd.DataFrame([r[:6] for r in rows], columns=cols)
    if with_detector:
        df["detector"] = [r[6] for r in rows]
    return df


def test_score_detectors_on_synthetic_pairs():
    gold = _spans([("d", "c1", 0, 0, 10, "name"), ("d", "c1", 0, 20, 30, "email"),
                   ("d", "c2", 0, 5, 15, "name"), ("d", "c3", 0, 0, 8, "phone")], with_detector=False)
    pred = _spans([("d", "c1", 0, 2, 10, "name", "presidio"),      # matches gold name in c1
                   ("d", "c1", 0, 20, 30, "name", "presidio"),     # right place, wrong category
                   ("d", "c2", 0, 50, 60, "name", "presidio"),     # no overlap
                   ("d", "c9", 0, 5, 15, "name", "presidio"),      # same offsets, different turn
                   ("d", "c1", 0, 21, 29, "email", "secrets"),
                   ("d", "c2", 0, 5, 15, "name", "selfid")])
    out = label.score_detectors(pred, gold).set_index(["detector", "category"])
    p = out.loc[("presidio", "name")]
    assert (p["n_gold"], p["n_pred"], p["tp_pred"], p["tp_gold"]) == (2, 4, 1, 1)
    assert p["precision"] == 0.25 and p["recall"] == 0.5 and p["f1"] == pytest.approx(1 / 3)
    assert out.loc[("presidio", "email"), "recall"] == 0 and np.isnan(out.loc[("presidio", "email"), "precision"])
    assert out.loc[("secrets", "email"), "f1"] == 1
    # The union of detectors finds both names and the email, but not the phone number.
    assert out.loc[("any", "name"), "recall"] == 1 and out.loc[("any", "phone"), "recall"] == 0
    a = out.loc[("any", "all")]
    assert (a["n_gold"], a["tp_gold"], a["n_pred"], a["tp_pred"]) == (4, 3, 6, 3)


def test_cohen_kappa():
    assert label.cohen_kappa(list("aabb"), list("aabb")) == 1
    assert label.cohen_kappa([True, True, False, False], [True, False, True, False]) == 0
    assert label.cohen_kappa(list("aabb"), list("bbaa")) == -1
    assert np.isnan(label.cohen_kappa([], [])) and np.isnan(label.cohen_kappa(list("aa"), list("aa")))


def test_gold_spans_from_offsets_or_text():
    text = "Ada met Ada in Paris"
    labels = pd.DataFrame([
        ("it1", "d", "c1", 0, text, "Ada", "", "", "name", "third_party"),     # both occurrences
        ("it1", "d", "c1", 0, text, "", "15", "20", "location", "user"),       # explicit offsets
        ("it1", "d", "c1", 0, text, "Zed", "", "", "name", "user"),            # not in the text
        ("it1", "d", "c1", 0, text, "Paris", "", "", "favourite", "user"),     # not a category
        ("it2", "d", "c2", 0, "nothing here", "", "", "", "none", ""),
    ], columns=["item_id", "dataset", "conversation_id", "turn_index", "text", "span_text", "start", "end",
                "pii_type", "about"])
    gold, skipped = label.gold_spans(labels, ["name", "location"])
    assert sorted(zip(gold["start"], gold["end"], gold["category"])) == [(0, 3, "name"), (8, 11, "name"),
                                                                          (15, 20, "location")]
    assert skipped == {"unlabeled": 0, "unknown_pii_type": 1, "span_not_found": 1}


@pytest.fixture()
def labeled(tmp_path, capfd):
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    cfg.update(run_name="lab", data_root=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    cfg["analysis"]["bootstrap_reps"] = 20
    key = load_or_create_key(tmp_path / "data")
    plain = [{"conversation_id": f"plain-{i}", "user": f"p{i}", "language": "English", "timestamp": "2023-05-01",
              "turns": [{"role": "user", "content": "how do magnets work, in simple terms?"}]} for i in range(8)]
    pipe = Pipeline(cfg, sources={"wildchat": InMemorySource("wildchat", _convs("wildchat", 30) + plain, key)},
                    llm_backend=llm.CallableBackend(
                        lambda s, u: '{"attributes": [{"type": "city", "value": "Denver", "evidence": "stated", '
                                     '"about": "user", "confidence": 0.9}]}' if "Denver" in u.split("USER TURN:")[1]
                        else '{"attributes": []}'))
    pipe.run(do_dedup=False)
    before = sorted(p for p in pipe.paths.results.rglob("*") if p.is_file())
    path = pipe.label_export(n=12)
    return pipe, path, before, capfd


def test_export_goes_to_restricted_only(labeled):
    pipe, path, before, capfd = labeled
    assert path.parent == pipe.paths.restricted / "labels" and path.name.startswith("batch_")
    assert (path.stat().st_mode & 0o777) == 0o600
    # Nothing new under results/, and nothing with raw text outside restricted/.
    assert sorted(p for p in pipe.paths.results.rglob("*") if p.is_file()) == before
    _assert_no_canaries(pipe, "".join(capfd.readouterr()))
    items = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(items.columns) == label.LABEL_COLUMNS and len(items) == 12
    assert items["item_id"].is_unique and (items[label.ANNOTATION_COLUMNS] == "").all().all()
    # Stratified on detection: turns with and without a detection both appear.
    spans = residual(pd.read_parquet(pipe.paths.restricted / "spans_wildchat.parquet"), 0.5)
    flagged = items["conversation_id"].isin(set(spans["conversation_id"]))
    assert flagged.any() and not flagged.all()
    assert any(c in " ".join(items["text"]) for c in CANARIES)          # it really is raw text
    assert pipe.label_export(n=5) != path                                # a second batch never overwrites


def test_label_score_end_to_end(labeled):
    pipe, path, _, capfd = labeled
    items = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for r in items.to_dict("records"):
        found = False
        for needle, typ, about in (("Quillon Varnadore", "name", "user"), ("(646) 391-2087", "phone", "user"),
                                   ("Denver", "location", "user"), ("Sir Bramwell", "name", "fictional")):
            if needle in r["text"]:
                rows.append({**r, "span_text": needle, "pii_type": typ, "about": about})
                found = True
        if not found:
            rows.append({**r, "pii_type": "none"})
    annotated = path.with_name("annotated.csv")
    pd.DataFrame(rows).to_csv(annotated, index=False)

    out = pipe.label_score(annotated, second_annotator="llm")
    written = pd.read_csv(pipe.paths.tables / "detector_validation.csv")
    assert len(written) == len(out) and (written["n_labeled_turns"] == 12).all()
    v = written.set_index(["detector", "category"])
    if ("presidio", "phone") in v.index and v.loc[("presidio", "phone"), "n_gold"] > 0:
        assert v.loc[("presidio", "phone"), "recall"] == 1
    assert ("any", "all") in v.index and v.loc[("any", "all"), "n_gold"] == sum(r["pii_type"] != "none" for r in rows)
    # LLM-vs-human agreement on `about` is reported as kappa.
    assert {"about_user_turn_level", "about_matched_spans"} <= set(written.loc[written.detector == "llm_vs_human",
                                                                               "category"])
    assert v.loc[("llm_vs_human", "about_user_turn_level"), "n_items"] == 12
    _assert_no_canaries(pipe, "".join(capfd.readouterr()))
