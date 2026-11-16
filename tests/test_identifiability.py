"""Stage 2: user-level sampling, profiles, uniqueness, linkage and extractor validation.

Every value below is fabricated. The end-to-end test also checks that no profile value or canary
reaches the aggregate results.
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from residpii import identifiability, linkage, llm_detect, uniqueness
from residpii.config import REPO_ROOT, Paths, load_config
from residpii.pipeline import Pipeline
from residpii.profiles import (AMBIGUOUS, ATTR_COLUMNS, DIMENSIONS, UNKNOWN, attribute_agreement,
                               attributes_from_records, build_profiles, load_coarsener, profile_summary, resolve)
from residpii.safety import assert_safe_table, load_or_create_key
from residpii.sampling import user_level_sample
from residpii.sources import InMemorySource


def _cfg() -> dict:
    return load_config(REPO_ROOT / "configs" / "user_level.yaml")


def _attrs(rows) -> pd.DataFrame:
    """rows: (conversation_id, attribute, value, kind)."""
    return pd.DataFrame([("d", c, 0, a, k, v) for c, a, v, k in rows], columns=ATTR_COLUMNS)


def _convs(mapping: dict, n_conv: dict | None = None) -> pd.DataFrame:
    return pd.DataFrame({"conversation_id": list(mapping), "user": list(mapping.values()),
                         "user_n_conversations": [(n_conv or {}).get(u, list(mapping.values()).count(u))
                                                  for u in mapping.values()]})


# ---------------------------------------------------------------- coarsening + profiles


def test_coarsening_maps_label_schema_values():
    c1, c2 = load_coarsener(_cfg()), load_coarsener(_cfg())
    cases = [("location_region", "NYC"), ("location_region", "Smallville"), ("location_region", "CA"),
             ("location_country", "U.S."), ("age_band", "27"), ("age_band", "mid 30s"), ("gender", "Woman"),
             ("education", "PhD student"), ("occupation", "software engineer"), ("occupation", "wizard")]
    for a, v in cases:
        assert c1.candidates(a, v) == c2.candidates(a, v)                       # deterministic
    assert dict(c1.candidates("location_region", "NYC")) == {"country": "united states", "region": "new york"}
    assert c1.candidates("location_region", "Smallville") == [("region", UNKNOWN)]   # unmatched -> unknown
    assert c1.candidates("location_region", "CA") == c1.candidates("location_region", "California")
    assert dict(c1.candidates("location_region", "Leeds, UK"))["country"] == "united kingdom"
    assert c1.candidates("location_country", "U.S.") == c1.candidates("location_country", "United States")
    assert c1.candidates("age_band", "27") == c1.candidates("age_band", "20-29") == [("age_band", "20-29")]
    # Occupations at mixed detail land in the same SOC major group.
    assert (c1.candidates("occupation", "nurse") == c1.candidates("occupation", "healthcare")
            == [("occupation_group", "29")])
    assert c1.candidates("occupation", "wizard") == [("occupation_group", UNKNOWN)]
    assert c1.candidates("education", "university student") == [("education_level", "some_college")]
    assert c1.candidates("education", "PhD") == [("education_level", "graduate")]


def test_free_text_attributes_are_categorised():
    c = load_coarsener(_cfg())
    # Different wordings of the same fact land in one category.
    assert c.candidates("health", "Type 2 diabetes") == c.candidates("health", "asthma") == [("health", "chronic_condition")]
    assert c.candidates("health", "ADHD") == [("health", "neurodevelopmental")]
    assert c.candidates("health", "a rare thing") == [("health", "other_condition")]       # fallback
    assert c.candidates("employer", " Brisvale Council. ") == [("employer", "government_military")]
    assert c.candidates("employer", "Acme Widgets") == [("employer", "other_employer")]
    assert c.candidates("religion", "Catholic") == c.candidates("religion", "Baptist") == [("religion", "christian")]
    assert c.candidates("nationality_ethnicity", "German") == [("nationality_ethnicity", "white_european")]
    assert c.candidates("income", "low income") == [("income", "low")]
    assert c.candidates("income", "about 40k") == [("income", UNKNOWN)]                   # no fallback
    assert c.candidates("relationship", "married, two kids") == [("relationship", "married")]
    assert c.candidates("health", " ") == [] and c.candidates("not_an_attribute", "x") == []
    assert set(DIMENSIONS) == {"region", "country", "occupation_group", "age_band", "gender", "education_level",
                               "employer", "relationship", "nationality_ethnicity", "religion", "political",
                               "income", "health"}
    assert set(DIMENSIONS) - {"region", "country", "occupation_group", "age_band", "gender",
                              "education_level"} <= set(_cfg()["labeling"]["attributes"])


def test_resolve_majority_and_ties():
    assert resolve(pd.Series({"texas": 3, "ohio": 1})) == ("texas", True)
    assert resolve(pd.Series({"ohio": 1, "texas": 1})) == (AMBIGUOUS, True)
    assert resolve(pd.Series({"texas": 2})) == ("texas", False)
    assert resolve(pd.Series({UNKNOWN: 5, "texas": 1})) == ("texas", False)      # unknown never outvotes
    assert resolve(pd.Series({UNKNOWN: 2})) == (UNKNOWN, False)


def test_profiles_conflicts_and_variants():
    attrs = _attrs([("c1", "location_region", "Texas", "stated"), ("c2", "location_region", "TX", "stated"),
                    ("c3", "location_region", "Ohio", "stated"), ("c1", "age_band", "27", "inferred"),
                    ("c4", "gender", "woman", "stated"), ("c5", "gender", "man", "stated"),
                    ("c9", "occupation", "nurse", "stated")])                    # not an included conversation
    convs = _convs({"c1": "u1", "c2": "u1", "c3": "u1", "c4": "u2", "c5": "u2", "c6": "u3"})
    prof = build_profiles("d", attrs, convs, load_coarsener(_cfg()))
    assert len(prof) == 3 * 2
    p = prof.set_index(["user", "variant"])
    assert p.loc[("u1", "stated_only"), "region"] == "texas"                    # majority across conversations
    assert p.loc[("u1", "stated_only"), "conflict_region"]
    assert p.loc[("u1", "stated_only"), "country"] == "united states"
    assert not p.loc[("u1", "stated_only"), "conflict_country"]
    assert pd.isna(p.loc[("u1", "stated_only"), "age_band"])                    # inferred: not in stated_only
    assert p.loc[("u1", "stated_plus_inferred"), "age_band"] == "20-29"
    assert p.loc[("u2", "stated_only"), "gender"] == AMBIGUOUS                  # a tie
    assert p.loc[("u3", "stated_plus_inferred"), list(DIMENSIONS)].isna().all()  # counted, revealed nothing
    assert p.loc[("u1", "stated_only"), "activity_bucket"] == "3"

    summ = profile_summary(prof, min_cell=1).set_index(["variant", "attribute"])
    assert summ.loc[("stated_only", "region"), "n_conflict"] == 1
    assert summ.loc[("stated_only", "gender"), "n_ambiguous"] == 1
    assert summ.loc[("stated_only", "gender"), "n_with_value"] == 0
    assert summ.loc[("stated_only", "any"), "share_conflict"] == pytest.approx(2 / 3)
    assert profile_summary(prof, min_cell=10)["suppressed"].all()


def test_attributes_from_records_skips_skipped_turns():
    recs = [{"conversation_id": "c1", "turn_index": 2, "skip": False,
             "attributes": [{"attribute": "occupation", "kind": "stated", "value": "nurse"}]},
            {"conversation_id": "c2", "turn_index": 0, "skip": True,
             "attributes": [{"attribute": "gender", "kind": "stated", "value": "man"}]}]
    df = attributes_from_records("d", recs)
    assert df.to_dict("records") == [{"dataset": "d", "conversation_id": "c1", "turn_index": 2,
                                      "attribute": "occupation", "kind": "stated", "value": "nurse"}]


# ---------------------------------------------------------------- uniqueness


def _profiles_with_known_k() -> pd.DataFrame:
    """20 users: 12 Texas + 8 Ohio; genders split so (region, gender) classes are 11, 1, 4, 4."""
    region = ["texas"] * 12 + ["ohio"] * 8
    gender = ["male"] * 11 + ["female"] + ["male"] * 4 + ["female"] * 4
    df = pd.DataFrame({"dataset": "d", "variant": "stated_only", "user": [f"u{i}" for i in range(20)],
                       "activity_bucket": ["3"] * 10 + ["4-5"] * 10, "region": region, "gender": gender})
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
    assert c.loc["region", "median_k"] == 12
    rg = c.loc["region+gender"]
    assert rg["n_users"] == 19 and rg["share_k1"] == pytest.approx(1 / 19)
    assert rg["share_k_lt5"] == pytest.approx(8 / 19) and rg["median_k"] == 11
    assert rg["share_k1_ci_low"] <= rg["share_k1"] <= rg["share_k1_ci_high"]
    assert c.loc["age_band", "suppressed"]                   # `unknown` is not a value
    assert len(by_combo) == sum(comb(len(DIMENSIONS), r) for r in range(1, 5))

    s = summary[summary["ci_kind"] == "bootstrap_users"].set_index("n_qi")
    assert s.loc[1, "share_unique"] == 0 and s.loc[2, "share_unique"] == pytest.approx(1 / 20)
    assert s.loc[4, "share_unique"] == pytest.approx(1 / 20)      # cumulative in n_qi
    a = by_act.set_index(["activity_bucket", "n_qi"])
    assert a.loc[("4-5", 2), "share_unique"] == pytest.approx(1 / 10) and a.loc[("3", 2), "share_unique"] == 0
    assert set(summary.loc[summary["ci_kind"] == "subsample_draws", "sample_users"]) == {10}
    for table in (by_combo, by_act, summary):
        assert_safe_table(table, "uniqueness")
        text = table.astype(str).to_string().casefold()
        assert not any(v in text for v in ("texas", "ohio", "female"))


def test_uniqueness_suppression():
    df = _profiles_with_known_k()
    df.loc[0, "activity_bucket"] = "6-10"                    # buckets now hold 9, 10 and 1 users
    by_combo, by_act, _ = uniqueness.analyze(df, {}, {"bootstrap_reps": 50, "ci_level": 0.95}, seed=1, min_cell=10)
    c = by_combo.set_index("combination")
    assert not c.loc["region", "suppressed"]
    assert c.loc["age_band", "suppressed"] and pd.isna(c.loc["age_band", "share_k1"])
    sup = by_act.groupby("activity_bucket")["suppressed"].all()
    assert sup["3"] and sup["6-10"] and not sup["4-5"]
    assert by_act.loc[by_act["suppressed"], ["n_users", "share_unique"]].isna().all().all()


def test_population_estimate_on_known_profiles():
    """Common values stop being unique in a large population; a rare pair stays unique."""
    n = 200
    df = pd.DataFrame({"dataset": "d", "variant": "stated_only", "user": [f"u{i}" for i in range(n)],
                       "activity_bucket": "3", "gender": ["male"] * 100 + ["female"] * 100,
                       "age_band": ["20-29"] * 150 + ["30-39"] * 50})
    for d in DIMENSIONS:
        if d not in df:
            df[d] = None
    df.loc[:9, "education_level"] = [f"e{i}" for i in range(10)]
    df.loc[:9, "religion"] = [f"r{i}" for i in range(10)]
    acfg, ucfg = {"bootstrap_reps": 50, "ci_level": 0.95}, {"sample_sizes": [100], "population_bootstrap_reps": 50}
    _, _, summary = uniqueness.analyze(df, ucfg, acfg, seed=1, min_cell=10)
    pop = uniqueness.population_estimate(df, summary, {"d": 20_000}, ucfg, acfg, seed=1)
    assert list(pop.columns) == uniqueness.POPULATION_COLUMNS
    est = pop[pop["subset"] == "population"].set_index("n_qi")
    assert est.loc[1, "share_unique_model"] < 1e-6
    expected = 10 / n * (1 - (0.5 / n) ** 2) ** (20_000 - 1)
    assert est.loc[2, "share_unique_model"] == pytest.approx(expected, rel=1e-6)
    assert (est["share_unique_model"].diff().dropna() >= -1e-12).all()
    check = pop[(pop["subset"] == "model_at_sample_size") & (pop["target_users"] == n)].set_index("n_qi")
    assert check.loc[2, "share_unique_observed"] == pytest.approx(10 / n)
    assert_safe_table(pop, "uniqueness_population")


def test_population_estimate_treats_region_and_country_as_one_factor():
    n = 100
    df = pd.DataFrame({"dataset": "d", "variant": "stated_only", "user": [f"u{i}" for i in range(n)],
                       "activity_bucket": "3", "region": ["texas"] * 2 + [None] * 98,
                       "country": ["united states"] * 2 + [None] * 98})
    for d in DIMENSIONS:
        if d not in df:
            df[d] = None
    acfg, ucfg = {"bootstrap_reps": 20, "ci_level": 0.95}, {"population_bootstrap_reps": 20}
    _, _, summary = uniqueness.analyze(df, ucfg, acfg, seed=1, min_cell=10)
    est = uniqueness.population_estimate(df, summary, {"d": 300}, ucfg, acfg, seed=1)
    est = est[est["subset"] == "population"].set_index("n_qi")["share_unique_model"]
    assert est[2] == pytest.approx(est[1])


# ---------------------------------------------------------------- linkage


def _linkage_inputs(n_users: int = 12):
    """Four conversations each. Users 0-5 state their own region every time, so any split links them.
    Users 6-9 all state the same gender and cannot be told apart. Users 10-11 state nothing but each
    repeats an email address of their own. Every conversation also has a shared Presidio name."""
    regions = ["Texas", "Ohio", "California", "Florida", "New York", "Illinois"]
    attrs, spans, convs = [], [], {}
    for u in range(n_users):
        for c in range(4):
            cid = f"c{u}_{c}"
            convs[cid] = f"u{u}"
            if u < 6:
                attrs.append((cid, "location_region", regions[u], "stated"))
            elif u < 10:
                attrs.append((cid, "gender", "woman", "stated"))
            else:
                spans.append((cid, "presidio", "PERSON", "email", f"hash{u}"))
            spans.append((cid, "presidio", "PERSON", "name", "shared"))      # not a direct key: ignored
            spans.append((cid, "llm_detector", "third_party", "name", "boss"))  # not the writer's: ignored
    spans = pd.DataFrame(spans, columns=["conversation_id", "detector", "entity", "category", "span_hmac"]).assign(
        dataset="d", in_placeholder=False, is_dummy=False)
    return _attrs(attrs), _convs(convs), spans


def test_linkage_links_only_distinguishable_users():
    attrs, convs, spans = _linkage_inputs()
    ucfg, acfg = {"linkage_splits": 3}, {"bootstrap_reps": 50, "ci_level": 0.95}
    out = linkage.analyze("d", attrs, convs, spans, load_coarsener(_cfg()), ucfg, acfg, seed=1, min_cell=1)
    assert_safe_table(out, "linkage")
    r = out[out["activity_bucket"] == "all"].set_index(["variant", "match_on"])
    assert r.loc[("stated_only", "profile"), "share_linked"] == pytest.approx(6 / 12)
    assert r.loc[("stated_only", "profile"), "share_both_halves"] == pytest.approx(10 / 12)
    assert r.loc[("stated_only", "profile"), "share_linked_given_both"] == pytest.approx(6 / 10)
    assert r.loc[("any", "direct_identifiers"), "share_linked"] == pytest.approx(2 / 12)
    assert r.loc[("stated_only", "profile_and_direct"), "share_linked"] == pytest.approx(8 / 12)
    assert np.allclose(r["chance"], 1 / 12)
    assert (r["ci_low"] <= r["share_linked"]).all() and (r["share_linked"] <= r["ci_high"]).all()
    small = linkage.analyze("d", attrs, convs, spans, load_coarsener(_cfg()), ucfg, acfg, seed=1, min_cell=13)
    assert small["suppressed"].all() and small["share_linked"].isna().all()


def test_llm_self_names_are_direct_keys():
    spans = pd.DataFrame({"conversation_id": ["a", "b"], "detector": "llm_detector", "entity": ["self", "fictional"],
                          "category": "name", "span_hmac": ["h1", "h2"], "in_placeholder": False, "is_dummy": False})
    keys = linkage.direct_keys(spans, pd.Series({"a": "u1", "b": "u2"}))
    assert keys["conversation_id"].tolist() == ["a"]


def test_direct_overlap_counts_users_not_values():
    attrs, convs, spans = _linkage_inputs()
    prof = build_profiles("d", attrs, convs, load_coarsener(_cfg()))
    out = linkage.direct_overlap(prof, convs, spans, {}, min_cell=1)
    assert_safe_table(out, "direct_overlap")
    r = out[out["variant"] == "stated_only"].set_index("category")
    assert r.loc["any", "n_direct"] == 2 and r.loc["email", "n_direct"] == 2 and r.loc["phone", "n_direct"] == 0
    assert r.loc["any", "n_direct_and_unique"] == 0
    assert linkage.direct_overlap(prof, convs, spans, {}, min_cell=10)["suppressed"].any()


# ---------------------------------------------------------------- extractor validation


def test_attribute_agreement_after_coarsening():
    def rec(*attrs, skip=False):
        return {"skip": skip, "attributes": [{"attribute": a, "kind": k, "value": v} for a, v, k in attrs]}
    gold = {"i1": rec(("occupation", "nurse", "stated")), "i2": rec(("location_country", "UK", "inferred")),
            "i3": rec(), "i4": rec(("gender", "man", "stated"), skip=True)}
    pred = {"i1": rec(("occupation", "healthcare", "stated")),          # same SOC group after coarsening
            "i2": rec(("location_country", "France", "inferred")),
            "i3": rec(("age_band", "34", "stated"))}
    t = attribute_agreement("d", ["i1", "i2", "i3", "i4"], gold, pred, load_coarsener(_cfg()), "gold", "llm")
    assert_safe_table(t, "identifiability_attribute_validation")
    r = t.set_index(["variant", "attribute"])
    assert r.loc[("stated_only", "occupation_group"), ["n_gold", "n_pred", "n_match"]].tolist() == [1, 1, 1]
    assert r.loc[("stated_plus_inferred", "country"), ["n_gold", "n_pred", "n_both", "n_match"]].tolist() == [1, 1, 1, 0]
    assert pd.isna(r.loc[("stated_only", "country"), "precision"])          # inferred only
    assert (r["n_items"] == 3).all()                                        # the skipped gold item is out
    a = r.loc[("stated_plus_inferred", "any")]
    assert (a["n_gold"], a["n_pred"], a["n_match"]) == (2, 3, 1)
    assert a["precision"] == pytest.approx(1 / 3) and a["recall"] == pytest.approx(1 / 2)


# ---------------------------------------------------------------- sampling + end to end


def _meta(counts: dict) -> pd.DataFrame:
    rows = [(f"{u}-{i}", u) for u, k in counts.items() for i in range(k)] + [("nouser-0", None)]
    df = pd.DataFrame(rows, columns=["conversation_id", "user_key"])
    return df.assign(dataset="d", n_turns=1, language="English", timestamp=pd.Timestamp("2023-05-01"),
                     _file=0, _rg=0, _row=range(len(df)))


def test_user_level_sample_keeps_users_whole_and_weights():
    cfg = _cfg()
    cfg["user_level"].update(n_users=3, min_conversations=3, max_conversations_per_user=4)
    meta = _meta({"a": 1, "b": 3, "c": 5, "d": 10, "e": 2})
    smp, summ = user_level_sample(meta, cfg, seed=7)
    assert set(smp["user_key"]) == {"b", "c", "d"}                         # exactly the eligible users
    assert smp.groupby("user_key").size().to_dict() == {"b": 3, "c": 4, "d": 4}   # capped at 4
    assert summ["eligible_users"] == 3 and summ["sampled_users_over_cap"] == 2
    w = smp.groupby("user_key")["weight"].first()
    assert w["b"] == pytest.approx(1.0) and w["d"] == pytest.approx(10 / 4)
    assert smp["weight"].sum() == pytest.approx(3 + 5 + 10)                # totals over eligible users' convs
    assert smp.equals(user_level_sample(meta, cfg, seed=7)[0])             # seeded


CANARIES = ["Quillon Varnadore", "quillon.varnadore@zephyrmailtest.org", "Brisvale Unity Hospital", "texas"]


def _conversations(n_users: int = 30, per_user: int = 4) -> list[dict]:
    texts = ["Hi, I'm Quillon Varnadore, a nurse at Brisvale Unity Hospital. Email quillon.varnadore@zephyrmailtest.org",
             "Help me plan a weekend in Texas, I live in Austin.", "Write a poem about the sea."]
    return [{"conversation_id": f"w-{u}-{c}", "user": f"user{u}", "language": "English",
             "timestamp": "2023-05-01", "turns": [{"role": "user", "content": texts[(u + c) % 3]},
                                                  {"role": "assistant", "content": "ok"}]}
            for u in range(n_users) for c in range(per_user)]


def _fake_llm_records(paths: Paths, cfg: dict, ds: str) -> None:
    """Writes LLM detector records as if `llm-detect` had run: occupation for every user, a region for
    some, and a self-name span located by offset."""
    turns = llm_detect._turns(paths, cfg, ds)
    with open(llm_detect.store_path(paths, ds), "w") as f:
        for t in turns.itertuples(index=False):
            attrs, spans = [], []
            if "nurse" in t.text:
                attrs.append({"attribute": "occupation", "kind": "stated", "value": "nurse"})
                start = t.text.index("Quillon")
                spans.append({"start": start, "end": start + len("Quillon Varnadore"), "category": "name",
                              "whose": "self"})
            if "Texas" in t.text:
                attrs.append({"attribute": "location_region", "kind": "stated", "value": "Texas"})
            f.write(json.dumps({"key": t.key, "conversation_id": t.conversation_id, "turn_index": int(t.turn_index),
                                "status": "ok", "spans": spans, "attributes": attrs, "skip": False,
                                "model": "fake", "prompt_version": "fake", "meta": {}}) + "\n")


def test_end_to_end_user_level(tmp_path, capfd):
    cfg = _cfg()
    cfg.update(data_root=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    cfg["user_level"].update(n_users=20, min_conversations=3)
    cfg["uniqueness"].update(sample_sizes=[10], subsample_draws=2, population_bootstrap_reps=10, linkage_splits=2)
    cfg["analysis"]["bootstrap_reps"] = 20
    key = load_or_create_key(Path(cfg["data_root"]))
    pipe = Pipeline(cfg, sources={"wildchat": InMemorySource("wildchat", _conversations(), key)})
    pipe.run(do_dedup=False)
    paths = Paths(cfg)
    smp = pd.read_parquet(paths.work / "sample_wildchat.parquet")
    assert smp["user_key"].nunique() == 20 and (smp.groupby("user_key").size() == 4).all()

    with pytest.raises(RuntimeError, match="no LLM detector records"):
        identifiability.run(cfg, ["wildchat"])
    _fake_llm_records(paths, cfg, "wildchat")
    tables = identifiability.run(cfg, ["wildchat"])
    cov = tables["identifiability_coverage"].iloc[0]
    assert cov["complete_users"] == cov["profiled_users"] == 20
    assert (paths.restricted / "profiles_wildchat.parquet").exists()
    for name in ("profile_summary", "uniqueness_by_combination", "uniqueness_summary", "uniqueness_population",
                 "linkage", "direct_overlap"):
        assert (paths.tables / f"{name}.csv").exists(), name
    # Everyone who said they are a nurse shares one SOC group: profiles are not unique on occupation alone.
    combo = tables["uniqueness_by_combination"].set_index(["variant", "combination"])
    assert combo.loc[("stated_only", "occupation_group"), "share_k1"] == 0

    # No canary or profile value in any published file or console output.
    out, err = capfd.readouterr()
    published = "".join(p.read_text(errors="ignore") for p in Path(cfg["results_dir"]).rglob("*") if p.is_file())
    for c in CANARIES:
        assert c.casefold() not in published.casefold(), c
        assert c not in out + err, c


def test_validate_requires_gold(tmp_path):
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    cfg.update(data_root=str(tmp_path / "data"), results_dir=str(tmp_path / "results"), run_name="v")
    with pytest.raises(FileNotFoundError):
        identifiability.validate(cfg, ["wildchat"])
