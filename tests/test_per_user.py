"""Per-user profiles, uniqueness and linkage, on fabricated values."""
from math import comb

import numpy as np
import pandas as pd
import pytest

from residpii import linkage, uniqueness
from residpii.config import REPO_ROOT, load_config
from residpii.profiles import (AMBIGUOUS, DIMENSIONS, UNKNOWN, build_profiles, load_coarsener, profile_summary,
                               resolve)
from residpii.safety import assert_safe_table


def _cfg() -> dict:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


# ---------------------------------------------------------------- coarsening + profiles


def test_coarsening_is_deterministic_and_closed():
    c1, c2 = load_coarsener(_cfg()), load_coarsener(_cfg())
    cases = [("city", "NYC"), ("city", "Smallville"), ("state_or_region", "CA"), ("country", "U.S."),
             ("age", "27"), ("age_range", "mid 30s"), ("gender", "Woman"), ("school", "Ohio State University"),
             ("education_level", "PhD student"), ("occupation", "software engineer"), ("occupation", "wizard")]
    for typ, val in cases:
        assert c1.candidates(typ, val) == c2.candidates(typ, val) == c1.candidates(typ, val)
    assert dict(c1.candidates("city", "NYC")) == {"country": "united states", "region": "new york"}
    assert c1.candidates("city", "Smallville") == [("region", UNKNOWN)]           # unmatched -> unknown
    assert c1.candidates("state_or_region", "CA") == c1.candidates("state_or_region", "California")
    assert c1.candidates("country", "U.S.") == c1.candidates("country", "United States")
    assert c1.candidates("age", "27") == [("age_band", "20-29")]
    assert c1.candidates("occupation", "software engineer") == [("occupation_group", "15-1")]
    assert c1.candidates("occupation", "anything", "29", "29-1") == [("occupation_group", "29-1")]
    assert c1.candidates("occupation", "anything", "29") == [("occupation_group", "29")]
    assert c1.candidates("occupation", "wizard") == [("occupation_group", UNKNOWN)]
    assert c1.candidates("country", "Germany") == [("country", "germany")]


def test_attributes_without_a_table_are_kept_as_stated():
    c = load_coarsener(_cfg())
    assert c.candidates("nationality", "German") == [("nationality", "german")]
    assert c.candidates("language_native", " Tagalog. ") == [("language_native", "tagalog")]
    assert c.candidates("employer_type", "Public hospital") == [("employer_type", "public hospital")]
    assert c.candidates("relationship_status", "Married") == [("relationship_status", "married")]
    # A school keeps its name and also gives the level of study it implies.
    assert dict(c.candidates("school", "Ohio State University")) == {"school": "ohio state university",
                                                                     "education_level": "some_college"}
    assert c.candidates("other", "owns a boat") == [] and c.candidates("nationality", " ") == []
    assert set(DIMENSIONS) == {"region", "country", "occupation_group", "age_band", "gender", "education_level",
                               "employer_type", "school", "relationship_status", "nationality", "language_native"}


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
    assert len(prof) == 3 * 2 and set(prof["level"]) == {"generalized"}   # users x variants, one level
    p = prof.set_index(["user", "variant", "level"])
    # Majority vote across conversations, with the conflict recorded.
    assert p.loc[("u1", "stated_only", "generalized"), "region"] == "texas"
    assert p.loc[("u1", "stated_only", "generalized"), "conflict_region"]
    assert p.loc[("u1", "stated_only", "generalized"), "country"] == "united states"
    assert not p.loc[("u1", "stated_only", "generalized"), "conflict_country"]
    # Inferred attributes are left out of stated_only and present in stated_plus_inferred.
    assert pd.isna(p.loc[("u1", "stated_only", "generalized"), "age_band"])
    assert p.loc[("u1", "stated_plus_inferred", "generalized"), "age_band"] == "20-29"
    # A tie is ambiguous.
    assert p.loc[("u2", "stated_only", "generalized"), "gender"] == AMBIGUOUS
    # Low-confidence and third-party attributes never reach a profile, but the user is still counted.
    assert p.loc[("u3", "stated_plus_inferred", "generalized"), list(DIMENSIONS)].isna().all()
    assert p.loc[("u1", "stated_only", "generalized"), "activity_bucket"] == "3"

    summ = profile_summary(prof, min_cell=1).set_index(["variant", "level", "attribute"])
    assert summ.loc[("stated_only", "generalized", "region"), "n_conflict"] == 1
    assert summ.loc[("stated_only", "generalized", "gender"), "n_ambiguous"] == 1
    assert summ.loc[("stated_only", "generalized", "gender"), "n_with_value"] == 0
    assert summ.loc[("stated_only", "generalized", "any"), "share_conflict"] == pytest.approx(2 / 3)
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
    assert len(by_combo) == sum(comb(len(DIMENSIONS), r) for r in range(1, 5))

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


def test_population_estimate_on_known_profiles():
    """Common values stop being unique in a large population; a rare pair stays unique."""
    n = 200
    df = pd.DataFrame({"dataset": "d", "variant": "stated_only", "level": "fine",
                       "user": [f"u{i}" for i in range(n)], "activity_bucket": "3",
                       "gender": ["male"] * 100 + ["female"] * 100, "age_band": ["20-29"] * 150 + ["30-39"] * 50})
    for d in DIMENSIONS:
        if d not in df:
            df[d] = None
    # Ten users each hold a one-off education level and a one-off occupation group.
    df.loc[:9, "education_level"] = [f"e{i}" for i in range(10)]
    df.loc[:9, "occupation_group"] = [f"o{i}" for i in range(10)]
    acfg = {"bootstrap_reps": 50, "ci_level": 0.95}
    ucfg = {"sample_sizes": [100], "population_bootstrap_reps": 50}
    _, _, summary = uniqueness.analyze(df, ucfg, acfg, seed=1, min_cell=10)
    pop = uniqueness.population_estimate(df, summary, {"d": 20_000}, ucfg, acfg, seed=1)
    assert list(pop.columns) == uniqueness.POPULATION_COLUMNS
    est = pop[pop["subset"] == "population"].set_index("n_qi")
    assert (est["target_users"] == 20_000).all() and (est["sample_users"] == n).all()
    # One dimension: a value seen once in 200 is expected ~50 times among 20,000 users.
    assert est.loc[1, "share_unique_model"] < 1e-6
    # Two one-off values together: match probability (0.5/200)^2, so the ten users stay mostly unique.
    expected = 10 / n * (1 - (0.5 / n) ** 2) ** (20_000 - 1)
    assert est.loc[2, "share_unique_model"] == pytest.approx(expected, rel=1e-6)
    assert est.loc[2, "ci_low"] <= est.loc[2, "share_unique_model"] <= est.loc[2, "ci_high"]
    assert (est["share_unique_model"].diff().dropna() >= -1e-12).all()        # more dimensions never hide a user
    # The model check sits beside the observed share at the same sample size, and never exceeds 1.
    check = pop[(pop["subset"] == "model_at_sample_size") & (pop["target_users"] == n)].set_index("n_qi")
    assert check.loc[2, "share_unique_observed"] == pytest.approx(10 / n)
    assert check["share_unique_model"].between(0, 1).all()
    assert (check["share_unique_model"] >= est["share_unique_model"]).all()   # uniqueness falls with population size
    assert_safe_table(pop, "uniqueness_population")
    assert not pop.astype(str).apply(lambda c: c.str.contains("female|20-29|e3|o3")).any().any()


def test_population_estimate_treats_region_and_country_as_one_factor():
    n = 100
    df = pd.DataFrame({"dataset": "d", "variant": "stated_only", "level": "coarse",
                       "user": [f"u{i}" for i in range(n)], "activity_bucket": "3",
                       "region": ["west"] * 2 + [None] * 98, "country": ["us"] * 2 + [None] * 98})
    for d in DIMENSIONS:
        if d not in df:
            df[d] = None
    acfg, ucfg = {"bootstrap_reps": 20, "ci_level": 0.95}, {"population_bootstrap_reps": 20}
    _, _, summary = uniqueness.analyze(df, ucfg, acfg, seed=1, min_cell=10)
    est = uniqueness.population_estimate(df, summary, {"d": 300}, ucfg, acfg, seed=1)
    est = est[est["subset"] == "population"].set_index("n_qi")["share_unique_model"]
    # A region that implies its country adds nothing: the pair is no more identifying than either alone.
    assert est[2] == pytest.approx(est[1])


# ---------------------------------------------------------------- linkage


def _linkage_inputs(n_users: int = 12):
    """Each user has four conversations. Users 0-5 state their own distinct region in every one, so
    any split links them. Users 6-9 all state the same gender, so they cannot be told apart. Users
    10-11 state nothing but each repeats an email address of their own."""
    regions = ["Texas", "Ohio", "California", "Florida", "New York", "Illinois"]
    attrs, spans, convs = [], [], {}
    for u in range(n_users):
        for c in range(4):
            cid = f"c{u}_{c}"
            convs[cid] = f"u{u}"
            if u < 6:
                attrs.append((cid, "state_or_region", regions[u]))
            elif u < 10:
                attrs.append((cid, "gender", "woman"))
            else:
                spans.append((cid, "presidio", "email", f"hash{u}"))
            spans.append((cid, "presidio", "name", "shared"))            # not a direct key: ignored
    attrs = pd.DataFrame(attrs, columns=["conversation_id", "type", "value"]).assign(
        dataset="d", turn_index=0, evidence="stated", about="user", confidence=0.9, occupation_group="",
        occupation_minor_group="")
    spans = pd.DataFrame(spans, columns=["conversation_id", "detector", "category", "span_hmac"]).assign(
        dataset="d", role="user", in_placeholder=False, is_dummy=False)
    sample = pd.DataFrame({"dataset": "d", "conversation_id": list(convs), "user_key": list(convs.values()),
                           "user_n_conversations": 4})
    tlog = pd.DataFrame({"dataset": "d", "conversation_id": list(convs), "turn_index": 0, "status": "ok"})
    return attrs, tlog, sample, spans


def test_linkage_links_only_distinguishable_users():
    attrs, tlog, sample, spans = _linkage_inputs()
    ucfg, acfg = {"linkage_splits": 3}, {"bootstrap_reps": 50, "ci_level": 0.95}
    out = linkage.analyze(attrs, tlog, sample, spans, load_coarsener(_cfg()), 0.6, ucfg, acfg, seed=1, min_cell=1)
    assert_safe_table(out, "linkage")
    r = out[out["activity_bucket"] == "all"].set_index(["variant", "match_on"])
    # Six of twelve users have a region of their own; the four who share a gender tie and do not link.
    assert r.loc[("stated_only", "profile"), "share_linked"] == pytest.approx(6 / 12)
    assert r.loc[("stated_only", "profile"), "share_both_halves"] == pytest.approx(10 / 12)
    assert r.loc[("stated_only", "profile"), "share_linked_given_both"] == pytest.approx(6 / 10)
    # The two users with a repeated email link on direct identifiers alone; a shared name span is not a key.
    assert r.loc[("any", "direct_identifiers"), "share_linked"] == pytest.approx(2 / 12)
    assert r.loc[("stated_only", "profile_and_direct"), "share_linked"] == pytest.approx(8 / 12)
    assert np.allclose(r["chance"], 1 / 12)
    assert (r["ci_low"] <= r["share_linked"]).all() and (r["share_linked"] <= r["ci_high"]).all()
    # With the real threshold a bucket of 12 users is published and nothing smaller would be.
    small = linkage.analyze(attrs, tlog, sample, spans, load_coarsener(_cfg()), 0.6, ucfg, acfg, seed=1, min_cell=13)
    assert small["suppressed"].all() and small["share_linked"].isna().all()


def test_direct_overlap_counts_users_not_values():
    attrs, tlog, sample, spans = _linkage_inputs()
    prof = build_profiles(attrs, tlog, sample, load_coarsener(_cfg()), 0.6)
    out = linkage.direct_overlap(prof, sample, spans, tlog, {}, min_cell=1)
    assert_safe_table(out, "direct_identifier_overlap")
    r = out[out["variant"] == "stated_only"].set_index("category")
    assert r.loc["any", "n_direct"] == 2 and r.loc["email", "n_direct"] == 2 and r.loc["phone", "n_direct"] == 0
    assert r.loc["any", "n_direct_and_unique"] == 0          # the email users revealed no profile attribute
    assert linkage.direct_overlap(prof, sample, spans, tlog, {}, min_cell=10)["suppressed"].any()
