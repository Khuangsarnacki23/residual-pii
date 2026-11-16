"""Stage 2: do the details a user leaves across conversations add up to something identifying?

Two commands:

  residpii identifiability run      --config configs/user_level.yaml
      Needs a user-level run (`residpii run --config configs/user_level.yaml`) and the LLM detector
      over it (`residpii llm-detect --config configs/user_level.yaml --allow-hosted`). Builds one
      profile per user from the detector's attributes, then writes aggregate tables: how many users
      are unique on combinations of 1-4 attributes (in the sample, at smaller sample sizes, and a
      model-based population estimate), whether a user's two halves of conversations link to each
      other, and how often users with a unique profile also left a direct identifier.

  residpii identifiability validate --config configs/default.yaml [--gold adjudicated] [--predictor llm_detect]
      Checks the attribute extractor against the human gold set on the Stage 1 label items, after
      the same coarsening Stage 2 uses. The predictor is the LLM detector's records for those turns
      (`llm_detect`, needs llm-detect on the default run) or any annotator, e.g. `llm` or `ethan`.

Profiles stay in the restricted directory. Tables hold only shares and counts of users over
attribute *names*, with small cells (< identifiability.min_cell users) blanked.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import labeling, linkage, llm_detect, profiles, uniqueness
from .config import Paths
from .report import write_table


def _min_cell(cfg: dict) -> int:
    return int((cfg.get("identifiability") or {}).get("min_cell", 10))


def _sample_summary(paths: Paths, ds: str) -> dict:
    side = paths.work / f"sample_{ds}.key.json"
    return json.loads(side.read_text()).get("extra", {}) if side.exists() else {}


def complete_users(cfg: dict, paths: Paths, ds: str) -> tuple[pd.DataFrame, dict, dict]:
    """(included conversations, LLM records, coverage counts). A user is included only when the LLM
    detector has a record for every non-empty user turn of every sampled conversation of theirs, so a
    budget stop never leaves a user with part of their profile missing."""
    sample = pd.read_parquet(paths.work / f"sample_{ds}.parquet")
    if "user_n_conversations" not in sample:
        raise RuntimeError(f"{ds}: the sample is not user-level; use a config with sampling.mode: user_level "
                           "(configs/user_level.yaml) and run `residpii run` with it first")
    recs = llm_detect.load_records(paths, ds)
    if not recs:
        raise RuntimeError(f"{ds}: no LLM detector records; run `residpii llm-detect` with this config first")
    turns = llm_detect._turns(paths, cfg, ds)
    done_turn = turns["key"].isin(recs.keys())          # failed calls are not recorded (retried next run)
    conv_done = done_turn.groupby(turns["conversation_id"]).all()
    sample["conv_done"] = sample["conversation_id"].map(conv_done).fillna(True).astype(bool)  # no text: nothing to do
    user_done = sample.groupby("user_key")["conv_done"].all()
    keep = sample[sample["user_key"].map(user_done)]
    convs = pd.DataFrame({"conversation_id": keep["conversation_id"].astype(str), "user": keep["user_key"],
                          "user_n_conversations": keep["user_n_conversations"].astype("int64")})
    refused = sum(r["status"] != "ok" for r in recs.values())
    coverage = {"dataset": ds, "sampled_users": int(sample["user_key"].nunique()),
                "sampled_conversations": len(sample), "complete_users": int(user_done.sum()),
                "complete_conversations": len(convs), "turns": len(turns), "turns_done": int(done_turn.sum()),
                "turns_refused": refused}
    return convs, recs, coverage


def _spans(cfg: dict, paths: Paths, ds: str) -> pd.DataFrame:
    """Presidio + secrets spans (at the analysis threshold) and the LLM detector's spans; no text."""
    det = pd.read_parquet(paths.restricted / f"spans_{ds}.parquet")
    thr = cfg["presidio"]["score_threshold"]
    det = det[(det["detector"] != "presidio") | (det["score"] >= thr)]
    llm = llm_detect.build_spans(cfg, ds)
    return pd.concat([det, llm], ignore_index=True) if len(llm) else det


def run(cfg: dict, datasets: list[str]) -> dict[str, pd.DataFrame]:
    paths = Paths(cfg).ensure()
    coarsener = profiles.load_coarsener(cfg)
    ucfg, acfg, seed, min_cell = cfg.get("uniqueness") or {}, cfg["analysis"], cfg["seed"], _min_cell(cfg)
    out: dict[str, list[pd.DataFrame]] = {}

    def add(name: str, df: pd.DataFrame) -> None:
        out.setdefault(name, []).append(df)

    for ds in datasets:
        convs, recs, coverage = complete_users(cfg, paths, ds)
        ok = [r for r in recs.values() if r["status"] == "ok"]
        attrs = profiles.attributes_from_records(ds, ok)
        prof = profiles.build_profiles(ds, attrs, convs, coarsener)
        target = paths.restricted / f"profiles_{ds}.parquet"
        prof.to_parquet(target, index=False)
        os.chmod(target, 0o600)
        coverage["profiled_users"] = int(prof["user"].nunique())
        add("identifiability_coverage", pd.DataFrame([coverage]))
        add("profile_summary", profiles.profile_summary(prof, min_cell))
        by_combo, by_activity, summary = uniqueness.analyze(prof, ucfg, acfg, seed, min_cell)
        add("uniqueness_by_combination", by_combo)
        add("uniqueness_by_activity", by_activity)
        add("uniqueness_summary", summary)
        eligible = int(_sample_summary(paths, ds).get("eligible_users") or 0)
        add("uniqueness_population", uniqueness.population_estimate(prof, summary, {ds: eligible}, ucfg, acfg,
                                                                     seed, min_cell))
        spans = _spans(cfg, paths, ds)
        add("linkage", linkage.analyze(ds, attrs, convs, spans, coarsener, ucfg, acfg, seed, min_cell))
        add("direct_overlap", linkage.direct_overlap(prof, convs, spans, ucfg, min_cell))
        print(f"[identifiability] {ds}: {coverage['profiled_users']:,} users profiled "
              f"({coverage['complete_users']:,} of {coverage['sampled_users']:,} fully covered by the LLM detector)")

    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in out.items()}
    for name, df in tables.items():
        write_table(df, paths.tables, name)
    return tables


def _predictions(paths: Paths, ds: str, items: pd.DataFrame, predictor: str) -> dict[str, dict]:
    if predictor != "llm_detect":
        return labeling.load_labels(paths, ds, predictor)
    recs = llm_detect.load_records(paths, ds)
    out = {}
    for it in items.itertuples(index=False):
        r = recs.get(f"{it.conversation_id}#{it.turn_index}")
        if r and r["status"] == "ok":
            out[it.item_id] = r
    return out


def validate(cfg: dict, datasets: list[str], gold: str | None = None,
             predictor: str = "llm_detect") -> pd.DataFrame:
    """Attribute agreement between a predictor and the gold annotator on the label items, by
    profile dimension. Template turns are dropped, as in `label score`."""
    paths = Paths(cfg).ensure()
    coarsener = profiles.load_coarsener(cfg)
    gold = gold or cfg["labeling"]["adjudicator"]
    parts = []
    for ds in datasets:
        items = pd.read_parquet(labeling.items_path(paths, ds))
        items, _ = labeling.drop_templates(cfg, paths, ds, items)
        g = labeling.load_labels(paths, ds, gold)
        if not g:
            raise RuntimeError(f"{ds}: no labels for gold annotator '{gold}'")
        p = _predictions(paths, ds, items, predictor)
        if not p:
            raise RuntimeError(f"{ds}: no predictions from '{predictor}' for the label items")
        parts.append(profiles.attribute_agreement(ds, items["item_id"].tolist(), g, p, coarsener, gold, predictor))
    table = pd.concat(parts, ignore_index=True)
    write_table(table, paths.tables, "identifiability_attribute_validation")
    return table
