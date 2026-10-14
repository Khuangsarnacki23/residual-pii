"""LLM detector over the full sample (every sampled user turn), with a hard spending cap.

OPT-IN ONLY: sends sampled user turns to Anthropic's API. Uses the same guideline and output schema
as the LLM annotator (llm_label.py), so its labels are comparable, but runs a cheaper model by default.

Outputs (all under the restricted directory except the aggregate tables):
  restricted/llm_detect/<ds>.jsonl     one record per turn: spans (offsets), attributes, token usage
  restricted/spans_llm_<ds>.parquet    SafeSpan rows (no text), two detector families:
                                         llm_detector      spans the model says are private (self/third party)
                                         llm_detector_all  every span it marked, whoever it belongs to
  results/<run>/tables/llm_prevalence.csv, llm_coverage.csv

Turns are processed in a seeded, conversation-shuffled order, so if the budget stops a run early the
labelled conversations are a random subset; prevalence is computed over fully covered conversations.
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis
from .config import Paths
from .llm_label import PRICES, _call, locate, prompt_version
from .redactions import compile_patterns, placeholder_spans
from .safety import SafeSpan, keyed_hash, load_or_create_key

DEFAULT_MODEL = "claude-haiku-5-5"
DETECTOR, DETECTOR_ALL = "llm_detector", "llm_detector_all"
EST_SYSTEM_TOKENS, EST_OUTPUT_TOKENS, CHARS_PER_TOKEN = 1400, 350, 3.5


def store_path(paths: Paths, ds: str) -> Path:
    d = paths.restricted / "llm_detect"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d / f"{ds}.jsonl"


def _turns(paths: Paths, cfg: dict, ds: str) -> pd.DataFrame:
    t = pd.read_parquet(paths.restricted / f"turns_{ds}.parquet")
    t = t[t["text"].str.strip().str.len() > 0].copy()
    t["text"] = t["text"].str.slice(0, cfg["presidio"]["max_chars_per_turn"])   # same scan limit as Presidio
    t["key"] = t["conversation_id"].astype(str) + "#" + t["turn_index"].astype(str)
    # Seeded conversation-level shuffle: an early stop leaves a random subset of whole conversations.
    convs = t["conversation_id"].drop_duplicates().to_numpy()
    order = {c: i for i, c in enumerate(np.random.default_rng(cfg["seed"] + zlib.crc32(ds.encode()) % 997).permutation(convs))}
    return t.assign(_o=t["conversation_id"].map(order)).sort_values(["_o", "turn_index"]).drop(columns="_o")


def load_records(paths: Paths, ds: str) -> dict[str, dict]:
    out = {}
    p = store_path(paths, ds)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def record_cost(r: dict) -> float:
    m = r.get("meta", {})
    pin, pout, pcache = PRICES.get(r.get("model", ""), PRICES["claude-opus-5-5"])
    return (m.get("input_tokens", 0) * pin + m.get("output_tokens", 0) * pout
            + m.get("cache_read_input_tokens", 0) * pcache + m.get("cache_creation_input_tokens", 0) * pin * 1.25) / 1e6


def spent(paths: Paths, datasets: list[str]) -> float:
    return sum(record_cost(r) for ds in datasets for r in load_records(paths, ds).values())


def estimate(cfg: dict, datasets: list[str], model: str = DEFAULT_MODEL) -> pd.DataFrame:
    paths = Paths(cfg).ensure()
    pin, pout, _ = PRICES[model]
    rows = []
    for ds in datasets:
        t = _turns(paths, cfg, ds)
        todo = t[~t["key"].isin(load_records(paths, ds))]
        tin = todo["text"].str.len().sum() / CHARS_PER_TOKEN + len(todo) * EST_SYSTEM_TOKENS
        rows.append({"dataset": ds, "turns": len(t), "remaining": len(todo),
                     "est_cost_usd": round((tin * pin + len(todo) * EST_OUTPUT_TOKENS * pout) / 1e6, 2)})
    return pd.DataFrame(rows)


def run(cfg: dict, ds: str, model: str = DEFAULT_MODEL, effort: str = "low", max_cost: float = 20.0,
        budget_datasets: list[str] | None = None, workers: int = 8, client=None) -> dict:
    """Labels every unprocessed turn of `ds` until done or until total spend reaches `max_cost`.
    `budget_datasets` are the datasets whose spend counts toward the cap (default: just `ds`)."""
    lcfg = cfg["labeling"]
    paths = Paths(cfg).ensure()
    turns = _turns(paths, cfg, ds)
    done = load_records(paths, ds)
    version = prompt_version(lcfg, model, effort)
    other = {r.get("prompt_version") for r in done.values()} - {version}
    if other:
        raise RuntimeError(f"{len(done)} existing records for {ds} were made with a different model, effort or "
                           f"prompt; move {store_path(paths, ds)} aside to start over.")
    todo = turns[~turns["key"].isin(done)]
    total = spent(paths, budget_datasets or [ds])
    pin, pout, _ = PRICES.get(model, PRICES["claude-opus-5-5"])
    # Reserve for in-flight requests so the cap holds even with parallel calls.
    per_call = (EST_SYSTEM_TOKENS * pin + EST_OUTPUT_TOKENS * pout) / 1e6

    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=4)

    stats = {"dataset": ds, "model": model, "turns": len(turns), "already_done": len(done), "processed": 0,
             "refused": 0, "failed": 0, "quotes_not_found": 0, "spent_before_usd": round(total, 4)}
    print(f"[llm-detect] {ds}: {len(todo):,} of {len(turns):,} turns to do with {model} (effort {effort}); "
          f"${total:.2f} of ${max_cost:.2f} budget spent", file=sys.stderr, flush=True)
    out = store_path(paths, ds)
    rows = iter(todo[["key", "conversation_id", "turn_index", "text"]].itertuples(index=False))

    def work(row):
        try:
            return row, *_call(client, row.text, lcfg, model, effort), None
        except Exception as e:
            return row, None, {}, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"

    pool = ThreadPoolExecutor(max_workers=workers)
    inflight, n, pending = set(), 0, None
    try:
        while True:
            while len(inflight) < workers * 2:
                row = pending if pending is not None else next(rows, None)
                pending = None
                if row is None:
                    break
                est = per_call + len(row.text) / CHARS_PER_TOKEN * pin / 1e6
                if total + sum(f.est for f in inflight) + est > max_cost:
                    pending = row   # retried as in-flight requests finish and free their reserve
                    break
                fut = pool.submit(work, row)
                fut.est = est
                inflight.add(fut)
            if not inflight:
                break
            finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                row, parsed, meta, err = fut.result()
                rec = {"key": row.key, "conversation_id": row.conversation_id, "turn_index": int(row.turn_index),
                       "model": model, "prompt_version": version, "meta": meta,
                       "at": datetime.now(timezone.utc).isoformat()}
                if err:
                    stats["failed"] += 1   # not recorded: retried on the next run
                    print(f"[llm-detect] {row.key} failed ({err})", file=sys.stderr, flush=True)
                    continue
                if parsed is None:
                    rec.update(status="refused" if meta.get("stop_reason") == "refusal" else "no_output",
                               spans=[], attributes=[], skip=False)
                    stats["refused"] += 1
                else:
                    spans, missing = locate(row.text, parsed["spans"])
                    stats["quotes_not_found"] += missing
                    rec.update(status="ok", spans=spans, attributes=parsed["attributes"], skip=parsed["skip"],
                               quotes_not_found=missing)
                fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += record_cost(rec)
                stats["processed"] += 1
                n += 1
                if n % 50 == 0 or n == len(todo):
                    print(f"[llm-detect] {ds} {n:,}/{len(todo):,}  ${total:.2f} spent", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        stats["interrupted"] = True
        print("\n[llm-detect] stopping: waiting for in-flight requests...", file=sys.stderr, flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    stopped_for_budget = pending is not None and not stats.get("interrupted")
    stats["stopped_for_budget"] = stopped_for_budget
    stats["spent_total_usd"] = round(total, 4)
    left = len(turns) - len(load_records(paths, ds))
    msg = f"[llm-detect] {ds}: {stats['processed']:,} processed this run, {left:,} left, ${total:.2f} spent in total."
    if stopped_for_budget:
        msg += f" Stopped at the ${max_cost:.2f} cap."
    print(msg, file=sys.stderr, flush=True)
    return stats


# ---------------------------------------------------------------- spans + prevalence


def build_spans(cfg: dict, ds: str) -> pd.DataFrame:
    """Converts stored records into a text-free SafeSpan table at restricted/spans_llm_<ds>.parquet."""
    paths = Paths(cfg).ensure()
    lcfg = cfg["labeling"]
    key = load_or_create_key(paths.data_root)
    direct = set(lcfg["span_categories"]["direct"])
    private = set(lcfg["private_whose"])
    turns = _turns(paths, cfg, ds).set_index("key")
    pats = compile_patterns(cfg, ds)
    rows = []
    for k, r in load_records(paths, ds).items():
        if r["status"] != "ok" or k not in turns.index:
            continue
        text = turns.at[k, "text"]
        ph = placeholder_spans(text, pats)
        for s in r["spans"]:
            tier = "direct" if s["category"] in direct else "quasi"
            in_ph = any(s["start"] < pe and ps < s["end"] for ps, pe in ph)
            for det in ([DETECTOR, DETECTOR_ALL] if s["whose"] in private else [DETECTOR_ALL]):
                rows.append(SafeSpan(ds, r["conversation_id"], int(r["turn_index"]), s["start"], s["end"], det,
                                     s["whose"], s["category"], tier, 1.0,
                                     keyed_hash(key, text[s["start"]:s["end"]]), in_ph, False).to_row())
    df = pd.DataFrame(rows, columns=[*SafeSpan.__dataclass_fields__, "length"])
    df.to_parquet(paths.restricted / f"spans_llm_{ds}.parquet", index=False)
    return df


def prevalence(cfg: dict, datasets: list[str]) -> dict[str, pd.DataFrame]:
    """Design-weighted share of conversations with LLM-detected PII, over fully covered conversations."""
    from .report import write_table

    paths = Paths(cfg).ensure()
    lcfg = cfg["labeling"]
    tax = {"categories": {c: {"tier": t, "labels": []} for t in ("direct", "quasi")
                          for c in lcfg["span_categories"][t]}}
    prev, cov = [], []
    for ds in datasets:
        spans = build_spans(cfg, ds)
        recs = load_records(paths, ds)
        turns = _turns(paths, cfg, ds)
        turns["status"] = turns["key"].map(lambda k: recs[k]["status"] if k in recs else "missing")
        full = turns.groupby("conversation_id")["status"].agg(lambda s: (s != "missing").all())
        covered = set(full[full].index)
        sample = pd.read_parquet(paths.work / f"sample_{ds}.parquet")
        sample = sample[sample["conversation_id"].isin(covered)]
        tasks = pd.read_parquet(paths.work / f"tasks_{ds}.parquet")
        cov.append({"dataset": ds, "turns": len(turns), "turns_done": int((turns["status"] != "missing").sum()),
                    "turns_refused": int((turns["status"] == "refused").sum()),
                    "conversations": int(full.size), "conversations_covered": len(covered),
                    "cost_usd": round(sum(record_cost(r) for r in recs.values()), 3)})
        if sample.empty:
            continue
        spans = spans[spans["conversation_id"].isin(covered)]
        conv = analysis.conversation_indicators(sample, spans, tax, 0.0)
        conv = conv.merge(sample[["dataset", "conversation_id", "weight", "stratum"]], on=["dataset", "conversation_id"])
        conv = conv.merge(tasks, on=["dataset", "conversation_id"], how="left")
        conv["is_story_roleplay"] = conv["is_story_roleplay"].fillna(False).astype(bool)
        prev.append(analysis.prevalence_table(conv, cfg, cfg["seed"]).query("detector != 'any'"))
    tables = {"llm_coverage": pd.DataFrame(cov)}
    if prev:
        tables["llm_prevalence"] = pd.concat(prev, ignore_index=True)
    for name, df in tables.items():
        write_table(df, paths.tables, name)
    return tables
