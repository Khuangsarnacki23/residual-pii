"""Stage orchestration with a content-keyed Parquet cache.

Each stage's output is stored next to a `.key` sidecar holding a hash of every input that can change
it (config slice + upstream keys + pipeline version). A rerun with an unchanged key reads the cache.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path
from typing import Callable

import pandas as pd

from . import analysis, dedup, detect, redactions, report
from .config import Paths, load_taxonomy, stable_hash
from .safety import load_or_create_key
from .sampling import eligible, stratified_sample
from .sources import HFSource

PIPELINE_VERSION = "0.1.0"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cached(path: Path, key: str, fn: Callable[[], tuple[pd.DataFrame, dict]], force: bool = False):
    """Returns (df, extra) from cache if `key` matches, else runs fn and stores both."""
    side = path.with_suffix(".key.json")
    if not force and path.exists() and side.exists():
        meta = json.loads(side.read_text())
        if meta.get("key") == key:
            return pd.read_parquet(path), meta.get("extra", {})
    df, extra = fn()
    df.to_parquet(path, index=False)
    side.write_text(json.dumps({"key": key, "extra": extra}, default=str))
    return df, extra


class Pipeline:
    def __init__(self, cfg: dict, sources: dict | None = None, force: bool = False):
        self.cfg = cfg
        self.paths = Paths(cfg).ensure()
        self.key = load_or_create_key(self.paths.data_root)
        self.taxonomy = load_taxonomy(cfg)
        self.force = force
        self._sources = sources or {}
        self.manifest: dict = {"pipeline_version": PIPELINE_VERSION, "run_name": cfg["run_name"],
                               "config_hash": stable_hash(cfg), "datasets": {}}

    # ------------------------------------------------------------------ helpers
    def enabled(self) -> list[str]:
        if self._sources:
            return list(self._sources)
        return [k for k, v in self.cfg["datasets"].items() if v.get("enabled")]

    def source(self, name: str, scan: dict | None = None):
        if name in self._sources:
            return self._sources[name]
        return HFSource(name, self.cfg["datasets"][name], scan if scan is not None else self.cfg["scan"], self.key)

    def seed(self, name: str) -> int:
        return self.cfg["seed"] + zlib.crc32(name.encode()) % 10_000

    # ------------------------------------------------------------------ stages
    def sample(self, name: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Returns (sample metadata, sampled user turns, cache key)."""
        src = self.source(name)
        base = {"v": PIPELINE_VERSION, "repo": src.repo, "rev": src.revision, "scan": self.cfg["scan"]}
        meta_key = stable_hash(base)
        meta, _ = cached(self.paths.work / f"meta_{name}.parquet", meta_key,
                         lambda: (src.scan_meta(), {}), self.force)
        n = self.cfg["datasets"].get(name, {}).get("sample_size", 100)
        s_key = stable_hash([meta_key, self.cfg["sampling"], self.cfg.get("languages"), n, self.seed(name)])
        smp, summary = cached(self.paths.work / f"sample_{name}.parquet", s_key,
                              lambda: stratified_sample(meta, n, self.cfg, self.seed(name)), self.force)
        turns, _ = cached(self.paths.restricted / f"turns_{name}.parquet", s_key,
                          lambda: (src.fetch_turns(smp), {}), self.force)
        self.manifest["datasets"][name] = {"repo": src.repo, "revision": src.revision, "sampling": summary}
        log(f"[{name}] sample: {len(smp):,} conversations, {len(turns):,} user turns")
        return smp, turns, s_key

    def detect(self, name: str, turns: pd.DataFrame, s_key: str):
        analysis_only = ("score_threshold", "sensitivity_thresholds")
        pcfg = {k: v for k, v in self.cfg["presidio"].items() if k not in analysis_only}
        d_key = stable_hash([s_key, pcfg, self.cfg.get("secrets"), self.cfg["placeholders"],
                             self.taxonomy, PIPELINE_VERSION])
        stats_path = self.paths.work / f"turnstats_{name}.parquet"
        spans, _ = cached(self.paths.restricted / f"spans_{name}.parquet", d_key,
                          lambda: self._detect_and_stats(turns, stats_path, d_key), self.force)
        stats = pd.read_parquet(stats_path)
        tasks, _ = cached(self.paths.work / f"tasks_{name}.parquet", stable_hash([s_key, self.taxonomy]),
                          lambda: (detect.tag_tasks(turns, self.taxonomy), {}), self.force)
        log(f"[{name}] detect: {len(spans):,} stored spans "
            f"({int((~spans['in_placeholder'] & ~spans['is_dummy']).sum()):,} after placeholder/dummy exclusion)")
        return spans, stats, tasks

    def _detect_and_stats(self, turns, stats_path, d_key):
        spans, stats = detect.detect(turns, self.cfg, self.taxonomy, self.key)
        stats.to_parquet(stats_path, index=False)
        return spans, {}

    def dedup_sample(self, lmsys_turns: pd.DataFrame) -> pd.DataFrame:
        """lmsys sample vs the full English Arena corpus (Arena is ~40 MB, so this is cheap)."""
        arena = self.source("arena", scan={})
        meta_key = stable_hash({"repo": arena.repo, "rev": arena.revision, "scan": {}, "v": PIPELINE_VERSION})
        ameta, _ = cached(self.paths.work / "meta_arena_full.parquet", meta_key,
                          lambda: (arena.scan_meta(), {}), self.force)
        ameta = eligible(ameta, self.cfg.get("languages"))
        key = stable_hash([meta_key, len(lmsys_turns), stable_hash(sorted(lmsys_turns["conversation_id"].unique())),
                           self.cfg["dedup"], self.cfg.get("languages")])

        def run():
            ref = dedup.conversation_texts(arena.fetch_turns(ameta))
            res = dedup.match(ref, dedup.conversation_texts(lmsys_turns), self.cfg["dedup"], self.cfg["seed"])
            return res, {"arena_reference_conversations": len(ref)}

        res, extra = cached(self.paths.work / "dedup_lmsys_vs_arena.parquet", key, run, self.force)
        self.manifest["dedup"] = {"scope": "lmsys sample vs full Arena (English)", **extra,
                                  **res["dedup_status"].value_counts().to_dict()}
        return res

    def dedup_corpus(self) -> pd.DataFrame:
        """Full English lmsys-chat-1m vs full English Arena. Streams ~1.5 GB; nothing kept but counts."""
        langs = self.cfg.get("languages")
        arena, lmsys = self.source("arena", scan={}), self.source("lmsys", scan={})
        ameta = eligible(arena.scan_meta(), langs)
        index = dedup.build_index(dedup.conversation_texts(arena.fetch_turns(ameta)), self.cfg["dedup"], self.cfg["seed"])
        lmeta = eligible(lmsys.scan_meta(), langs)
        parts = [dedup.match(index, dedup.conversation_texts(b), self.cfg["dedup"], self.cfg["seed"])
                 for b in lmsys.iter_turn_batches(lmeta) if len(b)]
        res = pd.concat(parts, ignore_index=True)
        out = (res["dedup_status"].value_counts().rename_axis("dedup_status").reset_index(name="n_conversations")
               .assign(scope="lmsys_full_vs_arena_full", arena_reference=len(index.ids)))
        report.write_table(out.rename(columns={"scope": "subset"}), self.paths.tables, "corpus_overlap_lmsys_arena")
        res[["conversation_id", "dedup_status", "best_jaccard"]].to_parquet(self.paths.work / "dedup_corpus.parquet")
        return out

    def placeholders(self, all_turns: pd.DataFrame) -> pd.DataFrame:
        disc = redactions.discover(all_turns)
        report.write_table(disc, self.paths.tables, "placeholder_shapes")
        return disc

    # ------------------------------------------------------------------ end to end
    def run(self, do_dedup: bool = True) -> dict:
        samples, spans, stats, tasks, turns_all = [], [], [], [], []
        for name in self.enabled():
            try:
                smp, turns, s_key = self.sample(name)
            except Exception as e:  # e.g. gated dataset without accepted terms / login
                log(f"[{name}] SKIPPED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
                self.manifest["datasets"][name] = {"skipped": f"{type(e).__name__}"}
                continue
            sp, st, tk = self.detect(name, turns, s_key)
            samples.append(smp), spans.append(sp), stats.append(st), tasks.append(tk), turns_all.append(turns)
        if not samples:
            raise RuntimeError("No dataset could be loaded.")
        sample = pd.concat(samples, ignore_index=True)
        spans_df = pd.concat(spans, ignore_index=True)
        stats_df = pd.concat(stats, ignore_index=True)
        tasks_df = pd.concat(tasks, ignore_index=True)
        turns_df = pd.concat(turns_all, ignore_index=True)

        self.placeholders(turns_df)
        thr = self.cfg["presidio"]["score_threshold"]
        conv = analysis.conversation_indicators(sample, spans_df, self.taxonomy, thr)
        conv = conv.merge(sample[["dataset", "conversation_id", "weight", "stratum"]], on=["dataset", "conversation_id"])
        conv = conv.merge(tasks_df, on=["dataset", "conversation_id"], how="left")
        conv["is_story_roleplay"] = conv["is_story_roleplay"].fillna(False).astype(bool)
        conv["task_type"] = conv["task_type"].fillna(self.taxonomy["default_task_type"])

        if do_dedup and "lmsys" in set(sample["dataset"]) and not self._sources:
            try:
                d = self.dedup_sample(turns_df[turns_df["dataset"] == "lmsys"]).assign(dataset="lmsys")
                conv = conv.merge(d[["dataset", "conversation_id", "dedup_status"]],
                                  on=["dataset", "conversation_id"], how="left")
            except Exception as e:
                log(f"[dedup] SKIPPED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")

        self.analyze(conv, sample, spans_df, stats_df)
        return self.manifest

    def analyze(self, conv, sample, spans_df, stats_df) -> None:
        seed, t = self.cfg["seed"], self.paths.tables
        prev = analysis.prevalence_table(conv, self.cfg, seed)
        report.write_table(prev, t, "prevalence")
        task_cols = [c for c in conv.columns if c in ("any::tier_direct", "any::tier_quasi", "presidio::name")]
        by_task = analysis.by_task_table(conv, self.cfg, seed, task_cols)
        report.write_table(by_task, t, "prevalence_by_task")
        thr = self.cfg["presidio"]["score_threshold"]
        report.write_table(analysis.rates_per_1k_tokens(spans_df, stats_df, sample, thr), t, "rates_per_1k_tokens")
        report.write_table(analysis.exclusion_table(spans_df, thr), t, "excluded_detections")
        report.write_table(analysis.threshold_sensitivity(sample, spans_df, self.taxonomy,
                                                          self.cfg["presidio"]["sensitivity_thresholds"]),
                           t, "presidio_threshold_sensitivity")
        report.write_table(analysis.placeholder_table(stats_df, sample), t, "existing_redactions")
        report.write_table(analysis.coverage_table(stats_df), t, "scan_coverage")
        if "dedup_status" in conv:
            ov = (conv[conv["dataset"] == "lmsys"]["dedup_status"].fillna("unique")
                  .value_counts().rename_axis("dedup_status").reset_index(name="n_conversations"))
            report.write_table(ov, t, "lmsys_arena_overlap")
        made = report.figures(prev, by_task, self.taxonomy, self.paths.figures)
        report.methods_summary(self.cfg, self.manifest, self.paths.results)
        report.disclosure_drafts(prev, self.manifest, self.paths.results / "disclosure")
        self.manifest["figures"] = made
        report.write_manifest(self.manifest, self.paths.results)
        log(f"[report] wrote tables, {len(made)} figures, methods.md and disclosure drafts to {self.paths.results}")
