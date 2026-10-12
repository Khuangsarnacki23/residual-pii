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

from . import analysis, dedup, detect, label, llm, profiles, redactions, report, uniqueness
from .config import Paths, load_taxonomy, stable_hash
from .safety import load_or_create_key, safe_count, suppress_small_cells
from .sampling import NoUserProxy, eligible, stratified_sample, user_activity, user_level_sample
from .sources import SPECS, HFSource

PIPELINE_VERSION = "0.2.0"


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
    def __init__(self, cfg: dict, sources: dict | None = None, force: bool = False, llm_backend=None):
        self.cfg = cfg
        self.paths = Paths(cfg).ensure()
        self.key = load_or_create_key(self.paths.data_root)
        self.taxonomy = load_taxonomy(cfg)
        self.force = force
        self._sources = sources or {}
        self.mode = cfg["sampling"].get("mode", "per_user_one_conv")
        if self.mode not in ("per_user_one_conv", "user_level"):
            raise ValueError(f"Unknown sampling.mode: {self.mode!r}")
        self.roles = tuple(cfg.get("roles") or ["user"])
        if "user" not in self.roles:
            raise ValueError("roles must include 'user'")
        self.min_cell = int(cfg.get("publication", {}).get("min_cell", 10))
        self.manifest: dict = {"pipeline_version": PIPELINE_VERSION, "run_name": cfg["run_name"],
                               "config_hash": stable_hash(cfg), "sampling_mode": self.mode,
                               "roles": list(self.roles), "datasets": {}}
        self._activity: dict[str, pd.DataFrame] = {}
        self.llm_enabled = bool(cfg.get("llm", {}).get("enabled"))
        self._llm_backend = llm_backend          # injected by tests; otherwise built from the config
        if self.llm_enabled and llm_backend is None:
            llm.check_backend_allowed(cfg["llm"])   # fail before any work if a hosted backend is not allowed

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
    def _meta(self, name: str):
        """Returns (source, cached metadata scan, cache key). Checks the user proxy in user_level mode."""
        if self.mode == "user_level" and name not in self._sources and SPECS[name].user is None:
            raise NoUserProxy(f"user_level sampling needs a user proxy and {name} has none")
        src = self.source(name)
        base = {"v": PIPELINE_VERSION, "repo": src.repo, "rev": src.revision, "scan": self.cfg["scan"]}
        meta_key = stable_hash(base)
        meta, _ = cached(self.paths.work / f"meta_{name}.parquet", meta_key,
                         lambda: (src.scan_meta(), {}), self.force)
        if self.mode == "user_level" and not meta["user_key"].notna().any():
            raise NoUserProxy(f"user_level sampling needs a user proxy and {name} has none")
        return src, meta, meta_key

    def sample(self, name: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Returns (sample metadata, sampled turns of the configured roles, cache key)."""
        src, meta, meta_key = self._meta(name)
        # The mode and its settings are part of the key, so the two samplers never share a cache entry.
        if self.mode == "user_level":
            s_key = stable_hash([meta_key, "user_level", self.cfg["user_level"], self.cfg["sampling"],
                                 self.cfg.get("languages"), self.seed(name)])
            draw = lambda: user_level_sample(meta, self.cfg, self.seed(name))  # noqa: E731
        else:
            n = self.cfg["datasets"].get(name, {}).get("sample_size", 100)
            s_key = stable_hash([meta_key, self.cfg["sampling"], self.cfg.get("languages"), n, self.seed(name)])
            draw = lambda: stratified_sample(meta, n, self.cfg, self.seed(name))  # noqa: E731
        smp, summary = cached(self.paths.work / f"sample_{name}.parquet", s_key, draw, self.force)
        t_key = stable_hash([s_key, self.roles])
        turns, _ = cached(self.paths.restricted / f"turns_{name}.parquet", t_key,
                          lambda: (src.fetch_turns(smp, self.roles), {}), self.force)
        summary = dict(summary)
        if "user_activity" in summary:
            self._activity[name] = pd.DataFrame(summary.pop("user_activity")).assign(dataset=name)
            for k in ("users_over_cap", "sampled_users_over_cap"):
                summary[k] = safe_count(summary[k], self.min_cell)
        self.manifest["datasets"][name] = {"repo": src.repo, "revision": src.revision, "sampling": summary}
        log(f"[{name}] sample: {len(smp):,} conversations, {len(turns):,} turns ({', '.join(self.roles)})")
        return smp, turns, t_key

    def write_user_activity(self) -> pd.DataFrame | None:
        """Bucketed users-by-conversation-count table for the datasets sampled at user level."""
        if not self._activity:
            return None
        act = pd.concat(self._activity.values(), ignore_index=True)
        act = act[["dataset", "activity_bucket", "meets_min_conversations", "n_users", "share_users"]]
        act = suppress_small_cells(act, "n_users", ["share_users"], self.min_cell)
        report.write_table(act, self.paths.tables, "user_activity")
        return act

    def user_activity_only(self) -> pd.DataFrame:
        """Metadata scan only (no text): how many users have how many conversations."""
        for name in self.enabled():
            try:
                _, meta, _ = self._meta(name)
                self._activity[name] = user_activity(meta, self.cfg).assign(dataset=name)
            except Exception as e:
                log(f"[{name}] SKIPPED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        act = self.write_user_activity()
        if act is None:
            raise RuntimeError("No dataset with a user proxy could be loaded.")
        return act

    def detect(self, name: str, turns: pd.DataFrame, s_key: str):
        analysis_only = ("score_threshold", "sensitivity_thresholds")
        pcfg = {k: v for k, v in self.cfg["presidio"].items() if k not in analysis_only}
        d_key = stable_hash([s_key, pcfg, self.cfg.get("secrets"), self.cfg.get("selfid"), self.cfg["placeholders"],
                             self.taxonomy, PIPELINE_VERSION])
        stats_path = self.paths.work / f"turnstats_{name}.parquet"
        spans, _ = cached(self.paths.restricted / f"spans_{name}.parquet", d_key,
                          lambda: self._detect_and_stats(turns, stats_path, d_key), self.force)
        stats = pd.read_parquet(stats_path)
        tasks, _ = cached(self.paths.work / f"tasks_{name}.parquet", stable_hash([s_key, self.taxonomy]),
                          lambda: (detect.tag_tasks(turns[turns["role"] == "user"], self.taxonomy), {}),
                          self.force)
        log(f"[{name}] detect: {len(spans):,} stored spans "
            f"({int((~spans['in_placeholder'] & ~spans['is_dummy']).sum()):,} after placeholder/dummy exclusion)")
        return spans, stats, tasks

    def _detect_and_stats(self, turns, stats_path, d_key):
        spans, stats = detect.detect(turns, self.cfg, self.taxonomy, self.key)
        stats.to_parquet(stats_path, index=False)
        return spans, {}

    def llm_backend(self):
        if self._llm_backend is None:
            self._llm_backend = llm.make_backend(self.cfg["llm"])
        return self._llm_backend

    def llm_extract(self, name: str, smp: pd.DataFrame, turns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """LLM quasi-identifier extraction on a subsample of user turns. Values stay in restricted/."""
        lcfg = self.cfg["llm"]
        selected = llm.select_turns(turns, smp, lcfg, self.seed(name), whole_users=self.mode == "user_level")
        attrs, turn_log = llm.extract(selected, lcfg, llm.load_prompt(self.cfg), self.llm_backend(), self.key,
                                      self.paths.restricted / f"llm_cache_{name}.jsonl")
        attrs.to_parquet(self.paths.restricted / f"llm_attrs_{name}.parquet", index=False)
        turn_log.to_parquet(self.paths.work / f"llm_turns_{name}.parquet", index=False)
        log(f"[{name}] llm: {len(turn_log):,} turns, {int((turn_log['status'] != 'ok').sum()):,} failures, "
            f"{len(attrs):,} attributes")
        return attrs, turn_log

    def identifiability(self, sample: pd.DataFrame, attrs: pd.DataFrame, turn_log: pd.DataFrame) -> list[str]:
        """LLM aggregates -> per-user profiles (restricted) -> uniqueness tables and figure."""
        t, lcfg, backend = self.paths.tables, self.cfg["llm"], self.llm_backend()
        report.write_table(llm.attribute_prevalence(attrs, turn_log, sample, self.min_cell), t,
                           "llm_attribute_prevalence")
        stats = llm.run_stats(turn_log, backend, lcfg)
        report.write_table(stats, t, "llm_run_stats")
        self.manifest["llm"] = {"backend": backend.name, "model": backend.model,
                                "hosted": backend.name in llm.HOSTED_BACKENDS,
                                "prompt_version": lcfg["prompt_version"],
                                "prompt_sha": llm.load_prompt(self.cfg).sha,
                                "turns_processed": int(stats["turns_processed"].sum()),
                                "failures": int(stats["failures"].sum())}

        # Profiles need a user to attach to; datasets without a user proxy stop at the tables above.
        with_proxy = sample.groupby("dataset")["user_key"].agg(lambda s: s.notna().any())
        keep = [ds for ds in turn_log["dataset"].unique() if with_proxy.get(ds, False)]
        prof = profiles.build_profiles(attrs[attrs["dataset"].isin(keep)], turn_log[turn_log["dataset"].isin(keep)],
                                       sample, profiles.load_coarsener(self.cfg), float(lcfg["min_confidence"]))
        for ds, g in prof.groupby("dataset"):
            g.to_parquet(self.paths.restricted / f"profiles_{ds}.parquet", index=False)
        if prof.empty:
            log("[uniqueness] SKIPPED: no dataset with a user proxy was processed by the LLM")
            return []
        report.write_table(profiles.profile_summary(prof, self.min_cell), t, "profile_summary")
        by_combo, by_activity, summary = uniqueness.analyze(prof, self.cfg.get("uniqueness", {}),
                                                            self.cfg["analysis"], self.cfg["seed"], self.min_cell)
        report.write_table(by_combo, t, "uniqueness_by_combination")
        report.write_table(by_activity, t, "uniqueness_by_activity")
        report.write_table(summary, t, "uniqueness_summary")
        log(f"[uniqueness] {prof['user'].nunique():,} users, {len(by_combo):,} combination rows")
        return report.uniqueness_figures(summary, self.paths.figures)

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
        k = self.cfg["placeholders"].get("discovery_min_conversations", 5)
        disc = redactions.discover(all_turns, min_conversations=k)
        patterns = {ds: redactions.compile_patterns(self.cfg, ds) for ds in disc["dataset"].unique()}
        disc["configured"] = [redactions.matches_configured(shp, patterns[ds])
                              for ds, shp in zip(disc["dataset"], disc["placeholder_shape"])]
        report.write_table(disc, self.paths.tables, "placeholder_shapes")
        return disc

    def discover_placeholders(self) -> pd.DataFrame:
        """Placeholder-format discovery on the sampled turns, without running any detector."""
        turns = []
        for name in self.enabled():
            try:
                turns.append(self.sample(name)[1])
            except Exception as e:
                log(f"[{name}] SKIPPED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        if not turns:
            raise RuntimeError("No dataset could be loaded.")
        return self.placeholders(pd.concat(turns, ignore_index=True))

    # ------------------------------------------------------------------ hand labeling
    def label_export(self, n: int, fmt: str = "csv") -> Path:
        """Writes a batch of user turns for annotation under restricted/labels/. Returns its path."""
        turns, spans, tasks = [], [], []
        for name in self.enabled():
            try:
                _, tn, s_key = self.sample(name)
            except Exception as e:
                log(f"[{name}] SKIPPED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
                continue
            sp, _, tk = self.detect(name, tn, s_key)
            turns.append(tn), spans.append(sp), tasks.append(tk)
        if not turns:
            raise RuntimeError("No dataset could be loaded.")
        items = label.export_sample(pd.concat(turns, ignore_index=True), pd.concat(spans, ignore_index=True),
                                    pd.concat(tasks, ignore_index=True), n, self.cfg["seed"],
                                    self.cfg["presidio"]["score_threshold"], int(self.cfg["label"]["max_text_chars"]))
        path = label.write_batch(items, self.paths.restricted, fmt)
        log(f"[label] wrote {len(items):,} turns for annotation (restricted directory, not the repo)")
        return path

    def label_score(self, labels_path: str | Path, second_annotator: str | None = None) -> pd.DataFrame:
        """Scores stored detector spans against an annotated batch -> tables/detector_validation.csv."""
        labels = label.read_labels(labels_path)
        min_overlap = float(self.cfg["label"]["match_min_overlap"])
        gold, skipped = label.gold_spans(labels, list(self.taxonomy["categories"]))
        items = labels.drop_duplicates(label.TURN)[[*label.TURN, "text"]]
        log(f"[label] {len(items):,} labeled turns, {len(gold):,} gold spans; skipped rows: {skipped}")

        pred, attrs, turn_logs = [], [], []
        for ds, it in items.groupby("dataset"):
            path = self.paths.restricted / f"spans_{ds}.parquet"
            if not path.exists():
                raise RuntimeError(f"No span store for {ds}; run `residpii run` with this config first.")
            sp = analysis.residual(pd.read_parquet(path), self.cfg["presidio"]["score_threshold"])
            sp = sp[sp["role"] == "user"].merge(it.assign(n_chars=it["text"].str.len())[[*label.TURN, "n_chars"]],
                                                on=label.TURN)
            pred.append(sp[sp["end"] <= sp["n_chars"]])      # only spans inside the exported text
            if second_annotator == "llm":
                # The LLM annotates exactly the labeled turns (the hosted opt-in rule applies as usual).
                selected = it.assign(role="user", context="")
                a, tl = llm.extract(selected, self.cfg["llm"], llm.load_prompt(self.cfg), self.llm_backend(),
                                    self.key, self.paths.restricted / f"llm_cache_{ds}.jsonl")
                attrs.append(a), turn_logs.append(tl)
            elif (self.paths.restricted / f"llm_attrs_{ds}.parquet").exists():
                attrs.append(pd.read_parquet(self.paths.restricted / f"llm_attrs_{ds}.parquet"))
                turn_logs.append(pd.read_parquet(self.paths.work / f"llm_turns_{ds}.parquet"))
        out = label.score_detectors(pd.concat(pred, ignore_index=True), gold, min_overlap)
        if attrs:
            out = pd.concat([out, label.about_agreement(labels, gold, pd.concat(attrs, ignore_index=True),
                                                        pd.concat(turn_logs, ignore_index=True), min_overlap)],
                            ignore_index=True)
        out.insert(0, "n_labeled_turns", len(items))
        report.write_table(out, self.paths.tables, "detector_validation")
        return out

    # ------------------------------------------------------------------ end to end
    def run(self, do_dedup: bool = True) -> dict:
        samples, spans, stats, tasks, turns_all, llm_attrs, llm_logs = [], [], [], [], [], [], []
        for name in self.enabled():
            try:
                smp, turns, s_key = self.sample(name)
            except NoUserProxy as e:
                log(f"[{name}] SKIPPED: {e}")
                self.manifest["datasets"][name] = {"skipped": "no user proxy (user_level sampling)"}
                continue
            except Exception as e:  # e.g. gated dataset without accepted terms / login
                log(f"[{name}] SKIPPED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
                self.manifest["datasets"][name] = {"skipped": f"{type(e).__name__}"}
                continue
            sp, st, tk = self.detect(name, turns, s_key)
            samples.append(smp), spans.append(sp), stats.append(st), tasks.append(tk), turns_all.append(turns)
            if self.llm_enabled:
                attrs, turn_log = self.llm_extract(name, smp, turns)
                llm_attrs.append(attrs), llm_logs.append(turn_log)
        if not samples:
            raise RuntimeError("No dataset could be loaded.")
        sample = pd.concat(samples, ignore_index=True)
        spans_df = pd.concat(spans, ignore_index=True)
        stats_df = pd.concat(stats, ignore_index=True)
        tasks_df = pd.concat(tasks, ignore_index=True)
        turns_df = pd.concat(turns_all, ignore_index=True)

        self.placeholders(turns_df)
        self.write_user_activity()
        thr = self.cfg["presidio"]["score_threshold"]
        # The prevalence tables keep their estimand (user turns). Other roles, when fetched, are
        # reported separately and serve as context for the LLM detector.
        if len(self.roles) > 1:
            report.write_table(analysis.by_role_table(sample, spans_df, self.taxonomy, thr),
                               self.paths.tables, "prevalence_by_role")
        turns_df = turns_df[turns_df["role"] == "user"]
        spans_df = spans_df[spans_df["role"] == "user"].reset_index(drop=True)
        stats_df = stats_df[stats_df["role"] == "user"].reset_index(drop=True)
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

        extra_figures = []
        if llm_logs:
            extra_figures = self.identifiability(sample, pd.concat(llm_attrs, ignore_index=True),
                                                 pd.concat(llm_logs, ignore_index=True))
        self.analyze(conv, sample, spans_df, stats_df, extra_figures)
        return self.manifest

    def analyze(self, conv, sample, spans_df, stats_df, extra_figures: list[str] | None = None) -> None:
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
        made = report.figures(prev, by_task, self.taxonomy, self.paths.figures) + (extra_figures or [])
        report.methods_summary(self.cfg, self.manifest, self.paths.results)
        report.disclosure_drafts(prev, self.manifest, self.paths.results / "disclosure")
        self.manifest["figures"] = made
        report.write_manifest(self.manifest, self.paths.results)
        log(f"[report] wrote tables, {len(made)} figures, methods.md and disclosure drafts to {self.paths.results}")
