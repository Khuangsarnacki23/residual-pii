"""Aggregate-only outputs: tables, figures, an auto-generated methods summary, disclosure drafts."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .safety import assert_safe_table  # noqa: E402

# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs, light mode).
DATASET_COLORS = {"wildchat": "#2a78d6", "lmsys": "#eb6834", "arena": "#1baf7a"}
DATASET_LABELS = {"wildchat": "WildChat-1M", "lmsys": "LMSYS-Chat-1M", "arena": "Chatbot Arena"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VARIANT_LABELS = {"all": "All conversations", "excl_story_roleplay": "Excluding story / role-play",
                  "excl_arena_overlap": "Excluding Arena overlap"}


TIER_LABELS = {"tier_direct": "any direct identifier", "tier_quasi": "any quasi-identifier"}


def _presidio_version() -> str:
    from importlib.metadata import version

    return version("presidio-analyzer")


def write_table(df: pd.DataFrame, tables_dir: Path, name: str) -> None:
    assert_safe_table(df, name)
    df.to_csv(tables_dir / f"{name}.csv", index=False, float_format="%.6g")


def _style(ax) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)


def dot_ci_panels(df: pd.DataFrame, row_col: str, panel_col: str, panels: list[str], panel_labels: dict,
                  row_order: list[str], xlabel: str, path: Path, title: str) -> None:
    """Horizontal dot + 95% CI plot; one row per category, datasets offset vertically, one panel per variant."""
    datasets = [d for d in DATASET_COLORS if d in df["dataset"].unique()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels) + 1.2, 0.42 * len(row_order) + 1.4),
                             sharey=True, squeeze=False)
    offsets = np.linspace(-0.22, 0.22, len(datasets)) if len(datasets) > 1 else [0.0]
    xmax = max(0.05, float(df["ci_high"].max(skipna=True) if df["ci_high"].notna().any() else df["prevalence"].max()))
    for ax, panel in zip(axes[0], panels):
        sub = df[df[panel_col] == panel]
        for off, ds in zip(offsets, datasets):
            d = sub[sub["dataset"] == ds].set_index(row_col).reindex(row_order)
            y = np.arange(len(row_order)) + off
            ax.hlines(y, d["ci_low"], d["ci_high"], color=DATASET_COLORS[ds], linewidth=2)
            ax.scatter(d["prevalence"], y, s=36, color=DATASET_COLORS[ds], edgecolor="white", linewidth=1.5,
                       zorder=3, label=DATASET_LABELS[ds])
        ax.set_title(panel_labels.get(panel, panel), fontsize=10, color=TEXT_PRIMARY, loc="left")
        ax.set_xlim(0, xmax * 1.05)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(xlabel, fontsize=9)
        _style(ax)
    axes[0][0].set_yticks(range(len(row_order)), [r.replace("_", " ") for r in row_order], color=TEXT_PRIMARY)
    axes[0][0].invert_yaxis()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(datasets), frameon=False, fontsize=9,
               labelcolor=TEXT_PRIMARY)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def figures(prev: pd.DataFrame, by_task: pd.DataFrame, taxonomy: dict, fig_dir: Path) -> list[str]:
    made = []
    producible = set(prev.loc[prev["detector"] != "any", "category"])   # categories some enabled detector emits
    cats = [c for c in taxonomy["categories"] if c in producible]
    p = prev[(prev["detector"] == "any") & prev["category"].isin(cats)]
    if not p.empty:
        dot_ci_panels(p, "category", "variant", ["all", "excl_story_roleplay"], VARIANT_LABELS, cats,
                      "Conversations with ≥1 residual detection", fig_dir / "prevalence_by_category",
                      "Residual detections by category (all rule-based detectors, design-weighted, 95% bootstrap CI)")
        made.append("prevalence_by_category")
    t = by_task[(by_task["detector"] == "any") & (by_task["category"] == "tier_direct")]
    if not t.empty:
        order = (t.groupby("task_type")["task_share"].mean().sort_values(ascending=False).index.tolist())
        dot_ci_panels(t.assign(panel="direct"), "task_type", "panel", ["direct"], {"direct": "Direct identifiers"},
                      order, "Conversations with ≥1 direct identifier", fig_dir / "direct_by_task",
                      "Direct-identifier prevalence by heuristic task type")
        made.append("direct_by_task")
    return made


VARIANT_COLORS = {"stated_only": "#2a78d6", "stated_plus_inferred": "#eb6834"}   # palette slots 1-2
PROFILE_VARIANT_LABELS = {"stated_only": "Stated only", "stated_plus_inferred": "Stated + inferred"}
LEVEL_LABELS = {"generalized": "11 attributes, generalised where a lookup table exists"}


def uniqueness_figures(summary: pd.DataFrame, fig_dir: Path, population: pd.DataFrame | None = None) -> list[str]:
    """Share of sampled users unique on >= 1 combination, against the number of quasi-identifiers.
    One line per profile variant, one panel per coarsening level, one file per dataset. Dashed lines
    are the modelled share among all eligible users, when `population` holds that estimate."""
    pop = None if population is None else population[population["subset"] == "population"]
    made = []
    full = summary[(summary["ci_kind"] == "bootstrap_users") & summary["share_unique"].notna()]
    for ds, d in full.groupby("dataset"):
        levels = sorted(d["level"].unique())
        fig, axes = plt.subplots(1, len(levels), figsize=(max(4.4 * len(levels) + 0.6, 7.2), 3.6), sharey=True, squeeze=False)
        for ax, level in zip(axes[0], levels):
            for variant, color in VARIANT_COLORS.items():
                v = d[(d["level"] == level) & (d["variant"] == variant)].sort_values("n_qi")
                if v.empty:
                    continue
                ax.fill_between(v["n_qi"], v["ci_low"], v["ci_high"], color=color, alpha=0.15, linewidth=0)
                ax.plot(v["n_qi"], v["share_unique"], color=color, linewidth=2, marker="o", markersize=6,
                        markeredgecolor="white", markeredgewidth=1.5, label=PROFILE_VARIANT_LABELS[variant])
                e = None if pop is None else pop[(pop["dataset"] == ds) & (pop["level"] == level)
                                                 & (pop["variant"] == variant)].sort_values("n_qi")
                if e is not None and len(e):
                    ax.plot(e["n_qi"], e["share_unique_model"], color=color, linewidth=1.5, linestyle=(0, (4, 2)))
            ax.set_title(LEVEL_LABELS.get(level, level), fontsize=10, color=TEXT_PRIMARY, loc="left")
            ax.set_xticks(sorted(d["n_qi"].unique()))
            ax.set_xlabel("Quasi-identifiers combined (up to)", fontsize=9)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))   # whole percents
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.set_ylim(0, max(0.05, float(d["ci_high"].max()) * 1.1))
            _style(ax)
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
        axes[0][0].set_ylabel("Sampled users unique on ≥1 combination", fontsize=9, color=TEXT_SECONDARY)
        handles, labels = axes[0][0].get_legend_handles_labels()
        big = None if pop is None or pop[pop["dataset"] == ds].empty else int(pop.loc[pop["dataset"] == ds, "target_users"].iloc[0])
        if big:
            handles.append(plt.Line2D([], [], color=TEXT_SECONDARY, linewidth=1.5, linestyle=(0, (4, 2))))
            labels.append(f"Modelled, all {big:,} eligible users")
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
        n = int(d["sample_users"].iloc[0])
        fig.suptitle(f"{DATASET_LABELS.get(ds, ds)}: within-sample uniqueness (n = {n:,} users, 95% bootstrap CI)",
                     x=0.01, ha="left", fontsize=11, color=TEXT_PRIMARY)
        fig.tight_layout(rect=(0, 0.07, 1, 0.92))
        name = f"uniqueness_by_n_qi_{ds}"
        for ext in ("png", "pdf"):
            fig.savefig((fig_dir / name).with_suffix(f".{ext}"), dpi=200)
        plt.close(fig)
        made.append(name)
    return made


METHODS_TEMPLATE = """# Methods summary (auto-generated {today}; adapt before use)

## Data
We analysed three public chat-log datasets, pinned to the following Hugging Face revisions:

{dataset_lines}

{roles_sentence} Conversations were restricted to those whose dataset-provided
language label was {languages}. Data were streamed column-wise from the Hub; only sampled
turns were retained locally, in an access-restricted directory outside the code repository.

## Sampling
{sampling_text}

{sampling_lines}

## Pre-existing redactions
Each dataset's own placeholders were located with the regular expressions listed in the
configuration file. Detections overlapping a placeholder were excluded from residual counts and
placeholder frequency is reported separately.

## Detection
Presidio Analyzer (v{presidio_version}) with spaCy `{spacy_model}`, score threshold
{threshold}, and Presidio's default entity mapping{org_note}. Turns were scanned in
{chunk}-character windows with {overlap}-character overlap, up to {max_chars} characters per
turn. A separate rule-based detector matched API keys, access tokens, private-key headers, and
hard-coded credentials, and another matched self-introductions ("my name is ...", "call me ..."),
reported as detector `selfid`. Detections were mapped onto direct and quasi-identifier categories
(configs/taxonomy.yaml). Detections matching common dummy or reserved values (e.g. example.com
addresses, 555 phone numbers, private IP ranges, payment-processor test cards) were excluded. Raw
span text was never written to disk; spans were stored as character offsets, category, length, and a
keyed HMAC-SHA256 of the normalised value.

Task type was assigned with keyword heuristics (not a validated classifier). As a sensitivity
analysis for fictional names, results are reported with and without conversations tagged as
story/role-play.

## Statistics
Prevalence is the design-weighted share of conversations with at least one residual detection.
95% confidence intervals use a stratified percentile bootstrap ({reps} replicates), resampling
conversations within strata.

{identifiability}## Limitations (summary)
Results are detector-reported prevalence and have not yet been validated against human labels, so
they mix true and false positives. Presidio was also the tool used to redact WildChat, so WildChat's
Presidio residual is low by construction. See docs/limitations.md.
"""

SAMPLING_PER_CONVERSATION = """For each dataset we drew a stratified random sample (target n = {targets}; seed {seed}). Where a
user proxy exists (WildChat `hashed_ip`, Arena `judge`), we first selected one conversation uniformly
at random per proxy, so prevalence estimates approximate a per-user rather than per-conversation
rate; LMSYS-Chat-1M has no proxy and is sampled per conversation. Strata were defined by
{strata} with proportional (largest-remainder) allocation, and design weights N_h / n_h were applied."""

SAMPLING_USER_LEVEL = """We sampled users rather than conversations (seed {seed}). Conversations were grouped by the
dataset's own user proxy (WildChat `hashed_ip`, Arena `judge`), which we replace with a keyed hash on
read; no other information links conversations. From users with at least {min_conv} eligible
conversations we drew up to {n_users} uniformly at random and kept all of their conversations, up
to {cap} per user (a random subset above that cap; such heavy users are likely shared or NAT
addresses). Datasets without a user proxy were not sampled. Conversation-level tables weight each
conversation by (eligible users / sampled users) x (the user's conversations / conversations kept);
their confidence intervals resample conversations and ignore clustering within users."""

IDENTIFIABILITY_TEMPLATE = """## LLM extraction of quasi-identifiers
For {llm_turns:,} sampled user turns ({llm_scope}), a language model ({backend} `{model}`, prompt
`qi_extract_{prompt_version}`, {hosted_note}) was given the turn, with the preceding turn as
context, and asked for the quasi-identifying attributes it reveals: type, value, whether it was
stated or inferred, whom it is about, and a confidence. Responses were validated against a strict
schema; a malformed response was retried once and otherwise recorded as a failure ({llm_failures:,}
of {llm_turns:,} turns). Turns longer than {max_turn_chars:,} characters were truncated. Extracted
values were stored only in the access-restricted directory.

## Per-user profiles and uniqueness
For each user we kept attributes about the user with confidence >= {min_confidence} and built a
profile on eleven dimensions. Six were generalised with fixed lookup tables (configs/coarsening.yaml)
so that different wordings of one fact match: state or region, country, occupation (SOC minor group
where available, else major group), 10-year age band, gender and education level; a value the
tables do not cover was set to `unknown`. Five have no table and were kept as stated (normalised
text): employer type, school, relationship status, nationality and native language. We report two
variants (stated attributes only; stated plus inferred). When a user's conversations disagreed we
took the majority value and treated ties as ambiguous.

For every combination of one to {max_combo} dimensions we computed the equivalence-class size k
among users with a usable value for each dimension in the combination, and report the share of
those users with k = 1 and k < 5 and the median k, with {reps}-replicate bootstrap intervals that
resample users. We also report the share of all sampled users who are unique on at least one
combination, overall and by number of conversations.

Uniqueness is measured within the sample and therefore overstates uniqueness in the population of
all users; we report it at several sample sizes to show how it falls as the sample grows (see Rocher,
Hendrickx & de Montjoye, 2019, for the population-level argument). To estimate uniqueness among all
eligible users we modelled the chance that another user matches a given user on a combination as
the product of the leave-one-out shares of sampled users with the same value on each dimension
(region and country jointly, the other dimensions independently), and took a user to be unique
among M users when none of the other M - 1 would match. The same model evaluated at the sample
sizes we observed is reported next to the observed shares as a check. Cells based on fewer than
{min_cell} users are suppressed.

To test whether these attributes would let records be linked, we split each user's conversations at
random into two halves, built a profile from each half, and matched every first half against all
second halves, scoring a shared value by the negative log of its share among second halves. A user
is linked when their own second half is the single best match. We repeated this with direct
identifiers detected in user turns (self-stated names, email addresses, phone numbers, credentials,
compared by keyed hash) and with both together, and averaged over repeated random splits. No outside
data was used, and we did not attempt to identify any user.

"""


def methods_summary(cfg: dict, manifest: dict, results_dir: Path) -> None:
    loaded = {k: m for k, m in manifest["datasets"].items() if "repo" in m}
    skipped = [k for k, m in manifest["datasets"].items() if "repo" not in m]
    ds_lines = "\n".join(f"- `{m['repo']}` @ `{m['revision'][:12]}`" for m in loaded.values())
    if skipped:
        ds_lines += f"\n\n(Not loaded in this run: {', '.join(skipped)}.)"
    smp = "\n".join(
        f"- {DATASET_LABELS.get(k, k)}: {m['sampling'].get('eligible_conversations', 0):,} eligible conversations, "
        f"sampling frame {m['sampling'].get('sampling_frame', 0):,}, sample {m['sampling'].get('sample_size', 0):,}"
        for k, m in loaded.items())
    p = cfg["presidio"]
    roles = cfg.get("roles") or ["user"]
    roles_sentence = ("Only user turns were analysed." if roles == ["user"] else
                      "Prevalence tables cover user turns; assistant turns were also scanned and are "
                      "reported separately (prevalence_by_role.csv).")
    if manifest.get("sampling_mode") == "user_level":
        u = cfg["user_level"]
        sampling_text = SAMPLING_USER_LEVEL.format(seed=cfg["seed"], min_conv=u["min_conversations"],
                                                   n_users=f"{u['n_users']:,}", cap=u["max_conversations_per_user"])
        smp = "\n".join(
            f"- {DATASET_LABELS.get(k, k)}: {m['sampling'].get('eligible_users', 0):,} eligible users, "
            f"{m['sampling'].get('sampled_users', 0):,} sampled, {m['sampling'].get('sample_size', 0):,} "
            f"conversations; eligible users above the cap: {m['sampling'].get('users_over_cap')}"
            for k, m in loaded.items())
    else:
        sampling_text = SAMPLING_PER_CONVERSATION.format(
            targets=", ".join(f"{k}={v['sample_size']}" for k, v in cfg["datasets"].items() if v.get("enabled")),
            seed=cfg["seed"], strata=" × ".join(cfg["sampling"]["strata"]))
    identifiability = ""
    if "llm" in manifest:
        m, lcfg = manifest["llm"], cfg["llm"]
        identifiability = IDENTIFIABILITY_TEMPLATE.format(
            llm_turns=m["turns_processed"], llm_failures=m["failures"], backend=m["backend"], model=m["model"],
            prompt_version=m["prompt_version"],
            llm_scope="all sampled user turns" if lcfg.get("sample_turns") is None
            else f"a random subsample capped at {lcfg['sample_turns']:,}",
            hosted_note="a hosted API, used with explicit opt-in" if m["hosted"] else "a test stub",
            max_turn_chars=lcfg["max_turn_chars"], min_confidence=lcfg["min_confidence"],
            max_combo=cfg.get("uniqueness", {}).get("max_combination_size", 4),
            reps=cfg["analysis"]["bootstrap_reps"], min_cell=cfg.get("publication", {}).get("min_cell", 10))
    text = METHODS_TEMPLATE.format(
        today=date.today().isoformat(), dataset_lines=ds_lines, languages=cfg.get("languages") or "any",
        roles_sentence=roles_sentence, sampling_text=sampling_text, sampling_lines=smp,
        identifiability=identifiability,
        presidio_version=_presidio_version(), spacy_model=p["spacy_model"],
        threshold=p["score_threshold"],
        org_note="" if p.get("include_organization") else " (ORGANIZATION ignored, as in Presidio's defaults)",
        chunk=p["chunk_chars"], overlap=p["chunk_overlap"], max_chars=p["max_chars_per_turn"],
        reps=cfg["analysis"]["bootstrap_reps"])
    (results_dir / "methods.md").write_text(text)


DISCLOSURE_TEMPLATE = """Subject: Residual personal information in {repo} — aggregate findings

Dear {label} maintainers,

As part of a class research project (with instructor approval) on de-identification in public
LLM chat-log datasets, we measured residual personal information in a random sample of
{n:,} English conversations from {repo} (revision {rev}).

Using Presidio (v{pv}) plus rule-based secret detection, and after excluding your existing
redaction placeholders and obvious dummy values, we estimate (design-weighted, 95% CI):

{lines}

These are automated detector counts and include false positives; they have not yet been validated
against human labels. We have not attempted to identify any individual, linked any data to
external sources, or tested any credential.

We are not attaching examples. If useful, we can share the list of affected conversation IDs through
a private channel of your choosing so you can review them.

Best regards,
[Name, affiliation, contact]
"""


def disclosure_drafts(prev: pd.DataFrame, manifest: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = prev[(prev["detector"] == "any") & (prev["variant"] == "all")]
    for ds, m in manifest["datasets"].items():
        if "repo" not in m:
            continue
        d = p[p["dataset"] == ds].sort_values("prevalence", ascending=False)
        if d.empty:
            continue
        d = d[d["category"] != "tier_url_unfiltered"]
        lines = "\n".join(f"- {TIER_LABELS.get(r.category, r.category.replace('_', ' '))}: {r.prevalence:.1%} "
                          f"({r.ci_low:.1%}–{r.ci_high:.1%}) of conversations"
                          for r in d.itertuples() if r.n_positive > 0)
        (out_dir / f"{ds}.md").write_text(DISCLOSURE_TEMPLATE.format(
            repo=m["repo"], label=DATASET_LABELS.get(ds, ds), n=int(d["n_conversations"].iloc[0]),
            rev=m["revision"][:12], pv=_presidio_version(), lines=lines or "- none"))


def write_manifest(manifest: dict, results_dir: Path) -> None:
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
