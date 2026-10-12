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
    xmax = max(0.002, float(df["ci_high"].max(skipna=True) if df["ci_high"].notna().any() else df["prevalence"].max()))
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
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0 if xmax >= 0.05 else 1))
        ax.set_xlabel(xlabel, fontsize=9)
        _style(ax)
    axes[0][0].set_yticks(range(len(row_order)), [r.replace("_", " ") for r in row_order], color=TEXT_PRIMARY)
    axes[0][0].invert_yaxis()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title, x=0.01, y=0.99, ha="left", va="top", fontsize=11, color=TEXT_PRIMARY)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.95), ncol=len(datasets),
               frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
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
                      "Residual detections by category (design-weighted, 95% bootstrap CI)")
        made.append("prevalence_by_category")
        rare = [c for c in ("email", "phone", "account_id", "ip_address", "api_key_token") if c in cats]
        if rare:
            dot_ci_panels(p[p["category"].isin(rare)], "category", "variant", ["all", "excl_story_roleplay"],
                          VARIANT_LABELS, rare, "Conversations with ≥1 residual detection",
                          fig_dir / "prevalence_rare_direct", "Rare direct identifiers (zoomed scale)")
            made.append("prevalence_rare_direct")
    t = by_task[(by_task["detector"] == "any") & (by_task["category"] == "tier_direct")]
    if not t.empty:
        order = (t.groupby("task_type")["task_share"].mean().sort_values(ascending=False).index.tolist())
        dot_ci_panels(t.assign(panel="direct"), "task_type", "panel", ["direct"], {"direct": "Direct identifiers"},
                      order, "Conversations with ≥1 direct identifier", fig_dir / "direct_by_task",
                      "Direct-identifier prevalence by heuristic task type")
        made.append("direct_by_task")
    return made


METHODS_TEMPLATE = """# Methods summary (auto-generated {today}; adapt before use)

## Data
We analysed three public chat-log datasets, pinned to the following Hugging Face revisions:

{dataset_lines}

Only user turns were analysed. Conversations were restricted to those whose dataset-provided
language label was {languages}. Data were streamed column-wise from the Hub; only sampled user
turns were retained locally, in an access-restricted directory outside the code repository.

## Sampling
For each dataset we drew a stratified random sample (target n = {targets}; seed {seed}). Where a
user proxy exists (WildChat `hashed_ip`, Arena `judge`), we first selected one conversation uniformly
at random per proxy, so prevalence estimates approximate a per-user rather than per-conversation
rate; LMSYS-Chat-1M has no proxy and is sampled per conversation. Strata were defined by
{strata} with proportional (largest-remainder) allocation, and design weights N_h / n_h were applied.

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
hard-coded credentials. Detections were mapped onto direct and quasi-identifier categories
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

## Limitations (summary)
Results are detector-reported prevalence and have not yet been validated against human labels, so
they mix true and false positives. Presidio was also the tool used to redact WildChat, so WildChat's
Presidio residual is low by construction. See docs/limitations.md.
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
    text = METHODS_TEMPLATE.format(
        today=date.today().isoformat(), dataset_lines=ds_lines, languages=cfg.get("languages") or "any",
        targets=", ".join(f"{k}={v['sample_size']}" for k, v in cfg["datasets"].items() if v.get("enabled")),
        seed=cfg["seed"], strata=" × ".join(cfg["sampling"]["strata"]), sampling_lines=smp,
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
