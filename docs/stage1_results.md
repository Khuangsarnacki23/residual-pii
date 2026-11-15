# Stage 1 results: residual PII in WildChat and how well Presidio finds it

Source tables: `results/default/tables/labels_*.csv` (aggregate only). Gold labels:
`labels/wildchat/adjudicated.jsonl`. All numbers are WildChat only; LMSYS and Arena have no label
sets yet. The LLM detector is Claude Haiku 5.5 (effort low) run over all 11,436 sampled user turns
under the annotators' guideline ($0.93; one turn refused).

## Summary

After WildChat's own Presidio scrubbing, about 1 in 13 user turns still contains private
quasi-identifying information and about 1 in 60 a private direct identifier. Against a jointly
adjudicated gold set, Presidio's flags are about 96% false positives and it finds under a third of
the real private spans. An LLM detector is about sixteen times more precise (60%) but finds no more
of the residual (29%); neither tool finds most occupation or education mentions.

Recall here is measured on text WildChat had already scrubbed with Presidio. It is the share of the
*remaining* PII a second pass would recover, not a tool's recall on raw text, which is not available.

## Gold set

- 300 WildChat user turns, stratified by what the detectors flagged (rare direct identifier, name,
  quasi-identifier only, nothing), with inclusion weights so estimates cover all eligible turns.
- Labelled blind and independently by two annotators (kai, ethan) under the guideline in
  `src/residpii/llm_label.py`.
- Adjudication: 238 items where both annotators' spans were identical were accepted automatically;
  the other 62 were resolved jointly. Gold attributes come from one annotator (ethan); the other
  annotator marked none.
- 40 items are shared prompt templates (e.g. a Midjourney prompt reused by about 1,000 users) and are
  excluded from scoring, leaving 260 scored items with 94 private spans.
- "Private" means whose = self or third party. Fictional, public-figure and non-personal spans are
  labelled but do not count as PII.

## Inter-annotator agreement (before adjudication, 242 items compared)

| Measure | Value |
|---|---|
| Cohen's kappa, item has a private direct identifier | 0.62 |
| Cohen's kappa, item has a private quasi-identifier | 0.41 |
| Span F1, all spans | 0.31 |

Agreement was high for names (κ 0.80), organizations (0.80), IP addresses (0.66) and health (0.66),
and near zero for occupation and location, which were the main topics of adjudication.

## Prevalence of private PII (weighted, 95% bootstrap CI)

| Category | Share of turns | 95% CI |
|---|---|---|
| Any private direct identifier | 1.6% | 0.5–2.6% |
| Any private quasi-identifier | 7.6% | 4.3–11.5% |
| Occupation | 4.5% | 1.9–7.9% |
| Organization | 2.2% | 0.3–4.6% |
| Education | 2.1% | 0.3–4.6% |
| Location | 1.6% | 0.5–2.8% |
| Name | 1.1% | 0.2–1.9% |

Quasi-identifiers are roughly five times as common as direct identifiers. Occupation and education,
the two most common, have no Presidio entity type at all.

## Presidio against the gold set (default threshold 0.5)

| | Precision | 95% CI | Recall | 95% CI |
|---|---|---|---|---|
| All categories | 3.8% | 2.1–5.9% | 31% | 19–46% |
| Names | 1.9% | 0.4–4.2% | 66% | 20–100% |
| Locations | 5.5% | 1.2–10.8% | 57% | 31–86% |
| IP addresses | 82% | 0–100% | 45% | 42–100% |

- Changing the threshold barely matters: across 0.3–0.85, precision is 3.5–4.3% and recall 28–31%.
- Most name flags fall on text the annotators did not mark at all; of the flagged names that were
  marked, most are fictional characters or public figures (`labels_flag_breakdown.csv`).
- The secrets rules fired 5 times (API keys/tokens): precision 40%, recall 50%, from 4–5 cases.

## LLM detector against the gold set

`llm_detector` counts only spans the model judged private (self or third party);
`llm_detector_all` counts every span it marked, whoever it belongs to.

| Detector | Precision | 95% CI | Recall | 95% CI | Flags |
|---|---|---|---|---|---|
| Presidio (0.5) | 3.8% | 2.1–5.9% | 31% | 19–46% | 721 |
| LLM, private spans | 60% | 43–84% | 29% | 13–46% | 46 |
| LLM, all spans | 14% | 7–22% | 49% | 32–66% | 248 |
| Presidio or LLM (any flag) | 8.8% | 4.8–13.5% | 57% | 40–74% | 1,020 |

- The LLM's gain is precision: it separates the writer and real third parties from characters and
  public figures (names: 81% precision against Presidio's 1.9%, at higher recall, 84% vs 66%).
- It does not recover most of the residual. Occupation (recall 0% private, 14% all spans) and
  education (0%) are missed by every tool. Health is the exception (recall 100%, 5 gold spans).
- It found 90% of IP addresses but did not mark them as private, so they count only under
  `llm_detector_all`.
- Full-sample LLM prevalence (`llm_prevalence.csv`) is per conversation and inherits these errors
  (29% recall, 60% precision), so it is not a substitute for the gold estimates above.

### Attribute extraction (input to Stage 2)

`identifiability validate` compares the LLM's per-turn attributes with the gold attributes after the
coarsening Stage 2 uses (stated plus inferred): precision 40%, recall 29% over 14 gold items with an
attribute; occupation recall 11% (1 of 9). Gold attributes come from one annotator and are sparse,
so this is indicative only. Profiles built from this extractor will under-count what users reveal,
which makes Stage 2 uniqueness a likely lower bound.

## Limitations

- One dataset, 260 scored items: several categories rest on 1–10 positive cases, and per-category
  intervals are wide. The overall precision and the direct-versus-quasi gap are the robust findings.
- Moderate agreement on quasi-identifiers (κ 0.41) means the gold set itself carries judgement,
  mostly about occupation and location.
- Estimates exclude shared prompt templates, so they describe non-templated turns.
- The LLM was run with the cheapest model and effort; a stronger model may recall more.
