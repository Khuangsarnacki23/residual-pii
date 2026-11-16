# Final report: rewrite plan

Every number below comes from `results/default/tables/` (Stage 1, full sample, sanitizer) and
`results/user_level/tables/` (Stage 2, pilot), after template exclusion. The finished synthesis with
charts is in `docs/findings.md`, and the figures are in `results/presentation/figures/`.
`midterm_report.tex` stays as it is; the final report goes in a new `final_report.tex`.

**Gold standards**
- **Stage 1 label set** (260 scored WildChat turns): the **adjudicated** set,
  `labels/wildchat/adjudicated.jsonl`. Two annotators (kai, ethan) labelled blind, and the
  disagreements were resolved jointly.
- **Sanitizer hand review** (100 fresh turns, `residpii review`): **Kai's review judgments** are the
  gold set for this second pass, with a single reviewer.

**Research questions** (agreed numbering)
1. RQ1: residual PII by type.
2. RQ2: detector accuracy.
3. RQ3: per-user profiles and linkage.
4. RQ4: the sanitizer.

## What changed since the midterm

1. **Shared prompt templates are left out of every estimate.**
   - One pasted prompt generator appears in 1,089 of the 5,000 sampled WildChat conversations, each from a different hashed IP.
   - With the four WildChat templates, 1,119 conversations (22.4% weighted) are excluded. LMSYS loses 329 (6.6%) and Arena 5 (0.1%). Counts are in `templates.csv`.
   - The old numbers survive as the `incl_templates` variant in `prevalence.csv` and `llm_prevalence.csv`.
2. **The WildChat gold set is double-labelled and adjudicated.**
   - 300 items; 260 remain after templates are removed, with 94 private spans.
   - Precision and recall are in `labels_precision_recall.csv` and gold prevalence in `labels_gold_prevalence.csv`, both with gold = adjudicated.
3. **Agreement.**
   - Between the two humans: `labels_agreement_wildchat.csv` (kai~ethan).
   - Between the adjudicated set and the Claude Sonnet 5.5 annotator: `labels_agreement_wildchat_llm.csv`.
4. **Stage 2 (RQ3) is done.** 1,982 users were profiled, with uniqueness and split-half linkage computed. See `docs/stage2_results.md`.
5. **The sanitizer (RQ4) has two parts.**
   - Kai's layered sanitizer, scored on the adjudicated gold set: `sanitizer_*.csv`.
   - A 50-user pilot comparing the user-level loop with the per-message loop: `adversarial_summary.csv`, written up in `docs/adversarial_pilot.md`.

## Numbers that change in the report (WildChat unless stated)

LLM detector rows come from our Haiku 5.5 run over all 11,436 sampled WildChat turns. The LMSYS
and Arena LLM rows come from Kai's run, the only one on those datasets. A second Haiku run over
WildChat gave 5.8% for private direct identifiers, so quote one run and note the variation.

| Claim in midterm | Midterm | Now (excl. templates) |
|---|---|---|
| Presidio flags a name | 44.4% | **28.8%** [27.5–30.2] |
| Any direct identifier (Presidio + rules) | 44.5% | **29.1%** [27.8–30.5] |
| … excl. story/role-play | 26.1% | 26.1% (unchanged) |
| Any quasi-identifier | 53.8% | **40.5%** |
| LLM, private name | 3.2% | **4.1%** [3.4–4.7] |
| LLM, private direct identifier | 4.5% | **5.5%** [4.8–6.3] |
| LLM, private quasi-identifier | 4.7% | **5.8%** [5.1–6.6] |
| LLM, all-spans direct identifier | 18.4% | 19.2% |
| API key / credential (rules) | 0.22% | 0.28% (≈1 in 350 users) |
| Has dataset placeholder | 0.7% | 1.0% |
| LMSYS / Arena, Presidio name | 12.0% / 14.8% | 12.3% / 14.8% |
| LMSYS / Arena, LLM private direct | 1.2% / 1.3% | 1.3% / 1.3% |

LLM-only categories, private, as WildChat / LMSYS / Arena:

| Category | WildChat | LMSYS | Arena |
|---|---|---|---|
| Employer / school | 2.2% | 0.8% | 0.3% |
| Occupation | 1.2% | 0.3% | 0.3% |
| Education | 1.2% | 0.2% | 0.2% |
| Username | 1.1% | 0.2% | 0.1% |
| Health | 0.6% | 0.6% | 0.4% |
| Age | 0.7% | 0.5% | 0.4% |
| Street address | 0.2% | 0.2% | 0.1% |
| Date of birth | 0.2% | 0.2% | 0.1% |

**Claims to rewrite, not just renumber:**
- *"Fiction drives much of the gap (44.5% → 26.1%)."* This is wrong: the template was tagged
  story/role-play. With templates excluded first, fiction only moves 29.1% to 26.1%. The template
  drove the gap.
- *"WildChat 3–4× the comparison datasets."* The Presidio gap shrinks to about 2× (29% against
  12–15%). The private-rate gap *grows* to 5.5% against 1.3%, about 4×. Real private PII, not
  flags, is what separates WildChat.
- *"Private is roughly a tenth of what Presidio flags."* It is now about a fifth (5.5% against
  29.1%). The gold set gives a sharper version: Presidio's precision is 3.8% (below).

## Gold-set results (RQ1, RQ2): WildChat, threshold 0.5, weighted, gold = adjudicated

| Detector | Precision | Recall |
|---|---|---|
| Presidio | 3.8% [2.1–5.9] (721 flags) | 31% [19–46] |
| Pattern rules (secrets) | 40% [0–100] (5 flags) | 0.1% of all private spans |
| LLM, private | 60% [43–84] (46 flags) | 29% [13–46] |
| LLM, all spans | 14% [7–22] (248 flags) | 49% [32–66] |
| Union of all | 8.8% [4.8–13.5] (1,020 flags) | 57% [40–74] |

- **Names.** Presidio's precision is 1.9% (226 flags) and its recall 66%, on 9 real names. The
  LLM's precision on names is 81%.
- **Gold prevalence.** A private direct identifier appears in 1.5% of eligible turns [0.5–2.6] and
  a private quasi-identifier in 7.6% [4.3–11.5]. Occupation is 4.5%, organization 2.1% and
  education 2.1%. Occupation and education have no Presidio entity type, and every tool scores
  0–14% recall on them.
- **Ownership.** Third-party spans are 9 of the 94 private spans. Counting only the writer's own
  spans moves LLM precision to 54% and recall to 25%, inside the confidence intervals.
- **What the LLM misses.** It found 90% of IP addresses but judged them not private. Its recall
  rises from 29% to 49% if ownership is ignored, so the main disagreement is *whose* information it
  is.

## Agreement (unweighted; 242 items compared between the humans, 237 with the LLM)

| Item-level: turn contains private … | Human–human κ (kai~ethan) | Adjudicated–LLM κ |
|---|---|---|
| any direct identifier | 0.62 (97.1%) | 0.79 (97.9%) |
| any quasi-identifier | 0.41 (93.8%) | 0.71 (95.4%) |
| name | 0.80 | 0.83 |
| API key / credential | | 0.80 |
| organization | 0.80 | 0.83 |
| education | | 0.75 |
| health | 0.66 | 1.00 |
| occupation | ≈0 | 0.69 |
| location | ≈0 | 0.76 |

- Span F1 over all spans is 0.31 between the humans and 0.37 between the adjudicated set and the
  LLM.
- The humans disagreed mostly on occupation and location, which is what adjudication resolved.
- The LLM annotator agrees with the adjudicated set better than either human agreed with the
  other. Report this as agreement with an LLM annotator, not as ground truth.

## Per-user identifiability (RQ3), 1,982 WildChat users

- **Disclosure.** 13.8% of users reveal at least one attribute, after coarsening to occupation
  group, country, age band and similar categories.
- **Uniqueness.** 5.9% of users [4.8–6.9] are unique on up to four attributes. The population model
  gives 4.8% of the 25,325 eligible users. One attribute alone singles out 2%.
- **Activity.** Uniqueness rises from 2.7% of users with 3 conversations to 15.2% of those with
  more than 50.
- **Linkage.** For 2.1% of users, the two halves of their conversations can be tied together from
  content (chance: 0.05%). This reaches 9.8% for users with 26–50 conversations.
- **Robustness.** Dropping repeated-prompt turns, 13% of the turns, leaves uniqueness at 5.9%.
- **Caveat.** The attribute extractor has 29% recall and 40% precision on gold attributes, so
  uniqueness is not a clean bound in either direction.

## Sanitizer (RQ4)

**Layered sanitizer on the gold set** (gold = adjudicated; layers are cumulative; `sanitizer_leakage.csv`, `sanitizer_overredaction.csv`)

| Stage | Direct removed (of 41) | Quasi removed (of 53) | Clean turns changed | Rated worse (<4/5) | Attacker's original inferences still made |
|---|---|---|---|---|---|
| Presidio re-run | 74% | 3% | 19.7% | 7.7% | 50% |
| Layer 1: patterns | 5% | 0% | 0.1% | 0.0% | 55% |
| Layers 1–2: + LLM identifiers | 62% | 0% | 0.1% | 0.0% | 55% |
| Layers 1–3: + LLM generalise | 84% | 12% | 9.9% | 0.3% | 46% |
| Layers 1–4: + adversarial loop | 84% | 63% [39–83] | 14.4% | 3.2% | 5% |
| No LLM: rule identifiers | 38% | 1% | 6.1% | 0.7% | 53% |
| No LLM: + rule generalise | 57% | 12% | 6.3% | 1.7% | 48% |

- Sources: `sanitizer_exposure.csv` and `sanitizer_utility.csv` alongside the two tables above.
- Layers 3–4 cost $4.93 for the 260 turns.
- The held-out attacker is Sonnet 5.5, and it made 52 confident inferences on the original text.
  With all four layers, turns with any confident inference fall from 10.5% to 5.8%.
- Removal is an exact-string check, so paraphrases count as removed. Lean on the attacker column
  for what stays inferable.
- Headline: only the adversarial layer removes most quasi-identifiers. The full pipeline still
  changes fewer clean turns and damages fewer requests than a Presidio re-run.

**Hand review of 100 fresh turns** (gold = Kai's review judgments; `review_scores_fresh.csv`)

| Stage | Precision | Pooled recall | Clean turns edited |
|---|---|---|---|
| Presidio | 13% | 41% | 12% |
| Layers 1–2: LLM identifiers | 71% | 24% | 0.3% |
| No LLM: rule identifiers | 57% | 23% | 1.5% |
| No LLM: + rule generalise | 61% | 34% (quasi: 38%) | 2.4% |

**User-level adversarial pilot** (50 users; a held-out Sonnet attacker grades each user's whole history)

| Version | Still at risk (of 23) | Messages rewritten | Rewrites rated ≥4 |
|---|---|---|---|
| Per-message (Staab et al.) | 13 | 14.4% | 76% |
| User-level, strict stop | 17 | 8.3% | 100% |
| User-level, improved rewrites + passages | 17 | 10.2% | 100% |

The difference in users protected is not significant (p = 0.73). The user-level loop edits about
half as much. A stronger rewrite step (gentle retries, more rounds, long messages rewritten passage
by passage) did not change protection (17 still at risk): for most remaining users every rewrite that
hid the identifying detail broke the request, because the request needs that detail.

LaTeX table to drop in:

```latex
\begin{table}
  \caption{Agreement on whether a WildChat user turn contains private information (242 and 237 turns,
    unweighted): between the two human annotators, and between the adjudicated gold set and an LLM
    annotator (Claude Sonnet 5.5).}
  \label{tab:agreement}
  \small
  \begin{tabular}{lrr}
    \toprule
    Turn contains private \ldots & Human--human $\kappa$ & Gold--LLM $\kappa$ \\
    \midrule
    Any direct identifier & 0.62 & 0.79 \\
    Any quasi-identifier & 0.41 & 0.71 \\
    \quad Name & 0.80 & 0.83 \\
    \quad Organization & 0.80 & 0.83 \\
    \quad Health & 0.66 & 1.00 \\
    \quad Occupation & 0.00 & 0.69 \\
    \quad Location & 0.00 & 0.76 \\
    \bottomrule
  \end{tabular}
\end{table}
```

## Section-by-section plan

| Section | Action |
|---|---|
| Title / subtitle | Drop "Midterm Project Report". The title can stay. |
| Abstract | Rewrite as findings. 1 in 13 turns still holds a private quasi-identifier after WildChat's scrubbing. Presidio's precision is 3.8% and its recall 31%; the LLM's precision is 60% with no better recall. 6% of users are unique across their conversations. Identifier redaction leaves almost every quasi-identifier, and a user-level sanitizer edits half as much as the per-message loop. |
| 1 Problem + RQs | Keep, with the agreed numbering: RQ3 is profiles and linkage, RQ4 is the sanitizer. Keep all four. |
| 1.3 Literature | Keep, with minor tense edits. |
| 2 Planned Methods → **Methods** | Past tense. Add template exclusion (the rule and the counts). Add the label set: 300 turns, 4 strata, inclusion weights, at most 4,000 characters, 260 after templates. Add **two blind annotators plus joint adjudication**, private = self or third party, and the matching rule (`min_overlap`). Add the Stage 2 user-level sample, coarsening, uniqueness and linkage. Add the sanitizer layers, the hand review (one reviewer, Kai) and the adversarial pilot design. |
| Ethics | Keep. Add that the label set (900 turns), the full sample and the pilot users' turns were sent to Anthropic, that no identities were sought, and that only aggregates were published. |
| Preliminary results → **Results** | (a) the templates finding; (b) RQ1 gold prevalence; (c) RQ2 precision/recall and the agreement table; (d) the dataset comparison; (e) RQ3 uniqueness and linkage; (f) RQ4 gold-set layers, hand review and pilot. The figures are in `results/presentation/figures/`. |
| Figure 1 | Replace `report/figures/residual_by_category_log.png` with a figure regenerated from the new `prevalence.csv`. Its caption *n* values change (3,881 WildChat conversations in total). |
| Next Steps / Expected Output → **Discussion, Limitations, Conclusion** | **Discussion:** why Presidio's precision is so low (fiction, public figures, code identifiers, templates); what everything misses (occupation, education); and why the user, not the message, is the unit of risk. **Limitations:** moderate human agreement on quasi-identifiers (κ 0.41); 9 real names in gold, so CIs are wide; turns of at most 4,000 characters; LMSYS and Arena have LLM labels only; the LLM detector varies between runs; the attribute extractor is weak; `hashed_ip` is a rough user proxy; the hand review has one reviewer; the pilot has 50 users. **Conclusion:** 3–4 sentences. |
| Appendix | Keep. Add the template-rule snippet (`labeling.templated_conversations`). |

## Open items

1. LMSYS/Arena label scores against the LLM annotator need a small change, so that `label score`
   can use a different gold set per dataset without overwriting WildChat's tables. Do this only if
   the comparison section needs precision and recall there.
