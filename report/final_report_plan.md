# Final report: rewrite plan

Source of every number below: `results/default/tables/` after template exclusion (2026-11-15).
`midterm_report.tex` stays as is; the final report goes in a new `final_report.tex`.

## What changed since the midterm

1. **Shared prompt templates are now left out of every estimate.** One pasted prompt generator
   appears in 1,089 of the 5,000 sampled WildChat conversations, each from a different hashed IP.
   With the four WildChat templates, 1,119 conversations (22.4% weighted) are excluded; LMSYS 329
   (6.6%), Arena 5 (0.1%). Counts: `templates.csv`. The old numbers survive as variant
   `incl_templates` in `prevalence.csv` / `llm_prevalence.csv`.
2. **Human labels exist for WildChat** (kai, 300 items; 260 after templates). Precision/recall in
   `labels_precision_recall.csv` (gold = kai), gold prevalence in `labels_gold_prevalence.csv`.
3. **Human–LLM agreement** is in `labels_agreement_wildchat.csv` (kai vs Claude Sonnet 5.5).

## Numbers that change in the report (WildChat unless stated)

| Claim in midterm | Midterm | Now (excl. templates) |
|---|---|---|
| Presidio flags a name | 44.4% | **28.8%** [27.5–30.2] |
| Any direct identifier (Presidio + rules) | 44.5% | **29.1%** [27.8–30.5] |
| … excl. story/role-play | 26.1% | 26.1% (unchanged) |
| Any quasi-identifier | 53.8% | **40.5%** |
| LLM, private name | 3.2% | **4.1%** [3.5–4.7] |
| LLM, private direct identifier | 4.5% | **5.8%** [5.0–6.5] |
| LLM, private quasi-identifier | 4.7% | **6.0%** |
| LLM, all-spans direct identifier | 18.4% | 18.8% |
| API key / credential (rules) | 0.22% | 0.28% (≈1 in 350 users) |
| Has dataset placeholder | 0.7% | 1.0% |
| LMSYS / Arena, Presidio name | 12.0% / 14.8% | 12.3% / 14.8% |
| LMSYS / Arena, LLM private direct | 1.2% / 1.3% | 1.3% / 1.3% |

LLM-only categories (private, WildChat / LMSYS / Arena): employer/school 2.2 / 0.8 / 0.3%,
occupation 1.3 / 0.3 / 0.3%, education 1.3 / 0.2 / 0.2%, username 1.1 / 0.2 / 0.1%,
health 0.7 / 0.6 / 0.4%, age 0.7 / 0.5 / 0.4%, street address 0.2 / 0.2 / 0.1%,
date of birth 0.2 / 0.2 / 0.1%.

**Claims to rewrite, not just renumber:**
- *"Fiction drives much of the gap (44.5% → 26.1%)."* Wrong: the template was tagged story/role-play.
  Excluding templates first, fiction only moves 29.1% → 26.1%. The template drove the gap.
- *"WildChat 3–4× the comparison datasets."* Presidio gap shrinks to ~2× (29% vs 12–15%). Private
  rate gap *grows*: 5.8% vs 1.3% (≈4.5×). Real private PII, not flags, is what separates WildChat.
- *"Private is roughly a tenth of what Presidio flags."* Now ~a fifth (5.8% vs 29.1%). The human
  labels give a sharper version: Presidio precision 2.6% (below).

## Human-label results (RQ1), WildChat, threshold 0.5, weighted, gold = kai

| Detector | Precision | Recall |
|---|---|---|
| Presidio | 2.6% [1.3–4.4] (721 flags) | 35.9% [20.2–58.0] |
| Pattern rules (secrets) | 40% [0–100] (5 flags) | 0.2% |
| LLM, private | 40.1% [20.9–57.6] (78) | 34.6% [14.7–59.0] |
| LLM, all spans | 11.6% [5.2–19.6] (253) | 64.5% [46.3–81.6] |
| Union of all | 6.7% [3.2–11.1] | 74.1% [60.7–88.4] |

- Names: Presidio precision 1.4% (226 flags); recall ~75% but only 6 real names in gold.
- Gold prevalence: real direct identifier in 1.1% of eligible turns [0.3–2.1], quasi 4.0% [1.8–6.7];
  organization 2.4%, occupation 2.4%.
- Remaining near-duplicates in the label set (2 pairs, 4 items) change Presidio precision from
  2.60% to 2.61%: no effect.

## Agreement, kai vs LLM (237 items both labelled; unweighted)

| Item-level: turn contains private … | Cohen's κ |
|---|---|
| any direct identifier | 0.68 (97.0% raw agreement) |
| any quasi-identifier | 0.65 (95.8%) |
| name | 0.72 |
| API key / credential | 0.80 |
| employer / organization | 0.83 |
| education | 1.00 |
| health | 0.66 |
| occupation | 0.41 |
| location | 0.28 |

Span F1 over *all* spans is 0.16: kai marked only the writer's own information, the LLM marked
fictional, public and not-personal spans too, so span-level and `whose` agreement are not
comparable. Report the item-level private κ only and say why.

LaTeX table to drop in:

```latex
\begin{table}
  \caption{Agreement between the human annotator and the LLM annotator on whether a WildChat user
    turn contains private information (237 turns, unweighted).}
  \label{tab:agreement}
  \small
  \begin{tabular}{lrr}
    \toprule
    Turn contains private \ldots & Agreement & Cohen's $\kappa$ \\
    \midrule
    Any direct identifier & 97.0\% & 0.68 \\
    Any quasi-identifier & 95.8\% & 0.65 \\
    \quad Name & & 0.72 \\
    \quad API key / credential & & 0.80 \\
    \quad Employer / school & & 0.83 \\
    \quad Occupation & & 0.41 \\
    \quad Location & & 0.28 \\
    \bottomrule
  \end{tabular}
\end{table}
```

## Section-by-section plan

| Section | Action |
|---|---|
| Title / subtitle | Drop "Midterm Project Report". Title can stay. |
| Abstract | Rewrite as findings: Presidio precision 2.6%; private direct identifiers in 5.8% of WildChat conversations vs 1.3% in LMSYS/Arena; a fifth of WildChat's sample is one template; LLM detector agrees with a human at κ≈0.65–0.68. |
| 1 Problem + RQs | Keep. Trim RQ4 to "future work" or drop it (no sanitizer was built). Decide RQ3 (below). |
| 1.3 Literature | Keep; minor tense edits. |
| 2 Planned Methods → **Methods** | Past tense. Add: template exclusion (rule, counts); label set (300 turns, 4 strata, inclusion weights, ≤4,000 chars, 260 after templates); single annotator + LLM as second annotator; private = self or third party; matching rule (`min_overlap`). Remove "two annotators" and "adjudicated". Cut Step 3 (sanitizer). |
| Ethics | Keep; add that the label set (900 turns) and full sample were sent to Anthropic. |
| Preliminary results → **Results** | New structure: (a) templates finding; (b) RQ1 precision/recall table + agreement table; (c) RQ2 residual PII by type (Tables 1–3 renumbered, Figure 1 regenerated); (d) dataset comparison; (e) RQ3 if included. |
| Figure 1 | Replace `report/figures/residual_by_category_log.png` with a regenerated one from the new `prevalence.csv`. Its caption *n* values change (WildChat 3,881 conversations total). |
| Next Steps / Expected Output → **Discussion, Limitations, Conclusion** | Discussion: why Presidio's precision is so low (fiction, public figures, code identifiers, templates); what Presidio misses (employer, occupation, handles, credentials). Limitations: single annotator who marked only self spans (third-party PII likely undercounted → recall overstated); 6 real names in gold → wide CIs; ≤4,000-char turns only; LMSYS/Arena have LLM labels only; LLM detector graded partly by an LLM. Conclusion: 3–4 sentences. |
| Appendix | Keep; add the template rule snippet (`labeling.templated_conversations`). |

## Open decisions

1. **RQ3 (profiles / uniqueness / linkability):** Ethan's `origin/ethan/identifiability-wip` has a
   2,000-user run and a split-half linkage test, unmerged. Include its results (need numbers from
   Ethan), or move RQ3 to future work.
2. **Third-party spans:** add them in a ~20-item pass (items the LLM marked third party), or define
   gold as the writer's own information and state it.
3. **LMSYS/Arena label scores against the LLM:** need a small change so `label score` can use a
   different gold per dataset without overwriting WildChat's tables. Only if the comparison section
   needs precision/recall there.
