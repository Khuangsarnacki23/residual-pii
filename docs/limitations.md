# Risks and limitations (draft for the paper)

## Current scope
- **Presidio + rule-based secrets only.** No transformer NER or LLM detector yet, so there is no
  detector-agreement analysis and no sensitive-topic tier.
- **No human labels collected yet** (the workflow exists: `residpii label`). Until then, All numbers are *detector-reported* prevalence: they mix true and
  false positives. Precision, recall and F1 are unknown. State this prominently.
- **Task types are keyword heuristics**, not a validated classifier.

## Detector behaviour seen in the smoke test
- **Circularity:** WildChat was redacted with Presidio, so a low WildChat Presidio residual is
  expected by construction. Read it as version/threshold drift, not "WildChat is clean".
- **Threshold choice matters for pattern recognizers, not for NER.** spaCy-based entities (PERSON,
  LOCATION, DATE_TIME, NRP) get a fixed score of 0.85, so the threshold doesn't filter them.
  Pattern recognizers vary: a phone number with no nearby context words scores 0.4, and
  driver-licence/bank-number patterns produce many low-score hits. WildChat's threshold isn't
  documented. Report `presidio_threshold_sensitivity.csv`.
- **Code prompts inflate "name".** spaCy tags identifiers in source code as PERSON. In the smoke
  sample, code conversations had name detections at a rate similar to story/role-play.
- **URLs are reported separately** (`url_unfiltered` tier), because most are public links.
- **Dummy-value filtering is heuristic** (example.com, 555 numbers, private IPs, test cards,
  sequential digits, "DAN"). It will miss some placeholders and occasionally drop real values.

## Fictional names
Story/role-play prompts produce many fictional names. We report results with and without them,
but the task tagger is heuristic, and fictional names also appear in other task types. Public
figures are not separated from private individuals.

## Sampling
- One conversation per user proxy changes the estimand toward a per-user rate. `hashed_ip` is a
  noisy proxy (shared, dynamic, VPN IPs). lmsys-chat-1m has no proxy, so heavy users can dominate
  its sample, which makes it less comparable on this dimension.
- With small samples, strata allocated zero conversations drop out of the weighted population.
- The smoke configuration reads only the first row groups of one file, so it is **not
  representative** and exists only to exercise the code.

## Datasets
- **Survivorship:** we measure only what survived each release's filtering (WildChat deletions
  and toxic-content split; LMSYS PII-based removals). Results apply to the pinned revisions.
- **lmsys-chat-1m and Chatbot Arena overlap** (same platform, 2023). We flag lmsys conversations
  that exactly or nearly duplicate an Arena conversation (MinHash, Jaccard ≥ 0.8) and report
  results with and without them. The same prompt sent by different users is also merged by
  text deduplication.
- **Language:** filtering uses each dataset's own language label, which is itself a classifier
  output. Presidio's recognizers are US/English-centric (SSN, US phone formats), and spaCy NER
  under-detects non-Western names.

## Interpretation
- Prevalence is per conversation and is confounded by length. See `rates_per_1k_tokens.csv`.
- Detection is not identification: a detected name or city doesn't mean the person is
  identifiable, and we never test that (no re-identification by design).

## Human-label benchmark
- Turns over 4,000 characters (about 3.6% of WildChat user turns) are not eligible for labelling, so
  precision and recall describe shorter turns. The rare-direct stratum is small (19 turns at a
  Presidio threshold of 0.5), so recall on rare categories will have wide intervals.
- Precision counts a detector span as correct if it overlaps a private gold span of the same
  category by at least half of the shorter span (`min_overlap`). `precision_any_category` relaxes
  the category requirement.
- Agreement statistics are unweighted (they describe the label set as drawn). Precision, recall and
  gold prevalence are weighted.
- The labels cover user turns only. Assistant turns and non-English text are out of scope.
- **LLM-only labels are not ground truth.** If precision and recall are computed against `--gold llm`,
  they measure agreement with Claude, not accuracy, and the LLM detector (once built) would be
  graded against a model of the same kind. Report them that way, and validate a subset by hand.
- Using `label llm` sends the label-set turns to a third party (Anthropic). The paper's ethics section
  must say so.
