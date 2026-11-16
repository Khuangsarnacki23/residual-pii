# Risks and limitations (draft for the paper)

## Current scope
- **Presidio + rule-based secrets only.** No transformer NER or LLM detector yet, so there is no
  detector-agreement analysis and no sensitive-topic tier.
- **No human labels collected yet** (the workflow exists: `residpii label`). Until then, All numbers are *detector-reported* prevalence: they mix true and
  false positives. Precision, recall and F1 are unknown. State this prominently.
- **Task types are keyword heuristics**, not a validated classifier.

## Shared prompt templates
One pasted prompt generator appears in 1,089 of the 5,000 sampled WildChat conversations, each from a
different hashed IP, so it is likely an app reusing the chat interface, not 1,089 people. Including it
inflated Presidio's name rate from 28.8% to 44.4%. Conversations whose user turn shares its first 200
normalized characters with at least 4 other sampled conversations are left out of every estimate
(WildChat 22.4% of the sample, LMSYS 6.6%, Arena 0.1%; `templates.csv`). The rule misses templates
shorter than 200 characters and ones that vary in their opening; `prevalence.csv` keeps the
`incl_templates` variant. Estimates describe non-templated conversations.

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

## Layered sanitizer (RQ4)
- **Tiny evaluation set.** Only 21 of the 260 label items hold private PII in the gold (24 direct,
  28 quasi spans), so leak rates have very wide intervals. Over-redaction is measured on all 260.
- **Leakage is string presence.** A gold value counts as leaked if it still appears in the text
  (values under four characters as whole words). A paraphrase that keeps the meaning ("my employer,
  the big Seattle coffee chain") is not caught; the attacker metric is there for that.
- **Gold covers the writer only.** The human labels mark the writer's own information, so leaks of
  third-party PII are not measured by `sanitizer_leakage.csv`.
- **Layer 2 reuses the LLM detector's spans** (Claude Haiku 5.5 over the full sample), not a fresh call.
- **The loop attacker and the evaluation attacker are both LLMs.** Using a different, stronger model
  for evaluation limits, but does not remove, the advantage of optimising against an attacker. The
  attacker has no ground truth: exposure counts confident inferences, not correct ones.
- **Utility is an LLM judge's rating**, not a user study.
- **Rule-based layers are tuned on the evaluation set.** The name and organisation rules were
  adjusted after looking at their errors on the 260 label items (counts only), so their scores there
  are optimistic. A fresh hand-labelled set is needed for an unbiased comparison with the LLM layers.
- **Rules depend on spaCy's entity labels.** Names spaCy tags as an organisation or place, names in
  lowercase, and non-English text are missed. The lookup tables are small and English/US-centred: an
  unlisted city becomes [LOCATION] instead of its country, and occupation groups are coarse
  ("developer" in "curriculum developer" maps to IT).
- **The keyword task tagger gates the rules.** Fiction it does not tag as story/role-play is treated
  as ordinary text, so its character names can be redacted when they appear in first-person sentences.
- **The hand review judges spans, not replacements.** A reviewer decides whether a highlighted span
  should change, not whether the replacement ("a healthcare professional", "the UK") is apt. LLM
  rewrite layers (generalize, adversarial) are not in the review, since their edits are not spans.
- **Pooled recall is an upper bound.** Personal information that no tool flagged and the reviewer also
  missed is not counted, as in pooled IR evaluation.
- **One reviewer.** Without a second reviewer there is no agreement statistic for the review.
- **Path rule is shape-based.** A path is redacted if it contains a home directory, is a Windows
  absolute path, or has three or more segments not starting at a system root or common web route.
  Relative paths (`./src/...`), short paths and paths inside URLs are kept, so a personal folder name
  in a short relative path is missed; deep but generic paths (`/content/drive/MyDrive/data/x.csv`) are
  redacted anyway.
- **Name lists are hand-written and incomplete.** `configs/public_names.yaml` lists about 340 public
  figures, 170 fictional characters, placeholders and 250 common first names, English-centred. An
  unlisted celebrity or character is treated like anyone else, and a real person who shares a listed
  name (a private "Michael Jordan") is not redacted. A few entries first written after seeing review
  data were removed so the review scores are not inflated.
