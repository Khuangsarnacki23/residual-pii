# Risks and limitations (draft for the paper)

## Current scope
- **Rule-based detectors and spaCy NER have run on real data; the LLM layer has not.** The LLM
  detector is implemented and tested with a fake backend only. No number from it exists yet.
- **No human validation yet.** All numbers are *detector-reported* prevalence: they mix true and
  false positives. Precision, recall and F1 are unknown until a batch is labeled
  (`residpii label-export` / `label-score`). State this prominently.
- **Task types are keyword heuristics**, not a validated classifier.
- **No sensitive-topic tier.** The LLM prompt extracts quasi-identifiers, not sensitive topics.

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
- **`selfid` is a narrow pattern.** It catches "my name is X", "call me X", "this is X speaking"
  and "I'm X" with a capitalized name of up to three tokens. It misses lower-case names, other
  phrasings and other languages, and it cannot tell a real name from a role-play persona.

- **`rules` patterns are untested on real data.** They were written against made-up examples and
  have no measured precision. Known weak spots: a year followed by a street name ("2008 Wall
  Street"), public figures' birth dates and profile links, IDs in code or sample data, and street
  suffixes outside the US/UK. Street addresses in other formats and handles without a platform word
  nearby are missed.
- **Counting once changes Presidio's own counts slightly.** A Presidio URL, date, location or name
  that overlaps a `rules` span is dropped in its favour, and a `rules` span that overlaps a specific
  Presidio, `secrets` or `selfid` detection is dropped. Both are counted in
  `excluded_detections.csv`. The rule is applied after the score threshold, so an international
  phone number that Presidio scored below the threshold is counted under `rules`.

## Fictional names
Story/role-play prompts produce many fictional names. We report results with and without them,
but the task tagger is heuristic, and fictional names also appear in other task types. Public
figures are not separated from private individuals.

## Sampling
- One conversation per user proxy changes the estimand toward a per-user rate. `hashed_ip` is a
  noisy proxy (shared, dynamic, VPN IPs). lmsys-chat-1m has no proxy, so heavy users can dominate
  its sample, which makes it less comparable on this dimension.
- With small samples, strata allocated zero conversations drop out of the weighted population.
- The smoke configurations read only the first row groups of one file, so they are **not
  representative** and exist only to exercise the code.

## User-level sampling and the user proxy
- **A `hashed_ip` is not a person.** One address can be many people (households, campuses,
  carrier-grade NAT, VPN exits) and one person has many addresses (home, work, mobile, dynamic
  reassignment). Shared addresses merge strangers into one "user", which produces conflicting
  attributes. Split users
  hide accumulation across conversations. Both errors are unmeasured.
- **Heavy users are capped.** Users above `max_conversations_per_user` (default 50) are likely
  shared or NAT addresses; a random subset of their conversations is kept. In the full English
  WildChat scan, 940 of 25,325 eligible users (3.7%) are above the cap. Report this, and consider
  a sensitivity run that drops them.
- **The eligible population is a minority.** Requiring three or more conversations keeps about a
  quarter of proxied users (see `user_activity.csv`). Results describe repeat users, not all users.
- **Conversation-level tables in user-level mode** weight conversations back to the eligible
  population, but their bootstrap resamples conversations and ignores clustering within users, so
  those intervals are too narrow.
- **Language filter.** Only conversations labeled English are counted and sampled, so a
  multilingual user's other conversations are invisible. Presidio is English-only; the LLM is not.
- **Assistant turns.** When fetched, they are scanned and reported in `prevalence_by_role.csv`.
  For Arena only model A's replies are read.

## LLM extraction
- **Quoted spans depend on the model copying text exactly.** A quote that is not in the turn
  verbatim is discarded (`quotes_not_found` in `llm_run_stats.csv`), which loses real detections. A
  quote is located at every exact occurrence in the turn, so a short name can also match the same
  string used in another sense. LLM spans are reported in their own tables and are not part of
  `prevalence.csv`, because the LLM may run on a subsample of turns.
- **Extraction errors.** The model misses attributes, misreads who they are about (the user, a
  third party, a fictional persona the user wants text for), and can output attributes that are
  not in the text. Small, cheap models will be noticeably worse at this than large ones.
- **Inferred attributes are guesses.** "Inferred" covers anything from a near-certain reading to a
  stereotype (guessing gender from topic, nationality from spelling). Stated attributes are the
  conservative result; stated plus inferred is an upper bound on what a motivated reader might
  assume, right or wrong.
- **Confidence is uncalibrated**, and no confidence threshold is applied to the reported counts.
- **Strict parsing drops whole responses.** One bad field fails the turn after a single retry.
  Report the failure rate from `llm_run_stats.csv`; failures bias attribute counts downward.
- **Truncation.** Turns beyond `llm.max_turn_chars` are cut, and only the immediately preceding
  turn is given as context, so references further back are lost.
- **Prompt injection.** User text can contain instructions. The parser only accepts values from a
  closed schema, which limits the damage to wrong labels.
- **Hosted only.** There is no local backend, so every LLM run sends sampled user text to a third
  party (Fireworks by default). That is a disclosure in its own right: it needs the dataset terms,
  the instructor/IRB and both partners to allow it, and it depends on the provider's retention
  terms. State the provider and model in the paper.
- **Model dependence.** Results depend on the model. A larger model would likely extract and infer
  more. Numbers from different models are not comparable.
- **Subsampling.** With `llm.sample_turns` set (the spend cap), only some users are processed, so
  the LLM sample is then smaller than the Presidio sample.

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
