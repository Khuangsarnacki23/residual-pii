# Risks and limitations (draft for the paper)

## Current scope
- **Rule-based detectors and spaCy NER have run on real data; the LLM layer has not.** The LLM
  detector, profiles and uniqueness analysis are implemented and tested with a fake backend only.
  No number from them exists yet.
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
  attributes and can make a profile look more or less unique than any real person. Split users
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
- **Extraction errors.** The model misses attributes, misreads who they are about (the user, a
  third party, a fictional persona the user wants text for), and can output attributes that are
  not in the text. Small, cheap models will be noticeably worse at this than large ones.
- **Inferred attributes are guesses.** "Inferred" covers anything from a near-certain reading to a
  stereotype (guessing gender from topic, nationality from spelling). The `stated_only` variant is
  the conservative result; `stated_plus_inferred` is an upper bound on what a motivated reader
  might assume, right or wrong. A wrong inference still splits users into smaller classes, so it
  inflates measured uniqueness.
- **Confidence is uncalibrated.** `llm.min_confidence` is a threshold on the model's own number.
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
  more, raising measured uniqueness. Numbers from different models are not comparable.
- **Subsampling.** With `llm.sample_turns` set (the spend cap), only some users are processed; the
  uniqueness sample is then smaller than the Presidio sample.

## Profiles and coarsening
- **Lookup tables are small and US/English-centred.** Cities, regions and occupations outside
  them become `unknown` and drop out, which *under*states uniqueness for users elsewhere.
- **Occupation groups** come from the LLM's SOC code when valid, otherwise from a keyword map, and
  fall back to the major group when no minor group is available, so they mix granularities.
- **Schools are mapped to an education level** assuming current attendance.
- **Majority vote hides change.** A user who moved or changed jobs looks like a conflict, and
  ties become `ambiguous` and drop out.
- **`other` is not used.** Free-text attributes of that type cannot be matched across users.

## Uniqueness
- **Within-sample uniqueness overstates population uniqueness.** A user unique among 2,000 sampled
  users is often not unique among all users of the service, let alone the general population. The
  share falls as the sample grows; `uniqueness_summary.csv` shows this at several sample sizes. Do
  not report the within-sample share as "X% of users are identifiable". For the population-level
  argument and a way to estimate it, see Rocher, Hendrickx & de Montjoye (2019).
- **The population estimate is a model, and a simple one.** `uniqueness_population.csv` assumes the
  dimensions are independent of each other (region and country apart). Where attributes go together
  in reality (a job that is common in one country), a match is more likely than the model says and
  the estimate is too high; where they rarely go together it is too low. A value seen once in the
  sample is given half a pseudo-user (`uniqueness.population_prior`), and the result moves with that
  choice. The check rows compare the model with observed uniqueness at the sample sizes we have,
  which says the model fits there, not that it extrapolates. Its bootstrap interval is skewed
  upward, because a resample holds fewer distinct users than the sample. "Population" means users
  of this dataset with enough conversations to be sampled, not the public.
- **One profile definition, with judgment calls in it.** Six dimensions are generalised with lookup
  tables (state or region, country, SOC occupation group, 10-year age band, gender, education
  level); a value the tables miss is dropped, which understates what users revealed. Five are kept
  as stated because there is no table for them (employer type, school, relationship status,
  nationality, native language); two users who word the same fact differently look different on
  these, which overstates uniqueness. Employer type and school are free text and are the most
  exposed to this. Earlier versions of the analysis reported a stricter six-attribute profile and a
  looser as-written one; on 600 users they gave 3-6% and 8-11% unique among all eligible users, and
  the current definition should be read as sitting inside that range, not as exact.
- **Unique is not identified.** k = 1 means no other sampled user shares the combination. Turning
  that into an identity needs outside data, which we never use.
- **Bootstrap intervals hold class sizes fixed** and resample users, so they show sampling noise in
  the share, not uncertainty in the classes, in the extraction, or in the proxy.
- **Missing is not absent.** A combination is only evaluated for users who revealed every
  attribute in it; users who revealed less count as not unique in the headline share.
- **Small-cell suppression** blanks cells under 10 users, so smoke runs show mostly blanks.

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
