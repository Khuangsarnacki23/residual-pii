# Methods summary (auto-generated 2026-10-12; adapt before use)

## Data
We analysed three public chat-log datasets, pinned to the following Hugging Face revisions:

- `allenai/WildChat-1M` @ `7d6490e46228`

Prevalence tables cover user turns; assistant turns were also scanned and are reported separately (prevalence_by_role.csv). Conversations were restricted to those whose dataset-provided
language label was ['English']. Data were streamed column-wise from the Hub; only sampled
turns were retained locally, in an access-restricted directory outside the code repository.

## Sampling
We sampled users rather than conversations (seed 20261012). Conversations were grouped by the
dataset's own user proxy (WildChat `hashed_ip`, Arena `judge`), which we replace with a keyed hash on
read; no other information links conversations. From users with at least 3 eligible
conversations we drew up to 50 uniformly at random and kept all of their conversations, up
to 50 per user (a random subset above that cap; such heavy users are likely shared or NAT
addresses). Datasets without a user proxy were not sampled. Conversation-level tables weight each
conversation by (eligible users / sampled users) x (the user's conversations / conversations kept);
their confidence intervals resample conversations and ignore clustering within users.

- WildChat-1M: 131 eligible users, 50 sampled, 426 conversations; eligible users above the cap: <10

## Pre-existing redactions
Each dataset's own placeholders were located with the regular expressions listed in the
configuration file. Detections overlapping a placeholder were excluded from residual counts and
placeholder frequency is reported separately.

## Detection
Presidio Analyzer (v2.2.364) with spaCy `en_core_web_lg`, score threshold
0.5, and Presidio's default entity mapping (ORGANIZATION ignored, as in Presidio's defaults). Turns were scanned in
5000-character windows with 200-character overlap, up to 50000 characters per
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
95% confidence intervals use a stratified percentile bootstrap (200 replicates), resampling
conversations within strata.

## LLM extraction of quasi-identifiers
For 1,038 sampled user turns (all sampled user turns), a language model (fireworks `accounts/fireworks/models/gpt-oss-120b`, prompt
`qi_extract_v1`, a hosted API, used with explicit opt-in) was given the turn, with the preceding turn as
context, and asked for the quasi-identifying attributes it reveals: type, value, whether it was
stated or inferred, whom it is about, and a confidence. Responses were validated against a strict
schema; a malformed response was retried once and otherwise recorded as a failure (0
of 1,038 turns). Turns longer than 6,000 characters were truncated. Extracted
values were stored only in the access-restricted directory.

## Limitations (summary)
Results are detector-reported prevalence and have not yet been validated against human labels, so
they mix true and false positives. Presidio was also the tool used to redact WildChat, so WildChat's
Presidio residual is low by construction. See docs/limitations.md.
