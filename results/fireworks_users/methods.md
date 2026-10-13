# Methods summary (auto-generated 2026-10-13; adapt before use)

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
conversations we drew up to 650 uniformly at random and kept all of their conversations, up
to 50 per user (a random subset above that cap; such heavy users are likely shared or NAT
addresses). Datasets without a user proxy were not sampled. Conversation-level tables weight each
conversation by (eligible users / sampled users) x (the user's conversations / conversations kept);
their confidence intervals resample conversations and ignore clustering within users.

- WildChat-1M: 25,325 eligible users, 650 sampled, 6,765 conversations; eligible users above the cap: 940

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
95% confidence intervals use a stratified percentile bootstrap (1000 replicates), resampling
conversations within strata.

## LLM extraction of quasi-identifiers
For 14,985 sampled user turns (a random subsample capped at 15,000), a language model (fireworks `accounts/fireworks/models/gpt-oss-120b`, prompt
`qi_extract_v1`, a hosted API, used with explicit opt-in) was given the turn, with the preceding turn as
context, and asked for the quasi-identifying attributes it reveals: type, value, whether it was
stated or inferred, whom it is about, and a confidence. Responses were validated against a strict
schema; a malformed response was retried once and otherwise recorded as a failure (0
of 14,985 turns). Turns longer than 6,000 characters were truncated. Extracted
values were stored only in the access-restricted directory.

## Per-user profiles and uniqueness
For each user we kept attributes about the user with confidence >= 0.6 and built a
profile on eleven dimensions. Six were generalised with fixed lookup tables (configs/coarsening.yaml)
so that different wordings of one fact match: state or region, country, occupation (SOC minor group
where available, else major group), 10-year age band, gender and education level; a value the
tables do not cover was set to `unknown`. Five have no table and were kept as stated (normalised
text): employer type, school, relationship status, nationality and native language. We report two
variants (stated attributes only; stated plus inferred). When a user's conversations disagreed we
took the majority value and treated ties as ambiguous.

For every combination of one to 4 dimensions we computed the equivalence-class size k
among users with a usable value for each dimension in the combination, and report the share of
those users with k = 1 and k < 5 and the median k, with 1000-replicate bootstrap intervals that
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
10 users are suppressed.

To test whether these attributes would let records be linked, we split each user's conversations at
random into two halves, built a profile from each half, and matched every first half against all
second halves, scoring a shared value by the negative log of its share among second halves. A user
is linked when their own second half is the single best match. We repeated this with direct
identifiers detected in user turns (self-stated names, email addresses, phone numbers, credentials,
compared by keyed hash) and with both together, and averaged over repeated random splits. No outside
data was used, and we did not attempt to identify any user.

## Limitations (summary)
Results are detector-reported prevalence and have not yet been validated against human labels, so
they mix true and false positives. Presidio was also the tool used to redact WildChat, so WildChat's
Presidio residual is low by construction. See docs/limitations.md.
