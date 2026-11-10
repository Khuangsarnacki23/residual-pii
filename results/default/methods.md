# Methods summary (auto-generated 2026-10-13; adapt before use)

## Data
We analysed three public chat-log datasets, pinned to the following Hugging Face revisions:

- `allenai/WildChat-1M` @ `7d6490e46228`

Only user turns were analysed. Conversations were restricted to those whose dataset-provided
language label was ['English']. Data were streamed column-wise from the Hub; only sampled
turns were retained locally, in an access-restricted directory outside the code repository.

## Sampling
For each dataset we drew a stratified random sample (target n = wildchat=5000; seed 20261012). Where a
user proxy exists (WildChat `hashed_ip`, Arena `judge`), we first selected one conversation uniformly
at random per proxy, so prevalence estimates approximate a per-user rather than per-conversation
rate; LMSYS-Chat-1M has no proxy and is sampled per conversation. Strata were defined by
month × turn_bucket with proportional (largest-remainder) allocation, and design weights N_h / n_h were applied.

- WildChat-1M: 474,405 eligible conversations, sampling frame 98,094, sample 5,000

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
reported as detector `selfid`. A third, `rules`, matched types Presidio has no recognizer for:
profile links and handles on social and code platforms, street addresses, full dates of birth,
identifiers named in context ("student ID: ...") and international-format phone numbers. Text found
by more than one detector was counted once: a `rules` span overlapping another detector's specific
detection was dropped, and a Presidio URL, date, location or name overlapping a remaining `rules`
span was dropped in its favour (excluded_detections.csv). Detections were mapped onto direct and quasi-identifier categories
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

## Limitations (summary)
Results are detector-reported prevalence and have not yet been validated against human labels, so
they mix true and false positives. Presidio was also the tool used to redact WildChat, so WildChat's
Presidio residual is low by construction. See docs/limitations.md.
