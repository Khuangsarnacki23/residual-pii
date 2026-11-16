# Stage 2 results: can WildChat users be singled out by what they reveal?

Source tables: `results/user_level/tables/` (aggregate only; cells under 10 users blanked). Config:
`configs/user_level.yaml`. WildChat only. Attributes and identifiers come from the LLM detector
(Claude Haiku 5.5, effort low; $4.64 for 46,072 user turns, 12 refused, 53 failed). No profile was
linked to a real person, and no lookups were made.

## Summary

Among 1,982 WildChat users with at least 3 conversations, about 14% revealed at least one
attribute the extractor could read (occupation, country, age, and so on). About 6% of all users are
unique in the sample on some combination of up to four attributes, and a population model puts it at
about 5% among the 25,325 eligible users. Every attribute is first reduced to a broad category
(occupation group, country, health condition type, employer sector, and so on), so this is not an
artefact of wording. Uniqueness rises steeply with activity: 15% of users with more than 50
conversations, against 3% of those with 3. A user's conversations can be tied back together from their content for
about 2% of users (chance: 0.05%), and about 10% of users with 26–50 conversations, almost all of it through direct identifiers.

## Sample and coverage

| | Users | Conversations | User turns |
|---|---|---|---|
| Sampled (≥3 conversations, at most 50 kept per user) | 2,000 | 19,517 | 46,072 |
| Fully covered by the LLM detector, profiled | 1,982 | 19,174 | |

Users are linked only through WildChat's own `hashed_ip`, so one "user" can be a shared IP and one
person can appear as several users. 25,325 users in WildChat meet the 3-conversation threshold.

## What users reveal (stated plus inferred)

| Attribute | Users with a value |
|---|---|
| Any attribute | 13.8% |
| Occupation group | 6.6% |
| Country | 5.0% |
| Education level | 3.4% |
| Age band | 2.8% |
| Health | 1.9% |
| Employer | 1.8% |
| Nationality / ethnicity | 1.3% |
| Relationship | 1.2% |

Religion, political views and income are each revealed by fewer than 10 users (cells blanked).
With stated values only, 10.5% reveal any attribute.

## Uniqueness

Share of all profiled users who are unique (k = 1) on at least one combination of at most n attributes.

| Variant | n = 1 | n = 2 | n = 4 | 95% CI (n = 4) |
|---|---|---|---|---|
| Stated only | 1.9% | 4.1% | 4.1% | 3.3–5.0% |
| Stated plus inferred | 2.0% | 5.9% | 5.9% | 4.8–6.9% |

- Coarsening (`configs/coarsening.yaml`): region (US state or listed region), country, SOC
  occupation major group, 10-year age band, gender, education level, and keyword categories for the
  rest: health (e.g. mental health, chronic condition, neurodevelopmental), employer sector
  (government, education, technology, …), income band, relationship status, religion, political
  leaning, and broad nationality or ethnic group. Values that match nothing are dropped, except
  health and employer, which fall into one "other" category.
- Before the free-text attributes were categorised, the headline was 7.7%: about a quarter of it
  came only from distinct wording.
- One attribute rarely singles anyone out (2%). Uniqueness comes from pairs of attributes, and the curve is flat after two because almost no user reveals three.
- **Population estimate** (`uniqueness_population.csv`): 4.8% (4.1–5.6%) of the 25,325 eligible
  users, or 3.1% with stated values only.

### By activity (stated plus inferred, n = 4)

| Conversations per user | Users | Unique | Mean known attributes |
|---|---|---|---|
| 3 | 559 | 2.7% | 0.15 |
| 4–5 | 510 | 4.7% | 0.22 |
| 6–10 | 448 | 4.7% | 0.21 |
| 11–25 | 283 | 12.4% | 0.48 |
| 26–50 | 103 | 9.7% | 0.56 |
| >50 | 79 | 15.2% | 0.71 |

Heavy users reveal more and are singled out more: accumulation across conversations is the main risk.

## Linkage: can a user's conversations be tied together from content?

Each user's conversations are split at random into two halves (20 splits). A user is linked when
their two halves match each other better than any other user's half.

| Matched on | Users linked | 95% CI | Linked, when both halves reveal something |
|---|---|---|---|
| Profile attributes (stated plus inferred) | 0.7% | 0.5–1.1% | 16% |
| Direct identifiers (keyed hashes) | 1.5% | 1.0–2.0% | 59% |
| Both | 2.1% | 1.6–2.7% | 31% |

Chance is 0.05%. For users with 26–50 conversations, 9.8% are linked on both. Linkage needs both
halves to reveal something, which only 7% of users do, so most of the low overall share is users
who revealed nothing.

## Direct identifiers

8.5% of profiled users left at least one private direct identifier (as judged by the LLM: name,
email, phone, handle, URL, address, API key). Names (4.3%) and usernames or handles (3.4%) are the
only categories above the 10-user cell; the rest are blanked. 2.3% of users are both unique on their
attributes and left a direct identifier.

## How far to trust these numbers

- **Extractor error runs both ways.** On the Stage 1 gold set the attribute extractor has 29% recall
  (occupation 11%) and 40% precision (`identifiability_attribute_validation.csv`, 14 gold items,
  one annotator). Missed attributes push uniqueness down; wrong or noisily worded values push it
  up. So these figures are neither a clean lower nor a clean upper bound.
- The LLM's direct-identifier calls have about 60% precision and 29% recall on the gold set.
- `hashed_ip` is a weak user proxy: shared IPs merge people, and changing IPs split them.
- Uniqueness is within this dataset only. It says a user is distinguishable, not who they are; no
  identities were sought.
- Cheapest model and effort; one dataset.

## Next

- Re-check extraction against a larger gold attribute set (both annotators), or a stronger model.
