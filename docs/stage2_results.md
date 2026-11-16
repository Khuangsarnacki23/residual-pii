# Stage 2 results: can WildChat users be singled out by what they reveal?

Source tables: `results/user_level/tables/` (aggregate only; cells under 10 users blanked). Config:
`configs/user_level.yaml`. WildChat only. Attributes and identifiers come from the LLM detector
(Claude Haiku 5.5, effort low; $4.64 for 46,072 user turns, 12 refused, 53 failed). No profile was
linked to a real person, and no lookups were made.

## Summary

Among 1,982 WildChat users with at least 3 conversations, about 14% revealed at least one
attribute the extractor could read (occupation, country, age, and so on). About 8% of all users are
unique in the sample on some combination of up to four attributes, and a population model puts it at
about 5% among the 25,325 eligible users. About half of that comes from free-text attributes kept
as stated (health, employer, income), so the conservative figure on coarsened attributes alone is
about 4%. Uniqueness rises steeply with activity: 23% of users with more than 50 conversations,
against 4% of those with 3. A user's conversations can be tied back together from their content for
about 2% of users (chance: 0.1%), and about 10% of users with 26–50 conversations.

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
| Any attribute | 14.1% |
| Occupation group | 6.6% |
| Country | 5.0% |
| Education level | 3.4% |
| Age band | 2.8% |
| Nationality / ethnicity | 1.7% |
| Employer | 1.6% |
| Health | 1.5% |

Users have 0.27 known attributes on average; with stated values only, 10.5% reveal any attribute.

## Uniqueness

Share of all profiled users who are unique (k = 1) on at least one combination of at most n attributes.

| Variant | n = 1 | n = 2 | n = 4 | 95% CI (n = 4) |
|---|---|---|---|---|
| Stated only | 5.0% | 5.8% | 5.8% | 4.8–6.8% |
| Stated plus inferred | 6.0% | 7.7% | 7.7% | 6.6–8.8% |
| Stated plus inferred, coarsened attributes only | | | 4.1% | |

- The curve is flat after two attributes, because almost no user reveals three.
- Free-text attributes (health, employer, income, religion, political) are kept as stated, so
  nearly every value is distinct and anyone with one is unique by construction. The coarsened-only
  row (region, country, occupation group, age band, gender, education level) removes that effect.
- **Population estimate** (`uniqueness_population.csv`): 5.2% (4.5–6.2%) of the 25,325 eligible
  users, or 3.6% with stated values only. Within-sample uniqueness falls as the sample grows (9.1%
  at 500 users, 8.3% at 1,000), so the sample figure overstates population uniqueness, as expected.

### By activity (stated plus inferred, n = 4)

| Conversations per user | Users | Unique | Mean known attributes |
|---|---|---|---|
| 3 | 559 | 3.6% | 0.15 |
| 4–5 | 510 | 5.5% | 0.22 |
| 6–10 | 448 | 6.2% | 0.21 |
| 11–25 | 283 | 14.8% | 0.48 |
| 26–50 | 103 | 16.5% | 0.56 |
| >50 | 79 | 22.8% | 0.71 |

Heavy users reveal more and are singled out more: accumulation across conversations is the main risk.

## Linkage: can a user's conversations be tied together from content?

Each user's conversations are split at random into two halves (20 splits). A user is linked when
their two halves match each other better than any other user's half.

| Matched on | Users linked | 95% CI | Linked, when both halves reveal something |
|---|---|---|---|
| Profile attributes (stated plus inferred) | 0.8% | 0.5–1.1% | 17% |
| Direct identifiers (keyed hashes) | 1.5% | 1.0–2.0% | 59% |
| Both | 2.2% | 1.6–2.7% | 32% |

Chance is 0.1%. For users with 26–50 conversations, 9.8% are linked on both. Linkage needs both
halves to reveal something, which only 7% of users do, so most of the low overall share is users
who revealed nothing.

## Direct identifiers

8.5% of profiled users left at least one private direct identifier (as judged by the LLM: name,
email, phone, handle, URL, address, API key). Names (4.3%) and usernames or handles (3.4%) are the
only categories above the 10-user cell; the rest are blanked. 2.8% of users are both unique on their
attributes and left a direct identifier.

## How far to trust these numbers

- **Extractor error runs both ways.** On the Stage 1 gold set the attribute extractor has 29% recall
  (occupation 11%) and 40% precision (`identifiability_attribute_validation.csv`, 14 gold items,
  one annotator). Missed attributes push uniqueness down; wrong or noisily worded values push it
  up. So these figures are neither a clean lower nor a clean upper bound. The coarsened-only figure is
  the most robust to wording noise.
- The LLM's direct-identifier calls have about 60% precision and 29% recall on the gold set.
- `hashed_ip` is a weak user proxy: shared IPs merge people, and changing IPs split them.
- Uniqueness is within this dataset only. It says a user is distinguishable, not who they are; no
  identities were sought.
- Cheapest model and effort; one dataset.

## Next

- Coarsen the free-text attributes (health into broad categories, income into bands, employer into
  sector or size) so the headline does not rely on string variation.
- Re-check extraction against a larger gold attribute set (both annotators), or a stronger model.
