# Pilot: a user-level adversarial sanitizer (Stage 3)

Code: `src/residpii/user_adversarial.py` (`residpii adversarial`), settings under `adversarial:` in
`configs/user_level.yaml`. Source table: `results/user_level/tables/adversarial_summary.csv`
(aggregate only). WildChat only, 50 users, $8.60 in total. No identity was looked up; sanitized text
and model outputs stay in the restricted directory.

## Summary

Stage 2 found that users are rarely identifiable from one message but become so across their
conversations. We tested whether a sanitizer that works on the whole user, rewriting only the
messages that keep that user identifiable, beats the per-message adversarial loop of Staab et al.
(2025), which is layer 4 of Kai's sanitizer.

On 50 users, the best user-level version changed 42% fewer messages (161 against 279) and kept
every kept rewrite usable. It left more users identifiable, though: 17 of 23 at-risk users against 13
for the per-message loop, a gap that is not significant at this size (p = 0.73). The user-level stopping
rule works once it is strict; what limits the loop now is the rewrite step.

## Setup

- **Users.** 40 users who are identifiable in the Stage 2 sample (their smallest class on up to four
  attributes is under 5) and 10 who are not (controls), drawn at random with a fixed seed. 1,934
  user messages.
- **Starting text.** Kai's layers 1–2: pattern rules plus the direct identifiers the LLM detector
  judged private, redacted. Every version starts from the same text.
- **Risk.** A profile is coarsened exactly as in Stage 2 (occupation group, country, age band, broad
  health category, and so on). Risk is the user's smallest class: for every combination of at most
  four known attributes, 1 + the number of the other 1,982 Stage 2 users who share it.
- **Models.** Loop attacker Claude Haiku 5.5; rewrites Claude Sonnet 5.5; utility judge Haiku (1–5,
  "could a chatbot still fulfil the request"). All at medium effort, confidence threshold 4 of 5.
- **Held-out grader.** Claude Sonnet 5.5 reads each user's whole history in one call and infers what
  it can. The loops never see it. A user is **at risk** if its profile puts them in a class under 5.

| Version | How it works |
|---|---|
| Per-message (Staab et al., Kai's layer 4) | Each message alone: while Haiku makes any confident inference, rewrite with its cues (up to 3 rewrites). No utility check. |
| User-level | Per-message attacks combined into one user profile. While the user's class is under 5, pick the attribute whose removal helps most and rewrite only the messages that reveal it. A rewrite is kept only if the judge rates it at least 4 and it changes at most half the message. Up to 6 rounds, 2 tries per attribute. |
| User-level, whole history | As above, but the loop attacker reads the whole history each round, and a failed rewrite gets one gentler retry. |
| User-level, strict stop | Core user-level loop. The user counts as safe only when both attackers (per-message combined, and whole history) put them in a class of at least 10. |

## Results

Held-out attacker, all 50 users (23 at risk before rewriting).

| Version | At-risk users still at risk | Users at risk | Users unique | Messages changed | Edit size* | Usefulness | Rewrites rated ≥ 4 |
|---|---|---|---|---|---|---|---|
| Before rewriting | 23 of 23 | 46% | 36% | 0 | | | |
| Per-message | **13** | **32%** | **22%** | 279 (14.4%) | 18% | 4.1 | 76% |
| User-level | 18 | 40% | 26% | 126 (6.5%) | 12% | 4.6 | 100% |
| User-level, whole history | 20 | 42% | 30% | **67 (3.5%)** | 10% | 4.6 | 100% |
| User-level, strict stop | 17 | 36% | 26% | 161 (8.3%) | 12% | 4.6 | 100% |
| User-level, improved rewrites | 17 | 38% | 26% | 183 (9.5%) | 13% | 4.5 | 100% |
| User-level, improved + passages | 17 | 38% | 26% | 198 (10.2%) | 13% | 4.5 | 100% |

\* Mean share of a changed message's characters that the rewrite did not keep.

- **Paired tests** (sign test on users whose at-risk status changed): per-message against no
  rewriting, p = 0.09; strict user-level against no rewriting, p = 0.13; per-message against strict
  user-level, p = 0.73. With 50 users none is significant.
- **Controls** stayed safe under every version, and every version left them almost untouched (at
  most 0.3 messages changed per user).
- **Some users got worse.** In one to three users per version, the held-out attacker inferred more
  after rewriting than before. Part of this is the attacker's own variability; part is rewrites that
  add new cues.
- The 100% usefulness for the user-level versions is partly by construction: the loop discards
  rewrites the judge rates below 4 (47–56 per version). The per-message loop has no such check.

## What we learned

1. **A loop is only as good as its in-loop attacker.** The first user-level version stopped when
   Haiku could no longer narrow a user down; for 10 users the held-out Sonnet attacker still could.
   Reading the whole history inside the loop made this worse (15 users declared safe too early),
   because Haiku reading a long history infers less, and 27 users stopped before any rewrite.
   Rewrites that fool the weaker attacker do not fool the stronger one, the known risk of optimising
   against a fixed adversary.
2. **A strict stop fixes the stopping problem.** Requiring both Haiku attackers to agree, with a
   class of at least 10, leaves only 1 user that the loop calls safe and the held-out attacker does not.
3. **The rewrite step is now the bottleneck.** Only 25 of 50 users reached the strict target: 13 ran
   out of rounds and 12 ran out of attributes. The utility check undid 56 rewrites, and 29 times the
   identifying cue sat in a message over 8,000 characters, which the loop does not rewrite. The
   per-message loop has neither limit, which is why it protects more users and also why one in four
   of its edits makes the request noticeably worse.
4. **Targeting works.** Every user-level version edits far less than the per-message loop and keeps
   rewrites usable, so the cost of protection is much lower per message changed.

## Follow-up: a stronger rewrite step (`user_level_improved`)

The strict loop with three changes to the rewrite step, run on the same 50 users (about $1 more;
earlier calls came from the cache):

- a rejected rewrite gets one gentler retry;
- in a message over 8,000 characters, only the passage around the attacker's cue is rewritten;
- up to 10 rounds and 3 tries per attribute (instead of 6 and 2).

| | Strict stop | Improved rewrites |
|---|---|---|
| At-risk users still at risk (of 23) | 17 | 17 |
| Users reaching the target | 25 | 27 |
| Rewrites kept | 228 | 279 |
| Rewrites rejected by the utility check | 56 | 139 |
| Kept after a gentler retry | 0 | 28 |
| Long-message passages rewritten | 0 | 0 |
| Stopped: out of rounds / no attribute left | 13 / 12 | 4 / 19 |
| Messages changed | 161 (8.3%) | 183 (9.5%) |

It did not protect more users. The gentle retry recovered 28 rewrites, but the extra rounds mostly
produced rewrites the utility check rejected, and more users ran out of attributes to try instead
of rounds. The long-message change never fired. The attacker prompt asks for a cue as "a few words
from the message or a short description of the evidence"; on the 57 long messages, all 45 confident
cues were descriptions (median 111 characters), none appeared verbatim, so no passage could be
located.

**Passage by passage (`user_level_passages`).** The improved loop, but a long message is split into
passages of about 4,000 characters and the loop attacker reads each one, so the passages that reveal
the target attribute can be found without relying on the cue text. Same 50 users, about $1.80 more.

| | Strict stop | Improved | Improved + passages |
|---|---|---|---|
| At-risk users still at risk (of 23) | 17 | 17 | 17 |
| Users reaching the target | 25 | 27 | 28 |
| Rewrites kept | 228 | 279 | 316 |
| Rejected by the utility check | 56 | 139 | 146 |
| Long-message passages rewritten | 0 | 0 | 53 (156 passages attacked) |
| Cue out of reach (long message, not rewritten) | 29 | 48 | 11 |
| Stopped: out of rounds / no attribute left | 13 / 12 | 4 / 19 | 5 / 17 |
| Messages changed | 161 (8.3%) | 183 (9.5%) | 198 (10.2%) |

Locating the passage works: long messages are no longer out of reach. Protection still does not move.
Of the 17 users still at risk, 10 stopped because no attribute was left that the loop could change,
after every rewrite of it was rejected as unusable. For these users the identifying details are part
of what the request needs (a CV, a cover letter, a visa application), so they cannot be hidden
without breaking it. The per-message loop protects more of them because it keeps rewrites the judge
rates as damaging.

## Limitations

- 50 users and one dataset: a pilot, not an estimate. Differences of 4–5 users are within noise.
- "At risk" is measured against the 1,982-user Stage 2 sample, not all 25,325 eligible users, so the
  target is strict; classes in the full population would be larger.
- Both the loop and the grader are LLM attackers. The held-out grader is a different, stronger
  model, which limits but does not remove the advantage of optimising against an attacker. Its
  inferences are not checked against ground truth.
- Utility is one LLM judge's rating, not a user study.
- Attribute extraction has low recall and precision on the Stage 1 gold set (29% and 40%), so
  profiles are noisy in both directions.
- The per-message baseline starts after layer 2, not after Kai's layer 3 (generalise every message),
  so it is weaker here than in his full pipeline.
- The pilot users were drawn from users already flagged as at risk, so the results describe
  at-risk users, not the population.

## Next

- Long messages are now handled (passage by passage). The remaining gap is requests whose identifying
  details are what the task needs: test generalising those details (a region for a city, a sector for an
  employer) rather than removing them, and a stronger loop attacker.
- A stronger loop attacker (Sonnet in the loop, Opus as the held-out grader), at roughly $15–20.
- A larger sample (200+ users) once one version is chosen, so differences can be tested.
- Kai's full pipeline is now scored on the adjudicated gold set (see `docs/findings.md`, RQ4).
