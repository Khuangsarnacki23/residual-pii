# Annotation guide

For hand-labeling a batch written by `residpii label-export`. The labels are the gold standard that
`residpii label-score` compares the detectors against.

## The quick way: the review page

```bash
residpii review --config <the config used for the run>
```

opens a page in your browser, served from this computer only. Tab 2 creates or opens a batch: select
text in a message, pick who it is about and what it is, and the span is saved to the batch file
described below. Tab 1 is the sanity check of the LLM detector: a sample of its extractions, each
shown next to the message it came from. Both write only to `restricted/labels/`. The rules below
apply either way; the spreadsheet route still works.

## Before you start

- The batch file contains raw user text. It lives in
  `~/.residual-pii/runs/<run>/restricted/labels/`. **Annotate it there.** Do not copy it into the
  repo, email it, upload it, or paste text from it into a hosted tool.
- **Do not look anyone up.** Label only what the text says. No web search, no checking whether a
  name, address or key is real.
- If you open a CSV in a spreadsheet program, import the `text` column as text. A cell that starts
  with `=`, `+`, `-` or `@` would otherwise be read as a formula.
- Never test a credential you see.

## The file

One row per exported turn to begin with. Columns `item_id`, `dataset`, `conversation_id`,
`turn_index` and `text` are filled in; leave them as they are. You fill in:

| Column | What to enter |
|---|---|
| `span_text` | The exact characters of one piece of personal information, copied from `text`. |
| `start`, `end` | Optional character offsets (0-based, end exclusive). Leave blank unless the same string appears more than once and only one occurrence counts. If blank, every occurrence of `span_text` in the turn is treated as labeled. |
| `pii_type` | One type from the table below, or `none`. |
| `about` | `user`, `third_party`, `public` or `fictional` (below). |
| `sensitive_topic` | Optional: `health`, `sexual_orientation`, `religion`, `politics`, `finances`, `criminal_record`, `immigration`, or blank. Describes the turn, not the span. |
| `notes` | Anything a second annotator should know. No need to repeat the text. |

**One row per span.** If a turn has several spans, copy the row (at least its `item_id`) once per
extra span. If a turn has no personal information, keep its single row and set `pii_type` to
`none`. Every item must end up with at least one row that has a `pii_type`.

## PII types

Use these names exactly; they are the categories in `configs/taxonomy.yaml`.

| `pii_type` | Label | Do not label |
|---|---|---|
| `name` | A person's full or partial name, including a first name alone when it refers to a specific person. | Company, product and model names. Honorifics alone ("Sir"). Generic roles ("my boss"). |
| `email` | Email addresses. | |
| `phone` | Phone and fax numbers. | Other long numbers (order numbers, IDs). |
| `account_id` | Government and financial identifiers: SSN, passport, driver's licence, bank account, IBAN, card number, crypto wallet, medical licence. | |
| `ip_address` | IPv4 and IPv6 addresses. | Version numbers that look like IPs ("1.2.3.4" of a library). |
| `handle` | Usernames and @handles on a named platform ("instagram @...", "discord name#1234"). | Handles of companies and public accounts mentioned as a topic. |
| `personal_url` | A link to a person's profile or account on a social or code platform (LinkedIn, GitHub, Facebook, Instagram, X, TikTok, Telegram, Reddit). | Links to a repository, post or company page. |
| `url` | Other URLs that point to a person (personal site, private document link). | Links to public sites, docs and news. |
| `api_key_token` | API keys, access tokens, private keys, passwords. | Variable names (`API_KEY`) without a value. |
| `street_address` | A street or postal address, with the house number. | A street or city named without a house number (`location`). |
| `location` | Places: country, region, city, named building. | Places in a general-knowledge question with no person attached ("capital of France") still count as `location`; mark `about` as `public`. |
| `date_of_birth` | A full date of birth (day, month and year). | A birth year or age alone (`date`). |
| `date` | Other dates tied to a person or event in their life (appointment, hire date, birth year). | Dates in general text ("in 1066"), durations, times of day. |
| `nrp` | Nationality, religion, political affiliation, ethnicity of a person or group of people. | Languages ("translate into French"). |
| `organization` | Employers, schools, hospitals, clubs and other organisations a person is linked to. | Well-known companies mentioned as a topic ("how does Google rank pages"). |
| `none` | The turn contains no personal information. | |

Label the smallest span that carries the information: `Ada Lovelace`, not `my name is Ada
Lovelace`. A full address is one `location` span. Label every occurrence only if they differ; a
repeated identical string needs one row.

## `about`

| Value | Use when |
|---|---|
| `user` | The information is about the person writing: "I'm a nurse in Leeds", "my email is ...". Text the user wants written *in their own name* (their cover letter, their bio) counts as `user`. |
| `third_party` | A real private person other than the user: a colleague, family member, customer, the recipient of a letter. |
| `public` | A public figure, or a public entity or place mentioned as a topic. |
| `fictional` | A character in a story, role-play or hypothetical; an invented persona. |

If you cannot tell whether a person is real, prefer `third_party` when the text reads as a real
situation and `fictional` when it reads as creative writing, and say so in `notes`.

## Edge cases

- **Code identifiers.** Variable, function and class names are not names of people (`john_doe =
  ...` is a variable). A real-looking name, email or key inside a string literal or comment *is*
  labeled. Author lines and commit signatures are `name`/`email`, `third_party` unless the user
  says the code is theirs.
- **Fictional characters.** Label the name with `about = fictional`. Do not skip it: the detectors
  flag these and we need to know how often.
- **Public figures.** Label the name with `about = public`. A politician's office address is
  `public`; their home address is `third_party`.
- **Existing placeholders.** `NAME_1`, `<PRESIDIO_ANONYMIZED_EMAIL_ADDRESS>` and similar tokens
  were inserted by the dataset's own scrubbing. Do not label them.
- **Dummy values.** `john@example.com`, `555-0100`, `123-45-6789`, `John Doe`, `xxxx-xxxx`,
  `YOUR_API_KEY`: do not label. If a value looks like a template but could be real, label it and
  add `possibly dummy` to `notes`.
- **Pasted documents.** A pasted email, CV or chat log is labeled like any other text; `about`
  follows whom each span describes, which is often `third_party`.
- **Non-English text.** Label it if you can read it; otherwise set `pii_type` to `none` and write
  `not read: <language>` in `notes` so the item can be excluded.
- **Truncated text.** Long turns are cut at `label.max_text_chars`. Label what is there.
- **Indirect information.** "I just turned thirty", "the only cardiologist in my town": there is no
  single PII span. Leave spans out and describe it in `notes`. The LLM detector measures these;
  the span labels do not.

## Scoring

```bash
residpii label-score --config <the config used for the export> --labels <annotated file>
```

A detector span matches a gold span when both have the same category and they overlap by at least
half of the shorter span. The output, `results/<run>/tables/detector_validation.csv`, holds counts
and precision, recall and F1 per detector and category, plus Cohen's kappa between the LLM and the
annotator on `about`. It contains no text. Add `--second-annotator llm` to run the LLM on exactly
the labeled turns first.

Two annotators should label an overlapping subset independently so that human agreement can be
reported alongside.

## Attributes (what the message reveals about a person)

Besides spans, record each quasi-identifying attribute the message reveals: occupation, employer
type, city, state or region, country, age or age range, gender, school, education level,
relationship status, nationality, native language. One entry per attribute type and person.

- **stated**: the message says it outright ("I'm a nurse", "my wife and I"). Also select the words
  as a span when a span type fits (a city is a `location`).
- **inferred**: never said, but a careful reader would conclude it with reasonable confidence from
  topic, wording or circumstances (asking how to prepare for a first-year calculus midterm implies a
  student). Do not record guesses from stereotype: a cooking question says nothing about gender,
  and a British spelling alone does not settle a country.
- **about**: the same four choices as for spans. Attributes of a character in a story are
  `fictional`.
- The value is optional and is not compared between annotators; type, stated/inferred and about are.

Inferred labels are where two annotators differ most. That disagreement is a result, so do not
discuss individual messages until both of you have finished the batch.

## Two annotators

1. Both read this guide and label the same 20-message trial batch, then compare
   (`residpii label-agreement`) and settle any rule that was read differently. Update this guide.
2. Both label the main batch independently, each with `--annotator <name>`.
3. First pass blind. Use "Compare with the detectors" only after labeling a message yourself;
   anything added then is marked and left out of recall.
4. Run `label-agreement`. Resolve disagreements together into one adjudicated file, and report the
   agreement figures from before adjudication.
