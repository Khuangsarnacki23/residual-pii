# residual-pii

A reproducible pipeline that measures how much personal information remains in public LLM
chat-log datasets after each dataset's own de-identification. All outputs are aggregate.

| Key | Dataset | Existing scrubbing | User proxy |
|---|---|---|---|
| `wildchat` | allenai/WildChat-1M (ODC-BY) | Presidio, in place: `<PRESIDIO_ANONYMIZED_*>` | `hashed_ip` |
| `lmsys` | lmsys/lmsys-chat-1m (gated) | PII conversations removed; names as `NAME_n` | none |
| `arena` | lmsys/chatbot_arena_conversations (gated) | PII conversations removed | `judge` |

**Detection layers**

1. Rule-based: Presidio pattern recognizers, `secrets` (API keys, tokens, passwords), `selfid`
   (self-introductions such as "my name is ...") and `rules` (types Presidio has no recognizer for:
   profile links, handles, street addresses, dates of birth, IDs named in context, international
   phone numbers). Text that `rules` and another detector both find is counted once.
2. Statistical NER: spaCy through Presidio.
3. LLM (off by default; see below): direct identifiers quoted from the turn and stored as located
   spans, each with whom it belongs to, and quasi-identifiers, stated and inferred.

**Status:** no human validation yet, so all numbers are detector-reported. The labeling export and
scoring commands exist but no batch has been labeled. The LLM layer has been tested with a fake
backend only. See [docs/limitations.md](docs/limitations.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
.venv/bin/pip install --no-deps -e .
```

For the two gated LMSYS datasets: accept the terms on both dataset pages on Hugging Face, then
run `.venv/bin/hf auth login` (your token stays in your HF config; the pipeline never reads it).

## Usage

```bash
.venv/bin/python -m pytest -q                              # unit tests + canary leak test
.venv/bin/residpii run --config configs/smoke.yaml         # ~100 conversations/dataset
.venv/bin/residpii run --config configs/default.yaml       # full 5,000/dataset sample
.venv/bin/residpii dedup-corpus --config configs/default.yaml --confirm   # full lmsys vs Arena overlap
.venv/bin/residpii placeholders --config configs/smoke.yaml               # candidate placeholder formats
```

Restrict to some datasets with `--datasets wildchat arena`, and ignore the cache with `--force`.

### User-level sampling

```bash
.venv/bin/residpii user-activity --config configs/user_level.yaml         # users by conversation count (metadata only)
.venv/bin/residpii run --config configs/user_level_smoke.yaml             # ~50 users, exercises the path
.venv/bin/residpii run --config configs/user_level.yaml                   # 2,000 users with >= 3 conversations
```

`sampling.mode: user_level` samples users instead of conversations: users with at least
`user_level.min_conversations` conversations, up to `user_level.n_users` of them, each with up to
`user_level.max_conversations_per_user` conversations. Conversations are grouped only by the
dataset's own user proxy. Datasets without one (lmsys) are skipped in this mode. `roles: [user,
assistant]` also fetches assistant turns, which the LLM uses as context.

With `llm.enabled: true` the run continues with LLM extraction of quasi-identifiers.

### The LLM detector is opt-in

`llm.enabled` is `false` by default. There is no local backend: `llm.backend` is `fireworks`
(default), `anthropic` or `openai`, and each one sends sampled user text to a third party. The run
**aborts** unless `llm.allow_hosted: true` is also set. Only set it if the dataset terms, your
instructor/IRB and your partner all allow it.

```bash
export FIREWORKS_API_KEY=...          # read from the environment; never logged or written to disk
# set llm.allow_hosted: true in configs/fireworks_pilot.yaml, then:
.venv/bin/residpii run --config configs/fireworks_pilot.yaml --datasets wildchat
```

`configs/fireworks_pilot.yaml` runs the detector on the user-level smoke sample (about 50 users,
1,000 user turns), which is the place to compare models before a larger run. `llm.model` is any
serverless chat model id from the Fireworks library.

**Cost control.** One call is roughly 1,300 input tokens (950 of them the fixed prompt) and a
short JSON reply. `llm.sample_turns` caps how many turns are sent (whole users are kept together
in user-level mode); the full user-level sample is roughly 40,000 user turns. Responses are cached,
so a rerun, or a run stopped by an exhausted balance, only pays for turns not yet done.
`llm_run_stats.csv` reports the input and output tokens billed; multiply by the model's prices.

The `anthropic` and `openai` backends need `pip install anthropic` or `openai`, and those SDKs read
their credentials from the environment.

### Hand labeling

```bash
.venv/bin/residpii label-export --config configs/default.yaml --n 300
.venv/bin/residpii label-score  --config configs/default.yaml --labels <annotated file>
```

`label-export` writes a stratified batch of user turns to
`~/.residual-pii/runs/<run>/restricted/labels/`. Annotate it there, following
[docs/annotation_guide.md](docs/annotation_guide.md). `label-score` writes precision, recall and F1
per detector and category to `results/<run>/tables/detector_validation.csv`. A batch that is only
partly annotated is scored on the turns that have labels.

`residpii review --config <config>` does both steps in a browser page instead of a spreadsheet, and
adds a check of the LLM detector: a sample of its extractions (`--n-attrs`, default 20), each next
to the message it came from, to mark as right or wrong.

Labeling is keyboard-first: select text and press the key for its type (`N` name, `E` email, `P`
phone, ...; the page lists them), `1`-`4` to set whom it is about, `Enter` to repeat the last type.
The same text is labeled everywhere it appears in the message. With nothing selected, `N` marks the
message as holding no personal information and moves on, `Enter` goes to the next unlabeled message,
`Z` undoes. Once a message is labeled, `R` shows what the detectors flagged that you did not, so
misses can be caught without the detectors steering the first pass; a label added at that point is
saved with `after_reveal = yes`, and `label-score` reports recall against the blind labels alone
(`gold = blind_only`) next to the figures for all labels.

Below the spans, each message also takes **attribute labels**: what it reveals about a person (job,
place, age, ...), marked *stated* or *inferred* and with whom it is about. An inferred attribute
has no text to select, so these are stored beside the batch in `attrs_<batch>.csv`, and `label-score`
writes `attribute_validation.csv` (the LLM's attributes against them, by stated and inferred).

For a gold set, two people label the same batch independently:

```bash
.venv/bin/residpii review --config <config> --annotator ann     # writes batch_<date>__ann.csv
.venv/bin/residpii review --config <config> --annotator ben     # writes batch_<date>__ben.csv
.venv/bin/residpii label-agreement --config <config> --a <ann's file> --b <ben's file>
```

Each annotator works on their own copy of the batch; the exported batch is never written to.
`label-agreement` writes `annotator_agreement.csv`: span F1 per category, kappa on whether a turn
holds any personal information, and F1 on attribute labels for stated and inferred separately.
Both annotators need the same batch file and attribute file; move them between machines only by a
channel suitable for raw user text. The page is served on 127.0.0.1 only, needs
the one-time token in the address it prints, loads nothing from the network, and writes answers
only to `restricted/labels/`. Stop it with Ctrl-C.

### Storage
Data is streamed column by column from the Hub; nothing is downloaded in full. Text is read only
for row groups that contain sampled conversations, and only the user `content`/`role` fields.
Local disk use is the sampled user turns (a few MB in smoke runs, tens of MB in full runs).
A full WildChat run streams roughly 2–3 GB over the network.

## Layout

```
configs/      default.yaml, smoke.yaml, user_level.yaml, user_level_smoke.yaml (each extends the previous),
              taxonomy.yaml, prompts/qi_extract_v1.txt
src/residpii/
  sources.py     streaming Hub adapters → common schema; revision pinning; turn roles
  sampling.py    stratified one-conversation-per-user sample, or a user-level sample
  redactions.py  existing-placeholder matching + k-thresholded format discovery
  detect.py      Presidio + secrets + selfid + rules detectors → SafeSpan; dummy-value filter; task heuristics
  llm.py         LLM quasi-identifier extraction: backends, strict parser, cache, aggregates
  label.py       labeling export, span matching, precision/recall/F1, Cohen's kappa
  dedup.py       exact + MinHash-LSH near-duplicate matching
  analysis.py    weighted prevalence, stratified bootstrap CIs, threshold sensitivity
  report.py      tables, figures, methods.md, disclosure drafts
  safety.py      keyed span hashing, SafeSpan, output-table guard, small-cell suppression
  pipeline.py    stages + content-keyed cache
docs/            limitations.md, annotation_guide.md
results/<run>/   aggregate outputs (safe to commit)
~/.residual-pii/ raw sampled text, span store, LLM outputs, label batches (outside the repo, 0700)
```

### Output tables for user-level sampling, the LLM detector and validation

| Table | Contents |
|---|---|
| `user_activity.csv` | Users by conversations-per-user bucket. |
| `prevalence_by_role.csv` | Detector prevalence in user vs assistant turns (when both are fetched). |
| `llm_attribute_prevalence.csv` | Share of users and conversations with ≥1 attribute, by type × evidence × about. |
| `llm_span_prevalence.csv` | Share of users and conversations with ≥1 direct identifier quoted by the LLM, by category × about. |
| `llm_detector_agreement.csv` | Per category and detector: spans, how many overlap a span from the other side (LLM vs the rest), and what the LLM said the overlapped text is about. |
| `llm_run_stats.csv` | Turns processed, failures, latency, backend, model, prompt version. |
| `detector_validation.csv` | Precision, recall, F1 per detector × category; LLM-vs-human kappa on `about`. |
| `attribute_validation.csv` | LLM attributes against hand-labeled ones: precision, recall, F1 per attribute type, for stated and inferred. |
| `annotator_agreement.csv` | Two annotators on one batch: span F1 per category, turn-level kappa, attribute F1 (stated, inferred). |
| `llm_attribute_review.csv` | Manual check of LLM extractions: reviewed count and share where the value is right, `about` and evidence agree, or the model missed something. |

In these tables a cell based on fewer than `publication.min_cell` (10) users is blank and the row
has `suppressed = True`.

## Privacy and ethics rules (enforced in code where possible)
- **No re-identification.** No web search, no external lookups, no linking any record to a real
  identity. Conversations are grouped only by the dataset's own user proxy, which is replaced by a
  keyed HMAC (`user_key`) as it is read; nothing else links users. Detected credentials are never
  tested.
- **Raw spans are never written.** Spans are stored as offsets, category, length, and a keyed
  HMAC-SHA256 (key at `~/.residual-pii/hmac.key`, 0600).
- **Raw text and extracted values stay outside the repo**, in
  `~/.residual-pii/runs/<run>/restricted` (0700). That includes LLM outputs and
  label batches, which are personal data themselves.
- **Everything under `results/<run>/` is aggregate.** `assert_safe_table` refuses to write a
  results table with an unvetted text column; a text column is vetted only if it holds category
  labels, never values. New user-level tables suppress small cells.
- **The canary test covers every stage that touches text.** The tests inject fabricated canary PII
  (including through the fake LLM backend) and fail if any of it reaches results, work files, or
  console output.
- **Hosted LLMs are opt-in** (`llm.allow_hosted`, default `false`), because they send user text to
  a third party.
- Any example in the paper must be paraphrased and altered.
- `results/<run>/disclosure/*.md` are notification drafts for maintainers: aggregates only, with
  an offer to share conversation IDs privately.
- Respect each dataset's licence and terms; LMSYS datasets require accepting their agreements.

## Reproducibility
Dataset revisions are pinned in `configs/default.yaml` and recorded in `results/<run>/manifest.json`
along with the config hash and sampling counts. Seeds are fixed. `requirements.lock` pins the
tested environment (Python 3.14, macOS arm64).
