# residual-pii

A reproducible pipeline that measures how much personal information remains in public LLM
chat-log datasets after each dataset's own de-identification. All outputs are aggregate.

| Key | Dataset | Existing scrubbing | User proxy |
|---|---|---|---|
| `wildchat` | allenai/WildChat-1M (ODC-BY) | Presidio, in place: `<PRESIDIO_ANONYMIZED_*>` | `hashed_ip` |
| `lmsys` | lmsys/lmsys-chat-1m (gated) | PII conversations removed; names as `NAME_n` | none |
| `arena` | lmsys/chatbot_arena_conversations (gated) | PII conversations removed | `judge` |

**Status:** detection is Presidio + rule-based secrets only. No LLM detector yet. The human-labelling
workflow is built (below), but no labels have been collected; see [docs/limitations.md](docs/limitations.md).

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
```

Restrict to some datasets with `--datasets wildchat arena`, and ignore the cache with `--force`.

### Human labelling (precision / recall benchmark)

Requires a completed `run` on the same config. Each command reads `labeling:` in `configs/default.yaml`.

```bash
.venv/bin/residpii label build      --config configs/default.yaml                 # draw 300 WildChat user turns
.venv/bin/residpii label serve      --config configs/default.yaml --annotator kai   # each annotator, separately
.venv/bin/residpii label progress   --config configs/default.yaml
.venv/bin/residpii label agreement  --config configs/default.yaml                 # span F1, Cohen's kappa
.venv/bin/residpii label adjudicate --config configs/default.yaml                 # resolve disagreements
.venv/bin/residpii label score      --config configs/default.yaml                 # precision/recall vs gold
```

- **Sample.** User turns of up to 4,000 characters, stratified by what the detectors flagged:
  rare direct identifier, name, quasi-identifier only, or nothing. Rare strata are oversampled, and
  inclusion weights correct for that, so weighted precision and recall estimate rates over all
  eligible turns.
- **Blind.** `serve` opens a local page (127.0.0.1 only, with a per-session token in the URL). It never
  shows detector output. Annotators should not look at each other's labels.
- **What to mark.** Every identifier and quasi-identifier span, including fictional and public ones,
  with a category and whose it is (self / third party / public figure / fictional / not personal /
  unclear). Also per-user attributes (age band, occupation, …), each marked stated or inferred.
  Only *self* and *third party* count as real PII when scoring.
- **Adjudicate.** The page is prefilled with the spans both annotators agree on exactly, and each
  disagreement has an Accept button. The result is the gold set.
- **LLM labels.** `label llm --allow-hosted` labels the set with Claude (`claude-sonnet-5-5` by default; `--model claude-haiku-5-5` is cheapest) under the
  same guideline, writing annotator `llm`. Score against them with `label score --gold llm`. LLM labels
  are not human ground truth: report them as agreement with an LLM annotator. Needs
  `pip install -e '.[llm]'` and an `ANTHROPIC_API_KEY`. Use `--limit 20` for a pilot. Reruns resume.
- **Outputs.** Labels: `~/.residual-pii/runs/<run>/restricted/labels/<annotator>.jsonl`, append-only,
  latest record per item wins. Aggregate tables: `results/<run>/tables/labels_*.csv`, with precision
  and recall at each Presidio threshold, a breakdown of what flagged spans actually are, and gold
  prevalence. Before adjudication, `label score --gold kai` scores against a single annotator.

### Storage
Data is streamed column by column from the Hub; nothing is downloaded in full. Text is read only
for row groups that contain sampled conversations, and only the user `content`/`role` fields.
Local disk use is the sampled user turns (a few MB in smoke runs, tens of MB in full runs).
A full WildChat run streams roughly 2–3 GB over the network.

## Layout

```
configs/      default.yaml, smoke.yaml (extends default), taxonomy.yaml
src/residpii/
  sources.py     streaming Hub adapters → common schema; revision pinning
  sampling.py    stratified sample, one conversation per user proxy, design weights
  redactions.py  existing-placeholder matching + k-thresholded format discovery
  detect.py      Presidio + secrets detectors → SafeSpan; dummy-value filter; task heuristics
  dedup.py       exact + MinHash-LSH near-duplicate matching
  analysis.py    weighted prevalence, stratified bootstrap CIs, threshold sensitivity
  report.py      tables, figures, methods.md, disclosure drafts
  safety.py      keyed span hashing, SafeSpan, output-table guard
  labeling.py    label-set sampling, label store, agreement, precision/recall vs gold
  annotate.py    local blind labelling server (+ annotate.html)
  pipeline.py    stages + content-keyed cache
results/<run>/   aggregate outputs (safe to commit)
~/.residual-pii/ raw sampled text + masked span store (outside the repo, 0700)
```

## Privacy and ethics rules (enforced in code where possible)
- **No re-identification.** No web search, no external lookups, no linking across users beyond
  text deduplication. Detected credentials are never tested.
- **Raw spans are never written.** Spans are stored as offsets, category, length, and a keyed
  HMAC-SHA256 (key at `~/.residual-pii/hmac.key`, 0600). `assert_safe_table` refuses to write any
  results table with an unvetted text column. `tests/test_pipeline.py` injects fabricated canary
  PII and fails if any of it reaches results, work files, or console output.
- **Restricted data stays outside the repo**, in `~/.residual-pii/runs/<run>/restricted` (0700).
- Any example in the paper must be paraphrased and altered.
- `results/<run>/disclosure/*.md` are notification drafts for maintainers: aggregates only, with
  an offer to share conversation IDs privately.
- Respect each dataset's licence and terms; LMSYS datasets require accepting their agreements.
- **Hosted LLM: opt-in only.** `residpii label llm --allow-hosted` sends the sampled user turns
  (the 300-item label set, nothing else) to Anthropic's API to label them. Nothing else in the
  pipeline calls a hosted model. Its labels and raw responses stay in the restricted directory.

## Reproducibility
Dataset revisions are pinned in `configs/default.yaml` and recorded in `results/<run>/manifest.json`
along with the config hash and sampling counts. Seeds are fixed. `requirements.lock` pins the
tested environment (Python 3.14, macOS arm64).
