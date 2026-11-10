# Hand labelling, step by step (two annotators, two laptops)

We each label the same 300 WildChat user messages on our own laptop, without seeing the detectors'
output or each other's labels. Then we compare, settle disagreements into one gold set, and score the
detectors against it. Start with WildChat; LMSYS and Arena get their own 300 later.

**Nobody sends anybody the messages.** The label set is drawn the same way on both machines (same
code, same config, same seed; it does not depend on your private `hmac.key`), so each of us builds it
locally from our own run. Only the finished label files move between us, at the end.

## 1. Set up (once each)

Both of us must be on the **same commit** of `main`, because the detectors decide which messages are
drawn.

```bash
git switch main && git pull
.venv/bin/pip install --no-deps -e .
.venv/bin/residpii run --config configs/default.yaml --datasets wildchat   # ~2-3 GB streamed, cached after
.venv/bin/residpii label build --config configs/default.yaml --dataset wildchat
```

## 2. Check we have the same 300 messages

Each of us runs this and compares the 12 characters it prints (over chat is fine: it is a hash of
item ids, no text):

```bash
.venv/bin/python -c "import pandas as pd, hashlib, pathlib; d = pd.read_parquet(pathlib.Path.home() / '.residual-pii/runs/default/work/labels/wildchat/items.parquet'); print(len(d), hashlib.sha256(d[['item_id','conversation_id','turn_index']].to_csv(index=False).encode()).hexdigest()[:12])"
```

If they differ, stop: one of us is on a different commit or an old run. Pull, rerun step 1 with
`--force` on `run` and `label build`, and check again.

## 3. Label

```bash
.venv/bin/residpii label serve --config configs/default.yaml --dataset wildchat --annotator ethan   # or kai
```

Open the address it prints (it only works on your own computer). Then for each message:

1. **Select** the exact text of an identifier with the mouse: just the value, `Sarah Jones`, not
   `my name is Sarah Jones`.
2. In the popup, pick a **category** and **whose** it is, then **Add span**. Each distinct value only
   needs marking once.
3. Under **Attributes about the user**, add what the message says or strongly implies about the
   *writer* (age band, occupation, country, ...), marked **stated** or **inferred**.
4. **Save & next** (or Cmd+Enter). A message with nothing in it is saved with no spans; that is a
   real answer, not a skip.

Labels save to `~/.residual-pii/runs/default/restricted/labels/wildchat/<you>.jsonl`. Stop the page
with Ctrl+C and restart it later; you resume where you were. Check how far you are with
`residpii label progress --config configs/default.yaml`.

### What to mark

Mark every identifier or quasi-identifier, **including fictional and public ones** (that is what
`whose` is for), and only count on `whose` to say whether it is private.

| Category | Examples |
|---|---|
| Direct: `name`, `email`, `phone`, `account_id`, `ip_address`, `api_key_token`, `street_address`, `username_handle`, `personal_url`, `date_of_birth` | `account_id` = card, bank, ID, passport, licence numbers. `personal_url` = a profile or personal page, not a docs/news link. |
| Quasi: `location`, `date`, `organization`, `occupation`, `age`, `nrp`, `education`, `health`, `other_quasi` | Only when tied to a person: where they live/work/study, a date in their life. Not places or dates as general topics. `nrp` = nationality, religion, political group. |

| Whose | Use for |
|---|---|
| `self` | The person typing ("I'm Sarah, a nurse in Leeds"). |
| `third_party` | A real private person they mention (boss, spouse, client, an email's recipient). |
| `public_figure` | Celebrities, politicians, well-known authors, historical figures. |
| `fictional` | Story and role-play characters, made-up examples. |
| `not_personal` | Looks like PII but is not about a person (code identifiers, product names). |
| `unclear` | You genuinely cannot tell. Use sparingly; add a note. |

- **Code:** a hard-coded key, token or password is `api_key_token` / `self`. Variable, function and
  class names are not names.
- **Do not mark** template placeholders (`[Your Name]`, `John Doe`, `NAME_1`,
  `<PRESIDIO_ANONYMIZED_PERSON>`) or the chatbot's own name.
- **Inferred attributes:** only what a careful reader would confidently conclude (fixing an NHS shift
  rota → country UK, occupation healthcare). Never guess gender or other traits from writing style.
- **Do not look anyone up**, and never test a key or password you see.
- Don't discuss individual messages until both of us have finished: disagreement is a result.

These are the same rules the LLM annotator gets (`GUIDELINES` in `src/residpii/llm_label.py`), so
human and LLM labels are comparable. If we change a rule, change it there too.

**Do a 20-message trial first:** both label items 1-20, swap files (step 4), run `label agreement`,
and agree on any rule we read differently before doing the rest.

## 4. Swap label files

The person who adjudicates (say Ethan) needs the other's file in their own labels folder:

```
~/.residual-pii/runs/default/restricted/labels/wildchat/kai.jsonl
```

These files hold character offsets, attribute values and notes from real messages, so send them
**only by AirDrop or an encrypted drive**: never email, chat, cloud storage, or the repo.

## 5. Agreement, adjudication, scoring

```bash
.venv/bin/residpii label agreement  --config configs/default.yaml --dataset wildchat   # report these, before adjudication
.venv/bin/residpii label adjudicate --config configs/default.yaml --dataset wildchat   # page prefilled with what we agree on
.venv/bin/residpii label score      --config configs/default.yaml --dataset wildchat   # detector precision/recall vs gold
```

Adjudicate together (screen side by side, same room): the page shows both sets of labels with an
**Accept** button per disagreement. Only the aggregate tables under `results/default/tables/labels_*.csv`
go in the repo.
