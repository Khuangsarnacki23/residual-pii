"""LLM detector for quasi-identifiers (stated and inferred).

Layer 3 of detection: for each sampled user turn, an LLM returns the quasi-identifying attributes it
reveals (occupation, city, age, ...), whether each was stated or inferred, and whom it is about.

Privacy rules this module enforces:
  - Every backend (fireworks, anthropic, openai) is a hosted API: it sends sampled user text to a
    third party. A run is refused unless `llm.allow_hosted` is true, and `llm.enabled` is off by
    default.
  - Extracted values are personal data. They are written only under the restricted directory
    (the response cache and `llm_attrs_<dataset>.parquet`). Everything this module returns for
    results/ is counts and shares over closed label sets.
  - Nothing here logs prompt or response text, and error messages carry reason codes only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import load_taxonomy
from .detect import is_dummy
from .redactions import compile_patterns, placeholder_spans
from .safety import SafeSpan, keyed_hash, suppress_small_cells

ATTRIBUTE_TYPES = ("occupation", "employer_type", "city", "state_or_region", "country", "age", "age_range",
                   "gender", "school", "education_level", "relationship_status", "nationality",
                   "language_native", "other")
EVIDENCE = ("stated", "inferred")
ABOUT = ("user", "third_party", "public_figure", "fictional", "unclear")
# Direct identifiers the model quotes from the turn (llm.extract_spans). Each type is a category in
# configs/taxonomy.yaml; the entity is the label that category already uses, so the taxonomy and the
# dummy-value filter apply unchanged.
SPAN_TYPES = {
    "name": ("PERSON", "a person's name, full or partial"),
    "email": ("EMAIL_ADDRESS", "an email address"),
    "phone": ("PHONE_NUMBER", "a phone or fax number"),
    "street_address": ("STREET_ADDRESS", "a street or postal address"),
    "account_id": ("CONTEXT_ID", "a government, financial, student, employee or other personal ID number"),
    "ip_address": ("IP_ADDRESS", "an IP address"),
    "personal_url": ("PERSONAL_URL", "a link to a person's profile, account or personal site"),
    "handle": ("SOCIAL_HANDLE", "a username or @handle"),
    "api_key_token": ("SECRET_TOKEN", "an API key, access token, private key or password"),
    "date_of_birth": ("DATE_OF_BIRTH", "a full date of birth"),
}
# What each task type in configs/taxonomy.yaml means, for the prompt. A type without an entry is
# listed by name alone.
TASK_DESCRIPTIONS = {
    "story_roleplay": "fiction, role-play, fan fiction, scripts, world-building, character or plot work",
    "code": "writing, fixing, explaining or running code, queries, configuration or other technical output",
    "translation": "translating text from one language to another",
    "job_application": "resumes, cover letters, applications, personal statements, interview preparation",
    "communication": "drafting or answering an email, letter, message or post meant for a real recipient",
    "editing": "rewriting, proofreading, paraphrasing or improving text the writer supplied",
    "summarization": "summarising or extracting points from text the writer supplied",
    "creative_other": "poems, lyrics, jokes, slogans, names and other short creative pieces",
    "info_or_other": "questions, explanations, advice, analysis and anything else",
}
MAX_SPAN_OCCURRENCES = 50   # a quoted identifier is located at most this many times in one turn

# SOC 2018 major groups, plus two codes for people outside the occupational classification.
OCCUPATION_GROUPS = {
    "11": "Management", "13": "Business and Financial Operations", "15": "Computer and Mathematical",
    "17": "Architecture and Engineering", "19": "Life, Physical, and Social Science",
    "21": "Community and Social Service", "23": "Legal", "25": "Educational Instruction and Library",
    "27": "Arts, Design, Entertainment, Sports, and Media", "29": "Healthcare Practitioners and Technical",
    "31": "Healthcare Support", "33": "Protective Service", "35": "Food Preparation and Serving Related",
    "37": "Building and Grounds Cleaning and Maintenance", "39": "Personal Care and Service",
    "41": "Sales and Related", "43": "Office and Administrative Support",
    "45": "Farming, Fishing, and Forestry", "47": "Construction and Extraction",
    "49": "Installation, Maintenance, and Repair", "51": "Production",
    "53": "Transportation and Material Moving", "55": "Military Specific",
    "ST": "Student", "NW": "Not working (retired, unemployed, homemaker)",
}
_MINOR_GROUP = re.compile(r"(\d\d)-(\d)\d{0,3}")   # "15-1", "15-1200" or a detailed code "15-1252"
MAX_VALUE_CHARS = 200
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
HOSTED_BACKENDS = ("fireworks", "anthropic", "openai")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- errors


class HostedBackendNotAllowed(RuntimeError):
    """A backend that sends text off this machine was configured without llm.allow_hosted."""


class BackendUnavailable(RuntimeError):
    """The backend could not be reached at all; the run stops instead of recording failures."""


class BackendRefusal(Exception):
    """The model declined to answer. Recorded as a failure; not retried."""


class ParseError(ValueError):
    """The response did not match the schema. The message is a reason code, never response text."""


# ---------------------------------------------------------------- schema


@dataclass(frozen=True, slots=True)
class Attribute:
    type: str
    value: str
    evidence: str
    about: str
    confidence: float
    occupation_group: str = ""
    occupation_minor_group: str = ""


def parse_response(raw: str) -> list[Attribute]:
    """Strict parser: any deviation from the schema rejects the whole response."""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        raise ParseError("not_json") from None
    if not isinstance(obj, dict) or not isinstance(obj.get("attributes"), list):
        raise ParseError("no_attributes_list")
    out = []
    for item in obj["attributes"]:
        if not isinstance(item, dict):
            raise ParseError("attribute_not_object")
        value = item.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)                                # models often give age as a number
        if not isinstance(value, str) or not value.strip() or len(value) > MAX_VALUE_CHARS:
            raise ParseError("bad_value")
        if item.get("type") not in ATTRIBUTE_TYPES:
            raise ParseError("bad_type")
        if item.get("evidence") not in EVIDENCE:
            raise ParseError("bad_evidence")
        if item.get("about") not in ABOUT:
            raise ParseError("bad_about")
        conf = item.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            raise ParseError("bad_confidence")
        group = item.get("occupation_group") or ""
        minor = item.get("occupation_minor_group") or ""
        if not isinstance(group, str) or not isinstance(minor, str):
            raise ParseError("bad_occupation_group")
        # The occupation codes are optional extras: an unusable code is dropped, not fatal.
        group = group.strip().upper()
        if item["type"] != "occupation" or group not in OCCUPATION_GROUPS:
            group = ""
        m = _MINOR_GROUP.fullmatch(minor.strip())
        minor = f"{m.group(1)}-{m.group(2)}" if m and group and m.group(1) == group else ""
        out.append(Attribute(item["type"], value.strip(), item["evidence"], item["about"], float(conf),
                             group, minor))
    return out


def parse_task(raw: str, allowed: tuple[str, ...]) -> str | None:
    """The "task_type" of a response; None when the answer has none. A value outside `allowed`
    rejects the response."""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        raise ParseError("not_json") from None
    task = obj.get("task_type") if isinstance(obj, dict) else None
    if task is not None and task not in allowed:
        raise ParseError("bad_task_type")
    return task


def parse_spans(raw: str) -> list[dict]:
    """The "spans" list of a response, as strictly as `parse_response`. A response without the key
    has no spans."""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        raise ParseError("not_json") from None
    items = obj.get("spans", []) if isinstance(obj, dict) else None
    if not isinstance(items, list):
        raise ParseError("no_spans_list")
    out = []
    for item in items:
        if not isinstance(item, dict):
            raise ParseError("span_not_object")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_VALUE_CHARS:
            raise ParseError("bad_span_text")
        if item.get("type") not in SPAN_TYPES:
            raise ParseError("bad_span_type")
        if item.get("about") not in ABOUT:
            raise ParseError("bad_about")
        out.append({"text": text, "type": item["type"], "about": item["about"]})
    return out


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"attributes": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": list(ATTRIBUTE_TYPES)},
            "value": {"type": "string"},
            "evidence": {"type": "string", "enum": list(EVIDENCE)},
            "about": {"type": "string", "enum": list(ABOUT)},
            "confidence": {"type": "number"},
            "occupation_group": {"type": "string"},
            "occupation_minor_group": {"type": "string"},
        },
        "required": ["type", "value", "evidence", "about", "confidence", "occupation_group",
                     "occupation_minor_group"],
        "additionalProperties": False,
    }}},
    "required": ["attributes"],
    "additionalProperties": False,
}


_SPAN_SCHEMA = {"type": "array", "items": {
    "type": "object",
    "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": list(SPAN_TYPES)},
                   "about": {"type": "string", "enum": list(ABOUT)}},
    "required": ["text", "type", "about"],
    "additionalProperties": False,
}}
RESPONSE_SCHEMA_SPANS = {**RESPONSE_SCHEMA, "required": ["task_type", "spans", "attributes"],
                         "properties": {"task_type": {"type": "string"}, "spans": _SPAN_SCHEMA,
                                        **RESPONSE_SCHEMA["properties"]}}


def response_schema(lcfg: dict) -> dict:
    return RESPONSE_SCHEMA_SPANS if lcfg.get("extract_spans") else RESPONSE_SCHEMA


# ---------------------------------------------------------------- prompt


@dataclass(frozen=True)
class Prompt:
    system: str
    user_template: str
    sha: str
    task_types: tuple[str, ...] = ()     # non-empty when the prompt asks for a task label

    def render(self, turn: str, context: str) -> tuple[str, str]:
        fill = {"turn": turn, "context": context or "(none)"}
        # One pass, so placeholder-looking text inside a turn is never substituted again.
        return self.system, re.sub(r"\{\{(turn|context)\}\}", lambda m: fill[m.group(1)], self.user_template)


def load_prompt(cfg: dict) -> Prompt:
    path = Path(cfg["prompts_dir"]) / f"qi_extract_{cfg['llm']['prompt_version']}.txt"
    raw = path.read_text()
    if "=====USER=====" not in raw:
        raise ValueError(f"{path} must contain a '=====USER=====' line separating system and user parts")
    if bool(cfg["llm"].get("extract_spans")) != ("{{span_types}}" in raw):
        raise ValueError(f"llm.extract_spans={cfg['llm'].get('extract_spans')!r} does not match {path.name}: "
                         "a prompt asks for spans exactly when it contains {{span_types}}")
    system, user = raw.split("=====USER=====", 1)
    span_types = "\n".join(f"     {name}: {desc}" for name, (_, desc) in SPAN_TYPES.items())
    system = system.replace("{{span_types}}", span_types)
    task_types: tuple[str, ...] = ()
    if "{{task_types}}" in raw:
        taxonomy = load_taxonomy(cfg)
        task_types = (*taxonomy["task_types"], taxonomy["default_task_type"])
        system = system.replace("{{task_types}}", "\n".join(
            f"     {t}: {TASK_DESCRIPTIONS[t]}" if t in TASK_DESCRIPTIONS else f"     {t}" for t in task_types))
    groups = "\n".join(f"    {code}  {name}" for code, name in OCCUPATION_GROUPS.items())
    system = system.replace("{{types}}", ", ".join(ATTRIBUTE_TYPES)).replace("{{occupation_groups}}", groups)
    # The hash keys the response cache, so it covers the task list the prompt was filled with.
    sha = hashlib.sha256((raw + "|".join(task_types)).encode()).hexdigest()[:16]
    return Prompt(system.strip(), user.strip(), sha, task_types)


# ---------------------------------------------------------------- backends


def check_backend_allowed(lcfg: dict) -> None:
    """Abort unless hosted use was explicitly switched on. Every real backend is hosted."""
    backend = lcfg.get("backend")
    if backend not in HOSTED_BACKENDS:
        raise ValueError(f"Unknown llm.backend: {backend!r} (expected one of {', '.join(HOSTED_BACKENDS)})")
    if not lcfg.get("allow_hosted"):
        raise HostedBackendNotAllowed(
            f"llm.backend={backend!r} sends sampled user text to a third party. That needs the dataset "
            "terms, your instructor/IRB and your partner to allow it; if they do, set "
            "llm.allow_hosted: true in your config.")


class FireworksBackend:
    """Hosted. Fireworks' OpenAI-compatible chat completions endpoint with a JSON-schema response
    format. The key comes from FIREWORKS_API_KEY and is never logged or put in an error message."""

    name = "fireworks"
    RETRY_STATUS = (408, 429, 500, 502, 503, 504)

    def __init__(self, lcfg: dict):
        self._key = os.environ.get("FIREWORKS_API_KEY")
        if not self._key:
            raise BackendUnavailable("llm.backend=fireworks needs the FIREWORKS_API_KEY environment variable")
        self.model = lcfg["model"]
        self.temperature = lcfg.get("temperature", 0)
        self.timeout = lcfg.get("timeout_s", 120)
        self.max_tokens = int(lcfg.get("max_output_tokens", 2000))
        # json_schema constrains the answer to the response schema (and turns reasoning output off);
        # json_object only guarantees valid JSON, for models without schema support.
        if lcfg.get("fireworks_response_format", "json_schema") == "json_schema":
            self.response_format = {"type": "json_schema",
                                    "json_schema": {"name": "quasi_identifiers", "schema": response_schema(lcfg)}}
        else:
            self.response_format = {"type": "json_object"}
        # Reasoning models bill their thinking as output tokens; for this extraction task most of the
        # spend is reasoning, so it is kept low. null leaves the model's default.
        self.reasoning_effort = lcfg.get("fireworks_reasoning_effort", "low")
        self._local = threading.local()

    def last_usage(self) -> tuple[int, int]:
        """(input tokens, output tokens) billed for this thread's most recent call."""
        return getattr(self._local, "usage", (0, 0))

    def complete(self, system: str, user: str) -> str:
        self._local.usage = (0, 0)
        body = {"model": self.model, "max_tokens": self.max_tokens, "response_format": self.response_format,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        req = urllib.request.Request(FIREWORKS_URL, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self._key}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in self.RETRY_STATUS and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                hint = {401: "check FIREWORKS_API_KEY", 403: "check FIREWORKS_API_KEY and model access",
                        402: "the account is out of credit", 404: "check llm.model",
                        400: "check llm.model; if it has no JSON-schema support set "
                             "llm.fireworks_response_format: json_object",
                        429: "rate limited or out of credit; lower llm.concurrency"}.get(e.code, "")
                # Stop the run: finished turns are cached, so a rerun picks up where this one ended.
                raise BackendUnavailable(f"Fireworks returned HTTP {e.code}" + (f" ({hint})" if hint else "")) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise BackendUnavailable(f"Could not reach Fireworks ({type(e).__name__})") from None
        usage = data.get("usage") or {}
        self._local.usage = (int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))
        choices = data.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""


class AnthropicBackend:
    """Hosted. Credentials are resolved by the SDK from the environment; this code never reads them."""

    name = "anthropic"

    def __init__(self, lcfg: dict):
        try:
            import anthropic
        except ImportError:
            raise BackendUnavailable("llm.backend=anthropic needs the SDK: pip install anthropic") from None
        self._sdk = anthropic
        self.client = anthropic.Anthropic()
        self.model = lcfg["model"]
        # Current Claude models reject a non-default temperature, so llm.temperature is not sent;
        # effort is the control they expose. Set llm.anthropic_effort to null for models without it.
        effort = lcfg.get("anthropic_effort", "low")
        self.output_config = {"format": {"type": "json_schema", "schema": response_schema(lcfg)}}
        if effort:
            self.output_config["effort"] = effort

    def complete(self, system: str, user: str) -> str:
        sdk = self._sdk
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=16000, system=system,
                messages=[{"role": "user", "content": user}], output_config=self.output_config)
        except (sdk.AuthenticationError, sdk.PermissionDeniedError, sdk.NotFoundError) as e:
            raise BackendUnavailable(f"Anthropic API rejected the request ({type(e).__name__}); "
                                     "check credentials and llm.model") from None
        except sdk.BadRequestError as e:
            raise BackendUnavailable(f"Anthropic API rejected the request (HTTP {e.status_code}); "
                                     "check llm.model and llm.anthropic_effort") from None
        except (sdk.RateLimitError, sdk.APIStatusError, sdk.APIConnectionError) as e:
            # The SDK has already retried; treat what is left as the service being unavailable.
            raise BackendUnavailable(f"Anthropic API unavailable ({type(e).__name__})") from None
        if resp.stop_reason == "refusal":
            raise BackendRefusal()
        return next((b.text for b in resp.content if b.type == "text"), "")


class OpenAIBackend:
    """Hosted. Credentials are resolved by the SDK from the environment; this code never reads them."""

    name = "openai"

    def __init__(self, lcfg: dict):
        try:
            import openai
        except ImportError:
            raise BackendUnavailable("llm.backend=openai needs the SDK: pip install openai") from None
        self._sdk = openai
        self.client = openai.OpenAI()
        self.model = lcfg["model"]
        self.temperature = lcfg.get("temperature", 0)

    def complete(self, system: str, user: str) -> str:
        kwargs = {} if self.temperature is None else {"temperature": self.temperature}
        try:
            resp = self.client.chat.completions.create(
                model=self.model, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
        except self._sdk.OpenAIError as e:
            raise BackendUnavailable(f"OpenAI API unavailable ({type(e).__name__})") from None
        return resp.choices[0].message.content or ""


class CallableBackend:
    """Wraps a function (system, user) -> str. Used by tests in place of a model."""

    def __init__(self, fn: Callable[[str, str], str], name: str = "fake", model: str = "fake"):
        self.complete, self.name, self.model = fn, name, model


def make_backend(lcfg: dict):
    check_backend_allowed(lcfg)
    return {"fireworks": FireworksBackend, "anthropic": AnthropicBackend, "openai": OpenAIBackend}[lcfg["backend"]](lcfg)


# ---------------------------------------------------------------- turn selection


def select_turns(turns: pd.DataFrame, sample: pd.DataFrame, lcfg: dict, seed: int, whole_users: bool) -> pd.DataFrame:
    """User turns to send to the LLM, each with the preceding turn (if it was fetched) as context.

    `llm.sample_turns` caps the work. In user-level mode whole users are kept or dropped together,
    because per-user shares from half a user's conversations would understate what they revealed."""
    turns = turns.sort_values(["dataset", "conversation_id", "turn_index"]).reset_index(drop=True)
    prev = turns.groupby(["dataset", "conversation_id"]).shift(1)
    adjacent = prev["turn_index"] == turns["turn_index"] - 1
    turns["context"] = prev["text"].where(adjacent, "").fillna("")
    user = turns[(turns["role"] == "user") & (turns["text"].str.strip() != "")]
    budget = lcfg.get("sample_turns")
    if budget is None or len(user) <= budget:
        return user.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if whole_users:
        owner = user["conversation_id"].map(sample.set_index("conversation_id")["user_key"])
        per_user = owner.value_counts().sort_index()
        order = per_user.index.to_numpy()[rng.permutation(len(per_user))]
        n_keep = max(1, int(np.searchsorted(np.cumsum(per_user[order].to_numpy()), budget, side="right")))
        user = user[owner.isin(order[:n_keep])]
    else:
        user = user.iloc[np.sort(rng.choice(len(user), size=budget, replace=False))]
    return user.reset_index(drop=True)


# ---------------------------------------------------------------- extraction

TURN_KEYS = ["dataset", "conversation_id", "turn_index"]
ATTR_COLUMNS = [*TURN_KEYS, *Attribute.__dataclass_fields__]
LOG_COLUMNS = [*TURN_KEYS, "status", "cached", "latency_s", "n_attributes", "truncated", "input_tokens",
               "output_tokens", "n_spans", "task_type"]
SPAN_ITEM_COLUMNS = [*TURN_KEYS, "text", "type", "about"]


class ResponseCache:
    """Append-only JSONL of parsed responses, keyed by an HMAC of everything that shapes the answer.
    It holds extracted values, so it lives in the restricted directory."""

    def __init__(self, path: Path):
        self.path, self._lock, self.entries = path, threading.Lock(), {}
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    self.entries[rec["k"]] = rec
                except (ValueError, KeyError):
                    continue      # a torn last line from an interrupted run

    def put(self, rec: dict) -> None:
        with self._lock:
            self.entries[rec["k"]] = rec
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")


def _call(backend, prompt: Prompt, turn: str, context: str, want_spans: bool = False) -> dict:
    """One turn: ask, parse strictly, retry once on a malformed answer. Token counts cover every
    attempt, for backends that report them."""
    system, user = prompt.render(turn, context)
    usage = getattr(backend, "last_usage", lambda: (0, 0))
    t0, status, attrs, spans, task, tok_in, tok_out = time.monotonic(), "parse_failure", [], [], None, 0, 0
    for attempt in (1, 2):
        try:
            raw = backend.complete(system, user)
        except BackendRefusal:
            status = "refusal"
            break
        tok_in, tok_out = tok_in + usage()[0], tok_out + usage()[1]
        try:
            parsed = [asdict(a) for a in parse_response(raw)]
            spans = parse_spans(raw) if want_spans else []
            task = parse_task(raw, prompt.task_types) if prompt.task_types else None
            attrs, status = parsed, "ok"
            break
        except ParseError:
            continue
    return {"status": status, "attrs": attrs, "spans": spans, "task_type": task, "attempts": attempt, "latency_s": time.monotonic() - t0,
            "input_tokens": tok_in, "output_tokens": tok_out}


def extract(selected: pd.DataFrame, lcfg: dict, prompt: Prompt, backend, key: bytes,
            cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (attributes: one row per extracted attribute, with raw values -> restricted only;
    turn log: one row per turn with status and latency, no text)."""
    return extract_with_spans(selected, lcfg, prompt, backend, key, cache_path)[:2]


def extract_with_spans(selected: pd.DataFrame, lcfg: dict, prompt: Prompt, backend, key: bytes,
                       cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`extract`, plus the quoted direct identifiers (one row each, raw text: keep it in memory and
    pass it to `locate_spans`). The third frame is empty unless llm.extract_spans is set."""
    want_spans = bool(lcfg.get("extract_spans"))
    cache = ResponseCache(cache_path)
    max_turn, max_ctx = int(lcfg["max_turn_chars"]), int(lcfg.get("max_context_chars", 1500))
    settings = [backend.name, backend.model, prompt.sha, lcfg.get("temperature"), max_turn, max_ctx,
                getattr(backend, "reasoning_effort", None)]
    jobs = []
    for row in selected.itertuples(index=False):
        turn, context = row.text[:max_turn], row.context[:max_ctx]
        k = hmac.new(key, json.dumps([settings, turn, context]).encode(), hashlib.sha256).hexdigest()
        jobs.append((row, k, turn, context))

    done = {k: {**cache.entries[k], "cached": True} for _, k, _, _ in jobs if k in cache.entries}
    todo = {k: (turn, context) for _, k, turn, context in jobs if k not in done}
    workers = int(lcfg.get("concurrency", 1))
    log(f"[llm] {backend.name}/{backend.model}: {len(jobs):,} turns, {len(jobs) - len(todo):,} served from cache")

    counter = {"n": 0, "failed": 0}
    lock = threading.Lock()

    def work(item):
        k, (turn, context) = item
        rec = {"k": k, **_call(backend, prompt, turn, context, want_spans)}
        cache.put(rec)
        with lock:
            counter["n"] += 1
            counter["failed"] += rec["status"] != "ok"
            if counter["n"] % 100 == 0 or counter["n"] == len(todo):
                log(f"[llm] {counter['n']:,}/{len(todo):,} new turns ({counter['failed']:,} failures)")
        return k, {**rec, "cached": False}

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            done.update(pool.map(work, todo.items()))
    else:
        done.update(map(work, todo.items()))

    new = [r for r in done.values() if not r["cached"]]
    if new:
        log(f"[llm] tokens billed this run: {sum(r.get('input_tokens', 0) for r in new):,} input, "
            f"{sum(r.get('output_tokens', 0) for r in new):,} output")

    attrs, turn_log, span_items, counted = [], [], [], set()
    for row, k, _, _ in jobs:
        rec = done[k]
        ident = (row.dataset, row.conversation_id, int(row.turn_index))
        # Identical turns share one request, so its tokens are counted once.
        tok = (0, 0) if k in counted else (rec.get("input_tokens", 0), rec.get("output_tokens", 0))
        counted.add(k)
        turn_log.append((*ident, rec["status"], rec["cached"], rec["latency_s"], len(rec["attrs"]),
                         len(row.text) > max_turn, *tok, len(rec.get("spans", [])), rec.get("task_type")))
        attrs += [(*ident, *(a[f] for f in Attribute.__dataclass_fields__)) for a in rec["attrs"]]
        span_items += [(*ident, s["text"], s["type"], s["about"]) for s in rec.get("spans", [])]
    return (pd.DataFrame(attrs, columns=ATTR_COLUMNS), pd.DataFrame(turn_log, columns=LOG_COLUMNS),
            pd.DataFrame(span_items, columns=SPAN_ITEM_COLUMNS))


def _occurrences(text: str, quote: str) -> list[int]:
    """Start offsets of non-overlapping occurrences of `quote`, exact first, then ignoring case."""
    for hay, needle in ((text, quote), (text.casefold(), quote.casefold())):
        if len(hay) != len(text):          # case folding changed the length; offsets would be wrong
            continue
        starts, i = [], hay.find(needle)
        while i != -1 and len(starts) < MAX_SPAN_OCCURRENCES:
            starts.append(i)
            i = hay.find(needle, i + len(needle))
        if starts:
            return starts
    return []


def locate_spans(span_items: pd.DataFrame, selected: pd.DataFrame, cfg: dict, taxonomy: dict,
                 key: bytes) -> tuple[pd.DataFrame, int]:
    """Turns the model's quotes into SafeSpan rows (detector `llm`): every occurrence of the quote in
    the turn, as offsets, category and keyed hash. The quoted text is not kept. Also returns how many
    quotes could not be found in their turn (the model changed or invented the text)."""
    max_turn = int(cfg["llm"]["max_turn_chars"])
    texts = {(r.dataset, r.conversation_id, int(r.turn_index)): r.text[:max_turn]
             for r in selected.itertuples(index=False)}
    patterns, rows, unlocated = {}, {}, 0
    for it in span_items.itertuples(index=False):
        ident = (it.dataset, it.conversation_id, int(it.turn_index))
        text, quote = texts[ident], it.text.strip()
        if it.dataset not in patterns:
            patterns[it.dataset] = compile_patterns(cfg, it.dataset)
        starts = _occurrences(text, quote)
        unlocated += not starts
        entity = SPAN_TYPES[it.type][0]
        tier = taxonomy["categories"].get(it.type, {}).get("tier", "unmapped")
        ph = placeholder_spans(text, patterns[it.dataset])
        for s in starts:
            e = s + len(quote)
            in_ph = any(s < pe and ps < e for ps, pe in ph)
            rows.setdefault((*ident, s, e, entity), SafeSpan(
                *ident, s, e, "llm", entity, it.type, tier, 1.0, keyed_hash(key, text[s:e]), in_ph,
                is_dummy(entity, text[s:e]), role="user", about_user=it.about == "user", about=it.about).to_row())
    cols = list(SafeSpan.__dataclass_fields__) + ["length"]
    return pd.DataFrame(list(rows.values()), columns=cols), int(unlocated)


# ---------------------------------------------------------------- aggregate outputs (safe)


def unit_keys(sample: pd.DataFrame) -> pd.Series:
    """conversation_id -> analysis unit: the user proxy, or the conversation where there is none."""
    s = sample.set_index("conversation_id")
    return s["user_key"].where(s["user_key"].notna(), "conv:" + s.index.to_series())


def attribute_prevalence(attrs: pd.DataFrame, turn_log: pd.DataFrame, sample: pd.DataFrame,
                         min_cell: int = 10) -> pd.DataFrame:
    """Share of LLM-processed users and conversations with >= 1 attribute, by type x evidence x about.
    Counts and shares only; the label columns come from closed sets."""
    rows = []
    for ds, tl in turn_log.groupby("dataset"):
        units = unit_keys(sample[sample["dataset"] == ds])
        n_conv = tl["conversation_id"].nunique()
        n_users = tl["conversation_id"].map(units).nunique()
        a = attrs[attrs["dataset"] == ds].assign(unit=lambda d: d["conversation_id"].map(units))
        for typ in (*ATTRIBUTE_TYPES, "any"):
            for ev in (*EVIDENCE, "any"):
                for about in (*ABOUT, "any"):
                    m = a
                    for col, val in (("type", typ), ("evidence", ev), ("about", about)):
                        if val != "any":
                            m = m[m[col] == val]
                    rows.append({"dataset": ds, "attribute_type": typ, "evidence": ev, "about": about,
                                 "n_users": m["unit"].nunique(), "share_users": m["unit"].nunique() / n_users,
                                 "n_conversations": m["conversation_id"].nunique(),
                                 "share_conversations": m["conversation_id"].nunique() / n_conv,
                                 "users_processed": n_users, "conversations_processed": n_conv})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    by_user = suppress_small_cells(out, "n_users", ["share_users"], min_cell)
    out = suppress_small_cells(by_user, "n_conversations", ["share_conversations"], min_cell)
    out["suppressed"] |= by_user["suppressed"]
    return out


def conversation_tasks(turn_log: pd.DataFrame) -> pd.DataFrame:
    """One task type per conversation from the per-turn labels: the most common one, and the earliest
    turn's on a tie. Conversations with no labeled turn are left out."""
    tl = turn_log[turn_log["task_type"].notna()].sort_values("turn_index")
    keys = ["dataset", "conversation_id"]
    counts = tl.groupby([*keys, "task_type"], sort=False).agg(n=("turn_index", "size"), first=("turn_index", "min"))
    best = counts.reset_index().sort_values(["n", "first"], ascending=[False, True]).drop_duplicates(keys)
    return best[[*keys, "task_type"]].reset_index(drop=True)


def _counts(spans: pd.DataFrame) -> pd.DataFrame:
    return spans[~spans["in_placeholder"] & ~spans["is_dummy"]]


def span_prevalence(llm_spans: pd.DataFrame, turn_log: pd.DataFrame, sample: pd.DataFrame,
                    min_cell: int = 10) -> pd.DataFrame:
    """Share of LLM-processed users and conversations with >= 1 quoted direct identifier, by category
    x about. "private" is user or third_party: a real person who is not a public figure."""
    rows, spans = [], _counts(llm_spans)
    groups = {**{a: [a] for a in ABOUT}, "private": ["user", "third_party"], "any": list(ABOUT)}
    for ds, tl in turn_log.groupby("dataset"):
        units = unit_keys(sample[sample["dataset"] == ds])
        n_conv = tl["conversation_id"].nunique()
        n_users = tl["conversation_id"].map(units).nunique()
        s = spans[spans["dataset"] == ds].assign(unit=lambda d: d["conversation_id"].map(units))
        for cat in (*SPAN_TYPES, "any"):
            for about, members in groups.items():
                m = s[s["about"].isin(members)]
                m = m if cat == "any" else m[m["category"] == cat]
                rows.append({"dataset": ds, "category": cat, "about": about,
                             "n_users": m["unit"].nunique(), "share_users": m["unit"].nunique() / n_users,
                             "n_conversations": m["conversation_id"].nunique(),
                             "share_conversations": m["conversation_id"].nunique() / n_conv,
                             "users_processed": n_users, "conversations_processed": n_conv})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    by_user = suppress_small_cells(out, "n_users", ["share_users"], min_cell)
    out = suppress_small_cells(by_user, "n_conversations", ["share_conversations"], min_cell)
    out["suppressed"] |= by_user["suppressed"]
    return out


def detector_agreement(llm_spans: pd.DataFrame, other_spans: pd.DataFrame, turn_log: pd.DataFrame,
                       max_turn_chars: int) -> pd.DataFrame:
    """Span-level agreement between the LLM and the other detectors, on the turns the LLM answered.

    `other_spans` are the other detectors' residual spans (analysis.residual). One row per category x
    detector: how many of that detector's spans overlap a span from the other side, and what the LLM
    said the overlapped text is about. For detector `llm` the other side is every other detector; for
    the rest it is the LLM. Counts of spans, no values."""
    ok = turn_log.loc[turn_log["status"] == "ok", TURN_KEYS].drop_duplicates()
    other = other_spans[(other_spans["role"] == "user") & (other_spans["end"] <= max_turn_chars)
                        & other_spans["category"].isin(SPAN_TYPES)].merge(ok, on=TURN_KEYS)
    mine = _counts(llm_spans).merge(ok, on=TURN_KEYS)
    pairs = (other.reset_index(drop=True).rename_axis("o").reset_index()
             .merge(mine.reset_index(drop=True).rename_axis("l").reset_index(), on=TURN_KEYS, suffixes=("_o", "_l")))
    pairs = pairs[(pairs["start_o"] < pairs["end_l"]) & (pairs["start_l"] < pairs["end_o"])]
    other, mine = other.reset_index(drop=True), mine.reset_index(drop=True)
    # What the LLM said about each of the other detectors' spans (the first overlapping LLM span).
    other_about = pairs.drop_duplicates("o").set_index("o")["about_l"]
    other = other.assign(matched=other.index.isin(pairs["o"]), llm_about=other.index.map(other_about))
    mine = mine.assign(matched=mine.index.isin(pairs["l"]), llm_about=mine["about"], detector="llm")
    rows = []
    for (ds, cat, det), g in pd.concat([other, mine]).groupby(["dataset", "category", "detector"]):
        hit = g[g["matched"]] if det != "llm" else g
        rows.append({"dataset": ds, "category": cat, "detector": det, "n_spans": len(g),
                     "n_matched": int(g["matched"].sum()), "share_matched": float(g["matched"].mean()),
                     **{f"n_about_{a}": int((hit["llm_about"] == a).sum()) for a in ABOUT}})
    cols = ["dataset", "category", "detector", "n_spans", "n_matched", "share_matched", *(f"n_about_{a}" for a in ABOUT)]
    return pd.DataFrame(rows, columns=cols)


def run_stats(turn_log: pd.DataFrame, backend, lcfg: dict) -> pd.DataFrame:
    rows = []
    for ds, tl in turn_log.groupby("dataset"):
        fresh = tl[~tl["cached"]]
        rows.append({"dataset": ds, "backend": backend.name, "model": backend.model,
                     "prompt_version": lcfg["prompt_version"], "turns_processed": len(tl),
                     "turns_from_cache": int(tl["cached"].sum()), "turns_truncated": int(tl["truncated"].sum()),
                     "failures": int((tl["status"] != "ok").sum()),
                     "refusals": int((tl["status"] == "refusal").sum()),
                     "failure_rate": float((tl["status"] != "ok").mean()),
                     "mean_latency_s": float(fresh["latency_s"].mean()) if len(fresh) else np.nan,
                     "attributes_extracted": int(tl["n_attributes"].sum()),
                     "spans_quoted": int(tl["n_spans"].sum()),
                     # Multiply by the model's per-token prices for the spend. "_new" is this run only;
                     # cached turns cost nothing again.
                     "input_tokens": int(tl["input_tokens"].sum()), "output_tokens": int(tl["output_tokens"].sum()),
                     "input_tokens_new": int(fresh["input_tokens"].sum()),
                     "output_tokens_new": int(fresh["output_tokens"].sum())})
    return pd.DataFrame(rows)
