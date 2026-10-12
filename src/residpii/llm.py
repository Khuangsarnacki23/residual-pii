"""LLM detector for quasi-identifiers (stated and inferred).

Layer 3 of detection: for each sampled user turn, an LLM returns the quasi-identifying attributes it
reveals (occupation, city, age, ...), whether each was stated or inferred, and whom it is about.

Privacy rules this module enforces:
  - The default backend is a local Ollama server. Hosted APIs (anthropic, openai, or an Ollama URL
    that is not on this machine) send user text to a third party and are refused unless
    `llm.allow_hosted` is true.
  - Extracted values are personal data. They are written only under the restricted directory
    (the response cache and `llm_attrs_<dataset>.parquet`). Everything this module returns for
    results/ is counts and shares over closed label sets.
  - Nothing here logs prompt or response text, and error messages carry reason codes only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
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
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .safety import suppress_small_cells

ATTRIBUTE_TYPES = ("occupation", "employer_type", "city", "state_or_region", "country", "age", "age_range",
                   "gender", "school", "education_level", "relationship_status", "nationality",
                   "language_native", "other")
EVIDENCE = ("stated", "inferred")
ABOUT = ("user", "third_party", "public_figure", "fictional", "unclear")

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
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
HOSTED_BACKENDS = ("anthropic", "openai")


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


# ---------------------------------------------------------------- prompt


@dataclass(frozen=True)
class Prompt:
    system: str
    user_template: str
    sha: str

    def render(self, turn: str, context: str) -> tuple[str, str]:
        fill = {"turn": turn, "context": context or "(none)"}
        # One pass, so placeholder-looking text inside a turn is never substituted again.
        return self.system, re.sub(r"\{\{(turn|context)\}\}", lambda m: fill[m.group(1)], self.user_template)


def load_prompt(cfg: dict) -> Prompt:
    path = Path(cfg["prompts_dir"]) / f"qi_extract_{cfg['llm']['prompt_version']}.txt"
    raw = path.read_text()
    if "=====USER=====" not in raw:
        raise ValueError(f"{path} must contain a '=====USER=====' line separating system and user parts")
    system, user = raw.split("=====USER=====", 1)
    groups = "\n".join(f"    {code}  {name}" for code, name in OCCUPATION_GROUPS.items())
    system = system.replace("{{types}}", ", ".join(ATTRIBUTE_TYPES)).replace("{{occupation_groups}}", groups)
    return Prompt(system.strip(), user.strip(), hashlib.sha256(raw.encode()).hexdigest()[:16])


# ---------------------------------------------------------------- backends


def _is_loopback(url: str) -> bool:
    return (urlparse(url).hostname or "") in ("localhost", "127.0.0.1", "::1")


def check_backend_allowed(lcfg: dict) -> None:
    """Abort unless the backend is local, or hosted use was explicitly switched on."""
    backend = lcfg.get("backend")
    if backend not in ("ollama", *HOSTED_BACKENDS):
        raise ValueError(f"Unknown llm.backend: {backend!r} (expected ollama, anthropic or openai)")
    leaves_machine = backend in HOSTED_BACKENDS or not _is_loopback(lcfg.get("ollama_url") or DEFAULT_OLLAMA_URL)
    if leaves_machine and not lcfg.get("allow_hosted"):
        raise HostedBackendNotAllowed(
            f"llm.backend={backend!r} would send user text off this machine. That needs the dataset "
            "terms, your instructor/IRB and your partner to allow it; if they do, set "
            "llm.allow_hosted: true. Otherwise use the local backend (llm.backend: ollama).")


class OllamaBackend:
    """Local model through Ollama's HTTP API, in JSON mode."""

    name = "ollama"

    def __init__(self, lcfg: dict):
        self.model = lcfg["model"]
        self.url = lcfg.get("ollama_url") or DEFAULT_OLLAMA_URL
        self.temperature = lcfg.get("temperature", 0)
        self.timeout = lcfg.get("timeout_s", 300)

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": self.temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["message"]["content"]
        except urllib.error.HTTPError as e:
            hint = f" Is the model pulled? Try: ollama pull {self.model}" if e.code == 404 else ""
            raise BackendUnavailable(f"Ollama returned HTTP {e.code} at {self.url}.{hint}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise BackendUnavailable(f"Could not reach Ollama at {self.url} ({type(e).__name__}). "
                                     "Is `ollama serve` running?") from None


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
        self.output_config = {"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}}
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
    return {"ollama": OllamaBackend, "anthropic": AnthropicBackend, "openai": OpenAIBackend}[lcfg["backend"]](lcfg)


# ---------------------------------------------------------------- turn selection


def select_turns(turns: pd.DataFrame, sample: pd.DataFrame, lcfg: dict, seed: int, whole_users: bool) -> pd.DataFrame:
    """User turns to send to the LLM, each with the preceding turn (if it was fetched) as context.

    `llm.sample_turns` caps the work. In user-level mode whole users are kept or dropped together,
    because a profile built from half a user's conversations would understate what they revealed."""
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
LOG_COLUMNS = [*TURN_KEYS, "status", "cached", "latency_s", "n_attributes", "truncated"]


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


def _call(backend, prompt: Prompt, turn: str, context: str) -> dict:
    """One turn: ask, parse strictly, retry once on a malformed answer."""
    system, user = prompt.render(turn, context)
    t0, error = time.monotonic(), "parse_failure"
    for attempt in (1, 2):
        try:
            attrs = parse_response(backend.complete(system, user))
            return {"status": "ok", "attrs": [asdict(a) for a in attrs], "attempts": attempt,
                    "latency_s": time.monotonic() - t0}
        except ParseError:
            continue
        except BackendRefusal:
            error = "refusal"
            break
    return {"status": error, "attrs": [], "attempts": attempt, "latency_s": time.monotonic() - t0}


def extract(selected: pd.DataFrame, lcfg: dict, prompt: Prompt, backend, key: bytes,
            cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (attributes: one row per extracted attribute, with raw values -> restricted only;
    turn log: one row per turn with status and latency, no text)."""
    cache = ResponseCache(cache_path)
    max_turn, max_ctx = int(lcfg["max_turn_chars"]), int(lcfg.get("max_context_chars", 1500))
    settings = [backend.name, backend.model, prompt.sha, lcfg.get("temperature"), max_turn, max_ctx]
    jobs = []
    for row in selected.itertuples(index=False):
        turn, context = row.text[:max_turn], row.context[:max_ctx]
        k = hmac.new(key, json.dumps([settings, turn, context]).encode(), hashlib.sha256).hexdigest()
        jobs.append((row, k, turn, context))

    done = {k: {**cache.entries[k], "cached": True} for _, k, _, _ in jobs if k in cache.entries}
    todo = {k: (turn, context) for _, k, turn, context in jobs if k not in done}
    workers = int(lcfg.get("concurrency", 1))
    if workers > 1 and backend.name == "ollama":
        log("[llm] concurrency > 1 is for hosted backends; using 1 for ollama")
        workers = 1
    log(f"[llm] {backend.name}/{backend.model}: {len(jobs):,} turns, {len(jobs) - len(todo):,} served from cache")

    counter = {"n": 0, "failed": 0}
    lock = threading.Lock()

    def work(item):
        k, (turn, context) = item
        rec = {"k": k, **_call(backend, prompt, turn, context)}
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

    attrs, turn_log = [], []
    for row, k, _, _ in jobs:
        rec = done[k]
        ident = (row.dataset, row.conversation_id, int(row.turn_index))
        turn_log.append((*ident, rec["status"], rec["cached"], rec["latency_s"], len(rec["attrs"]),
                         len(row.text) > max_turn))
        attrs += [(*ident, *(a[f] for f in Attribute.__dataclass_fields__)) for a in rec["attrs"]]
    return pd.DataFrame(attrs, columns=ATTR_COLUMNS), pd.DataFrame(turn_log, columns=LOG_COLUMNS)


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
                     "attributes_extracted": int(tl["n_attributes"].sum())})
    return pd.DataFrame(rows)
