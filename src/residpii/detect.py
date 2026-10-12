"""Detectors. Each returns SafeSpan records only; raw span text never leaves this module.

Families are kept separate (the `detector` field):
  presidio : Presidio AnalyzerEngine with spaCy NER + built-in pattern recognizers
  secrets  : regexes for API keys, tokens, private keys and hard-coded passwords
  selfid   : self-introductions ("my name is X"); a name the user gives as their own

Raw span text is used in-process for exactly two things: the keyed hash and the
dummy/placeholder-value check (`is_dummy`). Neither the text nor the matched value is returned.
"""
from __future__ import annotations

import ipaddress
import re
import sys
from typing import Iterable

import pandas as pd

from .redactions import compile_patterns, placeholder_spans
from .safety import SafeSpan, keyed_hash

# ---------------------------------------------------------------- taxonomy


def label_map(taxonomy: dict) -> dict[str, tuple[str, str]]:
    out = {}
    for category, spec in taxonomy["categories"].items():
        for label in spec["labels"]:
            out[label] = (category, spec["tier"])
    return out


# ---------------------------------------------------------------- dummy / non-personal values

_DUMMY_EMAIL_DOMAINS = {"example.com", "example.org", "example.net", "test.com", "domain.com",
                        "yourdomain.com", "email.com", "company.com", "mydomain.com", "website.com"}
_DUMMY_NAMES = {"john doe", "jane doe", "user", "username", "your name", "first last", "chatgpt",
                "gpt", "gpt-3", "gpt-4", "assistant"}
_TEST_CARDS = {"4111111111111111", "4242424242424242", "5555555555554444", "378282246310005",
               "6011111111111117", "4000056655665556"}
# Fragments that mark a value as dummy wherever they occur.
_PLACEHOLDERISH = re.compile(r"(?i)x{4,}|\*{4,}|your[_\- ]?(?:api|key|token|secret|password)|"
                             r"(.)\1{7,}|1234567|abcdef")
# Ellipses and <angle brackets> are dummy only when they are the whole value: "..." and "<name>" are
# placeholders, but "Dr. Smith..." is a real name followed by punctuation.
_PLACEHOLDER_WHOLE = re.compile(r"(?:\.{3,}|…+|<[^>]*>)+")


def is_dummy(entity: str, value: str) -> bool:
    v = value.strip()
    low = v.casefold()
    if _PLACEHOLDERISH.search(v) or _PLACEHOLDER_WHOLE.fullmatch(v):
        return True
    if entity in ("PERSON", "SELF_NAME"):
        return low in _DUMMY_NAMES or v == "DAN"   # "DAN" = widely shared jailbreak persona
    if entity == "EMAIL_ADDRESS":
        return low.rsplit("@", 1)[-1] in _DUMMY_EMAIL_DOMAINS
    if entity == "PHONE_NUMBER":
        d = re.sub(r"\D", "", v)
        return (len(set(d)) <= 2 or "1234567" in d or "0123456" in d
                or re.search(r"555\d{4}$", d) is not None)   # NANP 555 numbers are fictional
    if entity == "CREDIT_CARD":
        return re.sub(r"\D", "", v) in _TEST_CARDS
    if entity == "IP_ADDRESS":
        try:
            ip = ipaddress.ip_address(v)
            return ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_unspecified or ip.is_multicast
        except ValueError:
            return False
    if entity == "URL":
        return bool(re.search(r"(?i)\b(?:example\.(?:com|org|net)|localhost|127\.0\.0\.1)\b", v))
    return False


# ---------------------------------------------------------------- secrets


SECRET_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("openai_anthropic_key", re.compile(r"\bsk-(?:proj-|ant-(?:api\d\d-)?)?[A-Za-z0-9_\-]{20,}"), 0),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})"), 0),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), 0),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), 0),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), 0),
    ("stripe_live_key", re.compile(r"\b[rs]k_live_[0-9A-Za-z]{24,}\b"), 0),
    ("telegram_bot_token", re.compile(r"\b\d{8,10}:AA[0-9A-Za-z_\-]{33}\b"), 0),
    ("discord_bot_token", re.compile(r"\b[MN][A-Za-z\d]{23,25}\.[\w-]{6}\.[\w-]{27,38}\b"), 0),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), 0),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----"), 0),
    ("hardcoded_credential", re.compile(
        r"(?i)\b(?:api[_\-]?key|secret(?:[_\-]?key)?|access[_\-]?token|auth[_\-]?token|token|password|passwd|pwd)"
        r"\s*[:=]\s*['\"]([^'\"\s]{8,})['\"]"), 1),
]


def secret_matches(text: str) -> Iterable[tuple[int, int, str]]:
    for _name, pat, group in SECRET_PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span(group)
            yield s, e, "SECRET_TOKEN"


# ---------------------------------------------------------------- self-introductions

# A name is 1-3 capitalized tokens. The trigger is case-insensitive; the name is not, because
# capitalization is the only thing separating "I'm Dana" from "I'm done".
_NAME = r"[A-Z][A-Za-z'’\-]+(?:[ \t]+[A-Z][A-Za-z'’\-]+){0,2}"
_SELFID_PATTERNS: list[tuple[re.Pattern, bool]] = [   # (pattern, needs a clean right-hand boundary)
    (re.compile(rf"\b(?i:my (?:(?:full|first|last|real|given) )?name(?: is|'s|’s))[ \t]+({_NAME})"), False),
    (re.compile(rf"\b(?i:(?:you can |please |just )?call me)[ \t]+({_NAME})"), False),
    (re.compile(rf"\b(?i:this is)[ \t]+({_NAME})(?=[ \t]+(?i:speaking|here)\b)"), False),
    # "I'm X" is the noisy one ("I'm Sorry", "I'm American"), so it also needs the name to end the clause.
    (re.compile(rf"\b(?i:i am|i'm|i’m)[ \t]+({_NAME})"), True),
]
_SELFID_BOUNDARY = re.compile(r"\s*(?:$|[.,!?;:)\n]|(?i:and|from|here|speaking)\b)")
# Capitalized words that follow "I'm"/"call me" without being a name: sentence-case and title-case
# text, plus nationalities, religions and other proper adjectives (those belong to the nrp category).
_SELFID_STOP = {w.casefold() for w in """
    a an the not no so too very really just also still now here there back in on at up out off from
    sorry happy sad glad sure fine good great ok okay well ready done new old able afraid alone aware
    angry bored busy confused curious excited tired hungry lost late early free open single married
    pregnant sick ill hurt scared worried nervous proud grateful thankful interested looking going
    trying working writing using having getting making doing thinking wondering asking planning
    building learning studying creating feeling struggling hoping running living coming
    maybe later tomorrow today anything something nothing everyone someone anyone crazy stupid
    sir madam mr mrs ms miss dr doctor professor captain god lord king queen master daddy mommy
    american british english french german spanish italian chinese japanese korean indian russian
    canadian australian mexican brazilian irish scottish welsh dutch swedish polish turkish greek
    african asian european arab arabic persian iranian israeli pakistani nigerian filipino thai
    vietnamese indonesian egyptian ukrainian portuguese hispanic latino latina black white
    christian catholic protestant muslim jewish hindu buddhist atheist agnostic mormon sikh
    gay lesbian bisexual trans transgender straight vegan vegetarian democrat republican
    monday tuesday wednesday thursday friday saturday sunday
""".split()}


def selfid_matches(text: str) -> Iterable[tuple[int, int, str]]:
    """Spans of names the writer introduces as their own. Yields (start, end, "SELF_NAME")."""
    seen = set()
    for pat, needs_boundary in _SELFID_PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span(1)
            tokens = list(re.finditer(r"\S+", text[s:e]))
            if tokens[0].group().casefold() in _SELFID_STOP:
                continue
            # A stoplisted word after the name is the start of the next clause, not a surname.
            for i, tok in enumerate(tokens):
                if tok.group().casefold() in _SELFID_STOP:
                    tokens = tokens[:i]
                    break
            e = s + tokens[-1].end()
            if needs_boundary and not _SELFID_BOUNDARY.match(text, e):
                continue
            if (s, e) not in seen:
                seen.add((s, e))
                yield s, e, "SELF_NAME"


# ---------------------------------------------------------------- presidio


def build_presidio(pcfg: dict):
    import logging

    from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)  # silence non-English recognizer notices

    ignore = ["CARDINAL", "EVENT", "LANGUAGE", "LAW", "MONEY", "ORDINAL", "PERCENT", "PRODUCT",
              "QUANTITY", "WORK_OF_ART"]
    if not pcfg.get("include_organization"):
        ignore.append("ORGANIZATION")
    nlp_conf = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": pcfg["spacy_model"]}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON", "PERSON": "PERSON", "NORP": "NRP", "FAC": "LOCATION", "LOC": "LOCATION",
                "GPE": "LOCATION", "LOCATION": "LOCATION", "ORG": "ORGANIZATION",
                "ORGANIZATION": "ORGANIZATION", "DATE": "DATE_TIME", "TIME": "DATE_TIME"},
            "low_confidence_score_multiplier": 0.4,
            "labels_to_ignore": ignore,
        },
    }
    engine = NlpEngineProvider(nlp_configuration=nlp_conf).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=engine, supported_languages=["en"])
    return BatchAnalyzerEngine(analyzer_engine=analyzer)


def _chunks(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    if len(text) <= size:
        return [(0, text)]
    step = size - overlap
    return [(i, text[i:i + size]) for i in range(0, len(text), step) if text[i:i + size]]


# ---------------------------------------------------------------- driver


def detect(turns: pd.DataFrame, cfg: dict, taxonomy: dict, key: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all enabled detectors. Returns (spans, turn_stats); neither contains raw text."""
    pcfg = cfg["presidio"]
    lmap = label_map(taxonomy)
    entities = list(pcfg["entities"]) + (["ORGANIZATION"] if pcfg.get("include_organization") else [])
    patterns = {ds: compile_patterns(cfg, ds) for ds in turns["dataset"].unique()}
    max_chars = pcfg["max_chars_per_turn"]

    # Flatten every turn into (turn_row, offset, chunk) so Presidio can batch them through spaCy.
    chunk_index: list[tuple[int, int]] = []
    chunk_texts: list[str] = []
    stats = []
    ph_by_row: dict[int, list[tuple[int, int]]] = {}
    for row, (ds, text) in enumerate(turns[["dataset", "text"]].itertuples(index=False)):
        scanned = text[:max_chars]
        ph = placeholder_spans(text, patterns[ds])
        ph_by_row[row] = ph
        stats.append({"n_chars": len(text), "n_chars_scanned": len(scanned),
                      "n_tokens": len(text.split()), "n_placeholders": len(ph)})
        for off, chunk in _chunks(scanned, pcfg["chunk_chars"], pcfg["chunk_overlap"]):
            chunk_index.append((row, off))
            chunk_texts.append(chunk)

    found: dict[tuple, tuple] = {}   # (row, start, end, detector, entity) -> score

    batch = build_presidio(pcfg)
    print(f"[detect] presidio: {len(chunk_texts):,} chunks from {len(turns):,} turns",
          file=sys.stderr, flush=True)
    results = batch.analyze_iterator(chunk_texts, language="en", entities=entities,
                                     score_threshold=pcfg["store_threshold"], batch_size=32)
    for (row, off), res in zip(chunk_index, results):
        emails = [(r.start, r.end) for r in res if r.entity_type == "EMAIL_ADDRESS"]
        for r in res:
            # Presidio also reports fragments of an email address as URLs; drop those.
            if r.entity_type == "URL" and any(s <= r.start and r.end <= e for s, e in emails):
                continue
            k = (row, off + r.start, off + r.end, "presidio", r.entity_type)
            found[k] = max(found.get(k, 0.0), float(r.score))

    if cfg.get("secrets", {}).get("enabled", True):
        for row, text in enumerate(turns["text"]):
            for s, e, ent in secret_matches(text[:max_chars]):
                found[(row, s, e, "secrets", ent)] = 1.0

    if cfg.get("selfid", {}).get("enabled", True):
        # Only user turns: an assistant saying "I'm Claude" is not a user identifying themselves.
        for row, (role, text) in enumerate(turns[["role", "text"]].itertuples(index=False)):
            if role == "user":
                for s, e, ent in selfid_matches(text[:max_chars]):
                    found[(row, s, e, "selfid", ent)] = 1.0

    meta = turns[["dataset", "conversation_id", "turn_index", "role"]].to_numpy()
    texts = turns["text"].to_numpy()
    spans = []
    for (row, s, e, det, ent), score in found.items():
        category, tier = lmap.get(ent, (ent.lower(), "unmapped"))
        value = texts[row][s:e]
        in_ph = any(s < pe and ps < e for ps, pe in ph_by_row[row])
        ds, cid, tidx, role = meta[row]
        spans.append(SafeSpan(ds, cid, int(tidx), s, e, det, ent, category, tier, round(score, 4),
                              keyed_hash(key, value), in_ph, is_dummy(ent, value), role=role,
                              about_user=True if det == "selfid" else None).to_row())
    span_cols = list(SafeSpan.__dataclass_fields__) + ["length"]
    stats_df = pd.concat([turns[["dataset", "conversation_id", "turn_index", "role"]].reset_index(drop=True),
                          pd.DataFrame(stats)], axis=1)
    return pd.DataFrame(spans, columns=span_cols), stats_df


# ---------------------------------------------------------------- task type (heuristic)


def _task_rules(taxonomy: dict) -> dict[str, tuple]:
    def comp(pats) -> list[re.Pattern]:
        return [re.compile(p, re.IGNORECASE) for p in pats or []]

    rules = {}
    for task, spec in taxonomy["task_types"].items():
        if not isinstance(spec, dict):
            spec = {"patterns": spec}
        rules[task] = (comp(spec.get("patterns")), comp(spec.get("weak")), comp(spec.get("weak_context")),
                       int(spec.get("weak_min_distinct", 2)))
    return rules


def _task_hits(text: str, rule: tuple) -> int:
    patterns, weak, context, min_distinct = rule
    n = sum(len(p.findall(text)) for p in patterns)
    # Weak keywords are ordinary English words ("return", "class"); count them only with support.
    weak_hits = [m.group().casefold() for p in weak for m in p.finditer(text)]
    if weak_hits and (len(set(weak_hits)) >= min_distinct or any(p.search(text) for p in context)):
        n += len(weak_hits)
    return n


def tag_tasks(turns: pd.DataFrame, taxonomy: dict) -> pd.DataFrame:
    rules = _task_rules(taxonomy)
    order = list(rules)
    rows = []
    for (ds, cid), g in turns.groupby(["dataset", "conversation_id"], sort=False):
        text = "\n".join(g.sort_values("turn_index")["text"])[:20000]
        counts = {t: _task_hits(text, rule) for t, rule in rules.items()}
        best = max(order, key=lambda t: (counts[t], -order.index(t)))
        task = best if counts[best] > 0 else taxonomy["default_task_type"]
        rows.append({"dataset": ds, "conversation_id": cid, "task_type": task,
                     "is_story_roleplay": task == "story_roleplay"})
    return pd.DataFrame(rows)
