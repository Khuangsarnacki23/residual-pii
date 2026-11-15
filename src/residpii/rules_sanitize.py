"""Rule-based sanitizer layers: no model calls beyond spaCy, so it runs locally at dataset scale.

Two layers, applied on top of layer 1 (patterns) in `sanitize.py`:

  rules_identifiers  names and other direct identifiers, by context instead of by an LLM:
                     - spaCy PERSON entities, except in story/role-play and code conversations
                       (where most names are characters or identifiers), unless the name sits in a
                       self-introduction or a letter frame ("my name is", "Dear", "Regards,")
                     - @handles, social-profile URLs, street addresses, dates of birth
  rules_generalize   + quasi-identifiers in personal sentences (first person, or any sentence of a
                     CV / letter / application turn), generalised from lookup tables
                     (configs/generalization.yaml):
                     - places: city or state -> country; a listed country stays; unknown -> [LOCATION]
                     - organisations after an affiliation cue (work at, studied at, joined) -> kind
                       ("a university", "a healthcare organisation", "a company")
                     - job titles after "I am a", "I work as", "as a" -> occupation group
                     - ages -> 10-year band; full dates -> the year

Story/role-play and code turns are skipped by the generalisation rules. Every edit is a span with a
replacement, so the evaluation can count false edits the same way as for redaction layers.
"""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from .config import REPO_ROOT

FIRST_PERSON = re.compile(r"(?:\bI\b|\bI'm\b|\bI’m\b|(?i:\b(?:my|me|mine|myself|we|our|us|ours)\b))")
SELF_OR_LETTER = re.compile(
    r"(?i)(?:\bmy name is|\bi am|\bi'm|\bi’m|\bthis is|\bcall me|\bdear|\bhi|\bhello|\bregards|\bsincerely"
    r"|\bthanks|\bthank you|\byours(?: truly| faithfully| sincerely)?|\bbest|\bcheers|\bsigned|\bfrom|\bname)"
    r"[\s,:;\-–]*\Z")
# Any IPv4 address, private ranges included (layer 1 treats private ranges as dummy values, but an
# internal address pasted with a config still describes the writer's network).
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
HANDLE = re.compile(r"(?<![\w.@])@[A-Za-z0-9_]{3,30}\b")
PROFILE_URL = re.compile(
    r"(?i)\b(?:https?://)?(?:www\.)?(?:linkedin\.com/in|github\.com|facebook\.com|fb\.com|instagram\.com"
    r"|twitter\.com|x\.com|t\.me|tiktok\.com/@|youtube\.com/(?:@|c/|channel/|user/)|medium\.com/@)"
    r"/?[^\s)\]>\"']+")
STREET = re.compile(
    r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][a-z]+\.?\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln"
    r"|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Close|Crescent|Highway|Hwy)\b\.?")
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE = (rf"(?:\d{{1,2}}[/.\-]\d{{1,2}}[/.\-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}|{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
         rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\.?,?\s+\d{{4}})")
DOB = re.compile(rf"(?i)\b(?:born(?: on)?|date of birth|d\.?o\.?b\.?|birthday(?: is)?)\s*[:\-]?\s*(?P<date>{_DATE})")
FULL_DATE = re.compile(rf"(?i)\b{_DATE}\b")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
AGE = re.compile(
    r"(?i)(?:\b(?:i am|i'm|i’m|aged?|age\s*:)\s*(?P<a>\d{1,2})\b(?!\s*(?:%|percent|minutes|hours|days|weeks|months|am|pm|:))"
    r"|\b(?P<b>\d{1,2})(?:\s+|-)years?(?:\s+|-)old\b)")
AFFILIATION = re.compile(
    r"(?i)\b(?:work(?:s|ed|ing)?|employ(?:ed|ee)|job|intern(?:ship|ed|ing)?|position|role|joined|hired|"
    r"stud(?:y|ying|ied|ent)|graduat(?:e|ed|ing)|enrolled|attend(?:s|ed|ing)?|admitted|accepted|alumn\w*|"
    r"apply(?:ing)?|applied|offer|my (?:company|employer|school|university|college|firm|team|office))\b")
TITLE_CUE = r"(?:\bI am|\bI'm|\bI’m|\bI work as|\bworking as|\bworked as|\bwork as|\bmy (?:job|role|position) as|\bas|\bI've been|\bI have been)"
HONORIFIC = re.compile(r"(?:\b(?:mr|mrs|ms|miss|dr|prof|professor|sir|madam)\.?\s*)\Z", re.I)
RELATION = re.compile(
    r"(?i)\b(?:my|our|his|her|their)\s+(?:\w+\s+)?(?:wife|husband|partner|spouse|boyfriend|girlfriend|fianc[eé]e?|"
    r"son|daughter|child|kid|mother|mom|mum|father|dad|brother|sister|cousin|aunt|uncle|grand\w+|friend|"
    r"boss|manager|supervisor|colleague|coworker|co-worker|teacher|professor|student|client|customer|"
    r"landlord|tenant|neighbou?r|doctor|lawyer|roommate|classmate|ex)\W{0,3}(?:named\s+|called\s+)?\Z")
NAME_SHAPE = re.compile(r"^[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z][a-zA-Z'’\-.]*){0,3}$")
SIGNATURE = re.compile(
    r"(?im)^(?:best|regards|best regards|kind regards|warm regards|sincerely|yours sincerely|yours truly|"
    r"thanks|thank you|cheers|respectfully)[,!.]?\s*\n+\s*(?P<n>[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z][a-zA-Z'’\-.]*){0,2})\s*$")
CV_TASKS = {"job_application", "communication"}
SKIP_TASKS = {"story_roleplay", "code"}


@lru_cache(maxsize=1)
def tables() -> dict:
    t = yaml.safe_load((REPO_ROOT / "configs" / "generalization.yaml").read_text())
    t["occupation_keywords"] = [(re.compile(p, re.I), g) for p, g in t["occupation_keywords"]]
    t["organization_kinds"] = [(re.compile(p, re.I), g) for p, g in t["organization_kinds"]]
    return t


@lru_cache(maxsize=1)
def nlp(model: str = "en_core_web_lg"):
    import spacy

    return spacy.load(model, disable=["lemmatizer"])


def _edit(s: int, e: int, category: str, replacement: str | None = None) -> dict:
    return {"start": s, "end": e, "category": category,
            "replacement": replacement if replacement is not None else f"[{category.upper()}]"}


DEFAULT_OPTIONS = {"names_in_first_person": True, "organizations_in_first_person": False}


def identifier_edits(text: str, doc, task_type: str, options: dict | None = None) -> list[dict]:
    """Layer 2 (rules): names in context, handles, profile URLs, street addresses, dates of birth."""
    opt = {**DEFAULT_OPTIONS, **(options or {})}
    edits = []
    letter_like = task_type in CV_TASKS
    for ent in doc.ents:
        name = ent.text.strip()
        if ent.label_ != "PERSON" or not NAME_SHAPE.match(name) or name.isupper():
            continue
        before = text[max(0, ent.start_char - 40):ent.start_char]
        # A name is redacted only when its context marks it as a real person: a self-introduction or
        # letter frame, an honorific, a relationship word, or any name in a CV / letter / email turn
        # (outside those, most names in prompts are characters, public figures or examples).
        sent = ent.sent.text if doc.has_annotation("SENT_START") else ""
        real = (SELF_OR_LETTER.search(before) or HONORIFIC.search(before) or RELATION.search(before)
                or (letter_like and task_type not in SKIP_TASKS)
                or (opt["names_in_first_person"] and task_type not in SKIP_TASKS | {"creative_other"}
                    and FIRST_PERSON.search(sent)))
        if real:
            edits.append(_edit(ent.start_char, ent.end_char, "name"))
    # Signature lines in letters and emails: a short name-shaped line after a closing ("Best,\nAnna Li").
    if letter_like:
        for m in SIGNATURE.finditer(text):
            edits.append(_edit(m.start("n"), m.end("n"), "name"))
    for rx, cat in ((IPV4, "ip_address"), (HANDLE, "username_handle"), (PROFILE_URL, "personal_url"), (STREET, "street_address")):
        edits += [_edit(m.start(), m.end(), cat) for m in rx.finditer(text)]
    edits += [_edit(m.start("date"), m.end("date"), "date_of_birth") for m in DOB.finditer(text)]
    return edits


def _band(age: int) -> str:
    lo = age // 10 * 10
    return f"{lo}-{lo + 9}"


def _place(value: str, t: dict) -> str | None:
    """Generalised replacement for a place, or None to keep it (a listed country)."""
    v = re.sub(r"[^\w\s]", "", value.casefold()).strip()
    v = re.sub(r"^the\s+", "", v)
    if v in t["countries"]:
        return None
    return t["places"].get(v, "[LOCATION]")


def generalize_edits(text: str, doc, task_type: str, options: dict | None = None) -> list[dict]:
    """Layer 3 (rules): generalise quasi-identifiers in personal sentences."""
    if task_type in SKIP_TASKS:
        return []
    opt = {**DEFAULT_OPTIONS, **(options or {})}
    t = tables()
    cv_like = task_type in CV_TASKS
    edits = []
    for sent in doc.sents:
        s0, s1 = sent.start_char, sent.end_char
        stext = text[s0:s1]
        if not (cv_like or FIRST_PERSON.search(stext)):
            continue
        affiliated = AFFILIATION.search(stext) is not None
        for ent in sent.ents:
            if ent.label_ in ("GPE", "LOC", "FAC"):
                rep = _place(ent.text, t)
                if rep is not None:
                    edits.append(_edit(ent.start_char, ent.end_char, "location", rep))
            elif ent.label_ == "ORG" and (affiliated or opt["organizations_in_first_person"]):
                kind = next(g for rx, g in t["organization_kinds"] if rx.search(ent.text))
                edits.append(_edit(ent.start_char, ent.end_char, "organization", kind))
            elif ent.label_ == "DATE" and FULL_DATE.search(ent.text):
                y = YEAR.search(ent.text)
                edits.append(_edit(ent.start_char, ent.end_char, "date", y.group() if y else "[DATE]"))
        for m in AGE.finditer(stext):
            g = "a" if m.group("a") else "b"
            age = int(m.group(g))
            if 13 <= age <= 99:
                edits.append(_edit(s0 + m.start(g), s0 + m.end(g), "age", _band(age)))
        for rx, group in t["occupation_keywords"]:
            title = re.compile(TITLE_CUE + r"\s+(?:an?\s+)?(?P<t>(?:[\w-]+\s+){0,3}?" + rx.pattern + r")", re.I)
            for m in title.finditer(stext):
                edits.append(_edit(s0 + m.start("t"), s0 + m.end("t"), "occupation", group.split(" ", 1)[1]))
    return edits


def apply(text: str, edits: list[dict]) -> tuple[str, list[dict]]:
    """Applies non-overlapping edits (earlier/longer wins on overlap). Returns text and kept edits."""
    kept: list[dict] = []
    for e in sorted(edits, key=lambda e: (e["start"], -(e["end"] - e["start"]))):
        if kept and e["start"] < kept[-1]["end"]:
            continue
        kept.append(e)
    out, pos = [], 0
    for e in kept:
        out += [text[pos:e["start"]], e["replacement"]]
        pos = e["end"]
    out.append(text[pos:])
    return "".join(out), kept
