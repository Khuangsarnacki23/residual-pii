"""Rule-based sanitizer layers: no model calls beyond spaCy, so it runs locally at dataset scale.

Two layers, applied on top of layer 1 (patterns) in `sanitize.py`:

  rules_identifiers  names and other direct identifiers, by context instead of by an LLM:
                     - spaCy PERSON entities, except in story/role-play and code conversations
                       (where most names are characters or identifiers), unless the name sits in a
                       self-introduction or a letter frame ("my name is", "Dear", "Regards,")
                     - @handles, social-profile URLs, street addresses, dates of birth
                     - file paths that reveal the writer: home directories (/Users/<name>/...,
                       /home/<name>/..., C:\\Users\\<name>\\...), Windows absolute paths and deep
                       project paths become [PATH]; system paths (/usr/lib/...) and short web routes stay
                     Code is not mistaken for people: no handle matches on decorators (@app.route) or in
                     code, and no name/place/organisation rules on code-shaped tokens (np.array, my_var,
                     getName) or inside ``` blocks and `inline code`.
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
HANDLE_CUE = re.compile(r"(?i)\b(?:twitter|instagram|insta|ig|tiktok|telegram|discord|handle|username|user name|"
                        r"follow(?: me)?|dm|x\.com|threads|snapchat|reddit)\b[^\n]{0,25}\Z")
# File paths. Unix: two or more segments after a root, ~ or ./; Windows: drive or UNC root.
UNIX_PATH = re.compile(r"(?<![\w:/.])(?:~|\.{1,2})?/(?:[\w.\-@+]+/)+[\w.\-@+]*")
# Folder segments may contain spaces ("C:\\Users\\Maria Lopez\\"): a segment ends at the next backslash.
WIN_PATH = re.compile(r"(?:\b[A-Za-z]:|\\\\[\w.\-]+)\\(?:[^\\\n:*?\"<>|'`]+\\)*[^\\\s:*?\"<>|'`]*")
HOME_SEGMENT = re.compile(r"(?i)(?:^|[/\\])(?:Users|home|Documents and Settings|mnt/[a-z]/Users)[/\\](?!Shared\b|Public\b|Default\b)[^/\\\s]+")
SYSTEM_ROOTS = {"usr", "bin", "sbin", "lib", "lib64", "etc", "opt", "var", "tmp", "dev", "proc", "sys", "boot",
                "run", "srv", "system", "library", "applications", "private", "cores", "nix", "snap"}
ROUTE_ROOTS = {"api", "v1", "v2", "v3", "static", "assets", "public", "src", "app", "dist", "build", "node_modules",
               "images", "img", "css", "js", "auth", "admin", "login", "graphql", "docs", "index", "lib", "include"}
CODE_BLOCK = re.compile(r"```.*?(?:```|\Z)|`[^`\n]+`", re.S)
CODE_TOKEN = re.compile(r"[._()\[\]{}=<>/\\#$]|[a-z][A-Z]|^[a-z]+\d")
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


DEFAULT_OPTIONS = {"names_in_first_person": True, "organizations_in_first_person": False, "paths": True}


def code_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges inside ``` fences or `inline code`."""
    return [m.span() for m in CODE_BLOCK.finditer(text)]


def _inside(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in ranges)


def looks_like_code(token: str) -> bool:
    """np.array, df.head(), my_var, getName, x1: not a person, place or organisation."""
    return CODE_TOKEN.search(token.strip()) is not None


def code_context(text: str, start: int, end: int) -> bool:
    """Is the span code? Its own characters, or code punctuation glued to it: `np.` + span, span + `.attr`,
    span + `(` or `_`. A sentence-ending full stop or a comma does not count ("I live in Leeds.")."""
    if looks_like_code(text[start:end]):
        return True
    before, after = text[start - 1:start] if start else "", text[end:end + 2]
    return (before in (".", "_") or after[:1] in ("(", "_")
            or (after[:1] == "." and after[1:2].isalnum()))


def path_edits(text: str) -> list[dict]:
    """File paths that can reveal the writer. Home directories always; Windows absolute paths; Unix
    paths of three or more segments unless they start at a system root or a common web route."""
    edits = []
    for m in WIN_PATH.finditer(text):
        segs = [x for x in re.split(r"[\\]+", m.group()) if x]
        system = len(segs) > 1 and segs[1].lower() in {"windows", "program files", "program files (x86)", "programdata"}
        if HOME_SEGMENT.search(m.group()) or (len(segs) >= 3 and not system):
            edits.append(_edit(m.start(), m.end(), "file_path", "[PATH]"))
    for m in UNIX_PATH.finditer(text):
        p = m.group()
        if text[max(0, m.start() - 3):m.start()].endswith(("://", ":/")) or re.search(r"\w\.\w+$", text[max(0, m.start() - 40):m.start()]):
            continue   # part of a URL (scheme://host/path or host/path)
        segs = [x for x in p.split("/") if x and x not in ("~", ".", "..")]
        if HOME_SEGMENT.search(p) or p.startswith("~/"):
            edits.append(_edit(m.start(), m.end(), "file_path", "[PATH]"))
        elif len(segs) >= 3 and segs[0].lower() not in SYSTEM_ROOTS | ROUTE_ROOTS:
            edits.append(_edit(m.start(), m.end(), "file_path", "[PATH]"))
    return edits


def identifier_edits(text: str, doc, task_type: str, options: dict | None = None) -> list[dict]:
    """Layer 2 (rules): names in context, handles, profile URLs, street addresses, dates of birth."""
    opt = {**DEFAULT_OPTIONS, **(options or {})}
    edits = []
    letter_like = task_type in CV_TASKS
    code = code_spans(text)
    for ent in doc.ents:
        name = ent.text.strip()
        if ent.label_ != "PERSON" or not NAME_SHAPE.match(name) or name.isupper():
            continue
        if _inside(ent.start_char, code) or code_context(text, ent.start_char, ent.end_char):
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
    for rx, cat in ((IPV4, "ip_address"), (PROFILE_URL, "personal_url"), (STREET, "street_address")):
        edits += [_edit(m.start(), m.end(), cat) for m in rx.finditer(text)]
    # Handles: not decorators (@app.route, @property at a line start or followed by . or ( ), and in code
    # turns or code blocks only after a cue such as "my instagram is".
    for m in HANDLE.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        decorator = text[line_start:m.start()].strip() == "" or text[m.end():m.end() + 1] in (".", "(")
        in_code = task_type == "code" or _inside(m.start(), code)
        cued = HANDLE_CUE.search(text[max(0, m.start() - 60):m.start()]) is not None
        if (not decorator and not in_code) or cued:
            edits.append(_edit(m.start(), m.end(), "username_handle"))
    if opt["paths"]:
        edits += path_edits(text)
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
    code = code_spans(text)
    for sent in doc.sents:
        s0, s1 = sent.start_char, sent.end_char
        if _inside(s0, code):
            continue
        stext = text[s0:s1]
        if not (cv_like or FIRST_PERSON.search(stext)):
            continue
        affiliated = AFFILIATION.search(stext) is not None
        for ent in sent.ents:
            if code_context(text, ent.start_char, ent.end_char):
                continue
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
