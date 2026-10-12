"""Detect each dataset's existing redaction placeholders.

Two jobs: (1) locate configured placeholders so detections overlapping them are not counted as
residual PII, and (2) *discover* candidate placeholder formats so the config can be checked against
the data instead of assumed. Discovery reports only token *shapes* (digits collapsed to '#') that
occur in at least `min_conversations` distinct conversations, so a one-off all-caps name can never
surface in its output.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import pandas as pd

# Things that look like machine-inserted placeholders: PRESIDIO_ANONYMIZED_X, NAME_1, [EMAIL], <PHONE>.
CANDIDATE = re.compile(
    r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"          # SNAKE_CASE caps tokens (NAME_1, PRESIDIO_ANONYMIZED_EMAIL)
    r"|\[[A-Z][A-Z _]{2,60}\]"                   # [REDACTED], [EMAIL ADDRESS]
    r"|<[A-Z][A-Z _]{2,60}>"                     # <PERSON>
)


def compile_patterns(cfg: dict, dataset: str) -> re.Pattern | None:
    pats = [*cfg["placeholders"].get("common", []), *cfg["placeholders"].get(dataset, [])]
    return re.compile("|".join(f"(?:{p})" for p in pats)) if pats else None


def placeholder_spans(text: str, pattern: re.Pattern | None) -> list[tuple[int, int]]:
    return [m.span() for m in pattern.finditer(text)] if pattern else []


def shape(token: str) -> str:
    return re.sub(r"\d+", "#", token)


def discover(turns: pd.DataFrame, min_conversations: int = 5) -> pd.DataFrame:
    """Counts of candidate placeholder shapes per dataset, k-thresholded by distinct conversations."""
    convs: dict[tuple[str, str], set] = defaultdict(set)
    occ: Counter = Counter()
    for ds, cid, text in turns[["dataset", "conversation_id", "text"]].itertuples(index=False):
        for m in CANDIDATE.finditer(text):
            key = (ds, shape(m.group()))
            convs[key].add(cid)
            occ[key] += 1
    rows = [{"dataset": ds, "placeholder_shape": shp, "occurrences": occ[(ds, shp)],
             "conversations": len(c)} for (ds, shp), c in convs.items() if len(c) >= min_conversations]
    cols = ["dataset", "placeholder_shape", "occurrences", "conversations"]
    return pd.DataFrame(rows, columns=cols).sort_values(["dataset", "conversations"], ascending=[True, False])
