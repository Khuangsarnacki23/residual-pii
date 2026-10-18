"""Privacy guardrails: keyed span hashing and a span record that cannot carry raw text.

Detected spans are stored as *pointers* (dataset, conversation, turn, character offsets) plus
category, length and an HMAC of the normalized span text. The HMAC is keyed with a secret that lives
outside the repo: names and phone numbers are low-entropy, so an unkeyed or salt-only hash could be
brute-forced if the salt leaked. The HMAC exists only to match identical values across detectors
and annotators; it is not an anonymization guarantee.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

_KEY_CACHE: dict[Path, bytes] = {}


def load_or_create_key(data_root: Path) -> bytes:
    path = Path(data_root) / "hmac.key"
    if path in _KEY_CACHE:
        return _KEY_CACHE[path]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(secrets.token_bytes(32))
    key = path.read_bytes()
    _KEY_CACHE[path] = key
    return key


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def keyed_hash(key: bytes, text: str) -> str:
    return hmac.new(key, normalize_for_hash(text).encode(), hashlib.sha256).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class SafeSpan:
    """A detected span with no raw text. Safe to log; store only under the restricted directory."""

    dataset: str
    conversation_id: str
    turn_index: int
    start: int
    end: int
    detector: str
    entity: str
    category: str
    tier: str
    score: float
    span_hmac: str
    in_placeholder: bool
    is_dummy: bool

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_row(self) -> dict:
        row = asdict(self)
        row["length"] = self.length
        return row


# Columns that may legitimately appear in any table written outside the restricted directory.
# `assert_safe_table` refuses anything else, so a raw-text column can't leak into results by accident.
SAFE_TEXT_COLUMNS = {
    "dataset", "conversation_id", "detector", "entity", "category", "tier", "task_type",
    "stratum", "month", "turn_bucket", "model", "variant", "metric", "placeholder_shape",
    "language", "span_hmac", "user_key", "group", "subset", "pattern", "repo", "revision",
    "detectors", "dedup_status", "category_label",
    # labelling benchmark: vocabularies from configs/default.yaml, never data text
    "whose", "annotator", "label_stratum", "attribute", "gold",
}


def assert_safe_table(df, where: str) -> None:
    import pandas as pd

    def holds_text(col: pd.Series) -> bool:
        if pd.api.types.is_string_dtype(col) and not pd.api.types.is_object_dtype(col):
            return True
        return pd.api.types.is_object_dtype(col) and col.map(lambda v: isinstance(v, str)).any()

    bad = [c for c in df.columns if c not in SAFE_TEXT_COLUMNS and holds_text(df[c])]
    if bad:
        raise ValueError(f"Refusing to write {where}: unvetted text columns {bad}")
