"""Exact and near-duplicate matching of conversations on user-turn text (MinHash + LSH).

Used only to measure overlap between lmsys-chat-1m and chatbot_arena_conversations. Matching is
text-to-text; nothing about users is linked.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np
import pandas as pd

_PRIME = np.uint64((1 << 61) - 1)
_MASK = np.uint64(0xFFFFFFFF)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def conversation_texts(turns: pd.DataFrame) -> pd.Series:
    """One normalized string per conversation: all user turns in order."""
    t = turns.sort_values(["conversation_id", "turn_index"])
    return t.groupby("conversation_id", sort=False)["text"].agg(lambda s: normalize("\n".join(s)))


def exact_key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class MinHasher:
    def __init__(self, num_perm: int, shingle_words: int, seed: int):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, int(_PRIME), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_PRIME), size=num_perm, dtype=np.uint64)
        self.k = shingle_words

    def shingles(self, text: str) -> set[str]:
        words = text.split()
        if len(words) <= self.k:
            return {text}
        return {" ".join(words[i:i + self.k]) for i in range(len(words) - self.k + 1)}

    def signature(self, text: str) -> np.ndarray:
        hv = np.fromiter(
            (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little")
             for s in self.shingles(text)), dtype=np.uint64)
        # Universal hashing with uint64 wrap-around, as in datasketch; deterministic given the seed.
        with np.errstate(over="ignore"):
            ph = (hv[:, None] * self.a[None, :] + self.b[None, :]) % _PRIME & _MASK
        return ph.min(axis=0).astype(np.uint32)


class LSHIndex:
    def __init__(self, hasher: MinHasher, bands: int):
        self.h, self.bands = hasher, bands
        self.rows = len(hasher.a) // bands
        self.tables: list[dict] = [defaultdict(list) for _ in range(bands)]
        self.sigs: list[np.ndarray] = []
        self.ids: list[str] = []
        self.exact: dict[str, str] = {}

    def _bands(self, sig: np.ndarray):
        for b in range(self.bands):
            yield b, sig[b * self.rows:(b + 1) * self.rows].tobytes()

    def add(self, cid: str, text: str) -> None:
        sig = self.h.signature(text)
        idx = len(self.ids)
        self.ids.append(cid)
        self.sigs.append(sig)
        self.exact.setdefault(exact_key(text), cid)
        for b, key in self._bands(sig):
            self.tables[b][key].append(idx)

    def query(self, text: str) -> tuple[bool, float]:
        """(exact match?, best estimated Jaccard among LSH candidates)."""
        if exact_key(text) in self.exact:
            return True, 1.0
        sig = self.h.signature(text)
        cands = {i for b, key in self._bands(sig) for i in self.tables[b].get(key, ())}
        best = max((float(np.mean(self.sigs[i] == sig)) for i in cands), default=0.0)
        return False, best


def build_index(reference: pd.Series, dcfg: dict, seed: int) -> LSHIndex:
    index = LSHIndex(MinHasher(dcfg["num_perm"], dcfg["shingle_words"], seed), dcfg["bands"])
    for cid, text in reference.items():
        index.add(cid, text)
    return index


def match(reference: pd.Series | LSHIndex, query: pd.Series, dcfg: dict, seed: int) -> pd.DataFrame:
    """For each conversation in `query`, whether it duplicates any conversation in `reference`."""
    index = reference if isinstance(reference, LSHIndex) else build_index(reference, dcfg, seed)
    rows = []
    for cid, text in query.items():
        is_exact, j = index.query(text)
        status = "exact" if is_exact else ("near" if j >= dcfg["jaccard_threshold"] else "unique")
        rows.append({"conversation_id": cid, "dedup_status": status, "best_jaccard": round(j, 4)})
    return pd.DataFrame(rows, columns=["conversation_id", "dedup_status", "best_jaccard"])
