"""Dataset adapters that stream Parquet columns from the Hugging Face Hub into one common schema.

Nothing is downloaded in full. A metadata scan reads only small columns (ids, language, timestamp,
user proxy, ...); text is then fetched only for the row groups that contain sampled conversations,
reading only the nested `content`/`role` fields. Disk use is limited to the sampled user turns.

Common schemas
  meta : dataset, conversation_id, user_key, timestamp, model, n_turns, language, flag_redacted,
         flag_toxic, _file, _rg, _row
  turns: dataset, conversation_id, turn_index, role, text, language
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .safety import keyed_hash


@dataclass(frozen=True)
class Spec:
    id: str
    conv: str
    model: str
    turns: str
    lang: str
    user: str | None = None          # user/IP proxy column, if the dataset has one
    time: str | None = None
    time_unit: str | None = None     # "s" for epoch seconds; None for native timestamps
    flags: dict = field(default_factory=dict)   # common flag name -> source column
    turn_fields: tuple = ("content", "role")


SPECS = {
    # allenai/WildChat-1M: Presidio-redacted in place; hashed_ip is the user proxy.
    "wildchat": Spec(id="conversation_hash", conv="conversation", model="model", turns="turn",
                     lang="language", user="hashed_ip", time="timestamp",
                     flags={"redacted": "redacted", "toxic": "toxic"},
                     turn_fields=("content", "role", "language")),
    # lmsys/chatbot_arena_conversations: user turns are identical in conversation_a and _b.
    "arena": Spec(id="question_id", conv="conversation_a", model="model_a", turns="turn",
                  lang="language", user="judge", time="tstamp", time_unit="s"),
    # lmsys/lmsys-chat-1m: no user proxy and no timestamp in the public release.
    "lmsys": Spec(id="conversation_id", conv="conversation", model="model", turns="turn",
                  lang="language", flags={"redacted": "redacted"}),
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class HFSource:
    def __init__(self, name: str, ds_cfg: dict, scan_cfg: dict, key: bytes):
        from huggingface_hub import HfApi, HfFileSystem

        self.name = name
        self.spec = SPECS[name]
        self.repo = ds_cfg["repo"]
        self.key = key
        self.fs = HfFileSystem()
        self.revision = ds_cfg.get("revision") or HfApi().dataset_info(self.repo).sha
        self.max_files = scan_cfg.get("max_files")
        self.max_rgs = scan_cfg.get("max_row_groups_per_file")
        base = f"datasets/{self.repo}@{self.revision}"
        files = sorted(self.fs.glob(f"{base}/**/*.parquet"))
        if not files:
            raise RuntimeError(f"No parquet files found for {self.repo}@{self.revision}")
        self.files = files[: self.max_files] if self.max_files else files

    def _open(self, path: str) -> pq.ParquetFile:
        return pq.ParquetFile(self.fs.open(path, block_size=8 * 2**20), pre_buffer=True)

    def _check_columns(self, pf: pq.ParquetFile) -> None:
        names = set(pf.schema_arrow.names)
        s = self.spec
        wanted = [s.id, s.conv, s.model, s.turns, s.lang, s.user, s.time, *s.flags.values()]
        missing = [c for c in wanted if c and c not in names]
        if missing:
            raise RuntimeError(f"{self.name}: expected columns missing from {self.repo}: {missing}. "
                               f"Available: {sorted(names)}")

    def scan_meta(self) -> pd.DataFrame:
        s = self.spec
        cols = [c for c in [s.id, s.model, s.turns, s.lang, s.user, s.time, *s.flags.values()] if c]
        frames = []
        for fi, path in enumerate(self.files):
            pf = self._open(path)
            if fi == 0:
                self._check_columns(pf)
            n_rg = pf.metadata.num_row_groups
            rgs = list(range(min(n_rg, self.max_rgs) if self.max_rgs else n_rg))
            tb = pf.read_row_groups(rgs, columns=cols)
            sizes = [pf.metadata.row_group(i).num_rows for i in rgs]
            df = tb.to_pandas()
            df["_file"] = fi
            df["_rg"] = np.repeat(rgs, sizes)
            df["_row"] = np.concatenate([np.arange(n) for n in sizes])
            frames.append(df)
            log(f"[{self.name}] scanned metadata: file {fi + 1}/{len(self.files)}, "
                f"{len(rgs)} row groups, {len(df):,} conversations")
        raw = pd.concat(frames, ignore_index=True)
        return self._normalize_meta(raw)

    def _normalize_meta(self, raw: pd.DataFrame) -> pd.DataFrame:
        s = self.spec
        out = pd.DataFrame({
            "dataset": self.name,
            "conversation_id": raw[s.id].astype(str),
            "model": raw[s.model].astype(str),
            "n_turns": raw[s.turns].astype("int64"),
            "language": raw[s.lang].astype(str),
        })
        # The raw user proxy is replaced by a keyed hash immediately and never stored.
        out["user_key"] = (raw[s.user].map(lambda v: keyed_hash(self.key, str(v)) if pd.notna(v) else None)
                           if s.user else None)
        if s.time:
            ts = (pd.to_datetime(raw[s.time], unit=s.time_unit, utc=True) if s.time_unit
                  else pd.to_datetime(raw[s.time], utc=True))
            out["timestamp"] = ts
        else:
            out["timestamp"] = pd.NaT
        for flag in ("redacted", "toxic"):
            col = s.flags.get(flag)
            out[f"flag_{flag}"] = raw[col].astype("boolean") if col else pd.NA
        out[["_file", "_rg", "_row"]] = raw[["_file", "_rg", "_row"]]
        n_before = len(out)
        out = out.drop_duplicates("conversation_id", keep="first").reset_index(drop=True)
        if len(out) < n_before:
            log(f"[{self.name}] dropped {n_before - len(out):,} rows with duplicate conversation ids")
        return out

    def _turn_columns(self, pf: pq.ParquetFile) -> list[str]:
        """Leaf paths of the per-turn fields. Writers name list elements differently
        (`conv.list.element.content` vs `conv.list.item.content`), so look them up."""
        s = self.spec
        paths = [pf.metadata.schema.column(j).path for j in range(pf.metadata.num_columns)]
        cols = []
        for f in s.turn_fields:
            hits = [p for p in paths if p.startswith(f"{s.conv}.list.") and p.split(".")[3:] == [f]]
            if not hits:
                raise RuntimeError(f"{self.name}: no leaf column for {s.conv}[].{f}")
            cols.append(hits[0])
        return cols

    def iter_turn_batches(self, meta: pd.DataFrame):
        """Yields user turns one Parquet row group at a time (bounded memory for corpus passes)."""
        groups = meta.groupby(["_file", "_rg"], sort=True)
        cur_file, pf, cols = None, None, None
        for i, ((fi, rg), g) in enumerate(groups):
            if fi != cur_file:
                pf, cur_file = self._open(self.files[fi]), fi
                cols = self._turn_columns(pf)
            tb = pf.read_row_group(int(rg), columns=cols)
            if tb.num_columns != 1:
                raise RuntimeError(f"{self.name}: projected read returned {tb.num_columns} columns")
            conv = tb.column(0)
            picked = conv.take(g["_row"].to_numpy()).to_pylist()
            rows = []
            for cid, lang, turns in zip(g["conversation_id"], g["language"], picked):
                rows.extend(_user_turns(self.name, cid, lang, turns))
            if (i + 1) % 50 == 0 or i + 1 == groups.ngroups:
                log(f"[{self.name}] fetched text: {i + 1}/{groups.ngroups} row groups")
            yield pd.DataFrame(rows, columns=TURN_COLUMNS)

    def fetch_turns(self, meta: pd.DataFrame) -> pd.DataFrame:
        batches = list(self.iter_turn_batches(meta))
        return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=TURN_COLUMNS)


TURN_COLUMNS = ["dataset", "conversation_id", "turn_index", "role", "text", "language"]


def _user_turns(dataset: str, cid: str, conv_lang: str, turns) -> list[tuple]:
    out = []
    for idx, t in enumerate(turns or []):
        if (t.get("role") or "").lower() != "user":
            continue
        out.append((dataset, cid, idx, "user", t.get("content") or "", t.get("language") or conv_lang))
    return out


class InMemorySource:
    """Source backed by an in-memory list of conversations (used for tests and the canary check)."""

    def __init__(self, name: str, conversations: list[dict], key: bytes):
        self.name, self.key, self.revision = name, key, "in-memory"
        self.repo = f"synthetic/{name}"
        self._convs = {c["conversation_id"]: c for c in conversations}

    def scan_meta(self) -> pd.DataFrame:
        recs = []
        for i, c in enumerate(self._convs.values()):
            recs.append({
                "dataset": self.name, "conversation_id": c["conversation_id"],
                "model": c.get("model", "m"), "n_turns": sum(t["role"] == "user" for t in c["turns"]),
                "language": c.get("language", "English"),
                "user_key": keyed_hash(self.key, c["user"]) if c.get("user") else None,
                "timestamp": pd.Timestamp(c.get("timestamp", "2023-05-01"), tz="UTC"),
                "flag_redacted": pd.NA, "flag_toxic": pd.NA, "_file": 0, "_rg": 0, "_row": i,
            })
        df = pd.DataFrame(recs)
        df["flag_redacted"] = df["flag_redacted"].astype("boolean")
        df["flag_toxic"] = df["flag_toxic"].astype("boolean")
        return df

    def fetch_turns(self, meta: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for cid, lang in zip(meta["conversation_id"], meta["language"]):
            rows.extend(_user_turns(self.name, cid, lang, self._convs[cid]["turns"]))
        return pd.DataFrame(rows, columns=TURN_COLUMNS)

    def iter_turn_batches(self, meta: pd.DataFrame):
        yield self.fetch_turns(meta)
