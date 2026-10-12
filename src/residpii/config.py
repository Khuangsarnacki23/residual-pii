"""YAML config loading with `extends:` inheritance, plus stable hashing for the stage cache."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    cfg.setdefault("taxonomy_path", str(path.parent / "taxonomy.yaml"))
    return cfg


def load_taxonomy(cfg: dict) -> dict[str, Any]:
    with open(cfg["taxonomy_path"]) as f:
        return yaml.safe_load(f)


def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class Paths:
    """Resolves the on-disk layout for one run.

    data_root/
      hmac.key                      (0600) secret key for span hashing
      runs/<run>/work/              metadata, sample manifests, per-conversation features (no raw text)
      runs/<run>/restricted/        (0700) raw sampled user turns + masked span store
    <repo>/results/<run>/           aggregate tables, figures, methods, disclosure drafts
    """

    def __init__(self, cfg: dict):
        self.data_root = Path(os.path.expanduser(cfg["data_root"]))
        self.run = self.data_root / "runs" / cfg["run_name"]
        self.work = self.run / "work"
        self.restricted = self.run / "restricted"
        results = Path(cfg["results_dir"])
        self.results = (results if results.is_absolute() else REPO_ROOT / results) / cfg["run_name"]
        self.tables = self.results / "tables"
        self.figures = self.results / "figures"

    def ensure(self) -> "Paths":
        for p in (self.data_root, self.run, self.work, self.tables, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        self.restricted.mkdir(parents=True, exist_ok=True)
        os.chmod(self.data_root, 0o700)
        os.chmod(self.restricted, 0o700)
        return self
