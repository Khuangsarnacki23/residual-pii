"""Per-user quasi-identifier profiles from LLM attributes, coarsened with fixed lookup tables.

A profile is one row per (user, variant, coarsening level) with at most one value for each
dimension in DIMENSIONS. Profiles are personal data and are stored only in the restricted
directory; `profile_summary` is the aggregate view that may be published.

Nothing here looks anything up outside the tables in configs/coarsening.yaml, and users are grouped
only by the dataset's own user proxy (`user_key`, a keyed hash).
"""
from __future__ import annotations

import re

import pandas as pd
import yaml

from .llm import OCCUPATION_GROUPS, unit_keys
from .sampling import activity_bucket

DIMENSIONS = ("region", "occupation_group", "age_band", "gender", "education_level", "country")
LEVELS = ("fine", "coarse")
VARIANTS = ("stated_only", "stated_plus_inferred")
UNKNOWN, AMBIGUOUS = "unknown", "ambiguous"
NOT_KNOWN = (UNKNOWN, AMBIGUOUS)    # a dimension with one of these is treated as not available


def normalize(value: str) -> str:
    v = re.sub(r"[^\w\s/+-]", "", value.casefold().replace(".", ""))
    v = re.sub(r"\s+", " ", v).strip()
    return re.sub(r"^the ", "", v)


class Coarsener:
    """Maps one raw attribute onto (dimension, value) pairs at a coarsening level. Deterministic:
    the same input and tables always give the same output."""

    def __init__(self, spec: dict):
        self.band = spec["age_band_years"]
        self.top = int(spec["age_top_code"])
        self.state, self.census = {}, {}                 # alias -> state name; state name -> census region
        for region, pairs in spec["us_census_regions"].items():
            for pair in pairs:
                abbr, name = pair.split(":")
                self.state[normalize(abbr)] = self.state[normalize(name)] = name
                self.census[name] = f"us_{region}"
        self.country = {}
        for name, aliases in spec["countries"].items():
            for a in (name, *(aliases or [])):
                self.country[normalize(a)] = name
        self.region, self.city = {}, {}                  # region -> country; city -> (region, country)
        for country, regions in spec["cities"].items():
            for region, cities in regions.items():
                self.region[normalize(region)] = (region, country)
                for c in cities:
                    self.city[normalize(c)] = (region, country)
        for name in self.census:
            self.region[normalize(name)] = (name, "united states")
        self.gender = {normalize(a): g for g, aliases in spec["gender"].items() for a in aliases}
        self.education = [(lvl, re.compile(p, re.I)) for lvl, p in spec["education_level"]]
        self.school = [(lvl, re.compile(p, re.I)) for lvl, p in spec["school"]]
        self.education_coarse = spec["education_coarse"]
        self.occupation = [(re.compile(row[0], re.I), str(row[1]), row[2] if len(row) > 2 else "")
                           for row in spec["occupation_keywords"]]
        bad = {g for _, g, _ in self.occupation} - set(OCCUPATION_GROUPS)
        if bad:
            raise ValueError(f"coarsening.yaml: unknown occupation groups {sorted(bad)}")

    # -- single dimensions
    def _place(self, region: str | None, country: str | None, level: str) -> list[tuple[str, str]]:
        out = [("country", country)] if country else []
        if level == "fine":
            if region:
                out.append(("region", region))
        elif region and country == "united states":
            out.append(("region", self.census[region]))
        elif country and country != "united states":
            out.append(("region", country))              # coarse level outside the US: the country
        return out

    def _age(self, value: str, level: str) -> str:
        v = normalize(value)
        nums = [int(n) for n in re.findall(r"\d{1,3}", v)]
        if re.fullmatch(r"\d0s", v) or re.search(r"\b(?:early|mid|late) \d0s\b", v):
            age = nums[0] + {"early": 2, "late": 8}.get(v.split()[0], 5)
        elif len(nums) == 2 and nums[0] <= nums[1]:
            age = (nums[0] + nums[1]) / 2                # a range: use its midpoint
        elif len(nums) == 1:
            age = nums[0]
        elif "teen" in v:
            age = 16
        else:
            return UNKNOWN
        if not 0 < age < 120:
            return UNKNOWN
        if age >= self.top:
            return f"{self.top}+"
        width = int(self.band[level])
        lo = int(age // width * width)
        return f"{lo}-{min(lo + width, self.top) - 1}"

    def _first(self, rules: list, value: str) -> str:
        v = normalize(value)
        return next((lvl for lvl, pat in rules if pat.search(v)), UNKNOWN)

    def _occupation(self, value: str, group: str, minor: str, level: str) -> str:
        if group not in OCCUPATION_GROUPS:
            v = normalize(value)
            group, minor = next(((g, m) for pat, g, m in self.occupation if pat.search(v)), ("", ""))
        if not group:
            return UNKNOWN
        # Fine level uses the SOC minor group when one is known, else falls back to the major group.
        return minor if level == "fine" and minor else group

    # -- dispatch
    def candidates(self, attr_type: str, value: str, level: str, occupation_group: str = "",
                   occupation_minor_group: str = "") -> list[tuple[str, str]]:
        v = normalize(value)
        if attr_type == "city":
            region, country = self.city.get(v, (None, None))
            return self._place(region, country, level) or [("region", UNKNOWN)]
        if attr_type == "state_or_region":
            if v in self.state:
                return self._place(self.state[v], "united states", level)
            region, country = self.region.get(v, (None, None))
            return self._place(region, country, level) or [("region", UNKNOWN)]
        if attr_type == "country":
            country = self.country.get(v)
            return self._place(None, country, level) or [("country", UNKNOWN)]
        if attr_type in ("age", "age_range"):
            return [("age_band", self._age(value, level))]
        if attr_type == "gender":
            return [("gender", self.gender.get(v, UNKNOWN))]
        if attr_type in ("education_level", "school"):
            lvl = self._first(self.education if attr_type == "education_level" else self.school, value)
            if lvl != UNKNOWN and level == "coarse":
                lvl = self.education_coarse[lvl]
            return [("education_level", lvl)]
        if attr_type == "occupation":
            return [("occupation_group", self._occupation(value, occupation_group, occupation_minor_group, level))]
        return []          # employer_type, relationship_status, nationality, language_native, other


def load_coarsener(cfg: dict) -> Coarsener:
    with open(cfg["coarsening_path"]) as f:
        return Coarsener(yaml.safe_load(f))


def resolve(votes: pd.Series) -> tuple[str, bool]:
    """votes: value -> number of conversations supporting it. Returns (value, had a conflict).
    Majority vote over known values; a tie is `ambiguous`."""
    known = votes[~votes.index.isin(NOT_KNOWN)]
    if known.empty:
        return UNKNOWN, False
    top = known[known == known.max()]
    return (top.index[0] if len(top) == 1 else AMBIGUOUS), len(known) > 1


PROFILE_COLUMNS = ["dataset", "user", "variant", "level", "n_conversations", "user_n_conversations",
                   "activity_bucket", *DIMENSIONS, *(f"conflict_{d}" for d in DIMENSIONS)]


def build_profiles(attrs: pd.DataFrame, turn_log: pd.DataFrame, sample: pd.DataFrame, coarsener: Coarsener,
                   min_confidence: float) -> pd.DataFrame:
    """One row per (user, variant, level) for every user the LLM processed, including users who
    revealed nothing (all dimensions empty); they are the denominator of the uniqueness shares."""
    parts = []
    for ds, tl in turn_log.groupby("dataset"):
        smp = sample[sample["dataset"] == ds]
        units = unit_keys(smp)
        base = pd.DataFrame({"conversation_id": tl["conversation_id"].unique()})
        base["user"] = base["conversation_id"].map(units)
        users = base.groupby("user").size().rename("n_conversations").reset_index()
        if "user_n_conversations" in smp:
            full = smp.assign(user=smp["conversation_id"].map(units)).groupby("user")["user_n_conversations"].first()
            users["user_n_conversations"] = users["user"].map(full).astype("int64")
        else:
            users["user_n_conversations"] = users["n_conversations"]
        users["activity_bucket"] = activity_bucket(users["user_n_conversations"])

        a = attrs[(attrs["dataset"] == ds) & (attrs["about"] == "user") & (attrs["confidence"] >= min_confidence)]
        cand = [(units[r.conversation_id], r.conversation_id, r.evidence, level, dim, val)
                for r in a.itertuples(index=False) for level in LEVELS
                for dim, val in coarsener.candidates(r.type, r.value, level, r.occupation_group,
                                                     r.occupation_minor_group)]
        cand = pd.DataFrame(cand, columns=["user", "conversation_id", "evidence", "level", "dim", "value"])
        for variant in VARIANTS:
            c = cand if variant == "stated_plus_inferred" else cand[cand["evidence"] == "stated"]
            # One vote per conversation, so a long conversation cannot outvote the rest.
            votes = c.drop_duplicates(["user", "level", "dim", "value", "conversation_id"])
            resolved = {}
            for (user, level, dim), g in votes.groupby(["user", "level", "dim"]):
                resolved[(user, level, dim)] = resolve(g["value"].value_counts().sort_index())
            for level in LEVELS:
                p = users.assign(dataset=ds, variant=variant, level=level)
                for dim in DIMENSIONS:
                    got = [resolved.get((u, level, dim), (None, False)) for u in p["user"]]
                    p[dim] = [v for v, _ in got]
                    p[f"conflict_{dim}"] = [c_ for _, c_ in got]
                parts.append(p)
    if not parts:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    return pd.concat(parts, ignore_index=True)[PROFILE_COLUMNS]


def known_values(profiles: pd.DataFrame) -> pd.DataFrame:
    """The dimension columns with `unknown`/`ambiguous` blanked: what an observer could actually use."""
    dims = profiles[list(DIMENSIONS)]
    return dims.where(dims.notna() & ~dims.isin(NOT_KNOWN))


def profile_summary(profiles: pd.DataFrame, min_cell: int = 10) -> pd.DataFrame:
    """Aggregate, publishable view: per dimension, how many users have a usable value, how many had
    conflicting values, and how many ended up ambiguous or unmapped. Counts under min_cell are blanked."""
    rows = []
    for (ds, variant, level), g in profiles.groupby(["dataset", "variant", "level"]):
        known = known_values(g)
        for dim in (*DIMENSIONS, "any"):
            if dim == "any":
                counts = {"n_with_value": known.notna().any(axis=1).sum(),
                          "n_conflict": g[[f"conflict_{d}" for d in DIMENSIONS]].any(axis=1).sum(),
                          "n_ambiguous": g[list(DIMENSIONS)].eq(AMBIGUOUS).any(axis=1).sum(),
                          "n_unmapped": g[list(DIMENSIONS)].eq(UNKNOWN).any(axis=1).sum()}
            else:
                counts = {"n_with_value": known[dim].notna().sum(), "n_conflict": g[f"conflict_{dim}"].sum(),
                          "n_ambiguous": (g[dim] == AMBIGUOUS).sum(), "n_unmapped": (g[dim] == UNKNOWN).sum()}
            row = {"dataset": ds, "variant": variant, "level": level, "attribute": dim, "n_users": len(g)}
            suppressed = len(g) < min_cell
            for name, n in counts.items():
                small = suppressed or 0 < n < min_cell      # zero is not a disclosure; 1-9 is
                suppressed |= small
                row[name] = pd.NA if small else int(n)
                row[name.replace("n_", "share_", 1)] = pd.NA if small else n / len(g)
            rows.append({**row, "n_users": pd.NA if len(g) < min_cell else len(g), "suppressed": suppressed})
    out = pd.DataFrame(rows)
    for c in out.columns:
        if c.startswith("n_"):
            out[c] = out[c].astype("Int64")
        elif c.startswith("share_"):
            out[c] = out[c].astype("Float64")
    return out
