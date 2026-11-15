"""Per-user quasi-identifier profiles, coarsened with fixed lookup tables (Stage 2).

Input attributes use the labelling schema (configs/default.yaml `labeling.attributes`): one row per
(conversation, turn, attribute, kind, value), where kind is `stated` or `inferred` and the attribute
is about the person writing. The same rows come from human labels or from the LLM detector, so the
extractor can be checked against the gold set with the same code (`attribute_agreement`).

A profile is one row per (user, variant) with at most one value per dimension in DIMENSIONS.
Profiles are personal data and are stored only in the restricted directory; `profile_summary` is
the aggregate view that may be published.

Six dimensions are generalised with lookup tables (configs/coarsening.yaml), so that two users who
word the same fact differently still match: region (US state or a listed region), country,
occupation group (SOC major group), age band, gender and education level. A value the tables
do not recognise is `unknown` and never counts towards uniqueness. The remaining attributes have no
table and are kept as stated (normalised text): employer, relationship, nationality_ethnicity,
religion, political, income, health.

Nothing here looks anything up outside those tables, and users are grouped only by the dataset's
own user proxy (`user_key`, a keyed hash).
"""
from __future__ import annotations

import re

import pandas as pd
import yaml

from .sampling import activity_bucket

# SOC 2018 major groups, plus two non-SOC groups for people outside the labour force.
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

TABLE_DIMENSIONS = ("region", "country", "occupation_group", "age_band", "gender", "education_level")
# No lookup table exists for these: the stated value is kept. Dimension name == attribute name.
AS_STATED = ("employer", "relationship", "nationality_ethnicity", "religion", "political", "income", "health")
DIMENSIONS = (*TABLE_DIMENSIONS, *AS_STATED)
VARIANTS = ("stated_only", "stated_plus_inferred")
UNKNOWN, AMBIGUOUS = "unknown", "ambiguous"
NOT_KNOWN = (UNKNOWN, AMBIGUOUS)    # a dimension with one of these is treated as not available
ATTR_COLUMNS = ["dataset", "conversation_id", "turn_index", "attribute", "kind", "value"]


def normalize(value: str) -> str:
    v = re.sub(r"[^\w\s/+-]", "", str(value).casefold().replace(".", ""))
    v = re.sub(r"\s+", " ", v).strip()
    return re.sub(r"^the ", "", v)


class Coarsener:
    """Maps one attribute (labelling schema) onto (dimension, value) pairs. Deterministic:
    the same input and tables always give the same output."""

    def __init__(self, spec: dict):
        self.band = int(spec["age_band_years"])
        self.top = int(spec["age_top_code"])
        self.state = {}                                  # alias -> US state name
        for _, pairs in spec["us_census_regions"].items():
            for pair in pairs:
                abbr, name = pair.split(":")
                self.state[normalize(abbr)] = self.state[normalize(name)] = name
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
        self.gender = {normalize(a): g for g, aliases in spec["gender"].items() for a in aliases}
        self.education = [(lvl, re.compile(p, re.I)) for lvl, p in spec["education_level"]]
        self.school = [(lvl, re.compile(p, re.I)) for lvl, p in spec["school"]]
        self.occupation = [(re.compile(row[0], re.I), str(row[1]), row[2] if len(row) > 2 else "")
                           for row in spec["occupation_keywords"]]
        bad = {g for _, g, _ in self.occupation} - set(OCCUPATION_GROUPS)
        if bad:
            raise ValueError(f"coarsening.yaml: unknown occupation groups {sorted(bad)}")

    # -- single dimensions
    def _age(self, value: str) -> str:
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
        lo = int(age // self.band * self.band)
        return f"{lo}-{min(lo + self.band, self.top) - 1}"

    def _first(self, rules: list, value: str) -> str:
        v = normalize(value)
        return next((lvl for lvl, pat in rules if pat.search(v)), UNKNOWN)

    def _occupation(self, value: str) -> str:
        # SOC major group only. Values come at mixed detail ("nurse" vs "healthcare"); a minor group
        # for some users and not others would make the same job look like two values.
        v = normalize(value)
        return next((g for pat, g, _ in self.occupation if pat.search(v)), UNKNOWN)

    def _place(self, value: str) -> list[tuple[str, str]]:
        """A region-level location: a US state, a listed region, or a listed city (-> its region).
        "Leeds, UK" style values are tried part by part."""
        for part in [value, *value.split(",")]:
            v = normalize(part)
            if v in self.state:
                return [("region", self.state[v]), ("country", "united states")]
            for table in (self.region, self.city):
                if v in table:
                    region, country = table[v]
                    return [("region", region), ("country", country)]
        return [("region", UNKNOWN)]

    # -- dispatch
    def candidates(self, attribute: str, value: str) -> list[tuple[str, str]]:
        v = normalize(value)
        if not v:
            return []
        if attribute == "location_region":
            return self._place(value)
        if attribute == "location_country":
            hit = self.country.get(v) or next((self.country[normalize(p)] for p in value.split(",")
                                               if normalize(p) in self.country), None)
            return [("country", hit or UNKNOWN)]
        if attribute == "age_band":
            return [("age_band", self._age(value))]
        if attribute == "gender":
            return [("gender", self.gender.get(v, UNKNOWN))]
        if attribute == "education":
            level = self._first(self.education, value)
            return [("education_level", level if level != UNKNOWN else self._first(self.school, value))]
        if attribute == "occupation":
            return [("occupation_group", self._occupation(value))]
        return [(attribute, v)] if attribute in AS_STATED else []


def load_coarsener(cfg: dict) -> Coarsener:
    with open(cfg["coarsening_path"]) as f:
        return Coarsener(yaml.safe_load(f))


def attributes_from_records(ds: str, records: list[dict]) -> pd.DataFrame:
    """Flattens label-schema records ({conversation_id, turn_index, attributes: [...]}) to ATTR_COLUMNS."""
    rows = [(ds, str(r["conversation_id"]), int(r["turn_index"]), a["attribute"], a["kind"], a["value"])
            for r in records if not r.get("skip") for a in r.get("attributes") or []]
    return pd.DataFrame(rows, columns=ATTR_COLUMNS)


def resolve(votes: pd.Series) -> tuple[str, bool]:
    """votes: value -> number of conversations supporting it. Returns (value, had a conflict).
    Majority vote over known values; a tie is `ambiguous`."""
    known = votes[~votes.index.isin(NOT_KNOWN)]
    if known.empty:
        return UNKNOWN, False
    top = known[known == known.max()]
    return (top.index[0] if len(top) == 1 else AMBIGUOUS), len(known) > 1


PROFILE_COLUMNS = ["dataset", "user", "variant", "n_conversations", "user_n_conversations",
                   "activity_bucket", *DIMENSIONS, *(f"conflict_{d}" for d in DIMENSIONS)]


def candidate_table(attrs: pd.DataFrame, coarsener: Coarsener) -> pd.DataFrame:
    """(conversation_id, turn_index, kind, dim, value) after coarsening; one row per mapped value."""
    rows = [(r.conversation_id, r.turn_index, r.kind, dim, val)
            for r in attrs.itertuples(index=False) for dim, val in coarsener.candidates(r.attribute, r.value)]
    return pd.DataFrame(rows, columns=["conversation_id", "turn_index", "kind", "dim", "value"])


def build_profiles(ds: str, attrs: pd.DataFrame, convs: pd.DataFrame, coarsener: Coarsener) -> pd.DataFrame:
    """One row per (user, variant) for every user in `convs`, including users who revealed nothing
    (all dimensions empty); they are the denominator of the uniqueness shares.

    convs: one row per included conversation with columns conversation_id, user, user_n_conversations.
    attrs: ATTR_COLUMNS rows for (a superset of) those conversations."""
    convs = convs.drop_duplicates("conversation_id")
    users = (convs.groupby("user").agg(n_conversations=("conversation_id", "size"),
                                       user_n_conversations=("user_n_conversations", "first")).reset_index())
    users["user_n_conversations"] = users["user_n_conversations"].astype("int64")
    users["activity_bucket"] = activity_bucket(users["user_n_conversations"])
    cand = candidate_table(attrs[attrs["conversation_id"].isin(convs["conversation_id"])], coarsener)
    cand["user"] = cand["conversation_id"].map(convs.set_index("conversation_id")["user"])
    parts = []
    for variant in VARIANTS:
        c = cand if variant == "stated_plus_inferred" else cand[cand["kind"] == "stated"]
        # One vote per conversation, so a long conversation cannot outvote the rest.
        votes = c.drop_duplicates(["user", "dim", "value", "conversation_id"])
        resolved = {key: resolve(g["value"].value_counts().sort_index())
                    for key, g in votes.groupby(["user", "dim"])}
        p = users.assign(dataset=ds, variant=variant)
        for dim in DIMENSIONS:
            got = [resolved.get((u, dim), (None, False)) for u in p["user"]]
            p[dim] = [v for v, _ in got]
            p[f"conflict_{dim}"] = [c_ for _, c_ in got]
        parts.append(p)
    return pd.concat(parts, ignore_index=True)[PROFILE_COLUMNS]


def known_values(profiles: pd.DataFrame) -> pd.DataFrame:
    """The dimension columns with `unknown`/`ambiguous` blanked: what an observer could actually use."""
    dims = profiles[list(DIMENSIONS)]
    return dims.where(dims.notna() & ~dims.isin(NOT_KNOWN))


def profile_summary(profiles: pd.DataFrame, min_cell: int = 10) -> pd.DataFrame:
    """Aggregate, publishable view: per dimension, how many users have a usable value, how many had
    conflicting values, and how many ended up ambiguous or unmapped. Counts under min_cell are blanked."""
    rows = []
    for (ds, variant), g in profiles.groupby(["dataset", "variant"]):
        known, dims = known_values(g), list(DIMENSIONS)
        for dim in (*dims, "any"):
            if dim == "any":
                counts = {"n_with_value": known.notna().any(axis=1).sum(),
                          "n_conflict": g[[f"conflict_{d}" for d in dims]].any(axis=1).sum(),
                          "n_ambiguous": g[dims].eq(AMBIGUOUS).any(axis=1).sum(),
                          "n_unmapped": g[dims].eq(UNKNOWN).any(axis=1).sum()}
            else:
                counts = {"n_with_value": known[dim].notna().sum(), "n_conflict": g[f"conflict_{dim}"].sum(),
                          "n_ambiguous": (g[dim] == AMBIGUOUS).sum(), "n_unmapped": (g[dim] == UNKNOWN).sum()}
            row = {"dataset": ds, "variant": variant, "attribute": dim}
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


# ---------------------------------------------------------------- extractor validation

AGREEMENT_COLUMNS = ["dataset", "gold", "predictor", "variant", "attribute", "n_items", "n_gold", "n_pred",
                     "n_both", "n_match", "precision", "recall", "value_agreement"]


def _item_values(rec: dict | None, coarsener: Coarsener, variant: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if not rec or rec.get("skip"):
        return out
    for a in rec.get("attributes") or []:
        if variant == "stated_only" and a["kind"] != "stated":
            continue
        for dim, val in coarsener.candidates(a["attribute"], a["value"]):
            if val not in NOT_KNOWN:
                out.setdefault(dim, set()).add(val)
    return out


def attribute_agreement(ds: str, item_ids: list[str], gold: dict[str, dict], pred: dict[str, dict],
                        coarsener: Coarsener, gold_name: str, pred_name: str) -> pd.DataFrame:
    """Per profile dimension, how well a predictor's attributes match the gold attributes on the
    label-set items, compared after coarsening (what Stage 2 actually uses).

    n_gold / n_pred: items with a usable value; n_match: items where the two share a value.
    precision = n_match / n_pred, recall = n_match / n_gold, value_agreement = n_match / n_both."""
    ids = [i for i in item_ids if i in gold and not gold[i].get("skip")]
    rows = []
    for variant in VARIANTS:
        g = {i: _item_values(gold.get(i), coarsener, variant) for i in ids}
        p = {i: _item_values(pred.get(i), coarsener, variant) for i in ids}
        for dim in (*DIMENSIONS, "any"):
            def has(d, i):
                return bool(d[i]) if dim == "any" else dim in d[i]

            def match(i):
                dims = DIMENSIONS if dim == "any" else (dim,)
                return any(g[i].get(x, set()) & p[i].get(x, set()) for x in dims)
            n_g = sum(has(g, i) for i in ids)
            n_p = sum(has(p, i) for i in ids)
            n_b = sum(has(g, i) and has(p, i) for i in ids)
            n_m = sum(match(i) for i in ids)
            rows.append({"dataset": ds, "gold": gold_name, "predictor": pred_name, "variant": variant,
                         "attribute": dim, "n_items": len(ids), "n_gold": n_g, "n_pred": n_p, "n_both": n_b,
                         "n_match": n_m, "precision": n_m / n_p if n_p else float("nan"),
                         "recall": n_m / n_g if n_g else float("nan"),
                         "value_agreement": n_m / n_b if n_b else float("nan")})
    return pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)
