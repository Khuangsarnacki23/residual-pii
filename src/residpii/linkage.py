"""Can two sets of conversations by the same user be linked to each other?

Uniqueness says a user's profile is shared by nobody else in the sample. This module tests the step
an attacker would take next, without using any outside data: each user's conversations are split at
random into two halves, each half is profiled on its own, and every first half is matched against
all second halves. A user counts as linked when their own second half is the single best match.
Nothing is linked to a real identity; the only ground truth used is the dataset's own user proxy.

Halves are matched on shared keys: a profile dimension with the same value, or the same direct
identifier (compared by keyed hash, never by value). A shared key counts for more the fewer users
have it. Every output is a share or a count of users.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import uniqueness
from .llm import unit_keys
from .profiles import DIMENSIONS, LEVEL, VARIANTS, Coarsener, build_profiles, known_values
from .sampling import ACTIVITY_LABELS, activity_bucket

# (detector, category) pairs used as direct identifiers. Presidio's PERSON spans are left out: most
# are names of other people, characters or code identifiers, not the user's own name.
DIRECT_KEYS = (("selfid", "name"), ("presidio", "email"), ("presidio", "phone"), ("secrets", "api_key_token"))
MATCH_ON = ("profile", "direct_identifiers", "profile_and_direct")
NO_VARIANT = "any"            # direct identifiers do not depend on the profile variant
EPS = 1e-9


def direct_keys(spans: pd.DataFrame, units: pd.Series) -> pd.DataFrame:
    """(user, conversation_id, category, key) for direct identifiers in user turns; keys are hashes."""
    wanted = pd.Series(list(zip(spans["detector"], spans["category"])), index=spans.index).isin(DIRECT_KEYS)
    s = spans[wanted & (spans["role"] == "user") & ~spans["in_placeholder"] & ~spans["is_dummy"]
              & spans["conversation_id"].isin(units.index)]
    out = pd.DataFrame({"user": s["conversation_id"].map(units).to_numpy(),
                        "conversation_id": s["conversation_id"].to_numpy(), "category": s["category"].to_numpy(),
                        "key": (s["category"] + ":" + s["span_hmac"]).to_numpy()})
    return out.drop_duplicates().reset_index(drop=True)


def _profile_scores(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """a, b: known profile values of the two halves, same users in the same order. Returns the
    users x users score matrix and, per user, whether each half has any usable value."""
    n = len(a)
    S = np.zeros((n, n))
    for dim in DIMENSIONS:
        codes, _ = pd.factorize(pd.concat([a[dim], b[dim]], ignore_index=True))
        ca, cb = codes[:n], codes[n:]
        share = np.bincount(cb[cb >= 0], minlength=codes.max() + 2) / n
        w = -np.log(np.where(share > 0, share, 1.0))            # rarer value among second halves = more weight
        S += ((ca[:, None] == cb[None, :]) & (ca[:, None] >= 0)) * w[cb][None, :]
    return S, a.notna().any(axis=1).to_numpy(), b.notna().any(axis=1).to_numpy()


def _direct_scores(keys: pd.DataFrame, in_a: pd.Series, users: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(users)
    pos = pd.Series(np.arange(n), index=users)
    first = keys["conversation_id"].map(in_a).to_numpy(dtype=bool)
    codes, uniq = pd.factorize(keys["key"])
    A, B = np.zeros((n, len(uniq))), np.zeros((n, len(uniq)))
    rows = keys["user"].map(pos).to_numpy()
    A[rows[first], codes[first]] = 1
    B[rows[~first], codes[~first]] = 1
    share = B.sum(axis=0) / n
    w = -np.log(np.where(share > 0, share, 1.0))
    return (A * w) @ B.T, A.any(axis=1), B.any(axis=1)


def _linked(S: np.ndarray) -> np.ndarray:
    """True where a user's own second half is the single best-scoring match."""
    own, best = S.diagonal(), S.max(axis=1)
    return (own > EPS) & (own >= best - EPS) & ((S >= best[:, None] - EPS).sum(axis=1) == 1)


LINKAGE_COLUMNS = ["dataset", "variant", "level", "match_on", "activity_bucket", "n_users", "share_both_halves",
                   "share_linked", "ci_low", "ci_high", "share_linked_given_both", "chance", "suppressed"]


def analyze(attrs: pd.DataFrame, turn_log: pd.DataFrame, sample: pd.DataFrame, spans: pd.DataFrame,
            coarsener: Coarsener, min_confidence: float, ucfg: dict, acfg: dict, seed: int,
            min_cell: int = 10) -> pd.DataFrame:
    """Share of users whose two halves link to each other, by what they are matched on.
    `share_both_halves`: users with something to match on in each half. `share_linked` is out of all
    users with at least two conversations; `chance` is what picking a second half at random gives."""
    splits = int(ucfg.get("linkage_splits", 20))
    reps, level_ci = int(acfg["bootstrap_reps"]), float(acfg["ci_level"])
    rows = []
    for ds, tl in turn_log.groupby("dataset"):
        smp = sample[sample["dataset"] == ds]
        units = unit_keys(smp)
        convs = pd.DataFrame({"conversation_id": tl["conversation_id"].unique()})
        convs["user"] = convs["conversation_id"].map(units)
        convs = convs[convs.groupby("user")["user"].transform("size") >= 2].reset_index(drop=True)
        users = pd.Index(sorted(convs["user"].unique()))
        n = len(users)
        if n < 2:
            continue
        size = convs.groupby("user")["user"].transform("size")
        full = (smp.assign(user=smp["conversation_id"].map(units)).groupby("user")["user_n_conversations"].first()
                if "user_n_conversations" in smp else convs.groupby("user").size())
        bucket = activity_bucket(full.reindex(users)).to_numpy()
        a_ds = attrs[attrs["dataset"] == ds]
        keys = direct_keys(spans[spans["dataset"] == ds], units[units.index.isin(convs["conversation_id"])])
        cells = [(v, m) for v in VARIANTS for m in ("profile", "profile_and_direct")] + [(NO_VARIANT, "direct_identifiers")]
        linked = {c: np.zeros(n) for c in cells}
        both = {c: np.zeros(n) for c in cells}
        rng = np.random.default_rng(seed)
        for _ in range(splits):
            # First half: a random ceil(n/2) of each user's conversations.
            rank = pd.Series(rng.random(len(convs))).groupby(convs["user"]).rank(method="first")
            in_a = pd.Series((rank <= np.ceil(size / 2)).to_numpy(), index=convs["conversation_id"])
            Sd, da, db = _direct_scores(keys, in_a, users)
            linked[(NO_VARIANT, "direct_identifiers")] += _linked(Sd)
            both[(NO_VARIANT, "direct_identifiers")] += da & db
            half = []
            for first in (True, False):
                ids = in_a.index[in_a.to_numpy() == first]
                half.append(build_profiles(a_ds[a_ds["conversation_id"].isin(ids)],
                                           tl[tl["conversation_id"].isin(ids)], smp, coarsener, min_confidence))
            for variant in VARIANTS:
                ka, kb = (known_values(h[h["variant"] == variant].set_index("user")).reindex(users) for h in half)
                Sp, pa, pb = _profile_scores(ka, kb)
                linked[(variant, "profile")] += _linked(Sp)
                both[(variant, "profile")] += pa & pb
                linked[(variant, "profile_and_direct")] += _linked(Sp + Sd)
                both[(variant, "profile_and_direct")] += (pa | da) & (pb | db)

        W = uniqueness._boot_weights(n, reps, rng)
        for (variant, match_on), hit in linked.items():
            hit, has = hit / splits, both[(variant, match_on)] / splits
            for b in ("all", *ACTIVITY_LABELS):
                mask = np.ones(n, dtype=bool) if b == "all" else bucket == b
                m = int(mask.sum())
                if not m:
                    continue
                row = {"dataset": ds, "variant": variant, "level": LEVEL, "match_on": match_on,
                       "activity_bucket": b, "n_users": m, "share_both_halves": np.nan, "share_linked": np.nan,
                       "ci_low": np.nan, "ci_high": np.nan, "share_linked_given_both": np.nan,
                       "chance": 1.0 / n, "suppressed": False}
                if m < min_cell:
                    rows.append(uniqueness._blank(row, ("dataset", "variant", "level", "match_on", "activity_bucket")))
                    continue
                row["share_both_halves"], row["share_linked"] = float(has[mask].mean()), float(hit[mask].mean())
                row["ci_low"], row["ci_high"] = uniqueness._ci(W[:, mask] @ hit[mask], W[:, mask].sum(axis=1), level_ci)
                if has[mask].sum() > 0:
                    row["share_linked_given_both"] = float(hit[mask].sum() / has[mask].sum())
                rows.append(row)
    out = pd.DataFrame(rows, columns=LINKAGE_COLUMNS)
    out["n_users"] = out["n_users"].astype("Int64")
    return out


OVERLAP_COLUMNS = ["dataset", "variant", "level", "category", "n_users", "n_direct", "share_direct",
                   "n_direct_and_unique", "share_direct_and_unique", "suppressed"]


def direct_overlap(profiles: pd.DataFrame, sample: pd.DataFrame, spans: pd.DataFrame, turn_log: pd.DataFrame,
                   ucfg: dict, min_cell: int = 10) -> pd.DataFrame:
    """Users with a direct identifier in their own turns, and how many of them also have a unique
    profile. Counts between 1 and min_cell - 1 are blanked."""
    max_size = int(ucfg.get("max_combination_size", 4))
    rows = []
    for (ds, variant, level), g in profiles.groupby(uniqueness.GROUP):
        units = unit_keys(sample[sample["dataset"] == ds])
        convs = turn_log.loc[turn_log["dataset"] == ds, "conversation_id"].unique()
        keys = direct_keys(spans[spans["dataset"] == ds], units[units.index.isin(convs)])
        all_combos = uniqueness.combos(max_size)
        unique = uniqueness.unique_upto(uniqueness.k_matrix(known_values(g), all_combos), all_combos, max_size)[:, -1] > 0
        for cat in ("any", *sorted({c for _, c in DIRECT_KEYS})):
            has = g["user"].isin(keys["user"] if cat == "any" else keys.loc[keys["category"] == cat, "user"]).to_numpy()
            counts = {"n_direct": int(has.sum()), "n_direct_and_unique": int((has & unique).sum())}
            small = len(g) < min_cell or any(0 < c < min_cell for c in counts.values())
            row = {"dataset": ds, "variant": variant, "level": level, "category": cat,
                   "n_users": pd.NA if len(g) < min_cell else len(g), "suppressed": small}
            for name, c in counts.items():
                row[name] = pd.NA if small else c
                row[name.replace("n_", "share_", 1)] = pd.NA if small else c / len(g)
            rows.append(row)
    out = pd.DataFrame(rows, columns=OVERLAP_COLUMNS)
    for c in out.columns:
        if c.startswith("n_"):
            out[c] = out[c].astype("Int64")
        elif c.startswith("share_"):
            out[c] = out[c].astype("Float64")
    return out
