"""Presentation figures for the final report: reads only aggregate tables in results/ and writes
PNGs to results/presentation/figures/. No raw text is read.

    .venv/bin/python -m residpii.presentation          # report figures, with titles and footnotes
    .venv/bin/python -m residpii.presentation --deck   # slide figures in the midterm deck's style, no
                                                       # titles or footnotes (the slide carries them)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from residpii.config import REPO_ROOT  # noqa: E402

S1 = REPO_ROOT / "results" / "default" / "tables"
S2 = REPO_ROOT / "results" / "user_level" / "tables"
OUT = REPO_ROOT / "results" / "presentation" / "figures"

# Reference palette, light mode, fixed slot order (dataviz skill: references/palette.md).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

DECK = False


def apply_deck_style():
    """The midterm deck's look: white ground, navy ink, its blue / gold / teal accents, Arial."""
    global DECK, OUT, BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED, SURFACE, INK, INK2, MUTED, GRID, METHODS
    DECK = True
    OUT = REPO_ROOT / "results" / "presentation" / "deck_figures"
    BLUE, ORANGE, AQUA, VIOLET = "#2F5BB7", "#A16207", "#2A9D8F", "#14213D"
    YELLOW, RED = "#A16207", "#7A8594"
    SURFACE, INK, INK2, MUTED, GRID = "#FFFFFF", "#14213D", "#55606E", "#7A8594", "#E5E7EB"
    METHODS = [("Per-message\n(Staab et al.)", "per_message", BLUE), ("User-level", "user_level", ORANGE),
               ("User-level,\nwhole history", "user_level_history", AQUA),
               ("User-level,\nstrict stop", "user_level_strict", VIOLET),
               ("User-level,\nimproved rewrites", "user_level_passages", "#7A8594")]
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 13,
                         "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "xtick.color": INK2, "ytick.color": INK,
                         "text.color": INK, "grid.color": GRID})


def _title(target, text, **kw):
    """Chart title (report figures only: on a slide the headline is the title)."""
    if DECK:
        return
    if hasattr(target, "suptitle"):
        target.suptitle(text, **kw)
    else:
        target.set_title(text, **kw)


plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 14, "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
})


def _pct(ax, axis="x"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def _note(fig, text):
    if not DECK:
        fig._residpii_note = text


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    note = getattr(fig, "_residpii_note", None)
    if note:  # place the footnote under everything already drawn, so it never overlaps axis labels
        r = fig.canvas.get_renderer()
        bb = fig.get_tightbbox(r)
        x0 = bb.x0 / fig.get_figwidth()
        fig.text(x0, (bb.y0 - 0.12) / fig.get_figheight(), note, fontsize=9.5, color=MUTED, ha="left", va="top")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _err(lo, hi, v):
    return [[max(v - lo, 0)], [max(hi - v, 0)]]


# ---------------------------------------------------------------- RQ1: what remains

def fig_residual_prevalence():
    g = pd.read_csv(S1 / "labels_gold_prevalence.csv").query("gold == 'adjudicated'").set_index("category")
    rows = [("Any private direct identifier", "tier_direct", BLUE),
            ("Any private quasi-identifier", "tier_quasi", ORANGE),
            ("Occupation*", "occupation", ORANGE), ("Organization", "organization", ORANGE),
            ("Education*", "education", ORANGE), ("Location", "location", ORANGE),
            ("Name", "name", BLUE)]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (label, cat, col) in enumerate(rows):
        r = g.loc[cat]
        ax.barh(i, r.prevalence, color=col, height=0.62)
        ax.errorbar(r.prevalence, i, xerr=_err(r.ci_low, r.ci_high, r.prevalence), fmt="none",
                    ecolor=INK2, elinewidth=1.2, capsize=3)
        ax.text(r.ci_high + 0.003, i, f"{r.prevalence:.1%}", va="center", fontsize=11, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    _pct(ax)
    ax.set_xlim(0, 0.135)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of WildChat user turns (weighted, 95% CI)")
    _title(ax, "After WildChat's own scrubbing, 1 in 13 turns still holds a private quasi-identifier")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["Direct identifier", "Quasi-identifier"], loc="lower right")
    _note(fig, "Adjudicated gold set, 260 scored turns, 94 private spans. "
               "* No Presidio entity type exists for occupation or education.")
    _save(fig, "rq1_residual_prevalence")


# ---------------------------------------------------------------- RQ2: detector accuracy

def _pr(detector, category="any", threshold=0.5):
    pr = pd.read_csv(S1 / "labels_precision_recall.csv").query("gold == 'adjudicated'")
    pr = pr[(pr.detector == detector) & (pr.category == category) & (pr.threshold == threshold)]
    return {m: pr[pr.metric == m].iloc[0] for m in ("precision", "recall") if (pr.metric == m).any()}


def fig_detector_pr():
    # The LLM's "all spans" variant (every mark, whoever it belongs to) is left out: it only restates
    # that a superset of flags trades precision for recall; the ownership point is made in the text.
    dets = [("Presidio", "presidio", BLUE), ("LLM detector", "llm_detector", ORANGE),
            ("Presidio or LLM, any flag", "any", VIOLET)]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    offsets = {"presidio": (-12, 10), "llm_detector": (10, 8), "llm_detector_all": (-10, 28), "any": (10, -30)}
    for label, det, col in dets:
        d = _pr(det)
        p, r = d["precision"], d["recall"]
        ax.errorbar(r.value, p.value, xerr=_err(r.ci_low, r.ci_high, r.value),
                    yerr=_err(p.ci_low, p.ci_high, p.value), fmt="none", ecolor=col, alpha=0.45, elinewidth=1.4)
        ax.scatter(r.value, p.value, s=130, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        prec = f"{p.value:.1%}" if p.value < 0.1 else f"{p.value:.0%}"
        ax.annotate(f"{label}\n{prec} precise, finds {r.value:.0%}", (r.value, p.value),
                    xytext=offsets[det], textcoords="offset points", fontsize=11, color=INK,
                    ha="right" if det in ("llm_detector_all", "presidio") else "left")
    _pct(ax, "x"), _pct(ax, "y")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("Recall: share of the remaining private PII found")
    ax.set_ylabel("Precision: share of flags that are private PII")
    _title(ax, "An LLM is far more precise than Presidio,\nbut no tool finds most of what is left")
    _note(fig, "Adjudicated gold set; bars are 95% bootstrap CIs. Presidio at threshold 0.5 "
               "(0.3-0.85 gives 3.5-4.3% precision). LLM = Claude Haiku 5.5.")
    _save(fig, "rq2_detector_precision_recall")


def fig_recall_by_category():
    cats = [("Name", "name"), ("Location", "location"), ("IP address", "ip_address"),
            ("Organization", "organization"), ("Health", "health"),
            ("Occupation", "occupation"), ("Education", "education")]
    series = [("Presidio", "presidio", BLUE), ("LLM, private spans", "llm_detector", ORANGE),
              ("LLM, all spans", "llm_detector_all", AQUA)]
    pr = pd.read_csv(S1 / "labels_precision_recall.csv").query(
        "gold == 'adjudicated' and metric == 'recall' and threshold == 0.5")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.26
    for j, (label, det, col) in enumerate(series):
        for i, (_, cat) in enumerate(cats):
            row = pr[(pr.detector == det) & (pr.category == cat)]
            v = float(row.value.iloc[0]) if len(row) else 0.0
            x = i + (j - 1) * (w + 0.02)
            ax.bar(x, max(v, 0.004), width=w, color=col, label=label if i == 0 else None)
            if v == 0:
                ax.text(x, 0.02, "0", ha="center", fontsize=9, color=INK2)
    n = pr[pr.detector == "presidio"].set_index("category").n
    ax.set_xticks(range(len(cats)), [f"{lab}\n({int(n.get(c, 0))} spans)" for lab, c in cats])
    _pct(ax, "y")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Recall on the gold set")
    _title(ax, "Occupation and education, the most common leftovers, are missed by every tool")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    _title(ax, "Occupation and education, the most common leftovers, are missed by every tool", pad=36)
    ax.axvspan(4.55, 6.45, color=RED, alpha=0.06, zorder=0)
    _note(fig, "Recall of private spans (self or third party). Small counts: per-category values are "
               "indicative. 'All spans' counts the LLM's marks whoever it thought they belong to.")
    _save(fig, "rq2_recall_by_category")


# ---------------------------------------------------------------- RQ3: across conversations

BUCKETS = ["3", "4-5", "6-10", "11-25", "26-50", ">50"]


def fig_attributes():
    p = pd.read_csv(S2 / "profile_summary.csv").query("variant == 'stated_plus_inferred'").set_index("attribute")
    rows = [("Any attribute", "any"), ("Occupation group", "occupation_group"), ("Country", "country"),
            ("Education level", "education_level"), ("Age band", "age_band"), ("Health", "health"),
            ("Employer sector", "employer"), ("Nationality / ethnicity", "nationality_ethnicity"),
            ("Relationship", "relationship")]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, (lab, a) in enumerate(rows):
        v = p.loc[a].share_with_value
        ax.barh(i, v, color=ORANGE if a != "any" else INK2, height=0.62)
        ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=11)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    _pct(ax)
    ax.set_xlim(0, 0.16)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of 1,982 WildChat users (stated or inferred, across all their conversations)")
    _title(ax, "1 in 7 users reveals at least one attribute about themselves")
    _note(fig, "Religion, political views and income: fewer than 10 users each (not shown).")
    _save(fig, "rq3_attributes_revealed")


def fig_activity():
    u = pd.read_csv(S2 / "uniqueness_by_activity.csv").query("variant == 'stated_plus_inferred' and n_qi == 4")
    lk = pd.read_csv(S2 / "linkage.csv").query("variant == 'stated_plus_inferred' and match_on == 'profile_and_direct'")
    u, lk = u.set_index("activity_bucket").loc[BUCKETS], lk.set_index("activity_bucket").loc[BUCKETS]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    w = 0.38
    for j, (lab, df, val, col) in enumerate([("Unique on ≤4 attributes", u, "share_unique", BLUE),
                                              ("Conversations re-linked from content", lk, "share_linked", ORANGE)]):
        xs = [i + (j - 0.5) * (w + 0.03) for i in range(len(BUCKETS))]
        ax.bar(xs, df[val], width=w, color=col, label=lab)
        ax.errorbar(xs, df[val], yerr=[(df[val] - df.ci_low).clip(lower=0), (df.ci_high - df[val]).clip(lower=0)],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
    ax.set_xticks(range(len(BUCKETS)), [f"{b}\n(n={n})" for b, n in zip(BUCKETS, u.n_users)])
    _pct(ax, "y")
    ax.set_ylim(0, 0.24)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Conversations per user")
    ax.set_ylabel("Share of users (95% CI)")
    _title(ax, "The more a user chats, the more they can be singled out")
    ax.legend(loc="upper left")
    _note(fig, "1,982 WildChat users. Overall: 5.9% unique, 2.1% re-linked (chance 0.05%). "
               "Uniqueness is within the 1,982-user sample; no identities were sought.")
    _save(fig, "rq3_uniqueness_by_activity")


def fig_uniqueness_curve():
    s = pd.read_csv(S2 / "uniqueness_summary.csv").query("ci_kind == 'bootstrap_users'")
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab, var, col in [("Stated plus inferred", "stated_plus_inferred", BLUE), ("Stated only", "stated_only", ORANGE)]:
        d = s[s.variant == var].sort_values("n_qi")
        ax.fill_between(d.n_qi, d.ci_low, d.ci_high, color=col, alpha=0.12, linewidth=0)
        ax.plot(d.n_qi, d.share_unique, color=col, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=lab)
        ax.text(4.08, d.share_unique.iloc[-1], f"{d.share_unique.iloc[-1]:.1%}", va="center", fontsize=11)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.8, 4.5)
    _pct(ax, "y")
    ax.set_ylim(0, 0.08)
    ax.set_xlabel("Attributes combined (at most)")
    ax.set_ylabel("Users unique in the sample")
    _title(ax, "One attribute rarely singles anyone out; two do")
    ax.legend(loc="lower right")
    _note(fig, "1,982 users, coarsened attributes, bootstrap 95% CI. Population model: 4.8% of 25,325 eligible users.")
    _save(fig, "rq3_uniqueness_by_attributes")


# ---------------------------------------------------------------- RQ4: sanitizing

# Users at risk before rewriting under the held-out attacker who are still at risk after
# (of 23): from the pilot's per-user records, reported in docs/adversarial_pilot.md.
STILL_AT_RISK = {"per_message": 13, "user_level": 18, "user_level_history": 20, "user_level_strict": 17,
                 "user_level_passages": 17}
METHODS = [("Per-message\n(Staab et al.)", "per_message", BLUE),
           ("User-level", "user_level", ORANGE),
           ("User-level,\nwhole history", "user_level_history", AQUA),
           ("User-level,\nstrict stop", "user_level_strict", VIOLET),
           ("User-level,\nimproved rewrites", "user_level_passages", GREEN)]


def _adv():
    a = pd.read_csv(S2 / "adversarial_summary.csv")
    return a[(a.group == "all") & a.k.notna()].set_index("method")


def fig_sanitizer_tradeoff():
    a = _adv()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.scatter(0, 23, s=130, color=MUTED, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate("Before rewriting\n23 at risk", (0, 23), xytext=(10, -4), textcoords="offset points", fontsize=11)
    offs = {"per_message": (-10, 10), "user_level": (-12, 4), "user_level_history": (10, 2), "user_level_strict": (10, -26),
            "user_level_passages": (12, 6)}
    for lab, m, col in METHODS:
        x, y = a.loc[m].share_turns_changed, STILL_AT_RISK[m]
        ax.scatter(x, y, s=150, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ha = "right" if m in ("per_message", "user_level") else "left"
        ax.annotate(f"{lab.replace(chr(10), ' ')}\n{y} at risk, {x:.1%} of messages changed", (x, y),
                    xytext=offs[m], textcoords="offset points", fontsize=11, ha=ha)
    _pct(ax, "x")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    ax.set_xlim(-0.01, 0.17)
    ax.set_ylim(10, 25)
    ax.set_xlabel("Share of the 1,934 user messages rewritten")
    ax.set_ylabel("Users still at risk (of 23 at risk before)")
    _title(ax, "User-level rewriting edits far less, but protects fewer users")
    ax.annotate("better", xy=(0.002, 11.2), xytext=(0.03, 13.5), fontsize=10, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    _note(fig, "50-user pilot; at risk = held-out Claude Sonnet 5.5 attacker puts the user in a class under 5. "
               "Differences of 4-5 users are not significant (per-message vs strict: p = 0.73).")
    _save(fig, "rq4_sanitizer_tradeoff")


def fig_sanitizer_panels():
    a = _adv()
    panels = [("Users still at risk (of 23)", lambda m: STILL_AT_RISK[m], "{:.0f}", (0, 24)),
              ("Messages rewritten", lambda m: a.loc[m].share_turns_changed, "{:.1%}", (0, 0.17)),
              ("Rewrites the judge rates usable (≥4/5)", lambda m: a.loc[m].share_utility_ge4, "{:.0%}", (0, 1.12))]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (title, f, fmt, lim) in zip(axes, panels):
        for i, (lab, m, col) in enumerate(METHODS):
            v = f(m)
            ax.bar(i, v, color=col, width=0.66)
            ax.text(i, v + lim[1] * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=11)
        ax.set_xticks(range(len(METHODS)), [m[0] for m in METHODS], fontsize=10)
        ax.set_ylim(*lim)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=13)
        if "%" in fmt:
            _pct(ax, "y")
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04 if lim[1] < 0.5 else 0.2))
    _title(fig, "Four sanitizer loops on the same 50 users", x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _note(fig, "User-level loops discard rewrites rated below 4 (47-56 each), so their 100% is partly by construction.")
    _save(fig, "rq4_sanitizer_panels")


def fig_stop_reasons():
    a = pd.read_csv(S2 / "adversarial_summary.csv")
    s = a[(a.group == "all") & (a.method == "user_level_strict") & a.stop_reason.notna()].set_index("stop_reason").n_users
    parts = [("Reached the target", "safe", AQUA), ("Ran out of rounds", "max_rounds", YELLOW),
             ("No fixable attribute left", "no_target", RED)]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    left = 0
    for lab, key, col in parts:
        v = int(s.get(key, 0))
        ax.barh(0, v, left=left, color=col, height=0.5, edgecolor=SURFACE, linewidth=2)
        ax.text(left + v / 2, 0, f"{lab}\n{v} users", ha="center", va="center", fontsize=11, color=INK)
        left += v
    ax.set_xlim(0, 50), ax.set_yticks([]), ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Users (strict-stop loop, 50 users)")
    _title(ax, "Half the users never reach the target: the rewrite step is the bottleneck")
    _note(fig, "Over the run: 56 rewrites undone by the utility check; 29 times the cue was in a message over 8,000 characters.")
    _save(fig, "rq4_stop_reasons")


def fig_layered_sanitizer():
    """Kai's offline sanitizer stages scored against the adjudicated gold set."""
    lk = pd.read_csv(S1 / "sanitizer_leakage.csv").query("metric == 'span_leak_rate'")
    ov = pd.read_csv(S1 / "sanitizer_overredaction.csv").query("metric == 'clean_turns_changed'")
    stages = [("Presidio re-run", "presidio"), ("Layer 1: patterns", "patterns"),
              ("Layers 1-2: + LLM identifiers", "llm_identifiers"), ("Layers 1-3: + LLM generalise", "generalize"),
              ("Layers 1-4: + adversarial loop", "adversarial"),
              ("No-LLM: 1 + rule identifiers", "rules_identifiers"), ("No-LLM: + rule generalise", "rules_generalize")]
    ut = pd.read_csv(S1 / "sanitizer_utility.csv").query("metric == 'share_score_4_plus'").set_index("stage")
    # Rule stages appear once they have been re-scored (their utility rows exist); the deck leaves them out.
    stages = [st for st in stages if st[1] in ut.index and not (DECK and st[1].startswith("rules"))]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1.6, 0.8, 0.8]})
    h = 0.36
    for j, (lab, tier, col) in enumerate([("Direct identifiers", "direct", BLUE), ("Quasi-identifiers", "quasi", ORANGE)]):
        for i, (_, s) in enumerate(stages):
            r = lk[(lk.stage == s) & (lk.tier == tier)].iloc[0]
            v, lo, hi = 1 - r.value, 1 - r.ci_high, 1 - r.ci_low
            y = i + (j - 0.5) * (h + 0.04)
            a1.barh(y, max(v, 0.003), height=h, color=col, label=lab if i == 0 else None)
            a1.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
            a1.text(hi + 0.015, y, f"{v:.0%}", va="center", fontsize=10.5)
    a1.set_yticks(range(len(stages)), [s[0] for s in stages])
    a1.invert_yaxis()
    _pct(a1)
    a1.set_xlim(0, 1.05)
    a1.grid(axis="y", visible=False)
    a1.set_title("Private spans removed (of 41 direct, 53 quasi)", fontsize=13)
    fig.legend(*a1.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=2)
    for i, (_, s) in enumerate(stages):
        r = ov[ov.stage == s].iloc[0]
        a2.barh(i, max(r.value, 0.002), height=0.6, color=INK2)
        a2.text(r.value + 0.004, i, f"{r.value:.1%}", va="center", fontsize=10.5)
    a2.set_yticks(range(len(stages)), [""] * len(stages))
    a2.invert_yaxis()
    _pct(a2)
    a2.set_xlim(0, 0.25)
    a2.grid(axis="y", visible=False)
    a2.set_title("Clean turns changed anyway", fontsize=13)
    for i, (_, s_) in enumerate(stages):
        v = 1 - ut.loc[s_].value
        a3.barh(i, max(v, 0.001), height=0.6, color=INK2)
        a3.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=10.5)
    a3.set_yticks(range(len(stages)), [""] * len(stages))
    a3.invert_yaxis()
    _pct(a3)
    a3.set_xlim(0, 0.12)
    a3.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    a3.grid(axis="y", visible=False)
    a3.set_title("Turns the judge rates worse (<4/5)", fontsize=13)
    _title(fig, "Only the adversarial layer removes most quasi-identifiers, and it costs less than Presidio",
                 x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _note(fig, "Kai's layered sanitizer on the 260-turn adjudicated gold set (weighted, 95% CI). A span counts as removed "
               "if its exact value no longer appears, so rewrites that paraphrase count as removed.")
    _save(fig, "rq4_layered_sanitizer")


def main():
    if "--deck" in sys.argv[1:]:
        apply_deck_style()
    for f in (fig_layered_sanitizer,fig_residual_prevalence, fig_detector_pr, fig_recall_by_category, fig_attributes,
              fig_uniqueness_curve, fig_activity, fig_sanitizer_tradeoff, fig_sanitizer_panels, fig_stop_reasons):
        f()
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
