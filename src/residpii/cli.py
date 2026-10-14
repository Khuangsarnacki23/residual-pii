"""Command-line entry point.

  residpii run           --config configs/smoke.yaml     # sample -> detect -> analyze -> report
  residpii dedup-corpus  --config configs/default.yaml --confirm   # full lmsys vs Arena overlap
  residpii label build      --config configs/default.yaml           # draw the human-label set
  residpii label serve      --config configs/default.yaml --annotator kai
  residpii label adjudicate --config configs/default.yaml           # resolve the two annotators
  residpii label agreement  --config configs/default.yaml
  residpii label llm        --config configs/default.yaml --allow-hosted   # Claude as annotator 'llm'
  residpii label score      --config configs/default.yaml           # precision/recall vs gold
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="residpii")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Sample, detect, analyze and report")
    r.add_argument("--config", required=True)
    r.add_argument("--datasets", nargs="*", help="Subset of dataset keys to run")
    r.add_argument("--no-dedup", action="store_true")
    r.add_argument("--force", action="store_true", help="Ignore cached stage outputs")

    d = sub.add_parser("dedup-corpus", help="Full-corpus lmsys vs Arena overlap (streams ~1.5 GB)")
    d.add_argument("--config", required=True)
    d.add_argument("--confirm", action="store_true", help="Required: acknowledges a full-dataset pass")

    lab = sub.add_parser("label", help="Human-labelled benchmark")
    lsub = lab.add_subparsers(dest="label_cmd", required=True)
    lb = lsub.add_parser("build", help="Draw the label set from an existing run")
    lb.add_argument("--force", action="store_true", help="Redraw even if a label set exists")
    ls = lsub.add_parser("serve", help="Open the blind labelling page for one annotator")
    ls.add_argument("--annotator", required=True)
    ls.add_argument("--port", type=int)
    la = lsub.add_parser("adjudicate", help="Open the page with both annotators' labels")
    la.add_argument("--port", type=int)
    ll = lsub.add_parser("llm", help="Label the set with Claude as annotator 'llm' (sends text to Anthropic)")
    ll.add_argument("--allow-hosted", action="store_true",
                    help="Required: acknowledges that user turns are sent to Anthropic's API")
    ll.add_argument("--model", default="claude-sonnet-5-5",
                    help="claude-sonnet-5-5 (default), claude-haiku-5-5 (cheapest) or claude-opus-5-5")
    ll.add_argument("--restart", action="store_true",
                    help="Move existing llm labels made with another model/effort/prompt to a backup, start over")
    ll.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ll.add_argument("--limit", type=int, help="Label at most this many items (for a pilot)")
    ll.add_argument("--workers", type=int, default=4)
    lg = lsub.add_parser("agreement", help="Inter-annotator agreement table")
    lg.add_argument("--pair", nargs=2, metavar=("A", "B"), help="Annotators to compare (default: the configured two)")
    lsub.add_parser("progress", help="How many items each annotator has labelled")
    lc = lsub.add_parser("score", help="Detector precision/recall against gold labels")
    lc.add_argument("--gold", help="Score against this annotator instead of the adjudicated set")
    for p in (lb, ls, la, lc, ll, lg, lsub.choices["progress"]):
        p.add_argument("--config", required=True)
    for p in (lb, ll, lc, lg, lsub.choices["progress"]):
        p.add_argument("--dataset", nargs="*", help="Dataset keys (default: labeling.datasets in the config)")
    for p in (ls, la):
        p.add_argument("--dataset", required=True, help="Dataset key, e.g. wildchat")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "run":
        if args.datasets:
            for k in cfg["datasets"]:
                cfg["datasets"][k]["enabled"] = k in args.datasets
        Pipeline(cfg, force=args.force).run(do_dedup=not args.no_dedup)
        return 0

    if args.cmd == "dedup-corpus":
        if not args.confirm:
            print("This streams every English conversation in lmsys-chat-1m (~1.5 GB of network reads, "
                  "no disk) and Arena. Re-run with --confirm to proceed.", file=sys.stderr)
            return 2
        print(Pipeline(cfg).dedup_corpus().to_string(index=False))
        return 0

    if args.cmd == "label":
        from . import annotate, labeling

        c = args.label_cmd
        many = (args.dataset or cfg["labeling"]["datasets"]) if c not in ("serve", "adjudicate") else None
        if c == "build":
            for ds in many:
                try:
                    print(json.dumps(labeling.cmd_build(cfg, ds, force=args.force), indent=2))
                except FileExistsError as e:
                    print(f"[label] {ds}: {e}", file=sys.stderr)
        elif c == "serve":
            annotate.serve(cfg, args.dataset, args.annotator, "annotate", args.port)
        elif c == "adjudicate":
            annotate.serve(cfg, args.dataset, cfg["labeling"]["adjudicator"], "adjudicate", args.port)
        elif c == "llm":
            if not args.allow_hosted:
                print("This sends the text of each sampled user turn to Anthropic's API (a third party).\n"
                      "Re-run with --allow-hosted to proceed.", file=sys.stderr)
                return 2
            from . import llm_label

            for ds in many:
                try:
                    stats = llm_label.run(cfg, ds, args.model, args.effort, args.limit, args.workers,
                                          restart=args.restart)
                except RuntimeError as e:
                    print(f"[llm] {ds}: {e}", file=sys.stderr)
                    return 2
                print(json.dumps(stats, indent=2))
                if stats.get("interrupted"):
                    break
        elif c == "agreement":
            for ds in many:
                print(labeling.cmd_agreement(cfg, ds, args.pair).to_string(index=False))
        elif c == "progress":
            for ds in many:
                paths, items, labels = labeling._load_all(cfg, ds, ("llm",))
                print(labeling.progress(items, labels).assign(dataset=ds).to_string(index=False))
        elif c == "score":
            t = labeling.cmd_score(cfg, args.dataset, args.gold)["labels_precision_recall"]
            thr = cfg["presidio"]["score_threshold"]
            print(t[t["threshold"] == thr].drop(columns=["threshold"]).to_string(index=False))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
