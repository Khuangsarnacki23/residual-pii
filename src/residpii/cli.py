"""Command-line entry point.

  residpii run           --config configs/smoke.yaml     # sample -> detect -> analyze -> report
  residpii dedup-corpus  --config configs/default.yaml --confirm   # full lmsys vs Arena overlap
  residpii placeholders  --config configs/smoke.yaml     # discover candidate redaction-placeholder formats
  residpii user-activity --config configs/user_level.yaml   # users by conversation count (metadata only)
  residpii label-export  --config configs/default.yaml --n 300      # batch of turns to annotate (restricted dir)
  residpii label-score   --config configs/default.yaml --labels <file>   # detector precision/recall vs labels
  residpii review        --config configs/default.yaml   # local page for the manual checks and labeling
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .llm import BackendUnavailable, HostedBackendNotAllowed
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (HostedBackendNotAllowed, BackendUnavailable) as e:
        # Sampling and detection are cached, so a rerun resumes at the LLM stage.
        print(f"LLM backend: {e}", file=sys.stderr)
        return 2


def _main(argv: list[str] | None = None) -> int:
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

    p = sub.add_parser("placeholders", help="Discover candidate redaction-placeholder formats in the sample")
    p.add_argument("--config", required=True)
    p.add_argument("--datasets", nargs="*", help="Subset of dataset keys to run")
    p.add_argument("--force", action="store_true", help="Ignore cached stage outputs")

    u = sub.add_parser("user-activity", help="Users by conversations-per-user bucket (metadata scan, no text)")
    u.add_argument("--config", required=True)
    u.add_argument("--datasets", nargs="*", help="Subset of dataset keys to run")
    u.add_argument("--force", action="store_true", help="Ignore cached stage outputs")

    le = sub.add_parser("label-export", help="Write a stratified batch of user turns for hand labeling")
    le.add_argument("--config", required=True)
    le.add_argument("--n", type=int, default=300, help="Number of turns to export")
    le.add_argument("--format", choices=["csv", "xlsx"], default="csv")
    le.add_argument("--datasets", nargs="*", help="Subset of dataset keys to run")

    ls = sub.add_parser("label-score", help="Score detectors against an annotated batch (aggregate output)")
    ls.add_argument("--config", required=True)
    ls.add_argument("--labels", required=True, help="Annotated batch file (csv or xlsx)")
    ls.add_argument("--second-annotator", choices=["llm"],
                    help="Also run the LLM on the labeled turns for agreement statistics")

    rv = sub.add_parser("review", help="Open a local page for checking LLM extractions and labeling turns")
    rv.add_argument("--config", required=True)
    rv.add_argument("--datasets", nargs="*", help="Subset of dataset keys to run")
    rv.add_argument("--n-attrs", type=int, default=20, help="LLM extractions to check")
    rv.add_argument("--port", type=int, default=0, help="Port on 127.0.0.1 (default: any free port)")
    rv.add_argument("--no-browser", action="store_true", help="Print the address instead of opening it")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if getattr(args, "datasets", None):
        for k in cfg["datasets"]:
            cfg["datasets"][k]["enabled"] = k in args.datasets

    if args.cmd == "run":
        Pipeline(cfg, force=args.force).run(do_dedup=not args.no_dedup)
        return 0

    if args.cmd == "dedup-corpus":
        if not args.confirm:
            print("This streams every English conversation in lmsys-chat-1m (~1.5 GB of network reads, "
                  "no disk) and Arena. Re-run with --confirm to proceed.", file=sys.stderr)
            return 2
        print(Pipeline(cfg).dedup_corpus().to_string(index=False))
        return 0

    if args.cmd == "placeholders":
        # Shapes only (digits collapsed, k-thresholded), so this is safe to print.
        disc = Pipeline(cfg, force=args.force).discover_placeholders()
        print(disc.to_string(index=False) if len(disc) else "No candidate placeholder shapes found.")
        return 0

    if args.cmd == "user-activity":
        cfg["sampling"]["mode"] = "user_level"
        print(Pipeline(cfg, force=args.force).user_activity_only().to_string(index=False))
        return 0

    if args.cmd == "label-export":
        path = Pipeline(cfg).label_export(args.n, args.format)
        print(f"Annotate this file in place (it contains raw text; keep it where it is):\n{path}")
        return 0

    if args.cmd == "label-score":
        out = Pipeline(cfg).label_score(args.labels, args.second_annotator)
        print(out.to_string(index=False))
        return 0

    if args.cmd == "review":
        from .review import serve
        serve(Pipeline(cfg), args.port, args.n_attrs, open_browser=not args.no_browser)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
