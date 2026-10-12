"""Command-line entry point.

  residpii run           --config configs/smoke.yaml     # sample -> detect -> analyze -> report
  residpii dedup-corpus  --config configs/default.yaml --confirm   # full lmsys vs Arena overlap
"""
from __future__ import annotations

import argparse
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
