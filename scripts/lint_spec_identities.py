#!/usr/bin/env python3
"""CLI for read-only declarative mapping-identity lint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from lib.spec_identity_lint import SpecIdentityLintError, lint_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lint explicit declarative specs without opening source data")
    parser.add_argument("specs", nargs="*", type=Path,
                        help="explicit specification YAML path(s)")
    parser.add_argument("--spec-dir", action="append", type=Path, default=[],
                        help="explicit non-recursive directory of *.yaml specs")
    parser.add_argument("--plan", type=Path,
                        help="optional production_mapping_id_plan.yaml")
    parser.add_argument("--format", choices=("json", "yaml"), default="json")
    parser.add_argument("--output", type=Path,
                        help="optional report path; specifications are never modified")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    paths = list(args.specs)
    for directory in args.spec_dir:
        if not directory.is_dir():
            print(f"ERROR: spec directory does not exist: {directory}", file=sys.stderr)
            return 2
        paths.extend(sorted(directory.glob("*.yaml")))
    try:
        report = lint_paths(paths, plan_path=args.plan)
    except (OSError, SpecIdentityLintError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n"
               if args.format == "json"
               else yaml.safe_dump(report, sort_keys=False))
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if report["identity_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
