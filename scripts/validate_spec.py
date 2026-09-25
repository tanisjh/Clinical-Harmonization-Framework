#!/usr/bin/env python
"""validate_spec.py — deterministic pre-flight for a harmonization specification.

Read-only. Answers one question before any harmonization runs: **can this
document execute, and does everything it names actually exist?**

    python scripts/validate_spec.py specs/cimac_10021_v1.yaml
    python scripts/validate_spec.py specs/*.yaml --quiet
    python scripts/validate_spec.py proposed.yaml --json
    python scripts/validate_spec.py proposed.yaml --no-source-check   # grammar only

What it checks
    * the document parses as an EXECUTABLE specification (the parser's own
      allow-lists, enumerations, `$ref` closure, duplicate targets, declared
      sources, vocabulary references, and — because a schema is loaded — that
      every mapping target is a field of the target schema);
    * every construct is one the RUNTIME can drive, not merely one the grammar
      accepts. The grammar deliberately parses more than the executor
      implements, and a spec that validates but cannot run is the worst of both;
    * every `custom_rule` and post-rule resolves in the closed registry by id,
      version AND kind;
    * every declared source file exists, stays inside the approved input roots,
      and opens with its declared reader;
    * **every source column the document names is really in that file**, and
      every source carries the participant key it will be indexed by.

What it does NOT check
    Whether any of it is clinically right. A mapping can pass every check here
    and still be the wrong mapping; that is what human review is for. This
    command says the document is executable and internally honest about the
    data it points at — nothing more, and saying more would be a lie a reviewer
    might believe.

    It also does not recompute anything. See `trace_value.py` for lineage.

Exit codes
    0   PASS — no findings
    1   FAIL — the specification cannot execute as written
    2   usage error
    3   the target schema could not be loaded
    4   the specification could not be parsed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import spec_preflight                       # noqa: E402
from lib.schema_model import load as load_schema     # noqa: E402
from lib.spec_model import SpecError, load as load_spec  # noqa: E402

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2
EXIT_SCHEMA_LOAD = 3
EXIT_SPEC_LOAD = 4

DEFAULT_SCHEMA = "schemas/cimac_clinical_v1.yaml"


def _render(report, summary: bool) -> None:
    head = "PASS" if report.ok else "FAIL"
    print(f"{head}  {report.spec_ref}")
    if not report.ok:
        print(f"\n{len(report.findings)} finding(s):")
        for f in report.findings:
            print(f"  [{f.kind}] {f.message}")
    if summary and report.summary:
        s = report.summary
        print("\nStructure")
        print(f"  trial            {s['trial']}   status: {s['status']}")
        print(f"  target schema    {s['target_schema']}")
        print(f"  anchor           {s['anchor']}")
        print(f"  participant key  {s['participant_key']}")
        if s["record_key_column"] != s["participant_key"]:
            print(f"  provenance key   {s['record_key_column']}")
        print(f"  sources          {len(s['sources'])}")
        for src in s["sources"]:
            extra = f", sheet {src['sheet']}" if src["sheet"] else ""
            cols = (f", {src['columns_in_file']} columns"
                    if src["columns_in_file"] else "")
            print(f"    - {src['id']}: {src['path']} "
                  f"[{src['reader']}{extra}{cols}]")
            if src["participant_column"] != s["participant_key"]:
                print(f"        participant column: {src['participant_column']}")
        print(f"  target fields    {len(s['target_fields'])}")
        print(f"  operations       {s['mappings_by_op']}")
        if s["vocabularies"]:
            print(f"  vocabularies     {s['vocabularies']}")
        if s["registered_rules"]:
            print(f"  registered rules {s['registered_rules']}")
        if s["exclusions"]:
            print(f"  exclusions       {s['exclusions']}")
        if s["post_rules"]:
            print(f"  post rules       {s['post_rules']}")
        if s["additional_records"]:
            print(f"  declared records {s['additional_records']}")
        if s["open_questions"]:
            print(f"  OPEN QUESTIONS   {s['open_questions']}")
        print("\n  Pre-flight is deterministic only. It does not judge whether "
              "a mapping is\n  clinically correct — that remains human review.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Deterministic pre-flight for a harmonization specification.")
    p.add_argument("spec", nargs="+", help="path to a specification YAML")
    p.add_argument("--schema", default=DEFAULT_SCHEMA,
                   help=f"target schema YAML (default: {DEFAULT_SCHEMA})")
    p.add_argument("--no-source-check", action="store_true",
                   help="skip file and column checks (grammar and registry only)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--quiet", action="store_true",
                   help="one line per spec; omit the structural summary")
    args = p.parse_args(argv)

    try:
        schema = load_schema(PROJECT_ROOT / args.schema)
    except Exception as e:
        print(f"ERROR  could not load target schema {args.schema}: {e}",
              file=sys.stderr)
        return EXIT_SCHEMA_LOAD

    worst, payload = EXIT_OK, []
    for i, path in enumerate(args.spec):
        try:
            spec = load_spec(Path(path), target_schema=schema,
                             require_executable=True)
        except SpecError as e:
            worst = max(worst, EXIT_SPEC_LOAD)
            payload.append({"spec": path, "ok": False, "stage": "parse",
                            "findings": [{"kind": "spec.invalid",
                                          "message": str(e)}]})
            if not args.json:
                print(f"FAIL  {path}\n  [spec.invalid] {e}")
            continue

        report = spec_preflight.preflight(
            spec, PROJECT_ROOT, check_sources=not args.no_source_check)
        if not report.ok:
            worst = max(worst, EXIT_FINDINGS)
        payload.append({"spec": path, "ok": report.ok, "stage": "preflight",
                        "spec_ref": report.spec_ref,
                        "findings": [{"kind": f.kind, "message": f.message}
                                     for f in report.findings],
                        "summary": report.summary})
        if not args.json:
            if i:
                print()
            _render(report, summary=not args.quiet)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
