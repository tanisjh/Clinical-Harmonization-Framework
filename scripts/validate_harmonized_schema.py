"""validate_harmonized_schema.py — Validate a harmonized artifact against a target schema.

Read-only. It opens the schema and the artifact, reports findings, and writes
nothing unless asked to. It never modifies a harmonized value.

Usage
-----
    python scripts/validate_harmonized_schema.py \
        --schema schemas/cimac_clinical_v1.yaml \
        --artifact harmonization_outputs/harmonized_11trials.csv

    # every artifact the schema declares, machine-readable
    python scripts/validate_harmonized_schema.py \
        --schema schemas/cimac_clinical_v1.yaml --all --json

    # add the provenance-backed value-state check
    python scripts/validate_harmonized_schema.py \
        --schema schemas/cimac_clinical_v1.yaml --all \
        --provenance harmonization_outputs/provenance_long.csv

Exit codes are distinct by failure kind so a caller can tell a broken schema from
bad data without parsing text:

    0   valid — no error-severity findings (warnings and info do not fail)
    1   content validation errors (types, vocabularies, bounds, constraints)
    2   usage error
    3   schema could not be loaded
    4   artifact could not be loaded
    5   structural validation errors (columns, ordering, duplicate row keys)

5 outranks 1: when a table's shape is wrong, the per-value findings are mostly
consequences, and the caller should be told the shape failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd  # noqa: E402

from lib import schema_validator as sv  # noqa: E402
from lib.schema_model import ERROR, SchemaError, load as load_schema  # noqa: E402
from lib.source_ref import sha256_file  # noqa: E402

EXIT_OK = 0
EXIT_CONTENT_ERRORS = 1
EXIT_USAGE = 2
EXIT_SCHEMA_LOAD = 3
EXIT_ARTIFACT_LOAD = 4
EXIT_STRUCTURAL_ERRORS = 5

STRUCTURAL_PREFIX = "structure."


def _has_structural_errors(result: sv.ValidationResult) -> bool:
    return any(f.severity == ERROR and f.rule.startswith(STRUCTURAL_PREFIX)
               for f in result.findings)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate harmonized artifacts against a typed target schema.")
    ap.add_argument("--schema", required=True, help="path to a target-schema YAML file")
    ap.add_argument("--artifact", action="append", default=[],
                    help="path to a harmonized CSV; repeatable")
    ap.add_argument("--all", action="store_true",
                    help="validate every artifact the schema declares, resolved "
                         "relative to --outputs-dir")
    ap.add_argument("--outputs-dir", default=str(PROJECT_ROOT / "harmonization_outputs"),
                    help="directory used to resolve --all artifact names")
    ap.add_argument("--provenance", default="",
                    help="optional long-form provenance CSV; enables the "
                         "value-state check")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--out", default="", help="also write the JSON report to this path")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-finding examples in text output")
    ap.add_argument("--stamp", action="store_true",
                    help="record the schema identity and verdict in the run "
                         "manifest (pipeline gate mode)")
    args = ap.parse_args(argv)

    if not args.artifact and not args.all:
        ap.error("supply --artifact at least once, or --all")

    try:
        schema = load_schema(args.schema)
    except SchemaError as exc:
        print(f"SCHEMA ERROR: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_LOAD

    targets: list[Path] = [Path(a) for a in args.artifact]
    if args.all:
        if not schema.artifacts:
            print("SCHEMA ERROR: --all requires the schema to declare `artifacts`",
                  file=sys.stderr)
            return EXIT_SCHEMA_LOAD
        outputs_dir = Path(args.outputs_dir)
        targets += [outputs_dir / name for name in schema.artifacts]

    seen: set[str] = set()
    ordered: list[Path] = []
    for t in targets:
        if str(t) not in seen:
            seen.add(str(t))
            ordered.append(t)

    provenance: pd.DataFrame | None = None
    if args.provenance:
        try:
            provenance = pd.read_csv(args.provenance, low_memory=False)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            print(f"ARTIFACT ERROR: provenance could not be read: {exc}",
                  file=sys.stderr)
            return EXIT_ARTIFACT_LOAD

    results: list[sv.ValidationResult] = []
    for path in ordered:
        try:
            df = sv.load_artifact(path)
        except sv.ArtifactError as exc:
            print(f"ARTIFACT ERROR: {exc}", file=sys.stderr)
            return EXIT_ARTIFACT_LOAD
        result = sv.validate(schema, df, artifact=path.name,
                             artifact_sha256=sha256_file(path))
        if provenance is not None:
            result.findings.extend(
                sv.check_value_states(schema, provenance, artifact=path.name))
        results.append(result)

    payload = {
        "schema_id": schema.schema_id,
        "schema_version": schema.schema_version,
        "schema_ref": schema.ref,
        "schema_sha256": schema.sha256,
        "artifacts": [r.to_dict() for r in results],
        "verdict": "fail" if any(r.n_errors for r in results) else "pass",
        "counts": {
            "error": sum(r.n_errors for r in results),
            "warning": sum(r.n_warnings for r in results),
            "info": sum(r.n_info for r in results),
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for r in results:
            print(sv.render(r, show_examples=not args.quiet))
        c = payload["counts"]
        print(f"OVERALL  {payload['verdict'].upper()}  "
              f"{len(results)} artifact(s) · {c['error']} error · "
              f"{c['warning']} warning · {c['info']} info  "
              f"[{schema.ref}]")

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.stamp:
        from lib import run_manifest
        stamped = run_manifest.stamp_schema_validation(
            Path(args.outputs_dir),
            schema_id=schema.schema_id,
            schema_version=schema.schema_version,
            schema_sha256=schema.sha256,
            verdict=payload["verdict"],
            counts=payload["counts"],
        )
        if stamped is not None and not args.json:
            print(f"stamped {stamped}")

    if any(_has_structural_errors(r) for r in results):
        return EXIT_STRUCTURAL_ERRORS
    if any(r.n_errors for r in results):
        return EXIT_CONTENT_ERRORS
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
