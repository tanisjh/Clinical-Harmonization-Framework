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
from lib.project_profile import (  # noqa: E402
    ProjectProfile,
    ProjectProfileError,
    load as load_project_profile,
)
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


def _configured(explicit: str | None, profile_default: str,
                existing_default: str) -> str:
    """Apply CLI > explicit profile > existing behavior precedence."""
    if explicit is not None:
        return explicit
    return profile_default or existing_default


def _profile_reference(profile: ProjectProfile | None, reference: str) -> str:
    if profile is None or not reference:
        return ""
    return str(profile.resolve(reference))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate harmonized artifacts against a typed target schema.")
    ap.add_argument(
        "--project-profile",
        help=("explicit project-root-contained ProjectProfile; never discovered "
              "automatically"),
    )
    ap.add_argument(
        "--schema",
        default=None,
        help="path to a target-schema YAML file (overrides ProjectProfile)",
    )
    ap.add_argument("--artifact", action="append", default=[],
                    help="path to a harmonized CSV; repeatable")
    ap.add_argument("--all", action="store_true",
                    help="validate every artifact the schema declares, resolved "
                         "relative to --outputs-dir")
    ap.add_argument("--outputs-dir", default=None,
                    help=("directory used to resolve --all artifact names "
                          "(overrides ProjectProfile)"))
    ap.add_argument("--provenance", default=None,
                    help="optional long-form provenance CSV; enables the "
                         "value-state check (overrides ProjectProfile)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--out", default=None,
                    help="also write the JSON report here (overrides ProjectProfile)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-finding examples in text output")
    ap.add_argument("--stamp", action="store_true",
                    help="record the schema identity and verdict in the run "
                         "manifest (pipeline gate mode)")
    args = ap.parse_args(argv)

    if args.project_profile is None and args.schema is None:
        ap.error("the following arguments are required: --schema")
    if not args.artifact and not args.all:
        ap.error("supply --artifact at least once, or --all")

    profile = None
    if args.project_profile is not None:
        try:
            profile = load_project_profile(
                args.project_profile, project_root=PROJECT_ROOT)
        except ProjectProfileError as exc:
            print(f"PROFILE ERROR: {exc}", file=sys.stderr)
            return EXIT_USAGE

    profile_schema = _profile_reference(
        profile, profile.target_schema if profile else "")
    schema_path = _configured(args.schema, profile_schema, "")

    profile_outputs = _profile_reference(
        profile, profile.tooling.output_directory if profile else "")
    outputs_dir = _configured(
        args.outputs_dir,
        profile_outputs,
        str(PROJECT_ROOT / "harmonization_outputs"),
    )
    profile_provenance = (
        str(Path(outputs_dir) / profile.tooling.provenance_filename)
        if profile and profile.tooling.provenance_filename else ""
    )
    provenance_path = _configured(args.provenance, profile_provenance, "")
    profile_report = (
        str(Path(outputs_dir) / profile.tooling.report_filename)
        if profile and profile.tooling.report_filename else ""
    )
    report_path = _configured(args.out, profile_report, "")

    try:
        schema = load_schema(schema_path)
    except SchemaError as exc:
        print(f"SCHEMA ERROR: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_LOAD

    targets: list[Path] = [Path(a) for a in args.artifact]
    if args.all:
        if not schema.artifacts:
            print("SCHEMA ERROR: --all requires the schema to declare `artifacts`",
                  file=sys.stderr)
            return EXIT_SCHEMA_LOAD
        targets += [Path(outputs_dir) / name for name in schema.artifacts]

    seen: set[str] = set()
    ordered: list[Path] = []
    for t in targets:
        if str(t) not in seen:
            seen.add(str(t))
            ordered.append(t)

    provenance: pd.DataFrame | None = None
    if provenance_path:
        try:
            provenance = pd.read_csv(provenance_path, low_memory=False)
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

    if report_path:
        Path(report_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.stamp:
        from lib import run_manifest
        stamped = run_manifest.stamp_schema_validation(
            Path(outputs_dir),
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
