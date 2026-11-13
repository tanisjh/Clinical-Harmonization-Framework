#!/usr/bin/env python
"""harmonize_spec.py — run one approved specification through the reusable path.

A thin command over the existing generic API. It adds no execution logic of its
own: everything below is `canonical.harmonize_trial`, which is the same path the
eleven-trial assembly proof uses.

    python scripts/harmonize_spec.py specs/cimac_10021_v1.yaml \
        --output .scratch/canonical_10021.csv

    # with the Cell stream, so the run can be inspected afterwards
    python scripts/harmonize_spec.py specs/cimac_10021_v1.yaml \
        --output out.csv --provenance out_provenance.csv

    # refuse to run a specification that fails pre-flight (recommended)
    python scripts/harmonize_spec.py specs/cimac_10021_v1.yaml \
        --output out.csv --preflight

    # publish a replay-enabled run (output names live inside the fresh run dir)
    python scripts/harmonize_spec.py specs/example.yaml \
        --output delivered.csv --provenance provenance.csv \
        --replay-run-dir harmonization_outputs/run-example

WHAT IT DOES NOT TOUCH
No legacy extractor, no `migrated_trials.py`, no production orchestrator, no
`harmonization_config.yaml`, and no branch on a trial name. Whether a trial
needs a row anchor is read from its own `record.anchor` — a property of the
document, not a decision this script makes.

The row anchor is supplied only for specifications whose records come from an
external anchor. A grouping specification builds its own records from a source
and is given none; handing it one would silently override them, and the executor
says so.

Confidence thresholds come from the production configuration read-only, because
suppression is a delivery decision the trial did not author.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd                                    # noqa: E402
import yaml                                            # noqa: E402

from lib import canonical, spec_preflight              # noqa: E402
from lib.anchor_evidence import load_external_anchor   # noqa: E402
from lib.project_profile import (                     # noqa: E402
    ProjectProfileError,
    load as load_project_profile,
)
from lib.replay_orchestration import (                 # noqa: E402
    ReplayEvidenceCollector,
    ReplayOrchestrationError,
    ReplayPublication,
    ReplayRunContext,
    load_yaml_evidence,
)
from lib.schema_model import load as load_schema       # noqa: E402
from lib.spec_model import SpecError, load as load_spec  # noqa: E402
from lib.tooling_defaults import resolve_tool_default  # noqa: E402

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_USAGE = 2
EXIT_SCHEMA_LOAD = 3
EXIT_SPEC_LOAD = 4
EXIT_PREFLIGHT = 5

DEFAULT_SCHEMA = "schemas/cimac_clinical_v1.yaml"
DEFAULT_ANCHOR = "cross_trial_analysis_egk_april30_meta_9trials.csv"
DEFAULT_CONFIG = "scripts/config/harmonization_config.yaml"

#: Cell attributes worth writing out. The full model, minus the deprecated
#: positional locator that `lib/provenance.py` documents as superseded.
PROVENANCE_COLUMNS = (
    "trial", "cimac_part_id", "Cimac_id", "Collection_Event",
    "harmonized_field", "value", "confidence", "value_state",
    "rule_id", "rule_version", "run_id", "spec_ref", "mapping_id",
    "source_file", "source_logical_label", "source_resolved_path",
    "source_sha256", "source_sheet", "source_column", "source_row_number",
    "source_record_key", "source_ref_id", "raw_value", "raw_value_json",
    "evidence_refs", "contributions", "derivation",
    "authoritative_for", "extraction_method", "notes",
)


def _anchor_for(spec, anchor_path: Path, *, replay_enabled: bool = False):
    """The row anchor this specification needs — or none, as it declares."""
    if spec.record.anchor != "external":
        return None
    if not anchor_path.is_file():
        raise FileNotFoundError(
            f"{spec.ref} declares an external row anchor but the anchor table "
            f"was not found: {anchor_path}")
    if replay_enabled:
        anchor = load_external_anchor(
            anchor_path, project_root=PROJECT_ROOT, spec_ref=spec.ref,
            filter_column="trial", selector=spec.trial)
        if not anchor.filtered_positions:
            raise ValueError(
                f"{spec.ref}: the anchor table holds no rows for trial "
                f"{spec.trial!r}")
        return anchor
    table = pd.read_csv(anchor_path)
    rows = table[table["trial"] == spec.trial].reset_index(drop=True)
    if rows.empty:
        raise ValueError(
            f"{spec.ref}: the anchor table holds no rows for trial "
            f"{spec.trial!r}")
    return rows


def _replay_output_name(value: str, run_dir: Path, label: str) -> str:
    """Return an explicit filename, rejecting paths outside the run directory."""
    raw = Path(value)
    if raw.parent == Path("."):
        return raw.name
    resolved = (raw if raw.is_absolute() else PROJECT_ROOT / raw).resolve()
    if resolved.parent != run_dir or resolved.name != raw.name:
        raise ReplayOrchestrationError(
            f"{label} must be a filename or a direct child of {run_dir}")
    return resolved.name


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Harmonize one specification through the reusable path.")
    p.add_argument("spec", help="path to an approved specification YAML")
    p.add_argument("--output", required=True, help="canonical CSV to write")
    p.add_argument("--provenance", help="optional Cell-stream CSV to write")
    p.add_argument(
        "--project-profile",
        help="explicit ProjectProfile for the schema default; never auto-discovered",
    )
    p.add_argument("--schema", default=None,
                   help=("target schema YAML (overrides ProjectProfile; "
                         f"default without profile: {DEFAULT_SCHEMA})"))
    p.add_argument("--anchor", default=DEFAULT_ANCHOR,
                   help="row-anchor table (external-anchor specs only)")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="read-only source of confidence thresholds")
    p.add_argument("--run-id", default="",
                   help="run manifest ID to bind into emitted provenance")
    p.add_argument(
        "--replay-run-dir",
        help=("enable strict replay publication in a fresh immutable run "
              "directory; --provenance is required"),
    )
    p.add_argument(
        "--artifact-id", default="",
        help="delivered artifact identity in replay-enabled mode",
    )
    p.add_argument("--preflight", action="store_true",
                   help="refuse to run if deterministic pre-flight fails")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    replay_enabled = bool(args.replay_run_dir)
    if replay_enabled and not args.provenance:
        print("ERROR  --replay-run-dir requires --provenance", file=sys.stderr)
        return EXIT_USAGE

    profile_schema = ""
    if args.project_profile is not None:
        try:
            profile = load_project_profile(
                args.project_profile, project_root=PROJECT_ROOT)
            profile_schema = str(profile.resolve(profile.target_schema))
        except ProjectProfileError as exc:
            print(f"PROFILE ERROR: {exc}", file=sys.stderr)
            return EXIT_USAGE

    schema_path = resolve_tool_default(args.schema, profile_schema, DEFAULT_SCHEMA)
    try:
        schema = load_schema(PROJECT_ROOT / schema_path)
    except Exception as e:
        print(f"ERROR  could not load target schema: {e}", file=sys.stderr)
        return EXIT_SCHEMA_LOAD
    try:
        spec = load_spec(Path(args.spec), target_schema=schema,
                         require_executable=True)
    except SpecError as e:
        print(f"ERROR  specification is not executable: {e}", file=sys.stderr)
        return EXIT_SPEC_LOAD

    if args.preflight:
        report = spec_preflight.preflight(spec, PROJECT_ROOT)
        if not report.ok:
            print(f"PRE-FLIGHT FAILED  {report.spec_ref}", file=sys.stderr)
            for f in report.findings:
                print(f"  [{f.kind}] {f.message}", file=sys.stderr)
            return EXIT_PREFLIGHT
        if not args.quiet:
            print(f"pre-flight       PASS  {report.spec_ref}")

    context = None
    collector = None
    publication = None
    config_evidence = None
    try:
        if replay_enabled:
            context = ReplayRunContext.create(
                PROJECT_ROOT, schema, (spec,), run_id=args.run_id)
            collector = ReplayEvidenceCollector(context)
            publication = ReplayPublication(
                context, collector, args.replay_run_dir)
            delivered_name = _replay_output_name(
                args.output, publication.run_dir, "--output")
            provenance_name = _replay_output_name(
                args.provenance, publication.run_dir, "--provenance")
        else:
            delivered_name = provenance_name = ""

        thresholds = {}
        cfg = PROJECT_ROOT / args.config
        if cfg.is_file():
            if replay_enabled:
                config_evidence = load_yaml_evidence(PROJECT_ROOT, cfg)
                config_doc = config_evidence.document or {}
            else:
                config_doc = yaml.safe_load(cfg.read_text()) or {}
            if not isinstance(config_doc, dict):
                raise ReplayOrchestrationError(
                    f"configuration must be a YAML mapping: {cfg}")
            thresholds = config_doc.get("confidence_thresholds", {})
    except (OSError, ReplayOrchestrationError, yaml.YAMLError) as e:
        print(f"ERROR  replay setup failed: {e}", file=sys.stderr)
        return EXIT_USAGE if replay_enabled else EXIT_RUN_FAILED

    try:
        anchor = _anchor_for(
            spec, PROJECT_ROOT / args.anchor, replay_enabled=replay_enabled)
        result = canonical.harmonize_trial(spec, PROJECT_ROOT, anchor, schema,
                                           thresholds=thresholds,
                                           run_id=(context.run_id
                                                   if context else args.run_id))
        table = canonical.assemble([result], schema)
    except Exception as e:
        print(f"ERROR  harmonization failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return EXIT_RUN_FAILED

    out = Path(args.output)

    suppressed = [c for c in result.cells
                  if c.rule_id == "value_suppressed_below_threshold.v1"]
    if not args.quiet:
        print(f"spec             {spec.ref}  ({spec.trial})")
        print(f"pre-canonical    {result.intermediate_rows} rows")
        print(f"exclusions       {len(result.exclusions)} event(s), "
              f"{sum(e.n_rows for e in result.exclusions)} row(s) removed")
        print(f"suppression      {len(suppressed)} event(s)")
        print(f"canonical        {len(table)} rows x {len(table.columns)} columns")
        if not replay_enabled:
            print(f"wrote            {out}")

    frame = None
    if args.provenance:
        frame = pd.DataFrame([
            {c: getattr(cell, c, "") for c in PROVENANCE_COLUMNS}
            for cell in result.cells])
        # `Cimac_id` is the Cell attribute; the delivered column is `Cimac.id`.
        frame = frame.rename(columns={"Cimac_id": "Cimac.id"})
    if replay_enabled:
        try:
            collector.collect(spec, result.sources, result.cells)
            if config_evidence is not None:
                config_doc = config_evidence.document or {}
                config_ref = str(config_doc.get("config_id") or args.config)
                if config_doc.get("config_version"):
                    config_ref += f"@{config_doc['config_version']}"
                collector.add_evidence(config_evidence.binding(
                    evidence_ref_id="config:confidence-thresholds",
                    kind="thresholds",
                    reference=config_ref,
                    details={"section": "confidence_thresholds"},
                ))
            published = publication.publish(
                table, frame,
                artifact_id=(args.artifact_id or delivered_name),
                delivered_name=delivered_name,
                provenance_name=provenance_name,
            )
        except (OSError, ReplayOrchestrationError) as e:
            print(f"ERROR  replay publication failed: {e}", file=sys.stderr)
            return EXIT_RUN_FAILED
        if not args.quiet:
            print(f"run_id           {context.run_id}")
            print(f"wrote            {published.delivered_path}")
            print(f"provenance       {len(frame)} cells -> "
                  f"{published.provenance_path}")
            print(f"replay manifest  {published.manifest_path}")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        if args.provenance:
            prov = Path(args.provenance)
            prov.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(prov, index=False)
            if not args.quiet:
                print(f"provenance       {len(frame)} cells -> {prov}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
