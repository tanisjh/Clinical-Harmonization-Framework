"""Synthetic-only tests for production replay evidence orchestration."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import harmonize_spec
import lib.replay_orchestration as orchestration
from lib import artifact_stage, canonical, suppression
from lib.artifact_stage import (
    ArtifactResult,
    ExclusionEvent,
    OverrideEvent,
)
from lib.replay import ReplayRequest, replay_cell
from lib.replay_contract import ReplayStatus
from lib.replay_manifest import ReplayManifest, load as load_manifest
from lib.replay_orchestration import (
    ReplayEvidenceCollector,
    ReplayOrchestrationError,
    ReplayPublication,
    ReplayRunContext,
    is_generated_run_id,
    load_yaml_evidence,
)
from lib.schema_model import contract_for, load as load_schema
from lib.spec_executor import execute
from lib.spec_model import Exclusion, PostRule, RuleRef, load as load_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema.yaml"
SPEC = FIXTURE / "spec.yaml"
CONFIG = FIXTURE / "config.yaml"
SCHEMA_STAGE2 = FIXTURE / "schema_stage2.yaml"
SPEC_STAGE2 = FIXTURE / "spec_stage2.yaml"


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _anchor() -> pd.DataFrame:
    return pd.DataFrame([{
        "device_id": "DEVICE-A",
        "reading_id": "READ-001",
        "location": "bench-west",
        # BaseExtractor's compatibility anchor names remain part of Cell.
        "cimac_part_id": "DEVICE-A",
        "Cimac.id": "READ-001",
        "Collection_Event": "bench-west",
    }])


def _execution(run_id: str = "synthetic-orchestration-run"):
    schema = load_schema(SCHEMA)
    spec = load_spec(SPEC, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=run_id,
        application="synthetic-orchestration-test")
    cells, diagnostics = execute(
        spec, ROOT, _anchor(), contract=contract_for(schema),
        run_id=context.run_id)
    return context, schema, spec, cells, diagnostics


def _frames(schema, cells):
    row = {
        "device_id": "DEVICE-A",
        "reading_id": "READ-001",
        "location": "bench-west",
        "reading_text": None,
        "sequence_number": None,
        "enabled": None,
        "optional_note": None,
    }
    for cell in cells:
        value = cell.value
        try:
            if bool(pd.isna(value)):
                value = None
        except (TypeError, ValueError):
            pass
        row[cell.harmonized_field] = value
    delivered = pd.DataFrame([row], columns=schema.column_order)
    provenance = pd.DataFrame([
        {name: getattr(cell, name, "")
         for name in harmonize_spec.PROVENANCE_COLUMNS}
        for cell in cells
    ]).rename(columns={"Cimac_id": "Cimac.id"})
    return delivered, provenance


def _publication_bundle(tmp_path: Path,
                        run_id: str = "synthetic-publication-run"):
    context, schema, spec, cells, diagnostics = _execution(run_id)
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)
    delivered, provenance = _frames(schema, cells)
    run_dir = tmp_path / run_id
    publication = ReplayPublication(context, collector, run_dir)
    return (context, collector, publication, delivered, provenance,
            diagnostics, cells)


def test_run_context_rejects_blank_mapping_id_before_execution():
    schema = load_schema(SCHEMA)
    spec = load_spec(SPEC, target_schema=schema, require_executable=True)
    mappings = (replace(spec.mappings[0], mapping_id=""),
                *spec.mappings[1:])
    with pytest.raises(ReplayOrchestrationError, match="without mapping_id"):
        ReplayRunContext.create(
            ROOT, schema, (replace(spec, mappings=mappings),),
            run_id="missing-mapping-id-run")


def test_collector_rejects_blank_id_normal_mapping_cell():
    context, _schema, spec, cells, diagnostics = _execution()
    blank = replace(cells[0], mapping_id="")
    with pytest.raises(ReplayOrchestrationError,
                       match="normal specification mapping Cell"):
        ReplayEvidenceCollector(context).collect(
            spec, diagnostics.sources, (blank,))


def test_collector_exempts_declared_non_mapping_methods():
    context, _schema, spec, cells, diagnostics = _execution()
    base = cells[0]
    methods = (
        "spec:lookup_miss",
        suppression.EXTRACTION_METHOD,
        "post_pivot_override",
        "row_excluded",
    )
    special = tuple(replace(
        base, mapping_id="", extraction_method=method) for method in methods)
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, special)
    assert collector.mapping_usages == ()


def test_mapping_ids_survive_provenance_serialization():
    _context, schema, _spec, cells, _diagnostics = _execution()
    _delivered, provenance = _frames(schema, cells)
    assert provenance["mapping_id"].tolist() == [
        cell.mapping_id for cell in cells]


def test_run_context_generated_and_explicit_ids():
    schema = load_schema(SCHEMA)
    spec = load_spec(SPEC, target_schema=schema, require_executable=True)
    generated = ReplayRunContext.create(ROOT, schema, (spec,)).run_id
    explicit = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id="controlled-test-run").run_id
    assert generated
    assert is_generated_run_id(generated)
    assert explicit == "controlled-test-run"


def test_run_id_reaches_mapping_suppression_post_rule_and_exclusion():
    context, _schema, spec, cells, diagnostics = _execution(
        "synthetic-all-events-run")
    assert cells and {cell.run_id for cell in cells} == {context.run_id}
    assert {cell.spec_ref for cell in cells} == {spec.ref}

    base = next(c for c in cells if c.mapping_id == "direct.reading_text")
    withheld = [{
        "trial": base.trial,
        "cimac_part_id": base.cimac_part_id,
        "Cimac.id": base.Cimac_id,
        "Collection_Event": base.Collection_Event,
        "harmonized_field": base.harmonized_field,
        "input_rule_id": base.rule_id,
        "proposed": base.value,
        "confidence": base.confidence,
        "threshold": 1.1,
    }]
    suppressed = suppression.record_canonical_suppression(
        withheld, cells, thresholds={}, run_id=context.run_id,
        spec_ref=spec.ref)

    post = PostRule(
        id="synthetic.post", rule=RuleRef(
            "synthetic.post.v1", "1", approval_ref="synthetic-approval"),
        inputs=("sequence_number",), confidence=1.0,
        value_state="transformed", source_label="SPEC:synthetic.post",
        stage="post_pivot")
    exclusion = Exclusion(
        id="synthetic.exclusion", scope="participant",
        reason="synthetic exclusion", rule=RuleRef(
            "synthetic.exclusion.v1", "1",
            approval_ref="synthetic-approval"),
        source="readings", match={"field": "device_id"},
        source_label="SPEC:synthetic.exclusion")
    event_spec = replace(
        spec, post_rules=(post,), exclusions=(exclusion,))
    result = ArtifactResult(
        artifact="", frame=pd.DataFrame(),
        overrides=(OverrideEvent(
            artifact="", post_rule_id=post.id, trial=spec.trial,
            participant="DEVICE-A", sample="READ-001",
            collection_event="bench-west", field="reading_text",
            previous_value="001", value="changed", branch="synthetic",
            predicate="sequence_number >= 1", input_field="sequence_number",
            input_value=1.0, threshold=1.0,
            rule_id=post.rule.id, rule_version=post.rule.version,
            reason="synthetic post-rule event"),),
        exclusions=(ExclusionEvent(
            artifact="", exclusion_id=exclusion.id, trial=spec.trial,
            participant="DEVICE-A", reason=exclusion.reason,
            rule_id=exclusion.rule.id,
            rule_version=exclusion.rule.version, n_rows=1),),
    )
    special = artifact_stage.record_canonical_provenance(
        event_spec, result, cells, sources=diagnostics.sources,
        run_id=context.run_id)

    emitted = suppressed + special
    assert len(emitted) == 3
    assert {cell.run_id for cell in emitted} == {context.run_id}
    assert {cell.spec_ref for cell in emitted} == {spec.ref}


def test_collector_uses_actual_source_refs_without_reread(monkeypatch):
    context, _schema, spec, cells, diagnostics = _execution()
    collector = ReplayEvidenceCollector(context)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("collector attempted to reread or rehash a source")

    monkeypatch.setattr(orchestration, "sha256_file", forbidden)
    monkeypatch.setattr(type(next(iter(diagnostics.sources.values()))),
                        "verify", forbidden)
    collector.collect(spec, diagnostics.sources, cells)
    binding = collector.sources[0]
    actual = diagnostics.sources["readings"]
    assert binding.source_ref_id == actual.source_ref_id
    assert binding.path == actual.resolved_path
    assert binding.sha256 == actual.sha256


def test_collector_deduplicates_deterministically_and_scopes_evidence():
    context, _schema, spec, cells, diagnostics = _execution()
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)
    collector.collect(spec, diagnostics.sources, reversed(cells))
    loaded = load_yaml_evidence(ROOT, CONFIG)
    collector.add_evidence(loaded.binding(
        evidence_ref_id="config:unrelated", kind="configuration",
        reference="synthetic/unrelated@1.0.0"))

    assert len(collector.sources) == 1
    assert len(collector.rules) == 1
    assert [usage.mapping_id for usage in collector.mapping_usages] == sorted(
        cell.mapping_id for cell in cells)
    assert all(not usage.evidence_ref_ids
               for usage in collector.mapping_usages)
    assert collector.evidence[0].sha256 == loaded.sha256


def test_conflicting_spec_source_rule_and_mapping_identities_are_rejected():
    context, _schema, spec, cells, diagnostics = _execution()
    with pytest.raises(ReplayOrchestrationError, match="conflicting specification"):
        ReplayRunContext.create(
            ROOT, context.schema, (spec, replace(spec, sha256="0" * 64)),
            run_id="conflicting-spec-run")

    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)
    with pytest.raises(ReplayOrchestrationError, match="conflicting source"):
        collector.add_source(replace(
            collector.sources[0], logical_label="different.csv"))
    with pytest.raises(ReplayOrchestrationError, match="conflicting rule"):
        collector.add_rule(replace(
            collector.rules[0], parameters=(("different", True),)))
    with pytest.raises(ReplayOrchestrationError, match="names target"):
        collector.collect(
            spec, diagnostics.sources,
            (replace(cells[0], harmonized_field="optional_note"),))


def test_publication_hashes_counts_and_manifest_last(monkeypatch, tmp_path):
    (context, _collector, publication, delivered, provenance,
     _diagnostics, _cells) = _publication_bundle(tmp_path)
    observed = []
    original = ReplayManifest.write_atomic

    def checked_write(manifest, path):
        path = Path(path)
        assert (publication.run_dir / "delivered.csv").is_file()
        assert (publication.run_dir / "provenance.csv").is_file()
        assert not path.exists()
        observed.append("manifest-last")
        return original(manifest, path)

    monkeypatch.setattr(ReplayManifest, "write_atomic", checked_write)
    published = publication.publish(
        delivered, provenance, artifact_id="synthetic-delivered",
        delivered_name="delivered.csv", provenance_name="provenance.csv")

    assert observed == ["manifest-last"]
    assert published.manifest_path.is_file()
    assert published.manifest.delivered_artifacts[0].n_rows == len(delivered)
    assert published.manifest.provenance_artifact.n_rows == len(provenance)
    assert all(check.ok for check in published.manifest.verify_hashes(
        context.project_root))


def test_failed_writer_and_manifest_construction_leave_no_valid_manifest(
        monkeypatch, tmp_path):
    (_context, collector, publication, delivered, provenance,
     _diagnostics, _cells) = _publication_bundle(
        tmp_path, "synthetic-writer-failure")

    def writer_failure(*_args, **_kwargs):
        raise OSError("synthetic writer failure")

    monkeypatch.setattr(publication, "_write_csv", writer_failure)
    with pytest.raises(ReplayOrchestrationError, match="writer failure"):
        publication.publish(
            delivered, provenance, artifact_id="synthetic-delivered",
            delivered_name="delivered.csv",
            provenance_name="provenance.csv")
    assert not publication.run_dir.exists()

    (_context, collector, publication, delivered, provenance,
     _diagnostics, _cells) = _publication_bundle(
        tmp_path, "synthetic-manifest-failure")

    def manifest_failure(*_args, **_kwargs):
        raise ReplayOrchestrationError("synthetic manifest failure")

    monkeypatch.setattr(collector, "build_manifest", manifest_failure)
    with pytest.raises(ReplayOrchestrationError, match="manifest failure"):
        publication.publish(
            delivered, provenance, artifact_id="synthetic-delivered",
            delivered_name="delivered.csv",
            provenance_name="provenance.csv")
    assert not publication.run_dir.exists()


def test_failed_final_hash_verification_leaves_no_manifest(monkeypatch, tmp_path):
    (_context, _collector, publication, delivered, provenance,
     _diagnostics, _cells) = _publication_bundle(
        tmp_path, "synthetic-hash-failure")

    def hash_failure(*_args, **_kwargs):
        raise ReplayOrchestrationError("synthetic final hash failure")

    monkeypatch.setattr(publication, "_verify_final_hashes", hash_failure)
    with pytest.raises(ReplayOrchestrationError, match="hash failure"):
        publication.publish(
            delivered, provenance, artifact_id="synthetic-delivered",
            delivered_name="delivered.csv",
            provenance_name="provenance.csv")
    assert publication.run_dir.is_dir()
    assert not (publication.run_dir / "replay_manifest.json").exists()


def test_publication_refuses_overwrite_escape_and_missing_provenance(tmp_path):
    (context, collector, publication, delivered, provenance,
     _diagnostics, _cells) = _publication_bundle(
        tmp_path, "synthetic-immutable-run")
    publication.publish(
        delivered, provenance, artifact_id="synthetic-delivered",
        delivered_name="delivered.csv", provenance_name="provenance.csv")
    with pytest.raises(ReplayOrchestrationError, match="already exists"):
        ReplayPublication(context, collector, publication.run_dir)

    with pytest.raises(ReplayOrchestrationError, match="escapes project root"):
        ReplayPublication(context, collector, ROOT.parent / "escape")

    missing = ReplayPublication(
        context, collector, tmp_path / "synthetic-missing-provenance")
    with pytest.raises(ReplayOrchestrationError, match="requires provenance"):
        missing.publish(
            delivered, None, artifact_id="synthetic-delivered",
            delivered_name="delivered.csv",
            provenance_name="provenance.csv")
    assert not missing.run_dir.exists()


def _synthetic_pivot(cells, _thresholds, contract):
    row = {name: None for name in contract.columns}
    first = cells[0]
    row.update({
        "device_id": first.cimac_part_id,
        "reading_id": first.Cimac_id,
        "location": first.Collection_Event,
    })
    for cell in cells:
        row[cell.harmonized_field] = cell.value
    return pd.DataFrame([row], columns=contract.columns), [], []


def test_replay_enabled_harmonize_spec_publishes_replayable_bundle(
        monkeypatch, tmp_path):
    anchor_path = tmp_path / "anchor.csv"
    anchor = _anchor()
    anchor.insert(0, "trial", "synthetic_device_lab")
    anchor.to_csv(anchor_path, index=False)
    run_dir = tmp_path / "synthetic-cli-run"

    monkeypatch.setattr(canonical, "_pivot", lambda: _synthetic_pivot)
    exit_code = harmonize_spec.main([
        str(SPEC),
        "--schema", _relative(SCHEMA),
        "--anchor", _relative(anchor_path),
        "--config", _relative(CONFIG),
        "--output", "delivered.csv",
        "--provenance", "provenance.csv",
        "--replay-run-dir", _relative(run_dir),
        "--artifact-id", "synthetic-delivered",
        "--run-id", "synthetic-cli-run-id",
        "--quiet",
    ])
    assert exit_code == harmonize_spec.EXIT_OK

    delivered = run_dir / "delivered.csv"
    provenance = run_dir / "provenance.csv"
    manifest_path = run_dir / "replay_manifest.json"
    assert delivered.is_file()
    assert provenance.is_file()
    assert manifest_path.is_file()

    manifest = load_manifest(_relative(manifest_path), project_root=ROOT)
    assert manifest.run_id == "synthetic-cli-run-id"
    assert manifest.evidence
    assert all(check.ok for check in manifest.verify_hashes(ROOT))
    provenance_frame = pd.read_csv(provenance, keep_default_na=False)
    assert set(provenance_frame["run_id"]) == {manifest.run_id}

    result = replay_cell(ROOT, manifest, ReplayRequest(
        artifact_id="synthetic-delivered",
        row_key={"device_id": "DEVICE-A", "reading_id": "READ-001"},
        target_field="reading_text"))
    assert (result.status, result.reason_code) == (
        ReplayStatus.PASS, "REPLAY_MATCH")


def test_stage2_operations_use_production_orchestration(tmp_path):
    schema = load_schema(SCHEMA_STAGE2)
    spec = load_spec(
        SPEC_STAGE2, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id="synthetic-stage2-orchestration",
        application="synthetic-stage2-orchestration-test")
    cells, diagnostics = execute(
        spec, ROOT, _anchor(), contract=contract_for(schema),
        run_id=context.run_id)

    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)
    row = {name: None for name in schema.column_order}
    row.update({
        "device_id": "DEVICE-A",
        "reading_id": "READ-001",
        "location": "bench-west",
    })
    for cell in cells:
        row[cell.harmonized_field] = cell.value
    delivered = pd.DataFrame([row], columns=schema.column_order)
    provenance = pd.DataFrame([
        {name: getattr(cell, name, "")
         for name in harmonize_spec.PROVENANCE_COLUMNS}
        for cell in cells
    ]).rename(columns={"Cimac_id": "Cimac.id"})

    publication = ReplayPublication(
        context, collector, tmp_path / context.run_id)
    published = publication.publish(
        delivered, provenance, artifact_id="synthetic-stage2-delivered",
        delivered_name="delivered.csv", provenance_name="provenance.csv")

    usages = {usage.mapping_id: usage
              for usage in published.manifest.mapping_usages}
    assert usages["map.status"].rule_refs == ("map.inline.v1@1",)
    assert usages["unit.basic"].rule_refs == ("unit.convert.v1@1",)
    for field in ("status_label", "converted_measurement"):
        result = replay_cell(ROOT, published.manifest, ReplayRequest(
            artifact_id="synthetic-stage2-delivered",
            row_key={"device_id": "DEVICE-A", "reading_id": "READ-001"},
            target_field=field))
        assert (result.status, result.reason_code) == (
            ReplayStatus.PASS, "REPLAY_MATCH")


def test_replay_enabled_cli_requires_provenance():
    assert harmonize_spec.main([
        str(SPEC), "--output", "delivered.csv",
        "--replay-run-dir", ".scratch/synthetic-missing-provenance",
        "--quiet",
    ]) == harmonize_spec.EXIT_USAGE
