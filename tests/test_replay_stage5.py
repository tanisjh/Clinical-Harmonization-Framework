"""Synthetic-only tests for runtime-bound vocabulary replay (Stage 5)."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

import harmonize_spec
from lib import rule_registry, vocabularies
from lib.replay import (
    ReplayRequest,
    recompute_registered_rule,
    recompute_vocabulary_value_map,
    replay_cell,
)
from lib.replay_contract import ReplayStatus
from lib.replay_manifest import EvidenceBinding
from lib.replay_orchestration import (
    ReplayEvidenceCollector,
    ReplayOrchestrationError,
    ReplayPublication,
    ReplayRunContext,
    vocabulary_evidence_binding,
)
from lib.rule_registry import RegisteredRule
from lib.schema_model import contract_for, load as load_schema
from lib.source_ref import sha256_file
from lib.spec_executor import execute
from lib.spec_model import load as load_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema_stage5.yaml"
SPEC = FIXTURE / "spec_stage5.yaml"
SOURCE = FIXTURE / "sources/stage5_readings.csv"
ORDERED = FIXTURE / "vocabularies/stage5_ordered.yaml"
BUCKETS = FIXTURE / "vocabularies/stage5_buckets.yaml"
RUN_ID = "synthetic-stage5-run-0001"
ARTIFACT_ID = "synthetic-stage5-delivered"


def _bucket(value, *, rules):
    if value is None:
        return None, 0.2, "missing"
    for bucket, values in rules.items():
        if str(value).upper() in {str(item).upper() for item in values}:
            return bucket, 0.95, "synthetic bucket"
    return None, 0.4, "unmatched"


def _downstream(value):
    if value is None:
        return None, 0.2, "missing"
    return f"down:{value}", 0.88, "synthetic downstream"


@pytest.fixture(autouse=True)
def _rules(monkeypatch):
    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.bucket.v1", "1"), RegisteredRule(
            "synthetic.bucket.v1", "1", rule_registry.KIND_FIELD,
            _bucket, arity=1, params=("rules",)))
    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.downstream.v1", "1"), RegisteredRule(
            "synthetic.downstream.v1", "1", rule_registry.KIND_FIELD,
            _downstream, arity=1))


@dataclass
class Case:
    case_dir: Path
    manifest: object
    request: ReplayRequest
    spec: object
    cells: list
    sources: dict
    context: ReplayRunContext
    delivered: pd.DataFrame
    provenance: pd.DataFrame
    delivered_path: Path
    provenance_path: Path
    ordered_path: Path
    buckets_path: Path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _anchor():
    return pd.DataFrame([{
        "device_id": "DEVICE-A", "reading_id": "READ-001",
        "location": "bench-west", "cimac_part_id": "DEVICE-A",
        "Cimac.id": "READ-001", "Collection_Event": "bench-west",
    }])


def _write_rows(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def _bundle(tmp_path: Path, *, target="exact_label") -> Case:
    case_dir = tmp_path / target.replace(".", "_")
    case_dir.mkdir(parents=True)
    source_path = case_dir / "stage5_readings.csv"
    source_path.write_bytes(SOURCE.read_bytes())
    vocabulary_dir = case_dir / "vocabularies"
    vocabulary_dir.mkdir()
    ordered_path = vocabulary_dir / ORDERED.name
    buckets_path = vocabulary_dir / BUCKETS.name
    ordered_path.write_bytes(ORDERED.read_bytes())
    buckets_path.write_bytes(BUCKETS.read_bytes())

    spec_doc = yaml.safe_load(SPEC.read_text())
    spec_doc["sources"][0]["path"] = _relative(source_path)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_doc, sort_keys=False))
    schema = load_schema(SCHEMA)
    spec = load_spec(
        spec_path, target_schema=schema, require_executable=True,
        vocabulary_dir=vocabulary_dir)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage5-test")
    cells, diagnostics = execute(
        spec, ROOT, _anchor(), contract=contract_for(schema), run_id=RUN_ID)
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)

    row = {name: None for name in schema.column_order}
    row.update({"device_id": "DEVICE-A", "reading_id": "READ-001",
                "location": "bench-west"})
    for cell in cells:
        row[cell.harmonized_field] = cell.value
    delivered = pd.DataFrame([row], columns=schema.column_order)
    provenance = pd.DataFrame([
        {name: getattr(cell, name, "")
         for name in harmonize_spec.PROVENANCE_COLUMNS}
        for cell in cells
    ]).rename(columns={"Cimac_id": "Cimac.id"})
    published = ReplayPublication(
        context, collector, case_dir / "published").publish(
        delivered, provenance, artifact_id=ARTIFACT_ID,
        delivered_name="delivered.csv", provenance_name="provenance.csv")
    request = ReplayRequest(
        artifact_id=ARTIFACT_ID,
        row_key={"device_id": "DEVICE-A", "reading_id": "READ-001"},
        target_field=target)
    return Case(
        case_dir, published.manifest, request, spec, cells,
        diagnostics.sources, context, delivered, provenance,
        published.delivered_path, published.provenance_path,
        ordered_path, buckets_path)


def _event(rows, mapping_id):
    return next(row for row in rows if row["mapping_id"] == mapping_id)


def _mutate_provenance(case: Case, mapping_id: str, **updates) -> None:
    fields, rows = _read_rows(case.provenance_path)
    _event(rows, mapping_id).update(updates)
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(
        case.manifest,
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)))


def _mutate_delivery(case: Case, field: str, value: str) -> None:
    fields, rows = _read_rows(case.delivered_path)
    rows[0][field] = value
    _write_rows(case.delivered_path, fields, rows)
    binding = case.manifest.delivered_artifacts[0]
    case.manifest = replace(
        case.manifest,
        delivered_artifacts=(replace(
            binding, sha256=sha256_file(case.delivered_path)),))


def _usage(case: Case, mapping_id: str):
    return next(item for item in case.manifest.mapping_usages
                if item.mapping_id == mapping_id)


def _vocabulary_binding(case: Case, reference: str) -> EvidenceBinding:
    return next(item for item in case.manifest.evidence
                if item.kind == "vocabulary" and item.reference == reference)


def _replace_usage(case: Case, mapping_id: str, usage) -> None:
    case.manifest = replace(
        case.manifest,
        mapping_usages=tuple(
            usage if item.mapping_id == mapping_id else item
            for item in case.manifest.mapping_usages))


def _replace_binding(case: Case, old: EvidenceBinding,
                     new: EvidenceBinding) -> None:
    fields, rows = _read_rows(case.provenance_path)
    for row in rows:
        refs = [new.evidence_ref_id if ref == old.evidence_ref_id else ref
                for ref in row["evidence_refs"].split(";") if ref]
        row["evidence_refs"] = ";".join(refs)
    _write_rows(case.provenance_path, fields, rows)
    usages = tuple(replace(
        usage,
        evidence_ref_ids=tuple(
            new.evidence_ref_id if ref == old.evidence_ref_id else ref
            for ref in usage.evidence_ref_ids))
        for usage in case.manifest.mapping_usages)
    evidence = tuple(new if item.evidence_ref_id == old.evidence_ref_id else item
                     for item in case.manifest.evidence)
    case.manifest = replace(
        case.manifest, evidence=evidence, mapping_usages=usages,
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)))


def test_runtime_loaded_identity_is_exact_and_usage_scoped(tmp_path):
    case = _bundle(tmp_path)
    mapping = next(m for m in case.spec.mappings if m.mapping_id == "map.exact")
    loaded = mapping.fallback_vocabulary_object
    assert loaded.ref == "synthetic/stage5_ordered@1.0.0"
    assert loaded.kind == vocabularies.KIND_ORDERED_RULES
    assert loaded.source_sha256 == hashlib.sha256(
        case.ordered_path.read_bytes()).hexdigest()
    assert loaded.contained_path(ROOT) == _relative(case.ordered_path)
    assert loaded.parser_id == "pyyaml.safe_load"
    assert loaded.parser_version

    custom = next(m for m in case.spec.mappings if m.mapping_id == "custom.bucket")
    assert custom.params["rules"] == yaml.safe_load(
        case.buckets_path.read_bytes())["terms"]
    assert custom.parameter_vocabularies[0][0] == "rules"
    assert custom.parameter_vocabularies[0][1].ref == \
        "synthetic/stage5_buckets@1.0.0"

    cell = next(c for c in case.cells if c.mapping_id == "map.exact")
    usage = _usage(case, "map.exact")
    assert cell.evidence_refs == usage.evidence_ref_ids[0]
    assert _vocabulary_binding(
        case, "synthetic/stage5_ordered@1.0.0").evidence_ref_id == \
        cell.evidence_refs
    assert not _usage(case, "direct.response").evidence_ref_ids
    assert not _usage(case, "custom.downstream").evidence_ref_ids
    assert _usage(case, "custom.bucket").evidence_ref_ids


@pytest.mark.parametrize("target,expected", [
    ("inline_label", "inline-wins"),
    ("overlap_label", "broad-first"),
    ("exact_label", "exact-hit"),
    ("iexact_label", "iexact-hit"),
    ("prefix_label", "prefix-hit"),
    ("contains_label", "contains-hit"),
    ("regex_label", "regex-hit"),
    ("passthrough_label", "KEEP"),
    ("unmapped_label", None),
    ("no_value_label", None),
])
def test_ordered_vocabulary_value_map_branches_pass(tmp_path, target, expected):
    case = _bundle(tmp_path, target=target)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected)


def test_replay_never_uses_vocabulary_discovery(tmp_path, monkeypatch):
    case = _bundle(tmp_path)
    monkeypatch.setattr(vocabularies, "resolve", lambda *_a, **_k: (
        (_ for _ in ()).throw(AssertionError("discovery called"))))
    collector = ReplayEvidenceCollector(case.context)
    collector.collect(case.spec, case.sources, case.cells)
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


@pytest.mark.parametrize("updates,reason", [
    ({"rule_id": "map.inline.v1", "rule_version": "1"},
     "RULE_IDENTITY_MISMATCH"),
    ({"confidence": "0.99"}, "CONFIDENCE_CONTEXT_MISMATCH"),
])
def test_vocabulary_rule_and_confidence_are_independently_verified(
        tmp_path, updates, reason):
    case = _bundle(tmp_path)
    _mutate_provenance(case, "map.exact", **updates)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, reason)


def test_missing_cell_and_mapping_usage_refs_are_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, "map.exact", evidence_refs="")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "VOCABULARY_EVIDENCE_MISSING")

    case = _bundle(tmp_path / "usage")
    usage = _usage(case, "map.exact")
    _replace_usage(case, "map.exact", replace(usage, evidence_ref_ids=()))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "VOCABULARY_EVIDENCE_MISSING")


def test_missing_manifest_vocabulary_binding_is_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    binding = _vocabulary_binding(case, "synthetic/stage5_ordered@1.0.0")
    doc = case.manifest.to_dict()
    doc["evidence"] = [item for item in doc["evidence"]
                       if item["evidence_ref_id"] != binding.evidence_ref_id]
    path = case.case_dir / "missing-binding-manifest.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    result = replay_cell(ROOT, _relative(path), case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "VOCABULARY_EVIDENCE_MISSING")


@pytest.mark.parametrize("change", ["reference", "kind", "path"])
def test_vocabulary_binding_identity_contradictions_fail(tmp_path, change):
    case = _bundle(tmp_path)
    binding = _vocabulary_binding(case, "synthetic/stage5_ordered@1.0.0")
    if change == "reference":
        changed = replace(binding, reference="synthetic/wrong@1.0.0")
    elif change == "kind":
        details = dict(binding.details)
        details["vocabulary_kind"] = "buckets"
        changed = replace(binding, details=tuple(sorted(details.items())))
    else:
        copied = case.case_dir / "same-bytes-different-path.yaml"
        copied.write_bytes(case.ordered_path.read_bytes())
        changed = replace(binding, path=_relative(copied))
    case.manifest = replace(
        case.manifest,
        evidence=tuple(changed if item == binding else item
                       for item in case.manifest.evidence))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VOCABULARY_IDENTITY_MISMATCH")


@pytest.mark.parametrize("drift", [b"\n# drift\n", b"\n: invalid: yaml: [\n"])
def test_vocabulary_byte_drift_fails_without_searching_for_copy(tmp_path, drift):
    case = _bundle(tmp_path)
    case.ordered_path.write_bytes(case.ordered_path.read_bytes() + drift)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VOCABULARY_HASH_MISMATCH")


def test_unrelated_vocabulary_drift_does_not_block_usage_scoped_replay(tmp_path):
    case = _bundle(tmp_path)
    case.buckets_path.write_bytes(case.buckets_path.read_bytes() + b"\n# drift\n")
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


def test_collector_rejects_missing_or_conflicting_cell_identity(tmp_path):
    case = _bundle(tmp_path)
    target = next(cell for cell in case.cells if cell.mapping_id == "map.exact")
    without = replace(target, evidence_refs="")
    with pytest.raises(ReplayOrchestrationError, match="omits runtime"):
        ReplayEvidenceCollector(case.context).collect(
            case.spec, case.sources, (without,))
    conflict = replace(target, evidence_refs="vocab-" + "0" * 64)
    with pytest.raises(ReplayOrchestrationError, match="omits runtime|conflicting"):
        ReplayEvidenceCollector(case.context).collect(
            case.spec, case.sources, (conflict,))


def test_publication_fails_closed_when_registered_binding_disappears(tmp_path):
    case = _bundle(tmp_path)
    collector = ReplayEvidenceCollector(case.context)
    collector.collect(case.spec, case.sources, case.cells)
    collector._evidence.clear()  # simulate a producer losing runtime bindings
    run_dir = case.case_dir / "missing-evidence-publication"
    publication = ReplayPublication(case.context, collector, run_dir)
    with pytest.raises(ReplayOrchestrationError, match="unregistered evidence"):
        publication.publish(
            case.delivered, case.provenance, artifact_id="missing-evidence",
            delivered_name="delivered.csv", provenance_name="provenance.csv")
    assert not (run_dir / "replay_manifest.json").exists()


def test_bucket_parameter_and_downstream_recursive_replay_pass(tmp_path):
    parent = _bundle(tmp_path, target="response_bucket")
    result = replay_cell(ROOT, parent.manifest, parent.request)
    assert (result.status, result.recomputed_value) == (
        ReplayStatus.PASS, "accepted")

    downstream = _bundle(tmp_path / "down", target="downstream_bucket")
    result = replay_cell(ROOT, downstream.manifest, downstream.request)
    assert (result.status, result.recomputed_value) == (
        ReplayStatus.PASS, "down:accepted")


def test_wrong_bucket_vocabulary_identity_fails(tmp_path):
    case = _bundle(tmp_path, target="response_bucket")
    binding = _vocabulary_binding(case, "synthetic/stage5_buckets@1.0.0")
    changed = replace(binding, reference="synthetic/wrong-buckets@1.0.0")
    case.manifest = replace(
        case.manifest,
        evidence=tuple(changed if item == binding else item
                       for item in case.manifest.evidence))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VOCABULARY_IDENTITY_MISMATCH")


def test_registered_rule_receives_only_replayed_input_and_verified_buckets(
        tmp_path, monkeypatch):
    case = _bundle(tmp_path, target="response_bucket")
    seen = []

    def spy(value, *, rules):
        seen.append((value, rules))
        return _bucket(value, rules=rules)

    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.bucket.v1", "1"), RegisteredRule(
            "synthetic.bucket.v1", "1", rule_registry.KIND_FIELD,
            spy, arity=1, params=("rules",)))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS
    assert seen == [("GOOD", yaml.safe_load(case.buckets_path.read_bytes())["terms"])]


def test_historical_bucket_rule_without_cell_evidence_is_not_replayable(tmp_path):
    case = _bundle(tmp_path, target="response_bucket")
    _mutate_provenance(case, "custom.bucket", evidence_refs="")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "VOCABULARY_EVIDENCE_MISSING")


def test_delivered_and_provenance_outputs_cannot_drive_recomputation(tmp_path):
    case = _bundle(tmp_path)
    _mutate_delivery(case, "exact_label", "tampered")
    _mutate_provenance(case, "map.exact", value="tampered")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH", "exact-hit")


def test_stage5_recomputation_signatures_exclude_outputs():
    vocabulary_names = set(inspect.signature(
        recompute_vocabulary_value_map).parameters)
    assert vocabulary_names == {
        "raw_source_value", "mapping_table", "ordered_rules", "on_unmapped"}
    registered_names = set(inspect.signature(recompute_registered_rule).parameters)
    assert registered_names == {"rule", "replayed_inputs", "params"}
    assert not (vocabulary_names | registered_names) & {
        "delivered_value", "provenance_value", "output_value"}


def test_nested_vocabulary_marker_remains_unsupported(tmp_path):
    case = _bundle(tmp_path, target="response_bucket")
    spec_path = ROOT / case.manifest.specifications[0].path
    doc = yaml.safe_load(spec_path.read_text())
    mapping = next(m for m in doc["mappings"]
                   if m["mapping_id"] == "custom.bucket")
    mapping["params"]["rules"] = {
        "nested": {"$vocabulary": "synthetic/stage5_buckets@1.0.0"}}
    spec_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    spec_binding = replace(
        case.manifest.specifications[0], sha256=sha256_file(spec_path))
    case.manifest = replace(case.manifest, specifications=(spec_binding,))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "EXTERNAL_VOCABULARY_UNSUPPORTED")


def test_unsupported_vocabulary_kind_remains_unsupported(tmp_path):
    case = _bundle(tmp_path)
    old = _vocabulary_binding(case, "synthetic/stage5_ordered@1.0.0")
    case.ordered_path.write_text(yaml.safe_dump({
        "vocabulary_id": "synthetic/stage5_ordered",
        "version": "1.0.0",
        "kind": "buckets",
        "description": "same ref, unsupported kind",
        "evidence": "synthetic",
        "terms": {"bucket": ["EXACT"]},
    }, sort_keys=False))
    loaded = vocabularies.load_file(case.ordered_path)
    new = vocabulary_evidence_binding(ROOT, loaded)
    _replace_binding(case, old, new)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "EXTERNAL_VOCABULARY_UNSUPPORTED")
