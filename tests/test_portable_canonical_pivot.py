"""Real pivot execution/publication with fabricated, Cell-v1-compatible inputs."""
from __future__ import annotations

import ast
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import harmonize_spec as cli
from lib import canonical, project_profile
from lib.pivot import pivot_cells_to_wide
from lib.replay import ReplayRequest, replay_cell
from lib.replay_contract import ReplayStatus
from lib.replay_manifest import load as load_manifest
from lib.schema_model import load as load_schema
from lib.source_ref import sha256_file
from lib.spec_model import load as load_spec


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ["trial", "cimac_part_id", "Cimac.id", "Collection_Event"]
COLUMNS = ["reading_text", "trial", "cimac_part_id", "measurement_count",
           "Cimac.id", "Collection_Event", "review_note", "enabled_text"]


def _yaml(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


class _NoHistoricalOrchestrator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "extract_harmonized_clinical":
            raise AssertionError("portable execution imported the historical orchestrator")
        return None


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    assert "extract_harmonized_clinical" not in sys.modules
    monkeypatch.setattr(sys, "meta_path", [_NoHistoricalOrchestrator(), *sys.meta_path])
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    schema_path = _yaml(root / "contracts/lab.yaml", {
        "schema_id": "synthetic/portable-pivot", "schema_version": "1.0.0",
        "row_key": IDENTITY, "column_order": COLUMNS, "artifacts": ["delivery.csv"],
        "fields": [{"name": name, "type": "integer" if name == "measurement_count" else "string",
                    "nullable": name == "review_note"} for name in reversed(COLUMNS)],
    })
    source = root / "inputs/observations.csv"
    source.parent.mkdir()
    pd.DataFrame([
        {"subject": "A", "message": "alpha", "n": 0, "toggle": "no", "note": "candidate A"},
        {"subject": "B", "message": "beta", "n": 5, "toggle": "yes", "note": "candidate B"},
    ]).to_csv(source, index=False)
    anchor_path = root / "inputs/records.csv"
    anchor = pd.DataFrame([
        {"trial": "LAB", "cimac_part_id": "B", "Cimac.id": "S-B", "Collection_Event": "visit"},
        {"trial": "LAB", "cimac_part_id": "A", "Cimac.id": "S-A", "Collection_Event": "visit"},
        {"trial": "OTHER", "cimac_part_id": "Z", "Cimac.id": "S-Z", "Collection_Event": "visit"},
    ])
    anchor.to_csv(anchor_path, index=False)
    thresholds = {"default": .7, "per_field": {"review_note": .7}}
    config_path = _yaml(root / "settings/thresholds.yaml", {
        "config_id": "synthetic/thresholds", "config_version": "1.0.0",
        "confidence_thresholds": thresholds,
    })
    mappings = []
    for target, column in (("reading_text", "message"), ("measurement_count", "n"),
                           ("review_note", "note"), ("enabled_text", "toggle")):
        mapping = {
            "mapping_id": f"direct.{target}", "target": target, "op": "direct",
            "source": "observations", "column": column,
            "rule": {"id": "direct.copy.v1", "version": 1, "review_status": "approved"},
            "confidence": .6 if target == "review_note" else 1.0,
            "provenance": {"record_key_from": "participant", "row_number": "source"},
        }
        if target == "measurement_count":
            mapping["coerce"] = "numeric"
        mappings.append(mapping)
    spec_path = _yaml(root / "declarations/lab.yaml", {
        "spec_id": "synthetic/portable-pivot", "spec_version": "1.0.0",
        "project": "synthetic", "trial": "LAB", "status": "approved",
        "target_schema": {"id": "synthetic/portable-pivot", "version": "1.0.0"},
        "sources": [{"id": "observations", "path": "inputs/observations.csv",
                     "file_type": "csv", "reader": "cidc_csv@v1",
                     "record_keys": ["subject"], "participant_column": "subject",
                     "logical_label": "observations.csv"}],
        "record": {"anchor": "external", "participant_key": "cimac_part_id",
                   "sample_key": "Cimac.id", "collection_event_key": "Collection_Event",
                   "duplicates": {"policy": "error", "by": ["cimac_part_id"]}},
        "mappings": mappings,
    })
    profile = _yaml(root / "settings/project.yaml", {
        "profile_id": "synthetic-portable-pivot", "profile_format_version": 1,
        "profile_version": "1.0.0", "target_schema": "contracts/lab.yaml",
    })
    return SimpleNamespace(root=root, schema_path=schema_path, spec_path=spec_path,
                           source=source, anchor_path=anchor_path, anchor=anchor,
                           thresholds=thresholds, config_path=config_path, profile=profile)


def _args(case, *, run=None, profile=False):
    args = [str(case.spec_path), "--anchor", "inputs/records.csv", "--config",
            "settings/thresholds.yaml", "--preflight", "--quiet"]
    args += ["--project-profile", "settings/project.yaml"] if profile else [
        "--schema", "contracts/lab.yaml"]
    if run:
        args += ["--output", "delivery.csv", "--provenance", "cells.csv",
                 "--replay-run-dir", f"runs/{run}", "--run-id", run,
                 "--artifact-id", "new-project.delivery"]
    else:
        args += ["--output", str(case.root / "ordinary/delivery.csv"),
                 "--provenance", str(case.root / "ordinary/cells.csv")]
    return args


def _publish(case, run="portable-run", *, profile=False):
    assert cli.main(_args(case, run=run, profile=profile)) == cli.EXIT_OK
    return load_manifest(f"runs/{run}/replay_manifest.json", project_root=case.root)


def _request():
    return ReplayRequest(
        artifact_id="new-project.delivery",
        row_key={"trial": "LAB", "cimac_part_id": "B", "Cimac.id": "S-B",
                 "Collection_Event": "visit"}, target_field="reading_text")


def test_real_canonical_execution_normalizes_and_records_suppression(case):
    assert canonical._pivot() is pivot_cells_to_wide
    schema = load_schema(case.schema_path)
    spec = load_spec(case.spec_path, target_schema=schema, require_executable=True)
    result = canonical.harmonize_trial(
        spec, case.root, case.anchor.iloc[:2], schema,
        thresholds=case.thresholds, run_id="ordinary-real-pivot")
    table = canonical.assemble([result], schema)
    assert list(table.columns) == COLUMNS
    assert list(table.cimac_part_id) == ["B", "A"]
    assert list(table.reading_text) == ["beta", "alpha"]
    assert list(table.measurement_count) == [5, 0]
    assert all(type(value) is int for value in table.measurement_count)
    assert table.review_note.isna().all() and result.intermediate_rows == 2
    candidates = [c for c in result.cells if c.harmonized_field == "review_note"
                  and c.extraction_method == "spec:direct"]
    decisions = [c for c in result.cells if c.extraction_method == "value_suppressed"]
    assert len(candidates) == len(decisions) == 2
    assert {c.authoritative_for for c in candidates} == {"(none)"}
    for decision in decisions:
        assert decision.authoritative_for == "" and decision.value is None
        assert decision.rule_id == "value_suppressed_below_threshold.v1"
        assert decision.rule_version == "1" and decision.value_state == "suppressed"
        assert decision.run_id == "ordinary-real-pivot" and decision.spec_ref == spec.ref
        detail = json.loads(decision.derivation)
        assert detail["threshold"] == .7 and detail["confidence"] == .6
        assert detail["predicate"] == "confidence < 0.70"  # compatibility label
        assert detail["config_path"] == "confidence_thresholds.per_field.review_note"
        assert detail["input_rule_id"] == "direct.copy.v1"
        contribution, = json.loads(decision.contributions)
        assert contribution["status"] == "rejected"
        assert contribution["raw_value"] == f"candidate {decision.cimac_part_id}"
        assert decision.source_resolved_path == "inputs/observations.csv"
        assert decision.source_sha256 == sha256_file(case.source)
    assert "extract_harmonized_clinical" not in sys.modules


def test_real_cli_ordinary_publication_bindings_and_supported_replay(case):
    assert cli.main(_args(case)) == cli.EXIT_OK
    ordinary = pd.read_csv(case.root / "ordinary/delivery.csv", keep_default_na=False)
    assert list(ordinary.columns) == COLUMNS and list(ordinary.cimac_part_id) == ["B", "A"]
    assert list(ordinary.measurement_count) == [5, 0]
    ordinary_cells = pd.read_csv(case.root / "ordinary/cells.csv", keep_default_na=False)
    assert (ordinary_cells.extraction_method == "value_suppressed").sum() == 2
    assert not (case.root / "runs").exists()
    manifest = _publish(case)
    delivered = pd.read_csv(case.root / "runs/portable-run/delivery.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(ordinary, delivered)
    assert manifest.target_schema.path == "contracts/lab.yaml"
    assert manifest.target_schema.ref == "synthetic/portable-pivot@1.0.0"
    assert manifest.target_schema.sha256 == sha256_file(case.schema_path)
    assert manifest.specifications[0].path == "declarations/lab.yaml"
    assert manifest.specifications[0].sha256 == sha256_file(case.spec_path)
    assert manifest.specifications[0].ref == "synthetic/portable-pivot@1.0.0"
    assert manifest.delivered_artifacts[0].artifact_id == "new-project.delivery"
    assert manifest.delivered_artifacts[0].sha256 == sha256_file(
        case.root / "runs/portable-run/delivery.csv")
    assert manifest.provenance_artifact.sha256 == sha256_file(
        case.root / "runs/portable-run/cells.csv")
    sources = {source.source_id: source for source in manifest.sources}
    assert sources["observations"].path == "inputs/observations.csv"
    assert sources["observations"].sha256 == sha256_file(case.source)
    assert sources["$anchor"].path == "inputs/records.csv"
    assert sources["$anchor"].reader.reader_id == "external_anchor_csv@v1"
    assert sources["$anchor"].sha256 == sha256_file(case.anchor_path)
    anchor = next(binding for binding in manifest.evidence if binding.kind == "anchor")
    anchor_filter = dict(dict(anchor.details)["filter"])
    assert anchor_filter["column"] == "trial"
    assert dict(anchor_filter["selector"]) == {"kind": "string", "value": "LAB"}
    assert anchor_filter["equality"] == "pandas_scalar_eq@v1"
    assert anchor_filter["null_semantics"] == "parsed_null_never_matches"
    assert dict(anchor.details)["post_filter_row_count"] == 2
    config = next(binding for binding in manifest.evidence if binding.kind == "thresholds")
    assert config.path == "settings/thresholds.yaml"
    assert config.sha256 == sha256_file(case.config_path)
    assert config.reference == "synthetic/thresholds@1.0.0"
    assert all(check.ok for check in manifest.verify_hashes(case.root))
    result = replay_cell(case.root, manifest, _request())
    assert (result.status, result.reason_code) == (ReplayStatus.PASS, "REPLAY_MATCH")
    assert "extract_harmonized_clinical" not in sys.modules


def test_real_pivot_profile_independence_and_binding_equivalence(case, monkeypatch):
    baseline = _publish(case, "cli-schema")
    manifest = _publish(case, "profile-schema", profile=True)
    assert manifest.target_schema == baseline.target_schema
    assert manifest.specifications == baseline.specifications
    assert manifest.sources == baseline.sources and manifest.evidence == baseline.evidence
    assert "settings/project.yaml" not in {binding.path for binding in manifest.files_for()}
    initial = replay_cell(case.root, manifest, _request())
    assert initial.status == ReplayStatus.PASS

    def forbidden_profile(*args, **kwargs):
        pytest.fail("historical replay loaded a current profile")

    monkeypatch.setattr(cli, "load_project_profile", forbidden_profile)
    monkeypatch.setattr(project_profile, "load", forbidden_profile)
    case.profile.write_text("target_schema: elsewhere.yaml\n")
    for remove in (False, True):
        if remove:
            case.profile.unlink()
        assert all(check.ok for check in manifest.verify_hashes(case.root))
        assert replay_cell(case.root, manifest, _request()).to_dict() == initial.to_dict()


@pytest.mark.parametrize("delete", [False, True])
def test_real_publication_bound_schema_still_refuses_drift(case, delete):
    manifest = _publish(case, profile=True)
    if delete:
        case.schema_path.unlink()
    else:
        with case.schema_path.open("a") as handle:
            handle.write("\n# changed bound bytes\n")
    assert not all(check.ok for check in manifest.verify_hashes(case.root))
    result = replay_cell(case.root, manifest, _request())
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, "SCHEMA_HASH_MISMATCH")


def test_shared_pivot_imports_without_historical_orchestrator_in_fresh_process():
    code = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "extract_harmonized_clinical":
            raise AssertionError("historical import attempted")
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
from lib import canonical
from lib.pivot import pivot_cells_to_wide
assert canonical._pivot() is pivot_cells_to_wide
assert "extract_harmonized_clinical" not in sys.modules
'''
    subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "scripts")], check=True)


def test_runtime_import_and_pivot_dependency_boundaries():
    paths = [* (ROOT / "scripts").glob("*.py"), * (ROOT / "scripts/lib").glob("*.py")]
    for path in paths:
        if path.name == "extract_harmonized_clinical.py":
            continue  # user reference is never part of the runtime boundary
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "extract_harmonized_clinical", path
            elif isinstance(node, ast.Import):
                assert all(alias.name != "extract_harmonized_clinical" for alias in node.names), path
    tree = ast.parse((ROOT / "scripts/lib/pivot.py").read_text())
    assert {n.name for n in tree.body if isinstance(n, ast.FunctionDef)} == {
        "threshold_for", "pivot_cells_to_wide"}
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports == {"__future__", "lib.provenance", "lib.schema_model"}
    assert {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names} == {"pandas"}
    assert all(not isinstance(node, (ast.Assign, ast.AnnAssign)) for node in tree.body)


def test_historical_orchestrator_is_absent_from_public_runtime_tree():
    assert not (ROOT / "scripts/extract_harmonized_clinical.py").exists()
