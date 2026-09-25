"""Synthetic tests for read-only declarative mapping-identity lint."""
from __future__ import annotations

from pathlib import Path

import yaml

from lint_spec_identities import main as lint_main
from lib.spec_identity_lint import (
    compare_semantic_structure,
    lint_paths,
    lint_spec,
    semantic_structure,
)


def _doc(*, version="1.0.1", mapping_id="reading"):
    return {
        "spec_id": "synthetic/lint",
        "spec_version": version,
        "status": "draft",
        "sources": [{
            "id": "unopened",
            "path": "must/not/be/opened/clinical.csv",
        }],
        "mappings": [{
            "target": "reading",
            "op": "direct",
            "source": "unopened",
            "column": "raw",
            "mapping_id": mapping_id,
        }],
    }


def test_lint_reads_only_explicit_yaml_and_does_not_follow_source(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(_doc(), sort_keys=False))
    result = lint_paths((spec,))
    assert result["identity_ready"] is True
    assert result["mapping_count"] == result["present_id_count"] == 1
    assert not (tmp_path / "must/not/be/opened/clinical.csv").exists()


def test_cli_accepts_explicit_nonrecursive_directory(tmp_path, capsys):
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(_doc(), sort_keys=False))
    assert lint_main(["--spec-dir", str(tmp_path)]) == 0
    assert '"identity_ready": true' in capsys.readouterr().out


def test_lint_reports_missing_invalid_duplicate_branch_and_collision(tmp_path):
    doc = _doc(mapping_id="")
    doc["mappings"] = [
        {"target": "A.B", "op": "direct", "mapping_id": "same"},
        {"target": "A.B", "op": "constant", "mapping_id": "same"},
        {"target": "A_B", "op": "direct", "mapping_id": "Invalid ID"},
        {"target": "other", "op": "direct"},
    ]
    spec = tmp_path / "bad.yaml"
    spec.write_text(yaml.safe_dump(doc, sort_keys=False))
    result = lint_spec(spec)
    assert result["identity_ready"] is False
    assert result["duplicate_ids"] == ["same"]
    assert result["invalid_ids"] == ["Invalid ID"]
    assert result["missing_ids"]
    assert result["duplicate_target_branch_issues"]
    assert result["normalization_collisions"] == [
        {"slug": "a_b", "targets": ["A.B", "A_B"]}]


def test_optional_plan_detects_agreement_and_drift(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(_doc(), sort_keys=False))
    plan = {
        "mappings": [{
            "spec_ref": "synthetic/lint@1.0.0",
            "proposed_spec_version": "1.0.1",
            "target": "reading",
            "operation": "direct",
            "mapping_when": "",
            "proposed_mapping_id": "reading",
        }],
    }
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False))
    passed = lint_paths((spec,), plan_path=plan_path)
    assert passed["identity_ready"] is True
    assert passed["specs"][0]["plan_status"] == "pass"

    changed = _doc(mapping_id="different")
    spec.write_text(yaml.safe_dump(changed, sort_keys=False))
    drift = lint_paths((spec,), plan_path=plan_path)
    assert drift["identity_ready"] is False
    assert drift["specs"][0]["plan_status"] == "drift"


def test_semantic_comparison_removes_only_version_and_mapping_ids(tmp_path):
    old = tmp_path / "old.yaml"
    new = tmp_path / "new.yaml"
    old_doc = _doc(version="1.0.0", mapping_id="")
    new_doc = _doc(version="1.0.1", mapping_id="reading")
    old.write_text(yaml.safe_dump(old_doc, sort_keys=False))
    new.write_text(yaml.safe_dump(new_doc, sort_keys=False))
    assert semantic_structure(old_doc) == semantic_structure(new_doc)
    assert compare_semantic_structure(old, new)["equivalent"] is True

    new_doc["mappings"][0]["column"] = "changed"
    new.write_text(yaml.safe_dump(new_doc, sort_keys=False))
    assert compare_semantic_structure(old, new)["equivalent"] is False
