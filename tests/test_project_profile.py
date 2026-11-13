"""Synthetic tests for strict, explicitly loaded project-tooling settings."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from lib.project_profile import (
    PROFILE_FORMAT_VERSION,
    ProjectProfileError,
    load,
    parse,
)
from validate_project_profile import main as validate_main


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = REPOSITORY_ROOT / "tests/fixtures/synthetic_replay"
CURRENT_PROFILE = REPOSITORY_ROOT / "profiles/clinical_harmonization.yaml"


def _document(**updates):
    document = {
        "profile_id": "synthetic-project",
        "profile_format_version": PROFILE_FORMAT_VERSION,
        "profile_version": "1.0.0",
        "target_schema": "schema.yaml",
    }
    document.update(updates)
    return document


def _write_profile(root: Path, document=None, *, text: str | None = None) -> Path:
    profile = root / "profile.yaml"
    if text is None:
        text = yaml.safe_dump(document or _document(), sort_keys=False)
    profile.write_text(text, encoding="utf-8")
    return profile


def _synthetic_root(tmp_path: Path) -> Path:
    shutil.copyfile(SYNTHETIC / "schema.yaml", tmp_path / "schema.yaml")
    return tmp_path


def test_minimal_valid_profile_loads_and_exposes_identity(tmp_path):
    root = _synthetic_root(tmp_path)
    profile_path = _write_profile(root)
    profile = load(profile_path, project_root=root)

    assert profile.profile_id == "synthetic-project"
    assert profile.profile_format_version == 1
    assert profile.profile_version == "1.0.0"
    assert profile.ref == "synthetic-project@1.0.0"
    assert profile.target_schema_identity == "synthetic/device_readings@1.0.0"
    assert profile.validated_specifications == ()


def test_profile_sha256_is_exact_file_bytes_and_deterministic(tmp_path):
    root = _synthetic_root(tmp_path)
    path = _write_profile(root)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    assert load(path, project_root=root).sha256 == expected
    assert load(path, project_root=root).sha256 == expected


def test_profile_sha256_changes_when_only_profile_bytes_change(tmp_path):
    root = _synthetic_root(tmp_path)
    path = _write_profile(root)
    before = load(path, project_root=root).sha256
    path.write_text(path.read_text(encoding="utf-8") + "# audit comment\n",
                    encoding="utf-8")
    assert load(path, project_root=root).sha256 != before


@pytest.mark.parametrize(
    "document, message",
    [
        ({**_document(), "unexpected": True}, "unknown property"),
        ({**_document(), "tooling": {"unexpected": True}}, "unknown property"),
        ({**_document(), "artifacts": {"unexpected": True}}, "unknown property"),
        ({**_document(), "specifications": {"recursive": True}}, "unknown property"),
    ],
)
def test_unknown_top_level_and_nested_keys_are_rejected(document, message):
    with pytest.raises(ProjectProfileError, match=message):
        parse(document)


def test_nonstring_unknown_key_is_rejected_deterministically():
    document = _document()
    document[7] = "unexpected"
    with pytest.raises(ProjectProfileError, match="unknown property"):
        parse(document)


@pytest.mark.parametrize("missing", ["profile_id", "profile_version", "target_schema"])
def test_missing_required_fields_are_rejected(missing):
    document = _document()
    del document[missing]
    with pytest.raises(ProjectProfileError, match=missing):
        parse(document)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("profile_id", 4, "non-empty string"),
        ("profile_format_version", True, "integer"),
        ("profile_version", 1, "non-empty string"),
        ("target_schema", ["schema.yaml"], "non-empty string"),
    ],
)
def test_invalid_field_types_are_rejected(field, value, message):
    with pytest.raises(ProjectProfileError, match=message):
        parse(_document(**{field: value}))


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("profile_format_version", 2, "unsupported version"),
        ("profile_version", "1.0", "MAJOR.MINOR.PATCH"),
        ("profile_version", "v1.0.0", "MAJOR.MINOR.PATCH"),
    ],
)
def test_invalid_versions_are_rejected(field, value, message):
    with pytest.raises(ProjectProfileError, match=message):
        parse(_document(**{field: value}))


@pytest.mark.parametrize(
    "reference",
    [
        "/private/schema.yaml",
        "../schema.yaml",
        "~/schema.yaml",
        "schemas/**/*.yaml",
        "schemas\\schema.yaml",
        "https://example.test/schema.yaml",
    ],
)
def test_malformed_nonportable_or_recursive_references_are_rejected(reference):
    with pytest.raises(ProjectProfileError):
        parse(_document(target_schema=reference))


def test_missing_or_invalid_target_schema_reference_is_rejected(tmp_path):
    path = _write_profile(tmp_path)
    with pytest.raises(ProjectProfileError, match="target schema reference is invalid"):
        load(path, project_root=tmp_path)

    (tmp_path / "schema.yaml").write_text("schema_id: incomplete\n", encoding="utf-8")
    with pytest.raises(ProjectProfileError, match="target schema reference is invalid"):
        load(path, project_root=tmp_path)


def test_output_directory_cannot_resolve_outside_project_root(tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    shutil.copyfile(SYNTHETIC / "schema.yaml", root / "schema.yaml")
    (root / "escaped-output").symlink_to(outside, target_is_directory=True)
    path = _write_profile(root, _document(
        tooling={"output_directory": "escaped-output"}))
    with pytest.raises(ProjectProfileError, match="outside the explicit project root"):
        load(path, project_root=root)


def test_valid_artifact_role_alias_and_expectations_are_schema_backed(tmp_path):
    root = _synthetic_root(tmp_path)
    schema = yaml.safe_load((root / "schema.yaml").read_text(encoding="utf-8"))
    schema["artifacts"] = ["primary.csv", "secondary.csv"]
    (root / "schema.yaml").write_text(
        yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    path = _write_profile(root, _document(artifacts={
        "roles": {"primary_delivery": "primary.csv"},
        "aliases": {"review_copy": "secondary.csv"},
        "required": ["primary.csv"],
        "optional": ["secondary.csv"],
    }))
    profile = load(path, project_root=root)

    assert profile.artifacts.for_role("primary_delivery") == "primary.csv"
    assert profile.artifacts.for_alias("review_copy") == "secondary.csv"


def test_nonexistent_artifact_id_is_rejected(tmp_path):
    root = _synthetic_root(tmp_path)
    schema = yaml.safe_load((root / "schema.yaml").read_text(encoding="utf-8"))
    schema["artifacts"] = ["declared.csv"]
    (root / "schema.yaml").write_text(
        yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    path = _write_profile(root, _document(
        artifacts={"roles": {"primary_delivery": "not-declared.csv"}}))
    with pytest.raises(ProjectProfileError, match="not declared by target schema"):
        load(path, project_root=root)


def test_artifact_links_refuse_schema_without_artifact_ids(tmp_path):
    root = _synthetic_root(tmp_path)
    path = _write_profile(root, _document(
        artifacts={"aliases": {"primary": "output.csv"}}))
    with pytest.raises(ProjectProfileError, match="declares no artifact IDs"):
        load(path, project_root=root)


def test_duplicate_yaml_role_key_is_rejected(tmp_path):
    root = _synthetic_root(tmp_path)
    text = """\
profile_id: synthetic-project
profile_format_version: 1
profile_version: 1.0.0
target_schema: schema.yaml
artifacts:
  roles:
    delivery: one.csv
    delivery: two.csv
"""
    path = _write_profile(root, text=text)
    with pytest.raises(ProjectProfileError, match="duplicate key 'delivery'"):
        load(path, project_root=root)


def test_ambiguous_role_alias_and_expectation_are_rejected():
    with pytest.raises(ProjectProfileError, match="both roles and aliases"):
        parse(_document(artifacts={
            "roles": {"delivery": "one.csv"},
            "aliases": {"delivery": "one.csv"},
        }))
    with pytest.raises(ProjectProfileError, match="both required and optional"):
        parse(_document(artifacts={
            "required": ["one.csv"],
            "optional": ["one.csv"],
        }))


def test_explicit_approved_specification_is_validated_source_free(tmp_path):
    root = _synthetic_root(tmp_path)
    shutil.copyfile(SYNTHETIC / "spec.yaml", root / "spec.yaml")
    path = _write_profile(root, _document(
        specifications={"approved": ["spec.yaml"]}))
    profile = load(path, project_root=root)

    assert profile.validated_specifications == (
        "synthetic/device_readings_direct@1.0.0",
    )
    assert not (root / "tests/fixtures/synthetic_replay/sources/device_readings.csv").exists()


def test_explicit_invalid_specification_is_rejected(tmp_path):
    root = _synthetic_root(tmp_path)
    (root / "bad.yaml").write_text("spec_id: incomplete\n", encoding="utf-8")
    path = _write_profile(root, _document(
        specifications={"approved": ["bad.yaml"]}))
    with pytest.raises(ProjectProfileError, match="approved specification .* is invalid"):
        load(path, project_root=root)


def test_specification_roots_are_nonrecursive(tmp_path):
    root = _synthetic_root(tmp_path)
    specs = root / "specs"
    nested = specs / "nested"
    nested.mkdir(parents=True)
    shutil.copyfile(SYNTHETIC / "spec.yaml", specs / "approved.yaml")
    (nested / "bad.yaml").write_text("not: a supported spec\n", encoding="utf-8")
    path = _write_profile(root, _document(
        specifications={"roots": ["specs"]}))

    profile = load(path, project_root=root)
    assert profile.validated_specifications == (
        "synthetic/device_readings_direct@1.0.0",
    )


@pytest.mark.parametrize("forbidden", [
    "field_mappings", "rule_parameters", "clinical_values", "source_columns",
])
def test_forbidden_semantic_sections_fail_closed_as_unknown(forbidden):
    with pytest.raises(ProjectProfileError, match="unknown property"):
        parse(_document(**{forbidden: {}}))


def test_checked_in_profile_is_public_safe_and_valid():
    payload = CURRENT_PROFILE.read_text(encoding="utf-8")
    lowered = payload.lower()
    private_roots = ("/" + "vf/users/", "/" + "data/nextgen2/")
    assert not any(root in lowered for root in private_roots)
    assert "token" not in lowered
    assert "password" not in lowered

    profile = load(CURRENT_PROFILE, project_root=REPOSITORY_ROOT)
    assert profile.profile_id == "clinical-harmonization-framework"
    assert profile.target_schema_identity == "cimac/clinical_target@1.0.0"
    assert profile.validated_specifications == ()


def test_loading_profile_does_not_mutate_runtime_module_state(tmp_path):
    root = _synthetic_root(tmp_path)
    path = _write_profile(root)
    before = set(sys.modules)
    load(path, project_root=root)
    newly_loaded = set(sys.modules) - before

    assert "lib.spec_executor" not in newly_loaded
    assert "lib.replay" not in newly_loaded
    assert "lib.replay_manifest" not in newly_loaded


def test_existing_execution_and_replay_modules_do_not_import_project_profile():
    runtime_files = [
        REPOSITORY_ROOT / "scripts/lib/spec_executor.py",
        REPOSITORY_ROOT / "scripts/lib/canonical.py",
        REPOSITORY_ROOT / "scripts/lib/replay.py",
        REPOSITORY_ROOT / "scripts/lib/replay_contract.py",
        REPOSITORY_ROOT / "scripts/lib/replay_manifest.py",
        REPOSITORY_ROOT / "scripts/lib/replay_orchestration.py",
        REPOSITORY_ROOT / "scripts/lib/source_ref.py",
        REPOSITORY_ROOT / "scripts/lib/anchor_evidence.py",
        REPOSITORY_ROOT / "scripts/lib/provenance.py",
        REPOSITORY_ROOT / "scripts/lib/contributions.py",
        REPOSITORY_ROOT / "scripts/lib/cidc_io.py",
        REPOSITORY_ROOT / "scripts/lib/xlsx_reader.py",
        REPOSITORY_ROOT / "scripts/lib/trace_io.py",
        REPOSITORY_ROOT / "scripts/lib/confidence_policy.py",
    ]
    # Explicit CLI configuration is tested at its boundary; core authority is
    # independent of both the mutable profile and its tooling-choice helper.
    for path in runtime_files:
        source = path.read_text(encoding="utf-8")
        assert "project_profile" not in source, path
        assert "tooling_defaults" not in source, path


def test_read_only_validation_cli_reports_identity(tmp_path, capsys):
    root = _synthetic_root(tmp_path)
    path = _write_profile(root)
    assert validate_main([str(path), "--project-root", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["profile_id"] == "synthetic-project"
    assert result["target_schema_identity"] == "synthetic/device_readings@1.0.0"


def test_loading_does_not_change_profile_or_referenced_files(tmp_path):
    root = _synthetic_root(tmp_path)
    path = _write_profile(root)
    before = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in (path, root / "schema.yaml")
    }
    load(path, project_root=root)
    after = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in (path, root / "schema.yaml")
    }
    assert after == before
