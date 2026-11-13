"""P1D: explicit ProjectProfile adoption in read-only schema validation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import validate_harmonized_schema as cli


def _schema(path: Path, schema_id: str, artifact: str) -> None:
    path.write_text(yaml.safe_dump({
        "schema_id": schema_id,
        "schema_version": "1.0.0",
        "artifacts": [artifact],
        "column_order": ["record_id", "value"],
        "row_key": ["record_id"],
        "fields": [
            {"name": "record_id", "nullable": False},
            {"name": "value", "nullable": False},
        ],
    }, sort_keys=False))


def _artifact(path: Path) -> None:
    pd.DataFrame({"record_id": ["SYNTH-001"], "value": ["alpha"]}).to_csv(
        path, index=False)


def _profile(root: Path, *, schema: str = "profile_schema.yaml",
             output_directory: str = "profile_outputs",
             provenance_filename: str = "profile_provenance.csv",
             report_filename: str = "profile_report.json",
             extra: dict | None = None) -> Path:
    document = {
        "profile_id": "synthetic-project",
        "profile_format_version": 1,
        "profile_version": "1.0.0",
        "target_schema": schema,
        "tooling": {
            "output_directory": output_directory,
            "provenance_filename": provenance_filename,
            "report_filename": report_filename,
        },
    }
    if extra:
        document.update(extra)
    path = root / "profile.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def test_no_profile_behavior_keeps_existing_schema_and_output_defaults(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    schema = tmp_path / "schema.yaml"
    outputs = tmp_path / "harmonization_outputs"
    outputs.mkdir()
    _schema(schema, "synthetic/no_profile", "delivery.csv")
    _artifact(outputs / "delivery.csv")

    assert cli.main(["--schema", str(schema), "--all", "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_ref"] == "synthetic/no_profile@1.0.0"
    assert [item["artifact"] for item in payload["artifacts"]] == ["delivery.csv"]


def test_profile_supplies_missing_schema_output_provenance_and_report_defaults(
        tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    schema = tmp_path / "profile_schema.yaml"
    outputs = tmp_path / "profile_outputs"
    outputs.mkdir()
    _schema(schema, "synthetic/profile", "profile.csv")
    _artifact(outputs / "profile.csv")
    (outputs / "profile_provenance.csv").write_text(
        "harmonized_field,value_state\n")
    profile = _profile(tmp_path)

    assert cli.main([
        "--project-profile", _relative(profile, tmp_path), "--all", "--json",
    ]) == cli.EXIT_OK
    report = json.loads((outputs / "profile_report.json").read_text())
    assert report["schema_ref"] == "synthetic/profile@1.0.0"
    assert [item["artifact"] for item in report["artifacts"]] == ["profile.csv"]


def test_explicit_cli_values_override_every_adopted_profile_default(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    profile_schema = tmp_path / "profile_schema.yaml"
    profile_outputs = tmp_path / "profile_outputs"
    profile_outputs.mkdir()
    _schema(profile_schema, "synthetic/profile", "profile.csv")
    _artifact(profile_outputs / "profile.csv")
    profile = _profile(tmp_path)

    cli_schema = tmp_path / "cli_schema.yaml"
    cli_outputs = tmp_path / "cli_outputs"
    cli_outputs.mkdir()
    _schema(cli_schema, "synthetic/cli", "cli.csv")
    _artifact(cli_outputs / "cli.csv")
    cli_provenance = cli_outputs / "cli_provenance.csv"
    cli_provenance.write_text("harmonized_field,value_state\n")
    cli_report = cli_outputs / "cli_report.json"

    assert cli.main([
        "--project-profile", _relative(profile, tmp_path),
        "--schema", str(cli_schema),
        "--outputs-dir", str(cli_outputs),
        "--provenance", str(cli_provenance),
        "--out", str(cli_report),
        "--all", "--json",
    ]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_ref"] == "synthetic/cli@1.0.0"
    assert [item["artifact"] for item in payload["artifacts"]] == ["cli.csv"]
    assert cli_report.is_file()
    assert not (profile_outputs / "profile_report.json").exists()


def test_malformed_explicit_profile_fails_clearly(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    schema = tmp_path / "profile_schema.yaml"
    _schema(schema, "synthetic/profile", "delivery.csv")
    profile = _profile(tmp_path, extra={"clinical_values": {}})

    assert cli.main([
        "--project-profile", _relative(profile, tmp_path),
        "--artifact", "unused.csv",
    ]) == cli.EXIT_USAGE
    assert "PROFILE ERROR" in capsys.readouterr().err


def test_absent_optional_profile_defaults_are_deterministic(
        tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    schema = tmp_path / "schema.yaml"
    artifact = tmp_path / "delivery.csv"
    _schema(schema, "synthetic/minimal", artifact.name)
    _artifact(artifact)
    profile = _profile(
        tmp_path,
        schema=schema.name,
        output_directory="",
        provenance_filename="",
        report_filename="",
    )

    assert cli.main([
        "--project-profile", _relative(profile, tmp_path),
        "--artifact", str(artifact),
    ]) == cli.EXIT_OK
    assert not (tmp_path / "profile_report.json").exists()


def test_profile_is_never_discovered_implicitly(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    schema = tmp_path / "profile_schema.yaml"
    _schema(schema, "synthetic/profile", "delivery.csv")
    _profile(tmp_path)

    with pytest.raises(SystemExit) as exc:
        cli.main(["--artifact", "unused.csv"])
    assert exc.value.code == cli.EXIT_USAGE
    assert "required: --schema" in capsys.readouterr().err


def test_strict_replay_sources_remain_profile_independent(project_root):
    replay_paths = [
        project_root / "scripts/lib/replay.py",
        project_root / "scripts/lib/replay_contract.py",
        project_root / "scripts/lib/replay_manifest.py",
        project_root / "scripts/lib/replay_orchestration.py",
        project_root / "scripts/lib/source_ref.py",
    ]
    for path in replay_paths:
        assert "project_profile" not in path.read_text(encoding="utf-8"), path


def test_empty_explicit_profile_is_rejected_even_with_explicit_schema(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    schema = tmp_path / "schema.yaml"
    artifact = tmp_path / "delivery.csv"
    _schema(schema, "synthetic/empty_profile", artifact.name)
    _artifact(artifact)

    assert cli.main([
        "--project-profile", "", "--schema", str(schema),
        "--artifact", str(artifact),
    ]) == cli.EXIT_USAGE
    assert "PROFILE ERROR" in capsys.readouterr().err


def test_empty_explicit_schema_keeps_historical_schema_load_error(capsys):
    assert cli.main([
        "--schema", "", "--artifact", "unused.csv",
    ]) == cli.EXIT_SCHEMA_LOAD
    assert "SCHEMA ERROR" in capsys.readouterr().err


@pytest.mark.parametrize("override", ["schema", "outputs", "provenance", "report"])
def test_cli_overrides_are_independent_of_other_profile_defaults(
        tmp_path, monkeypatch, capsys, override):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    _schema(tmp_path / "profile_schema.yaml", "synthetic/profile", "delivery.csv")
    profile = _profile(tmp_path)
    outputs = tmp_path / "profile_outputs"
    schema_ref = "synthetic/profile@1.0.0"
    args = ["--project-profile", profile.name, "--all", "--json"]

    if override == "schema":
        schema = tmp_path / "cli_schema.yaml"
        _schema(schema, "synthetic/cli", "delivery.csv")
        args += ["--schema", str(schema)]
        schema_ref = "synthetic/cli@1.0.0"
    elif override == "outputs":
        outputs = tmp_path / "cli_outputs"
        args += ["--outputs-dir", str(outputs)]

    outputs.mkdir()
    _artifact(outputs / "delivery.csv")
    provenance = outputs / "profile_provenance.csv"
    if override == "provenance":
        provenance = tmp_path / "cli_provenance.csv"
        args += ["--provenance", str(provenance)]
    provenance.write_text("harmonized_field,value_state\n")

    report = outputs / "profile_report.json"
    if override == "report":
        report = tmp_path / "cli_report.json"
        args += ["--out", str(report)]

    assert cli.main(args) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_ref"] == schema_ref
    assert [item["artifact"] for item in payload["artifacts"]] == ["delivery.csv"]
    assert json.loads(report.read_text()) == payload
    if override in {"outputs", "report"}:
        assert not (tmp_path / "profile_outputs/profile_report.json").exists()


@pytest.mark.parametrize("option", ["--provenance", "--out"])
def test_empty_cli_value_disables_only_its_optional_profile_default(
        tmp_path, monkeypatch, option):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    _schema(tmp_path / "profile_schema.yaml", "synthetic/profile", "delivery.csv")
    profile = _profile(tmp_path)
    outputs = tmp_path / "profile_outputs"
    outputs.mkdir()
    _artifact(outputs / "delivery.csv")
    if option == "--out":
        (outputs / "profile_provenance.csv").write_text(
            "harmonized_field,value_state\n")

    assert cli.main([
        "--project-profile", profile.name, "--all", option, "",
    ]) == cli.EXIT_OK
    assert (outputs / "profile_report.json").exists() == (option != "--out")


def test_profile_references_resolve_from_project_root_not_profile_dir_or_cwd(
        tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    (tmp_path / "schemas").mkdir()
    _schema(tmp_path / "schemas/target.yaml", "synthetic/nested", "delivery.csv")
    outputs = tmp_path / "results"
    outputs.mkdir()
    _artifact(outputs / "delivery.csv")
    (outputs / "profile_provenance.csv").write_text(
        "harmonized_field,value_state\n")
    profile = _profile(
        tmp_path, schema="schemas/target.yaml", output_directory="results")
    (tmp_path / "profiles").mkdir()
    profile.rename(tmp_path / "profiles/current.yaml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert cli.main([
        "--project-profile", "profiles/current.yaml", "--all",
    ]) == cli.EXIT_OK
    payload = json.loads((outputs / "profile_report.json").read_text())
    assert payload["schema_ref"] == "synthetic/nested@1.0.0"
    assert list(elsewhere.iterdir()) == []


def test_profile_without_output_directory_uses_existing_output_default(
        tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    _schema(tmp_path / "profile_schema.yaml", "synthetic/profile", "delivery.csv")
    profile = _profile(tmp_path, output_directory="")
    outputs = tmp_path / "harmonization_outputs"
    outputs.mkdir()
    _artifact(outputs / "delivery.csv")
    (outputs / "profile_provenance.csv").write_text(
        "harmonized_field,value_state\n")

    assert cli.main([
        "--project-profile", profile.name, "--all",
    ]) == cli.EXIT_OK
    assert (outputs / "profile_report.json").is_file()


@pytest.mark.parametrize("reference", ["missing.yaml", "invalid.yaml", "../profile.yaml"])
def test_invalid_explicit_profile_fails_even_when_schema_is_overridden(
        tmp_path, monkeypatch, capsys, reference):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    schema = root / "schema.yaml"
    _schema(schema, "synthetic/invalid_profile", "delivery.csv")
    (root / "invalid.yaml").write_text("profile_id: [\n")

    assert cli.main([
        "--project-profile", reference, "--schema", str(schema),
        "--artifact", "unused.csv",
    ]) == cli.EXIT_USAGE
    assert "PROFILE ERROR" in capsys.readouterr().err


def test_profile_artifact_roles_aliases_and_requiredness_do_not_select_all(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    schema = tmp_path / "profile_schema.yaml"
    _schema(schema, "synthetic/artifacts", "primary.csv")
    document = yaml.safe_load(schema.read_text())
    document["artifacts"].append("secondary.csv")
    schema.write_text(yaml.safe_dump(document, sort_keys=False))
    profile = _profile(
        tmp_path, provenance_filename="", report_filename="",
        extra={"artifacts": {
            "roles": {"primary": "primary.csv"},
            "aliases": {"delivery": "primary.csv"},
            "required": ["primary.csv"],
            "optional": ["secondary.csv"],
        }})
    outputs = tmp_path / "profile_outputs"
    outputs.mkdir()
    for name in document["artifacts"]:
        _artifact(outputs / name)

    assert cli.main([
        "--project-profile", profile.name, "--all", "--json",
    ]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert [item["artifact"] for item in payload["artifacts"]] == document["artifacts"]


def test_profile_defaults_do_not_leak_into_later_no_profile_invocation(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    schema = tmp_path / "profile_schema.yaml"
    _schema(schema, "synthetic/profile", "delivery.csv")
    profile = _profile(tmp_path, provenance_filename="", report_filename="")
    outputs = tmp_path / "profile_outputs"
    outputs.mkdir()
    _artifact(outputs / "delivery.csv")
    assert cli.main([
        "--project-profile", profile.name, "--all",
    ]) == cli.EXIT_OK
    capsys.readouterr()

    def unexpected_profile_load(*args, **kwargs):
        pytest.fail("no-profile invocation attempted to load a profile")

    monkeypatch.setattr(cli, "load_project_profile", unexpected_profile_load)
    outputs.rename(tmp_path / "harmonization_outputs")
    assert cli.main([
        "--schema", str(schema), "--all", "--json",
    ]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert [item["artifact"] for item in payload["artifacts"]] == ["delivery.csv"]
