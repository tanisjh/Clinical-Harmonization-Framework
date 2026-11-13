"""Schema-only ProjectProfile adoption with fabricated spec-validation inputs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import validate_harmonized_schema as artifact_cli
import validate_spec as cli


def _yaml(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def _schema(path: Path, schema_id: str) -> Path:
    return _yaml(path, {
        "schema_id": schema_id,
        "schema_version": "1.0.0",
        "row_key": ["record_id"],
        "fields": [{"name": "record_id"}, {"name": "measurement"}],
    })


def _spec(path: Path, schema_id: str, *, spec_id: str = "synthetic/explicit",
          source: str = "inputs/readings.csv") -> Path:
    return _yaml(path, {
        "spec_id": spec_id,
        "spec_version": "1.0.0",
        "trial": "SYNTHETIC-DEVICE-LAB",
        "status": "approved",
        "target_schema": {"id": schema_id, "version": "1.0.0"},
        "sources": [{
            "id": "readings", "path": source, "file_type": "csv",
            "reader": "cidc_csv@v1", "record_keys": ["record_id"],
        }],
        "record": {
            "anchor": "external", "participant_key": "record_id",
            "duplicates": {"policy": "error", "by": ["record_id"]},
        },
        "mappings": [{
            "mapping_id": "reading.measurement", "target": "measurement",
            "op": "direct", "source": "readings", "column": "measurement",
            "rule": {"id": "direct.copy.v1", "version": 1,
                     "review_status": "approved"},
            "confidence": 1.0,
        }],
    })


def _profile(root: Path, **updates) -> Path:
    document = {
        "profile_id": "synthetic-tooling",
        "profile_format_version": 1,
        "profile_version": "1.0.0",
        "target_schema": "schemas/profile.yaml",
    }
    document.update(updates)
    return _yaml(root / "profiles/current.yaml", document)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    _schema(tmp_path / cli.DEFAULT_SCHEMA, "synthetic/legacy")
    _schema(tmp_path / "schemas/profile.yaml", "synthetic/profile")
    _schema(tmp_path / "schemas/cli.yaml", "synthetic/cli")
    _profile(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "readings.csv").write_text("record_id,measurement\nSYNTH-001,7\n")
    return tmp_path


def _payload(capsys):
    return json.loads(capsys.readouterr().out)


def test_no_profile_keeps_default_schema_without_profile_discovery(
        case, monkeypatch, capsys):
    spec = _spec(case / "explicit.yaml", "synthetic/legacy")

    def unexpected_profile_load(*args, **kwargs):
        pytest.fail("an omitted profile must never be discovered or loaded")

    monkeypatch.setattr(cli, "load_project_profile", unexpected_profile_load)
    assert cli.main([str(spec), "--no-source-check", "--json"]) == cli.EXIT_OK
    assert _payload(capsys)[0]["summary"]["target_schema"] == "synthetic/legacy@1.0.0"


def test_explicit_profile_supplies_schema_only(case, capsys):
    spec = _spec(case / "explicit.yaml", "synthetic/profile")
    profile = _profile(case, tooling={
        "output_directory": "unused_outputs",
        "provenance_filename": "absent.csv",
        "report_filename": "unused.json",
        "external_anchor": "unused_anchor.csv",
        "confidence_config": "unused_config.yaml",
    })
    (case / "unused_anchor.csv").write_text("not a source for validation\n")
    (case / "unused_config.yaml").write_text("[not valid YAML\n")

    assert cli.main([
        str(spec), "--project-profile", str(profile), "--no-source-check", "--json",
    ]) == cli.EXIT_OK
    assert _payload(capsys)[0]["summary"]["target_schema"] == "synthetic/profile@1.0.0"
    assert not (case / "unused_outputs").exists()


@pytest.mark.parametrize("with_profile", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_explicit_cli_schema_wins_and_keeps_project_root_path_semantics(
        case, monkeypatch, capsys, with_profile, absolute):
    spec = _spec(case / "explicit.yaml", "synthetic/cli")
    elsewhere = case / "elsewhere"
    elsewhere.mkdir()
    _schema(elsewhere / "schemas/cli.yaml", "synthetic/wrong_cwd")
    monkeypatch.chdir(elsewhere)
    schema = str(case / "schemas/cli.yaml") if absolute else "schemas/cli.yaml"
    args = [str(spec), "--schema", schema, "--no-source-check", "--json"]
    if with_profile:
        args += ["--project-profile", "profiles/current.yaml"]

    assert cli.main(args) == cli.EXIT_OK
    assert _payload(capsys)[0]["summary"]["target_schema"] == "synthetic/cli@1.0.0"


def test_nested_profile_schema_is_project_relative_from_different_cwd(
        case, monkeypatch, capsys):
    elsewhere = case / "elsewhere"
    elsewhere.mkdir()
    spec = _spec(elsewhere / "explicit.yaml", "synthetic/profile")
    monkeypatch.chdir(elsewhere)

    assert cli.main([
        "explicit.yaml", "--project-profile", "profiles/current.yaml",
        "--no-source-check", "--json",
    ]) == cli.EXIT_OK
    assert _payload(capsys)[0]["summary"]["target_schema"] == "synthetic/profile@1.0.0"


def test_harmonized_validator_keeps_its_distinct_cwd_relative_schema_contract(
        case, monkeypatch, capsys):
    elsewhere = case / "elsewhere"
    elsewhere.mkdir()
    _schema(elsewhere / "schemas/cli.yaml", "synthetic/cwd_contract")
    artifact = elsewhere / "readings.csv"
    artifact.write_text("record_id,measurement\nSYNTH-001,7\n")
    monkeypatch.chdir(elsewhere)

    assert artifact_cli.main([
        "--schema", "schemas/cli.yaml", "--artifact", str(artifact), "--json",
    ]) == artifact_cli.EXIT_OK
    assert _payload(capsys)["schema_ref"] == "synthetic/cwd_contract@1.0.0"


@pytest.mark.parametrize("invalid", [
    "empty", "missing", "malformed", "unknown_key", "duplicate_key",
    "escaping_schema", "outside_profile",
])
def test_invalid_profile_is_not_hidden_by_valid_cli_schema(case, capsys, invalid):
    spec = _spec(case / "explicit.yaml", "synthetic/cli")
    profile = case / "profiles/current.yaml"
    reference = str(profile)
    if invalid == "empty":
        reference = ""
    elif invalid == "missing":
        reference = "profiles/missing.yaml"
    elif invalid == "malformed":
        profile.write_text("profile_id: [\n")
    elif invalid == "unknown_key":
        _profile(case, source_columns={})
    elif invalid == "duplicate_key":
        profile.write_text(profile.read_text() + "profile_id: duplicate\n")
    elif invalid == "escaping_schema":
        _profile(case, target_schema="../outside.yaml")
    else:
        reference = "../outside.yaml"

    assert cli.main([
        str(spec), "--schema", "schemas/cli.yaml", "--project-profile", reference,
        "--no-source-check",
    ]) == cli.EXIT_USAGE
    assert "PROFILE ERROR" in capsys.readouterr().err


def test_empty_profile_without_cli_schema_is_also_a_profile_error(case, capsys):
    spec = _spec(case / "explicit.yaml", "synthetic/legacy")
    assert cli.main([
        str(spec), "--project-profile", "", "--no-source-check",
    ]) == cli.EXIT_USAGE
    assert "PROFILE ERROR" in capsys.readouterr().err


@pytest.mark.parametrize("with_profile", [False, True])
def test_empty_explicit_schema_retains_schema_load_exit(case, capsys, with_profile):
    spec = _spec(case / "explicit.yaml", "synthetic/legacy")
    args = [str(spec), "--schema", "", "--no-source-check"]
    if with_profile:
        args += ["--project-profile", "profiles/current.yaml"]
    assert cli.main(args) == cli.EXIT_SCHEMA_LOAD
    assert "could not load target schema" in capsys.readouterr().err


@pytest.mark.parametrize("selection", ["approved", "roots"])
def test_profile_lists_are_validated_but_not_selected_as_inputs(
        case, monkeypatch, capsys, selection):
    listed = _spec(
        case / "listed/declared.yaml", "synthetic/profile",
        spec_id="synthetic/listed", source="inputs/absent_listed.csv")
    reference = listed if selection == "approved" else listed.parent
    profile = _profile(case, specifications={
        selection: [reference.relative_to(case).as_posix()],
    })
    explicit = _spec(case / "explicit.yaml", "synthetic/profile")
    seen = []
    original_preflight = cli.spec_preflight.preflight

    def recording_preflight(spec, root, **kwargs):
        seen.append(spec.ref)
        return original_preflight(spec, root, **kwargs)

    monkeypatch.setattr(cli.spec_preflight, "preflight", recording_preflight)
    assert cli.main([
        str(explicit), "--project-profile", str(profile), "--no-source-check", "--json",
    ]) == cli.EXIT_OK
    assert seen == ["synthetic/explicit@1.0.0"]
    assert [item["spec"] for item in _payload(capsys)] == [str(explicit)]

    with pytest.raises(SystemExit) as exc:
        cli.main(["--project-profile", str(profile)])
    assert exc.value.code == cli.EXIT_USAGE
    assert "spec" in capsys.readouterr().err


def test_invalid_profile_listed_spec_is_still_strictly_validated(case, capsys):
    _yaml(case / "listed/invalid.yaml", {"spec_id": "incomplete"})
    profile = _profile(case, specifications={"approved": ["listed/invalid.yaml"]})
    explicit = _spec(case / "explicit.yaml", "synthetic/cli")
    assert cli.main([
        str(explicit), "--schema", "schemas/cli.yaml", "--project-profile", str(profile),
        "--no-source-check",
    ]) == cli.EXIT_USAGE
    assert "PROFILE ERROR" in capsys.readouterr().err


def test_schema_spec_mismatch_remains_specification_failure(case, capsys):
    spec = _spec(case / "explicit.yaml", "synthetic/legacy")
    assert cli.main([
        str(spec), "--project-profile", "profiles/current.yaml",
        "--no-source-check", "--json",
    ]) == cli.EXIT_SPEC_LOAD
    payload = _payload(capsys)
    assert payload[0]["stage"] == "parse"
    assert payload[0]["findings"][0]["kind"] == "spec.invalid"


def test_no_source_check_prevents_source_inspection(case, monkeypatch):
    spec = _spec(case / "explicit.yaml", "synthetic/profile")
    (case / "inputs/readings.csv").unlink()

    def unexpected_source_inspection(*args, **kwargs):
        pytest.fail("--no-source-check must not inspect source files")

    monkeypatch.setattr(cli.spec_preflight, "check_columns", unexpected_source_inspection)
    assert cli.main([
        str(spec), "--project-profile", "profiles/current.yaml", "--no-source-check",
    ]) == cli.EXIT_OK


@pytest.mark.parametrize("source_present", [False, True])
def test_default_preflight_still_checks_fabricated_source(
        case, monkeypatch, source_present):
    spec = _spec(case / "explicit.yaml", "synthetic/profile")
    if not source_present:
        (case / "inputs/readings.csv").unlink()
    calls = []
    original_check = cli.spec_preflight.check_columns

    def recording_check(*args, **kwargs):
        calls.append(True)
        return original_check(*args, **kwargs)

    monkeypatch.setattr(cli.spec_preflight, "check_columns", recording_check)
    result = cli.main([str(spec), "--project-profile", "profiles/current.yaml"])
    assert calls == [True]
    assert result == (cli.EXIT_OK if source_present else cli.EXIT_FINDINGS)


def test_profile_defaults_do_not_leak_into_subsequent_no_profile_call(
        case, monkeypatch, capsys):
    spec = _spec(case / "profile_spec.yaml", "synthetic/profile")
    assert cli.main([
        str(spec), "--project-profile", "profiles/current.yaml", "--no-source-check",
    ]) == cli.EXIT_OK
    capsys.readouterr()

    def unexpected_profile_load(*args, **kwargs):
        pytest.fail("profile settings leaked into a later invocation")

    monkeypatch.setattr(cli, "load_project_profile", unexpected_profile_load)
    legacy_spec = _spec(case / "legacy_spec.yaml", "synthetic/legacy")
    assert cli.main([
        str(legacy_spec), "--no-source-check", "--json",
    ]) == cli.EXIT_OK
    assert _payload(capsys)[0]["summary"]["target_schema"] == "synthetic/legacy@1.0.0"


def test_multiple_explicit_spec_results_keep_existing_exit_aggregation(case, capsys):
    good = _spec(case / "good.yaml", "synthetic/legacy")
    missing = case / "missing.yaml"
    assert cli.main([
        str(good), str(missing), "--no-source-check", "--json",
    ]) == cli.EXIT_SPEC_LOAD
    payload = _payload(capsys)
    assert [item["spec"] for item in payload] == [str(good), str(missing)]
    assert [item["ok"] for item in payload] == [True, False]
