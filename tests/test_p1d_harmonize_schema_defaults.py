"""Schema-only CLI defaults and historical replay using fabricated inputs.

The pivot seam matches the existing orchestration tests. It does not implement
or validate scientific confidence/suppression policy.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import harmonize_spec as cli
from lib import canonical, project_profile
from lib.replay import ReplayRequest, replay_cell
from lib.replay_contract import ReplayStatus
from lib.replay_manifest import load as load_manifest
from lib.replay_orchestration import is_generated_run_id
from lib.source_ref import sha256_file


FIXTURE = Path(__file__).parent / "fixtures/synthetic_replay"


def _yaml(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def _profile(case, **updates) -> Path:
    document = {
        "profile_id": "synthetic-harmonization",
        "profile_format_version": 1,
        "profile_version": "1.0.0",
        "target_schema": "schemas/profile.yaml",
    }
    document.update(updates)
    return _yaml(case.root / "profiles/current.yaml", document)


def _spec(case, schema="profile", *, path=None, spec_id="synthetic/explicit",
          source="inputs/readings.csv", status="approved") -> Path:
    document = yaml.safe_load((FIXTURE / "spec.yaml").read_text())
    document.update(spec_id=spec_id, status=status)
    document["target_schema"]["id"] = f"synthetic/{schema}"
    document["sources"][0]["path"] = source
    return _yaml(path or case.spec, document)


def _synthetic_pivot(cells, _thresholds, contract):
    first = cells[0]
    row = {name: None for name in contract.columns}
    row.update(device_id=first.cimac_part_id, reading_id=first.Cimac_id,
               location=first.Collection_Event)
    for cell in cells:
        row[cell.harmonized_field] = cell.value
    return pd.DataFrame([row], columns=contract.columns), [], []


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    schemas = {}
    base = yaml.safe_load((FIXTURE / "schema.yaml").read_text())
    base["artifacts"] = ["delivered.csv", "unused-profile.csv"]
    for name, reference in (
        ("default", cli.DEFAULT_SCHEMA),
        ("profile", "schemas/profile.yaml"),
        ("override", "schemas/override.yaml"),
        ("alternate", "schemas/alternate.yaml"),
    ):
        schemas[name] = _yaml(root / reference, {**base, "schema_id": f"synthetic/{name}"})
    inputs = root / "inputs"
    inputs.mkdir()
    (inputs / "readings.csv").write_bytes(
        (FIXTURE / "sources/device_readings.csv").read_bytes())
    anchor = pd.DataFrame([
        {"trial": "synthetic_device_lab", "device_id": "DEVICE-A",
         "reading_id": "READ-001", "location": "bench-west",
         "cimac_part_id": "DEVICE-A", "Cimac.id": "READ-001",
         "Collection_Event": "bench-west"},
        {"trial": "other_synthetic_lab", "device_id": "OTHER",
         "reading_id": "OTHER", "location": "other",
         "cimac_part_id": "OTHER", "Cimac.id": "OTHER", "Collection_Event": "other"},
    ])
    anchor.to_csv(root / cli.DEFAULT_ANCHOR, index=False)
    thresholds = {"default": 0.4}
    config = _yaml(root / cli.DEFAULT_CONFIG, {
        "config_id": "synthetic/confidence", "config_version": "1.0.0",
        "confidence_thresholds": thresholds,
    })
    value = SimpleNamespace(
        root=root, schemas=schemas, spec=root / "explicit.yaml", config=config,
        anchor=root / cli.DEFAULT_ANCHOR, thresholds=thresholds,
        seen=[], assembled=[], anchors=[],
    )
    _spec(value)
    value.profile = _profile(value)
    original_harmonize = canonical.harmonize_trial
    original_assemble = canonical.assemble
    original_anchor = cli._anchor_for

    def recording_harmonize(spec, project_root, anchor, schema, **kwargs):
        value.seen.append({"spec": spec, "schema": schema, "anchor": anchor, **kwargs})
        return original_harmonize(spec, project_root, anchor, schema, **kwargs)

    def recording_assemble(trials, schema):
        value.assembled.append(schema)
        return original_assemble(trials, schema)

    def recording_anchor(spec, path, **kwargs):
        value.anchors.append((path, kwargs["replay_enabled"]))
        return original_anchor(spec, path, **kwargs)

    monkeypatch.setattr(canonical, "_pivot", lambda: _synthetic_pivot)
    monkeypatch.setattr(canonical, "harmonize_trial", recording_harmonize)
    monkeypatch.setattr(canonical, "assemble", recording_assemble)
    monkeypatch.setattr(cli, "_anchor_for", recording_anchor)
    return value


def _ordinary(case):
    return [str(case.spec), "--output", str(case.root / "ordinary/delivered.csv"), "--quiet"]


def _replay_args(case, name, *, profile=True, explicit_ids=True):
    args = [str(case.spec), "--output", "delivered.csv", "--provenance", "provenance.csv",
            "--replay-run-dir", f"runs/{name}", "--quiet"]
    if profile:
        args += ["--project-profile", "profiles/current.yaml"]
    else:
        args += ["--schema", "schemas/profile.yaml"]
    if explicit_ids:
        args += ["--run-id", name, "--artifact-id", "synthetic-delivery"]
    return args


def _published(case, name="profile-run", **kwargs):
    assert cli.main(_replay_args(case, name, **kwargs)) == cli.EXIT_OK
    return load_manifest(f"runs/{name}/replay_manifest.json", project_root=case.root)


def _request(manifest):
    return ReplayRequest(
        artifact_id=manifest.delivered_artifacts[0].artifact_id,
        row_key={"device_id": "DEVICE-A", "reading_id": "READ-001"},
        target_field="reading_text")


def _unused_settings(case):
    unused_anchor = case.root / "unused/anchor.csv"
    unused_anchor.parent.mkdir()
    unused_anchor.write_text("not,the,selected,anchor\n")
    _yaml(case.root / "unused/config.yaml", {
        "config_id": "synthetic/unused", "confidence_thresholds": {"default": 99},
    })
    return _profile(case, tooling={
        "external_anchor": "unused/anchor.csv", "confidence_config": "unused/config.yaml",
        "output_directory": "unused_outputs", "provenance_filename": "unused.csv",
        "report_filename": "unused.json",
    }, artifacts={
        "roles": {"primary_delivery": "unused-profile.csv"},
        "aliases": {"delivery": "unused-profile.csv"},
    })


def test_no_profile_preserves_default_schema_and_never_discovers_settings(
        case, monkeypatch):
    _spec(case, "default")
    (case.root / "profiles/clinical_harmonization.yaml").write_bytes(case.profile.read_bytes())
    monkeypatch.setenv("HARMONIZATION_PROJECT_PROFILE", str(case.profile))

    def unexpected_load(*args, **kwargs):
        pytest.fail("profile loaded without an explicit --project-profile")

    monkeypatch.setattr(cli, "load_project_profile", unexpected_load)
    assert cli.main(_ordinary(case)) == cli.EXIT_OK
    assert case.seen[0]["schema"].source_path == str(case.schemas["default"])
    assert case.assembled[0] is case.seen[0]["schema"]


def test_profile_schema_reaches_existing_canonical_path_only(case):
    _unused_settings(case)
    provenance = case.root / "ordinary/explicit_provenance.csv"
    assert cli.main(_ordinary(case) + [
        "--project-profile", "profiles/current.yaml", "--provenance", str(provenance),
        "--run-id", "ordinary-explicit", "--preflight",
    ]) == cli.EXIT_OK
    assert case.seen[0]["schema"].ref == "synthetic/profile@1.0.0"
    assert case.assembled[0] is case.seen[0]["schema"]
    assert case.seen[0]["thresholds"] == case.thresholds
    assert list(case.seen[0]["anchor"]["trial"]) == ["synthetic_device_lab"]
    assert case.anchors == [(case.anchor, False)]
    assert set(pd.read_csv(provenance)["run_id"]) == {"ordinary-explicit"}
    assert not (case.root / "unused_outputs").exists()
    assert not list(case.root.rglob("replay_manifest.json"))


@pytest.mark.parametrize("with_profile", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_cli_schema_overrides_profile_and_retains_root_joining_from_another_cwd(
        case, monkeypatch, with_profile, absolute):
    _spec(case, "override")
    elsewhere = case.root / "elsewhere"
    elsewhere.mkdir()
    _yaml(elsewhere / "schemas/override.yaml", {"schema_id": "synthetic/wrong_cwd"})
    monkeypatch.chdir(elsewhere)
    schema = str(case.schemas["override"]) if absolute else "schemas/override.yaml"
    args = _ordinary(case) + ["--schema", schema]
    if with_profile:
        args += ["--project-profile", "profiles/current.yaml"]
    assert cli.main(args) == cli.EXIT_OK
    assert case.seen[0]["schema"].ref == "synthetic/override@1.0.0"


def test_nested_profile_is_root_relative_and_ordinary_destinations_stay_cwd_relative(
        case, monkeypatch):
    elsewhere = case.root / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert cli.main([
        str(case.spec), "--project-profile", "profiles/current.yaml", "--output", "out.csv",
        "--provenance", "prov.csv", "--quiet",
    ]) == cli.EXIT_OK
    assert (elsewhere / "out.csv").is_file() and (elsewhere / "prov.csv").is_file()
    assert not (case.root / "out.csv").exists()
    assert case.seen[0]["schema"].source_path == str(case.schemas["profile"])


def test_ordinary_absolute_cli_schema_outside_root_remains_supported(case):
    _spec(case, "override")
    outside = case.root.parent / "outside.yaml"
    outside.write_bytes(case.schemas["override"].read_bytes())
    assert cli.main(_ordinary(case) + [
        "--schema", str(outside), "--project-profile", "profiles/current.yaml",
    ]) == cli.EXIT_OK
    assert case.seen[0]["schema"].source_path == str(outside)


@pytest.mark.parametrize("invalid", [
    "empty", "missing", "malformed", "unknown", "duplicate", "escape", "symlink",
])
def test_invalid_profile_fails_before_execution_side_reads_despite_cli_override(
        case, monkeypatch, capsys, invalid):
    reference = str(case.profile)
    if invalid == "empty":
        reference = ""
    elif invalid == "missing":
        reference = "profiles/missing.yaml"
    elif invalid == "malformed":
        case.profile.write_text("profile_id: [\n")
    elif invalid == "unknown":
        _profile(case, source_columns={})
    elif invalid == "duplicate":
        case.profile.write_text(case.profile.read_text() + "profile_id: duplicate\n")
    elif invalid == "escape":
        _profile(case, target_schema="../outside.yaml")
    else:
        outside = case.root.parent / "outside.yaml"
        outside.write_bytes(case.schemas["profile"].read_bytes())
        case.schemas["profile"].unlink()
        case.schemas["profile"].symlink_to(outside)

    def unexpected_activity(*args, **kwargs):
        pytest.fail("invalid profile reached execution-side activity")

    for name in ("load_schema", "load_spec", "_anchor_for"):
        monkeypatch.setattr(cli, name, unexpected_activity)
    monkeypatch.setattr(cli.spec_preflight, "preflight", unexpected_activity)
    monkeypatch.setattr(canonical, "harmonize_trial", unexpected_activity)
    assert cli.main(_ordinary(case) + [
        "--project-profile", reference, "--schema", "schemas/override.yaml", "--preflight",
    ]) == cli.EXIT_USAGE
    assert "PROFILE ERROR" in capsys.readouterr().err
    assert not (case.root / "ordinary").exists()


@pytest.mark.parametrize("with_profile", [False, True])
def test_explicit_empty_schema_keeps_schema_load_error(case, capsys, with_profile):
    args = _ordinary(case) + ["--schema", ""]
    if with_profile:
        args += ["--project-profile", "profiles/current.yaml"]
    assert cli.main(args) == cli.EXIT_SCHEMA_LOAD
    assert "could not load target schema" in capsys.readouterr().err
    assert not case.seen


@pytest.mark.parametrize("missing", ["spec", "output"])
def test_explicit_spec_and_output_remain_required(case, missing):
    args = ["--project-profile", "profiles/current.yaml"]
    args += ["--output", "out.csv"] if missing == "spec" else [str(case.spec)]
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == cli.EXIT_USAGE


@pytest.mark.parametrize("selection", ["approved", "roots"])
def test_profile_specs_are_only_validated_not_selected_for_execution(case, selection):
    listed = _spec(case, path=case.root / "listed/declared.yaml",
                   spec_id="synthetic/listed", source="missing-listed-input.csv")
    reference = listed if selection == "approved" else listed.parent
    _profile(case, specifications={selection: [reference.relative_to(case.root).as_posix()]})
    assert cli.main(_ordinary(case) + [
        "--project-profile", "profiles/current.yaml",
    ]) == cli.EXIT_OK
    assert [item["spec"].spec_id for item in case.seen] == ["synthetic/explicit"]


def test_invalid_unused_profile_declaration_is_not_skipped(case, capsys):
    _yaml(case.root / "listed/invalid.yaml", {"spec_id": "incomplete"})
    _profile(case, specifications={"approved": ["listed/invalid.yaml"]})
    assert cli.main(_ordinary(case) + [
        "--project-profile", "profiles/current.yaml", "--schema", "schemas/profile.yaml",
    ]) == cli.EXIT_USAGE
    assert "PROFILE ERROR" in capsys.readouterr().err
    assert not case.seen


def test_profile_does_not_approve_an_explicit_draft_spec(case, capsys):
    listed = _spec(case, path=case.root / "listed/approved.yaml", spec_id="synthetic/listed")
    _profile(case, specifications={"approved": [listed.relative_to(case.root).as_posix()]})
    _spec(case, status="draft")
    assert cli.main(_replay_args(case, "draft-run")) == cli.EXIT_USAGE
    assert "not approved" in capsys.readouterr().err
    assert not (case.root / "runs").exists()


def test_profile_state_does_not_leak_into_later_invocations(case, monkeypatch):
    assert cli.main(_ordinary(case) + [
        "--project-profile", "profiles/current.yaml",
    ]) == cli.EXIT_OK
    _spec(case, "default")

    def unexpected_load(*args, **kwargs):
        pytest.fail("profile state leaked into a no-profile invocation")

    monkeypatch.setattr(cli, "load_project_profile", unexpected_load)
    assert cli.main(_ordinary(case)) == cli.EXIT_OK
    assert [item["schema"].schema_id for item in case.seen] == [
        "synthetic/profile", "synthetic/default"]


def test_anchor_and_config_cli_overrides_are_not_profile_defaulted(case):
    _unused_settings(case)
    anchor = case.root / "inputs/explicit_anchor.csv"
    anchor.write_bytes(case.anchor.read_bytes())
    config = _yaml(case.root / "inputs/explicit_config.yaml", {
        "confidence_thresholds": {"default": 0.7},
    })
    assert cli.main(_ordinary(case) + [
        "--project-profile", "profiles/current.yaml", "--anchor", str(anchor),
        "--config", str(config),
    ]) == cli.EXIT_OK
    assert case.anchors == [(anchor, False)]
    assert case.seen[0]["thresholds"] == {"default": 0.7}


@pytest.mark.parametrize("config", ["", "missing.yaml"])
def test_empty_or_missing_cli_config_keeps_existing_optional_behavior(case, config):
    _unused_settings(case)
    assert cli.main(_ordinary(case) + [
        "--project-profile", "profiles/current.yaml", "--config", config,
    ]) == cli.EXIT_OK
    assert case.seen[0]["thresholds"] == {}


def test_empty_cli_anchor_is_not_replaced_by_profile_anchor(case):
    _unused_settings(case)
    assert cli.main(_ordinary(case) + [
        "--project-profile", "profiles/current.yaml", "--anchor", "",
    ]) == cli.EXIT_RUN_FAILED
    assert not case.seen


@pytest.mark.parametrize("enabled", [False, True])
def test_preflight_remains_opt_in_and_retains_failure_exit(case, monkeypatch, enabled):
    calls = []

    def blocked(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(ok=False, spec_ref="synthetic/blocked",
                               findings=[SimpleNamespace(kind="synthetic", message="blocked")])

    monkeypatch.setattr(cli.spec_preflight, "preflight", blocked)
    args = _ordinary(case) + ["--project-profile", "profiles/current.yaml"]
    if enabled:
        args.append("--preflight")
    assert cli.main(args) == (cli.EXIT_PREFLIGHT if enabled else cli.EXIT_OK)
    assert calls == ([True] if enabled else [])
    assert bool(case.seen) == (not enabled)


@pytest.mark.parametrize("quiet", [False, True])
def test_quiet_remains_an_explicit_control(case, capsys, quiet):
    args = _ordinary(case)
    if not quiet:
        args.remove("--quiet")
    assert cli.main(args + ["--project-profile", "profiles/current.yaml"]) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert (output == "") if quiet else ("canonical" in output)


def test_replay_directory_still_requires_explicit_provenance(case):
    assert cli.main(_ordinary(case) + [
        "--project-profile", "profiles/current.yaml", "--replay-run-dir", "runs/no-prov",
    ]) == cli.EXIT_USAGE
    assert not (case.root / "runs").exists()


@pytest.mark.parametrize("explicit_ids", [False, True])
def test_replay_identity_controls_keep_explicit_and_generated_behavior(case, explicit_ids):
    _unused_settings(case)
    manifest = _published(case, explicit_ids=explicit_ids)
    if explicit_ids:
        assert manifest.run_id == "profile-run"
        assert manifest.delivered_artifacts[0].artifact_id == "synthetic-delivery"
    else:
        assert is_generated_run_id(manifest.run_id)
        assert manifest.delivered_artifacts[0].artifact_id == "delivered.csv"


def test_profile_and_cli_schema_publish_the_same_input_bindings(case):
    _unused_settings(case)
    selected = _published(case, "selected-profile")
    baseline = _published(case, "explicit-schema", profile=False)
    assert selected.target_schema == baseline.target_schema
    assert selected.target_schema.path == "schemas/profile.yaml"
    assert selected.target_schema.ref == "synthetic/profile@1.0.0"
    assert selected.target_schema.sha256 == sha256_file(case.schemas["profile"])
    assert selected.specifications == baseline.specifications
    assert selected.specifications[0].path == "explicit.yaml"
    assert selected.specifications[0].ref == "synthetic/explicit@1.0.0"
    assert selected.sources == baseline.sources
    assert selected.evidence == baseline.evidence
    anchor_source = next(source for source in selected.sources if source.source_id == "$anchor")
    assert anchor_source.path == cli.DEFAULT_ANCHOR
    assert anchor_source.sha256 == sha256_file(case.anchor)
    assert anchor_source.reader.reader_id == "external_anchor_csv@v1"
    anchor = next(item for item in selected.evidence if item.kind == "anchor")
    assert dict(dict(anchor.details)["filter"])["column"] == "trial"
    assert dict(anchor.details)["post_filter_row_count"] == 1
    config = next(item for item in selected.evidence if item.kind == "thresholds")
    assert config.path == cli.DEFAULT_CONFIG and config.sha256 == sha256_file(case.config)
    assert config.reference == "synthetic/confidence@1.0.0"


def test_profile_mutation_and_deletion_do_not_change_historical_replay(case, monkeypatch):
    manifest = _published(case)
    assert case.profile.relative_to(case.root).as_posix() not in {
        binding.path for binding in manifest.files_for()}
    request = _request(manifest)
    initial = replay_cell(case.root, manifest, request)
    assert initial.status == ReplayStatus.PASS

    def forbidden_profile_read(*args, **kwargs):
        pytest.fail("historical replay attempted to load current ProjectProfile")

    monkeypatch.setattr(cli, "load_project_profile", forbidden_profile_read)
    monkeypatch.setattr(project_profile, "load", forbidden_profile_read)
    _profile(case, target_schema="schemas/alternate.yaml")
    assert all(check.ok for check in manifest.verify_hashes(case.root))
    assert replay_cell(case.root, manifest, request).to_dict() == initial.to_dict()
    case.profile.unlink()
    assert all(check.ok for check in manifest.verify_hashes(case.root))
    assert replay_cell(case.root, manifest, request).to_dict() == initial.to_dict()


@pytest.mark.parametrize("missing", [False, True])
def test_current_profile_cannot_repair_a_changed_or_missing_bound_schema(
        case, monkeypatch, missing):
    manifest = _published(case)
    replacement = case.root / "schemas/replacement.yaml"
    replacement.write_bytes(case.schemas["profile"].read_bytes())
    _profile(case, target_schema="schemas/replacement.yaml")
    if missing:
        case.schemas["profile"].unlink()
    else:
        with case.schemas["profile"].open("a") as handle:
            handle.write("\n# changed historical bytes\n")

    def forbidden_profile_read(*args, **kwargs):
        pytest.fail("current profile attempted to repair historical schema evidence")

    monkeypatch.setattr(cli, "load_project_profile", forbidden_profile_read)
    monkeypatch.setattr(project_profile, "load", forbidden_profile_read)
    assert not all(check.ok for check in manifest.verify_hashes(case.root))
    result = replay_cell(case.root, manifest, _request(manifest))
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SCHEMA_HASH_MISMATCH")
