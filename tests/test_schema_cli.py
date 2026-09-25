"""The standalone CLI, the pipeline gate, and the CIMAC artifacts themselves.

Three concerns:

  * Exit codes. A caller must be able to tell a broken schema from a broken
    artifact from bad data without parsing text, so each failure kind gets its
    own code and each is provoked here.
  * The delivered CIMAC artifacts. All three must validate with zero errors —
    including 10013 and 14C0059G, which had no automated value validation at all
    before Phase 2a.
  * Manifest integration. A run must record which target contract validated it.

Run:  python -m pytest tests/test_schema_cli.py -v
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

import validate_harmonized_schema as cli
from lib import run_manifest, schema_validator as sv
from lib.schema_model import ERROR, load

SCHEMA_REL = "schemas/cimac_clinical_v1.yaml"
PRODUCTION_ARTIFACTS = (
    "harmonized_11trials.csv",
    "harmonized_9trials_gold.csv",
    "harmonized_9trials_reproduced.csv",
)
PORTABLE_ARTIFACTS = ("synthetic_primary.csv", "synthetic_secondary.csv")


@pytest.fixture(scope="module")
def portable_cli_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("portable-schema-cli")
    schema = root / "schema.yaml"
    outputs = root / "outputs"
    outputs.mkdir()
    schema.write_text(yaml.safe_dump({
        "schema_id": "synthetic/portable_target",
        "schema_version": "1.0.0",
        "artifacts": list(PORTABLE_ARTIFACTS),
        "column_order": ["record_id", "measurement_a", "category_x"],
        "row_key": ["record_id"],
        "fields": [
            {"name": "record_id", "nullable": False,
             "pattern": "^SYNTH-[0-9]{3}$"},
            {"name": "measurement_a", "type": "number", "nullable": False,
             "bounds": {"min": 0, "max": 200, "severity": "error"}},
            {"name": "category_x", "vocabulary": {
                "values": ["alpha", "beta"], "canonical": ["alpha", "beta"],
                "unknown_severity": "error"}},
        ],
    }, sort_keys=False))
    frame = pd.DataFrame({
        "record_id": ["SYNTH-001", "SYNTH-002"],
        "measurement_a": [123.45, 67.0],
        "category_x": ["alpha", "beta"],
    })
    for name in PORTABLE_ARTIFACTS:
        frame.to_csv(outputs / name, index=False)
    return schema, outputs


@pytest.fixture(scope="module")
def schema_path(portable_cli_case):
    return str(portable_cli_case[0])


@pytest.fixture(scope="module")
def portable_outputs_dir(portable_cli_case):
    return portable_cli_case[1]


@pytest.fixture(scope="module")
def production_schema(project_root):
    return load(project_root / SCHEMA_REL)


def _all_args(schema_path, outputs):
    return ["--schema", schema_path, "--all", "--outputs-dir", str(outputs)]


# ── exit codes ──────────────────────────────────────────────────────────────

def test_exit_zero_when_all_artifacts_validate(
        schema_path, portable_outputs_dir, capsys):
    assert cli.main(_all_args(schema_path, portable_outputs_dir)) == cli.EXIT_OK
    assert "OVERALL  PASS" in capsys.readouterr().out


def test_exit_usage_when_no_artifact_selected(schema_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--schema", schema_path])
    assert exc.value.code == cli.EXIT_USAGE


def test_exit_schema_load_on_bad_schema(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_id: x\nschema_version: nope\nfields: [{name: a}]\n")
    assert cli.main(["--schema", str(bad), "--artifact", "whatever.csv"]) \
        == cli.EXIT_SCHEMA_LOAD
    assert "SCHEMA ERROR" in capsys.readouterr().err


def test_exit_artifact_load_on_missing_artifact(schema_path, tmp_path, capsys):
    assert cli.main(["--schema", schema_path,
                     "--artifact", str(tmp_path / "absent.csv")]) \
        == cli.EXIT_ARTIFACT_LOAD
    assert "ARTIFACT ERROR" in capsys.readouterr().err


def test_exit_structural_on_a_shape_violation(schema_path, tmp_path):
    """A truncated artifact fails structurally, not merely on content."""
    broken = tmp_path / "broken.csv"
    broken.write_text("record_id,unexpected\nSYNTH-001,alpha\n")
    assert cli.main(["--schema", schema_path, "--artifact", str(broken)]) \
        == cli.EXIT_STRUCTURAL_ERRORS


def test_exit_content_on_a_value_violation(
        schema_path, portable_outputs_dir, tmp_path):
    """Right shape, wrong value: content code, not the structural one."""
    src = pd.read_csv(portable_outputs_dir / PORTABLE_ARTIFACTS[0],
                      dtype=str, keep_default_na=False)
    src.loc[0, "measurement_a"] = "999"
    out = tmp_path / "tampered.csv"
    src.to_csv(out, index=False)
    assert cli.main(["--schema", schema_path,
                     "--artifact", str(out)]) == cli.EXIT_CONTENT_ERRORS


def test_structural_code_outranks_content_code(schema_path, tmp_path):
    broken = tmp_path / "b.csv"
    broken.write_text("record_id\nSYNTH-001\n")
    assert cli.main(["--schema", schema_path, "--artifact", str(broken)]) \
        == cli.EXIT_STRUCTURAL_ERRORS


# ── output modes ────────────────────────────────────────────────────────────

def test_json_output_is_valid_and_deterministic(
        schema_path, portable_outputs_dir, capsys):
    args = _all_args(schema_path, portable_outputs_dir) + ["--json"]
    cli.main(args)
    first = capsys.readouterr().out
    cli.main(args)
    second = capsys.readouterr().out
    assert first == second
    payload = json.loads(first)
    assert payload["schema_id"] == "synthetic/portable_target"
    assert payload["schema_version"] == "1.0.0"
    assert len(payload["schema_sha256"]) == 64
    assert [a["artifact"] for a in payload["artifacts"]] == \
        list(PORTABLE_ARTIFACTS)
    assert payload["verdict"] == "pass"
    assert payload["counts"]["error"] == 0


def test_json_findings_are_grouped_and_ordered(
        schema_path, portable_outputs_dir, capsys):
    cli.main(_all_args(schema_path, portable_outputs_dir) + ["--json"])
    payload = json.loads(capsys.readouterr().out)
    for art in payload["artifacts"]:
        sev = [f["severity"] for f in art["findings"]]
        rank = {"error": 0, "warning": 1, "info": 2}
        assert sev == sorted(sev, key=lambda s: rank[s])
        for f in art["findings"]:
            assert {"severity", "rule", "field", "message", "count",
                    "examples"} <= set(f)


def test_out_file_is_written(schema_path, portable_outputs_dir, tmp_path):
    out = tmp_path / "report.json"
    cli.main(_all_args(schema_path, portable_outputs_dir)
             + ["--out", str(out)])
    assert json.loads(out.read_text())["verdict"] == "pass"


def test_quiet_suppresses_examples(schema_path, portable_outputs_dir,
                                   tmp_path, capsys):
    bad = pd.read_csv(portable_outputs_dir / PORTABLE_ARTIFACTS[0], dtype=str)
    bad.loc[0, "measurement_a"] = "999"
    artifact = tmp_path / "invalid.csv"
    bad.to_csv(artifact, index=False)
    args = ["--schema", schema_path, "--artifact", str(artifact)]
    cli.main(args + ["--quiet"])
    quiet = capsys.readouterr().out
    cli.main(args)
    loud = capsys.readouterr().out
    assert len(quiet) < len(loud)


# ── the delivered CIMAC artifacts ───────────────────────────────────────────

@pytest.mark.production_integration
@pytest.mark.parametrize("name", PRODUCTION_ARTIFACTS)
def test_delivered_artifact_has_no_schema_errors(
        production_schema, production_outputs_dir, name):
    df = sv.load_artifact(production_outputs_dir / name)
    result = sv.validate(production_schema, df, artifact=name)
    errors = [f for f in result.findings if f.severity == ERROR]
    assert not errors, [f"{f.rule} {f.field}: {f.message}" for f in errors]


@pytest.mark.production_integration
@pytest.mark.parametrize("name", PRODUCTION_ARTIFACTS)
def test_delivered_artifact_uses_only_empty_string_for_null(
        production_outputs_dir, name):
    """Regression guard for literal null tokens found during earlier review."""
    df = sv.load_artifact(production_outputs_dir / name)
    for col in df.columns:
        offenders = df[col][df[col].str.strip().str.lower().isin(sv.NULLISH)
                            & (df[col] != "")]
        assert offenders.empty, (
            f"{name}.{col} has {len(offenders)} nonempty null encodings; "
            f"the only allowed null encoding is an empty field")


@pytest.mark.production_integration
def test_the_two_previously_unvalidated_trials_are_now_covered(
        production_schema, production_outputs_dir, production_expectations):
    """10013 and 14C0059G are absent from the reference template, so extrinsic
    validation never saw them. Intrinsic validation must."""
    df = sv.load_artifact(production_outputs_dir / "harmonized_11trials.csv")
    expected_counts = production_expectations.get("trial_row_counts", {})
    assert set(expected_counts) >= {"10013", "14C0059G"}
    for trial in ("10013", "14C0059G"):
        sub = df[df["trial"] == trial].reset_index(drop=True)
        assert len(sub) == expected_counts[trial]
        result = sv.validate(
            production_schema, sub, artifact="harmonized_11trials.csv")
        errors = [f for f in result.findings if f.severity == ERROR]
        assert not errors, [f"{trial}: {f.rule} {f.field}" for f in errors]


@pytest.mark.production_integration
def test_value_states_declared_by_the_schema_match_provenance(
        production_schema, production_outputs_dir):
    prov = pd.read_csv(
        production_outputs_dir / "provenance_long.csv", low_memory=False)
    findings = sv.check_value_states(production_schema, prov)
    assert not findings, [f"{f.rule} {f.field}: count={f.count}" for f in findings]


@pytest.mark.production_integration
def test_open_questions_never_raise_severity(
        production_schema, production_outputs_dir):
    df = sv.load_artifact(production_outputs_dir / "harmonized_11trials.csv")
    result = sv.validate(
        production_schema, df, artifact="harmonized_11trials.csv")
    for f in result.findings:
        if f.rule == "schema.open_question":
            assert f.severity == "info"
    assert result.verdict == "pass"


# ── run manifest integration ────────────────────────────────────────────────

@pytest.mark.production_integration
def test_manifest_records_schema_identity_and_verdict(
        production_outputs_dir, project_root):
    m = run_manifest.load(production_outputs_dir)
    assert m["schema_version"] == run_manifest.SCHEMA_VERSION == 2
    ts = m["target_schema"]
    assert ts["id"] == "cimac/clinical_target"
    assert ts["version"] == "1.0.0"
    assert ts["sha256"] == load(project_root / SCHEMA_REL).sha256
    assert m["schema_validation_verdict"] == "pass"
    assert m["schema_validation_counts"]["error"] == 0
    assert set(m["schema_validation_counts"]) == {"error", "warning", "info"}


@pytest.mark.production_integration
def test_manifest_still_carries_the_pre_existing_verdicts(production_outputs_dir):
    """Schema validation must not displace the gates that already existed."""
    m = run_manifest.load(production_outputs_dir)
    assert m["validation_verdict"] == "pass"
    assert m["provenance_gate_verdict"] == "pass"


def test_stamping_preserves_other_manifest_keys(tmp_path):
    manifest = {"schema_version": 2, "run_id": "x", "provenance_gate_verdict": "pass",
                "validation_verdict": "pass", "inputs": [{"a": 1}]}
    run_manifest.write(manifest, tmp_path)
    run_manifest.stamp_schema_validation(
        tmp_path, schema_id="acme/other", schema_version="9.9.9",
        schema_sha256="f" * 64, verdict="fail",
        counts={"error": 2, "warning": 1, "info": 0})
    after = run_manifest.load(tmp_path)
    assert after["run_id"] == "x"
    assert after["provenance_gate_verdict"] == "pass"
    assert after["inputs"] == [{"a": 1}]
    assert after["target_schema"]["id"] == "acme/other"
    assert after["schema_validation_verdict"] == "fail"
    assert after["schema_validation_counts"]["error"] == 2


def test_stamping_without_a_manifest_is_a_no_op(tmp_path):
    assert run_manifest.stamp_schema_validation(
        tmp_path, schema_id="a", schema_version="1.0.0", schema_sha256="",
        verdict="pass", counts={}) is None


@pytest.mark.production_integration
def test_pipeline_declares_the_schema_gate(project_root):
    runner = (project_root / "scripts" / "run_full_harmonization.sh").read_text()
    assert "validate_harmonized_schema.py" in runner
    assert "--stamp" in runner
    assert "audit_migrated_provenance.py" in runner
    assert runner.index("validate_harmonized_schema.py") < \
        runner.index("audit_migrated_provenance.py"), (
            "the schema gate must stamp the manifest before the provenance "
            "audit rewrites it")


@pytest.mark.production_integration
def test_pipeline_writes_a_machine_readable_report(production_outputs_dir):
    report = production_outputs_dir / "schema_validation_report.json"
    assert report.is_file(), "pipeline step 10 should leave a JSON report"
    payload = json.loads(report.read_text())
    assert payload["verdict"] == "pass"
    assert payload["counts"]["error"] == 0
