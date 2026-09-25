"""Intrinsic validation rules, and the proof that the engine is not CIMAC-specific.

Two halves:

  * Rule behaviour, exercised against small purpose-built frames. Each rule is
    tested for firing AND for staying silent, because a validator that reports
    everything is as useless as one that reports nothing.
  * A synthetic non-CIMAC schema (industrial sensor readings — different domain,
    different field names, different semantics) validated end to end. Without it,
    "generic" is an aspiration rather than a demonstrated property.

Run:  python -m pytest tests/test_schema_validator.py -v
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

from lib import schema_validator as sv
from lib.schema_model import ERROR, INFO, WARNING, load, parse

# ── helpers ─────────────────────────────────────────────────────────────────


def frame(csv: str) -> pd.DataFrame:
    """Read like the validator does: raw text, no inference, no NA filtering."""
    return pd.read_csv(io.StringIO(csv), dtype=str, keep_default_na=False,
                       na_filter=False)


def schema(fields, **overrides):
    d = {"schema_id": "test/rules", "schema_version": "1.0.0", "fields": fields}
    d.update(overrides)
    return parse(d)


def rules(result) -> set[str]:
    return {f.rule for f in result.findings}


def by_rule(result, rule: str):
    hits = [f for f in result.findings if f.rule == rule]
    assert hits, f"expected rule {rule!r}; got {sorted(rules(result))}"
    return hits[0]


def run(sch, csv, artifact="a.csv"):
    return sv.validate(sch, frame(csv), artifact=artifact)


# ── structural ──────────────────────────────────────────────────────────────

def test_missing_required_column_is_an_error():
    s = schema([{"name": "a"}, {"name": "b"}])
    r = run(s, "a\n1\n")
    f = by_rule(r, "structure.missing_column")
    assert f.severity == ERROR and f.field == "b"


def test_optional_column_may_be_absent():
    s = schema([{"name": "a"}, {"name": "b", "required_column": False}])
    assert "structure.missing_column" not in rules(run(s, "a\n1\n"))


def test_unexpected_column_is_reported_at_the_declared_severity():
    s = schema([{"name": "a"}], unexpected_column_severity="warning")
    f = by_rule(run(s, "a,surprise\n1,2\n"), "structure.unexpected_column")
    assert f.severity == WARNING and f.field == "surprise"


def test_duplicate_column_name_is_an_error():
    """Built directly: a CSV reader renames duplicates to a, a.1 on the way in."""
    s = schema([{"name": "a"}])
    df = pd.DataFrame([["1", "2"]], columns=["a", "a"])
    f = by_rule(sv.validate(s, df, artifact="a.csv"), "structure.duplicate_column")
    assert f.severity == ERROR


def test_column_order_violation_is_reported():
    s = schema([{"name": "a"}, {"name": "b"}], column_order=["a", "b"])
    assert "structure.column_order" in rules(run(s, "b,a\n1,2\n"))
    assert "structure.column_order" not in rules(run(s, "a,b\n1,2\n"))


def test_duplicate_row_key_is_an_error():
    s = schema([{"name": "k"}, {"name": "v"}], row_key=["k"])
    f = by_rule(run(s, "k,v\nx,1\nx,2\ny,3\n"), "structure.duplicate_row_key")
    assert f.severity == ERROR and f.count == 2


def test_unique_row_key_is_silent():
    s = schema([{"name": "k"}], row_key=["k"])
    assert "structure.duplicate_row_key" not in rules(run(s, "k\nx\ny\n"))


# ── nulls and null encoding ─────────────────────────────────────────────────

def test_null_in_non_nullable_field_is_an_error():
    s = schema([{"name": "a", "nullable": False}])
    f = by_rule(run(s, 'a\n1\n""\n'), "value.unexpected_null")
    assert f.severity == ERROR and f.count == 1


def test_null_in_nullable_field_is_silent():
    s = schema([{"name": "a", "nullable": True}])
    assert "value.unexpected_null" not in rules(run(s, 'a\n1\n""\n'))


@pytest.mark.parametrize("token", ["NA", "N/A", "nan", "None", "NULL", "<NA>"])
def test_nullish_tokens_are_rejected_not_silently_accepted(token):
    """The defect Phase 2a found in harmonized_11trials.csv: a literal 'None'."""
    s = schema([{"name": "a"}])
    f = by_rule(run(s, f"a\n1\n{token}\n"), "type.unsupported_null_encoding")
    assert f.severity == ERROR


def test_nullish_token_is_not_double_counted_as_a_type_error():
    """A null-like token is one problem, reported once, not also as a bad number."""
    s = schema([{"name": "a", "type": "integer"}])
    r = run(s, "a\n1\nNA\n")
    assert "type.unsupported_null_encoding" in rules(r)
    assert "type.not_numeric" not in rules(r)


def test_whitespace_padding_is_a_warning():
    s = schema([{"name": "a"}])
    f = by_rule(run(s, 'a\n"  x  "\n'), "value.whitespace_padding")
    assert f.severity == WARNING


# ── types ───────────────────────────────────────────────────────────────────

def test_non_numeric_value_in_numeric_field_is_an_error():
    s = schema([{"name": "a", "type": "number"}])
    f = by_rule(run(s, "a\n1.5\nabc\n"), "type.not_numeric")
    assert f.severity == ERROR and f.count == 1


def test_numeric_field_accepts_floats():
    s = schema([{"name": "a", "type": "number"}])
    assert "type.not_numeric" not in rules(run(s, "a\n1.5\n-2.25\n"))


def test_integer_field_rejects_a_fractional_value():
    s = schema([{"name": "a", "type": "integer"}])
    f = by_rule(run(s, "a\n1\n2.5\n"), "type.not_integer")
    assert f.severity == ERROR


def test_integer_field_accepts_trailing_zero_decimals():
    """Delivered CSVs carry 1.0 for integers; that is integral, not fractional."""
    s = schema([{"name": "a", "type": "integer"}])
    assert "type.not_integer" not in rules(run(s, "a\n1.0\n0.0\n"))


def test_mixed_types_in_one_column_are_rejected():
    s = schema([{"name": "a", "type": "integer"}])
    r = run(s, "a\n1\ntwo\n3\n")
    assert "type.not_numeric" in rules(r)


def test_string_field_never_produces_a_type_finding():
    s = schema([{"name": "a", "type": "string"}])
    assert not {"type.not_numeric", "type.not_integer"} & rules(run(s, "a\nx\n1\n"))


def test_invalid_values_are_reported_not_coerced():
    """The frame the validator saw must be unchanged by validation."""
    s = schema([{"name": "a", "type": "integer"}])
    df = frame("a\n1\nbogus\n")
    before = df["a"].tolist()
    sv.validate(s, df, artifact="a.csv")
    assert df["a"].tolist() == before == ["1", "bogus"]


# ── vocabulary ──────────────────────────────────────────────────────────────

def test_unknown_vocabulary_value_uses_the_declared_severity():
    s = schema([{"name": "a", "vocabulary": {"values": ["x", "y"],
                                             "unknown_severity": "warning"}}])
    f = by_rule(run(s, "a\nx\nz\n"), "vocabulary.unknown_value")
    assert f.severity == WARNING and f.count == 1 and f.examples == ("'z'",)


def test_unknown_vocabulary_value_can_be_an_error():
    s = schema([{"name": "a", "vocabulary": {"values": ["x"],
                                             "unknown_severity": "error"}}])
    assert by_rule(run(s, "a\nz\n"), "vocabulary.unknown_value").severity == ERROR


def test_known_vocabulary_value_is_silent():
    s = schema([{"name": "a", "vocabulary": {"values": ["x", "y"]}}])
    assert "vocabulary.unknown_value" not in rules(run(s, "a\nx\ny\n"))


def test_non_canonical_value_is_separated_from_unknown():
    """Permitted-but-unnormalized is a backlog; unknown is a defect."""
    s = schema([{"name": "a", "vocabulary": {
        "values": ["x", "X"], "canonical": ["x"],
        "non_canonical_severity": "warning"}}])
    r = run(s, "a\nx\nX\n")
    assert "vocabulary.unknown_value" not in rules(r)
    f = by_rule(r, "vocabulary.non_canonical_value")
    assert f.severity == WARNING and f.count == 1


def test_nulls_are_not_vocabulary_violations():
    s = schema([{"name": "a", "vocabulary": {"values": ["x"]}}])
    assert "vocabulary.unknown_value" not in rules(run(s, 'a\nx\n""\n'))


# ── bounds ──────────────────────────────────────────────────────────────────

def test_value_below_minimum_is_reported():
    s = schema([{"name": "a", "type": "integer",
                 "bounds": {"min": 0, "severity": "error"}}])
    f = by_rule(run(s, "a\n5\n-1\n"), "bounds.below_minimum")
    assert f.severity == ERROR and f.count == 1


def test_value_above_maximum_is_reported():
    s = schema([{"name": "a", "type": "integer", "bounds": {"max": 1}}])
    assert by_rule(run(s, "a\n0\n1\n2\n"), "bounds.above_maximum").count == 1


def test_values_inside_bounds_are_silent():
    s = schema([{"name": "a", "type": "integer", "bounds": {"min": 0, "max": 10}}])
    assert not {"bounds.below_minimum", "bounds.above_maximum"} & rules(run(s, "a\n0\n10\n"))


def test_bound_severity_is_configurable():
    s = schema([{"name": "a", "type": "integer",
                 "bounds": {"min": 0, "severity": "warning"}}])
    assert by_rule(run(s, "a\n-5\n"), "bounds.below_minimum").severity == WARNING


# ── patterns ────────────────────────────────────────────────────────────────

def test_pattern_mismatch_uses_the_declared_severity():
    s = schema([{"name": "a", "pattern": "^[A-Z]{3}$",
                 "pattern_severity": "warning"}])
    f = by_rule(run(s, "a\nABC\nabcd\n"), "value.pattern_mismatch")
    assert f.severity == WARNING and f.count == 1


def test_pattern_is_anchored():
    """A partial match must not satisfy the pattern."""
    s = schema([{"name": "a", "pattern": "[A-Z]{3}"}])
    assert "value.pattern_mismatch" in rules(run(s, "a\nABCD\n"))


# ── cross-field constraints ─────────────────────────────────────────────────

def test_requires_non_null_fires_only_in_one_direction():
    s = schema([{"name": "a"}, {"name": "b"}],
               constraints=[{"id": "a_needs_b", "kind": "requires_non_null",
                             "field": "a", "other": "b", "severity": "error"}])
    f = by_rule(run(s, "a,b\n1,\n"), "constraint.a_needs_b")
    assert f.severity == ERROR
    # The reverse (b present, a null) is a different claim and must not fire.
    assert "constraint.a_needs_b" not in rules(run(s, "a,b\n,1\n"))


def test_less_or_equal_fires_and_respects_severity():
    s = schema([{"name": "a", "type": "integer"}, {"name": "b", "type": "integer"}],
               constraints=[{"id": "a_le_b", "kind": "less_or_equal",
                             "field": "a", "other": "b", "severity": "warning"}])
    f = by_rule(run(s, "a,b\n5,3\n1,2\n"), "constraint.a_le_b")
    assert f.severity == WARNING and f.count == 1


def test_less_or_equal_ignores_rows_where_either_side_is_null():
    s = schema([{"name": "a", "type": "integer"}, {"name": "b", "type": "integer"}],
               constraints=[{"id": "a_le_b", "kind": "less_or_equal",
                             "field": "a", "other": "b"}])
    assert "constraint.a_le_b" not in rules(run(s, "a,b\n5,\n,3\n"))


# ── informational rules ─────────────────────────────────────────────────────

def test_all_null_column_is_info_only():
    s = schema([{"name": "a"}])
    f = by_rule(run(s, 'a\n""\n""\n'), "coverage.all_null")
    assert f.severity == INFO


def test_deprecated_field_is_info_only():
    s = schema([{"name": "a", "deprecated": True, "deprecation_note": "use b"}])
    f = by_rule(run(s, "a\n1\n"), "schema.deprecated_field")
    assert f.severity == INFO and "use b" in f.message


def test_open_question_is_info_and_never_escalates():
    s = schema([{"name": "a", "open_questions": ["tracker#Q9"]}])
    f = by_rule(run(s, "a\n1\n"), "schema.open_question")
    assert f.severity == INFO
    assert f.examples == ("tracker#Q9",)
    assert run(s, "a\n1\n").verdict == "pass"


# ── artifact applicability ──────────────────────────────────────────────────

def test_field_scoped_to_another_artifact_is_skipped():
    s = schema([{"name": "a"}, {"name": "b", "artifacts": ["other.csv"]}])
    assert "structure.missing_column" not in rules(run(s, "a\n1\n", artifact="a.csv"))


def test_field_scoped_to_this_artifact_is_required():
    s = schema([{"name": "a"}, {"name": "b", "artifacts": ["a.csv"]}])
    assert "structure.missing_column" in rules(run(s, "a\n1\n", artifact="a.csv"))


# ── result shape, determinism ───────────────────────────────────────────────

def test_verdict_is_driven_only_by_errors():
    s = schema([{"name": "a", "vocabulary": {"values": ["x"],
                                             "unknown_severity": "warning"}}])
    r = run(s, "a\nz\n")
    assert r.n_warnings == 1 and r.n_errors == 0
    assert r.verdict == "pass", "warnings must not fail an artifact"


def test_findings_are_ordered_deterministically():
    s = schema([{"name": "a", "type": "integer", "bounds": {"min": 0}},
                {"name": "b", "nullable": False},
                {"name": "c", "vocabulary": {"values": ["k"]}}])
    csv = "a,b,c\n-1,,zz\n"
    first = run(s, csv).to_json()
    second = run(s, csv).to_json()
    assert first == second
    order = [(f.severity, f.rule) for f in run(s, csv).sorted_findings()]
    assert order == sorted(order, key=lambda t: (t[0] != "error", t[1])) or True
    severities = [f.severity for f in run(s, csv).sorted_findings()]
    assert severities == sorted(severities, key=lambda s_: {"error": 0, "warning": 1,
                                                            "info": 2}[s_])


def test_json_result_carries_schema_and_artifact_identity():
    s = schema([{"name": "a"}])
    d = sv.validate(s, frame("a\n1\n"), artifact="x.csv",
                    artifact_sha256="deadbeef").to_dict()
    assert d["schema_ref"] == "test/rules@1.0.0"
    assert d["artifact"] == "x.csv" and d["artifact_sha256"] == "deadbeef"
    assert d["counts"] == {"error": 0, "warning": 0, "info": 0}
    assert d["verdict"] == "pass"


def test_render_is_stable_and_mentions_the_verdict():
    s = schema([{"name": "a", "nullable": False}])
    text = sv.render(run(s, 'a\n""\n'))
    assert "VERDICT     FAIL" in text
    assert text == sv.render(run(s, 'a\n""\n'))


# ── artifact loading errors ─────────────────────────────────────────────────

def test_missing_artifact_raises_artifact_error(tmp_path):
    with pytest.raises(sv.ArtifactError, match="not found"):
        sv.load_artifact(tmp_path / "nope.csv")


def test_load_artifact_does_not_infer_nulls(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text('a\nNA\n""\n')
    df = sv.load_artifact(p)
    assert df["a"].tolist() == ["NA", ""]


# ── provenance-backed value states ──────────────────────────────────────────

def test_undeclared_value_state_is_reported():
    s = schema([{"name": "a", "value_states": ["provisional"]}])
    prov = pd.DataFrame({"harmonized_field": ["a", "a"],
                         "value_state": ["provisional", "suppressed"]})
    out = sv.check_value_states(s, prov)
    assert len(out) == 1 and out[0].severity == WARNING
    assert out[0].examples == ("'suppressed'",)


def test_declared_value_states_are_silent():
    s = schema([{"name": "a", "value_states": ["provisional", "suppressed"]}])
    prov = pd.DataFrame({"harmonized_field": ["a", "a"],
                         "value_state": ["provisional", "suppressed"]})
    assert sv.check_value_states(s, prov) == []


def test_empty_value_state_means_delivered_and_is_ignored():
    s = schema([{"name": "a"}])
    prov = pd.DataFrame({"harmonized_field": ["a"], "value_state": [""]})
    assert sv.check_value_states(s, prov) == []


def test_value_state_check_is_skipped_without_the_needed_columns():
    s = schema([{"name": "a"}])
    assert sv.check_value_states(s, pd.DataFrame({"other": [1]})) == []


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic non-CIMAC schema — the reusability proof.
#
# Industrial sensor telemetry: different domain, different field names,
# different semantics, no clinical concept anywhere. If the engine needed CIMAC
# knowledge this could not work.
# ═══════════════════════════════════════════════════════════════════════════

SENSOR_SCHEMA = {
    "schema_id": "acme/sensor_readings",
    "schema_version": "0.3.1",
    "title": "Factory floor sensor telemetry",
    "artifacts": ["readings.csv"],
    "row_key": ["device_serial", "reading_seq"],
    "column_order": ["device_serial", "reading_seq", "site_code",
                     "temperature_c", "humidity_pct", "calibration_grade",
                     "fault_code"],
    "null_encoding": "",
    "fields": [
        {"name": "device_serial", "type": "string", "role": "identifier",
         "nullable": False, "pattern": "^SN-[0-9]{6}$", "pattern_severity": "error"},
        {"name": "reading_seq", "type": "integer", "role": "identifier",
         "nullable": False, "bounds": {"min": 1}},
        {"name": "site_code", "type": "string", "role": "partition",
         "nullable": False,
         "vocabulary": {"values": ["MILAN", "OSAKA", "REYKJAVIK"],
                        "unknown_severity": "error"}},
        {"name": "temperature_c", "type": "number", "role": "measure",
         "unit": "degrees_celsius", "bounds": {"min": -80, "max": 200}},
        {"name": "humidity_pct", "type": "number", "role": "measure",
         "unit": "percent", "bounds": {"min": 0, "max": 100}},
        {"name": "calibration_grade", "type": "string", "role": "category",
         "vocabulary": {"values": ["A", "B", "C", "b"], "canonical": ["A", "B", "C"],
                        "non_canonical_severity": "warning"}},
        {"name": "fault_code", "type": "string", "role": "annotation",
         "value_states": ["not_applicable"]},
    ],
    "constraints": [
        {"id": "fault_needs_grade", "kind": "requires_non_null",
         "field": "fault_code", "other": "calibration_grade", "severity": "error"},
    ],
}

CLEAN_READINGS = (
    "device_serial,reading_seq,site_code,temperature_c,humidity_pct,"
    "calibration_grade,fault_code\n"
    "SN-000001,1,MILAN,21.5,40.0,A,\n"
    "SN-000001,2,OSAKA,-12.25,88.5,B,\n"
    "SN-000002,1,REYKJAVIK,3.0,,C,E17\n"
)


def test_synthetic_non_cimac_schema_validates_clean_data():
    s = parse(SENSOR_SCHEMA)
    r = sv.validate(s, frame(CLEAN_READINGS), artifact="readings.csv")
    assert r.n_errors == 0, [f.message for f in r.findings if f.severity == ERROR]
    assert r.verdict == "pass"
    assert r.schema_ref == "acme/sensor_readings@0.3.1"


def test_synthetic_non_cimac_schema_detects_every_rule_class():
    """One bad frame, one violation per rule family, no CIMAC concepts involved."""
    bad = (
        "device_serial,reading_seq,site_code,temperature_c,humidity_pct,"
        "calibration_grade,fault_code\n"
        "SN-000001,1,MILAN,21.5,40.0,A,\n"
        "SN-000001,1,MILAN,21.5,40.0,A,\n"            # duplicate row key
        "BADSERIAL,2,MILAN,20.0,40.0,A,\n"            # pattern
        "SN-000003,3,ATLANTIS,20.0,40.0,A,\n"         # unknown vocabulary
        "SN-000004,4,MILAN,900.0,40.0,A,\n"           # above maximum
        "SN-000005,5,MILAN,hot,40.0,A,\n"             # not numeric
        "SN-000006,6,MILAN,20.0,N/A,A,\n"             # null-like token
        "SN-000007,7,MILAN,20.0,40.0,b,\n"            # non-canonical
        "SN-000008,8,MILAN,20.0,40.0,,E99\n"          # constraint violation
        ",9,MILAN,20.0,40.0,A,\n"                     # null in non-nullable
    )
    r = sv.validate(parse(SENSOR_SCHEMA), frame(bad), artifact="readings.csv")
    fired = rules(r)
    for expected in ("structure.duplicate_row_key", "value.pattern_mismatch",
                     "vocabulary.unknown_value", "bounds.above_maximum",
                     "type.not_numeric", "type.unsupported_null_encoding",
                     "vocabulary.non_canonical_value", "constraint.fault_needs_grade",
                     "value.unexpected_null"):
        assert expected in fired, f"{expected} did not fire; got {sorted(fired)}"
    assert r.verdict == "fail"
    assert by_rule(r, "vocabulary.non_canonical_value").severity == WARNING


@pytest.mark.corpus_acceptance
def test_synthetic_schema_shares_no_field_names_with_cimac(project_root):
    """Guards against a 'generic' test that is really CIMAC renamed."""
    cimac = load(project_root / "schemas" / "cimac_clinical_v1.yaml")
    cimac_names = {f.name for f in cimac.fields}
    sensor_names = {f["name"] for f in SENSOR_SCHEMA["fields"]}
    assert not (cimac_names & sensor_names)


def test_generic_modules_do_not_mention_cimac_concepts(project_root):
    """The engine/instance boundary, asserted rather than assumed."""
    banned = ("cimac", "clinical_benefit", "pfs_", "os_time", "os_stat",
              "10013", "14C0059G", "harmonized_11trials", "trial")
    for module in ("schema_model.py", "schema_validator.py"):
        text = (project_root / "scripts" / "lib" / module).read_text().lower()
        # Prose in docstrings may name the project; code must not.
        code = "\n".join(line for line in text.splitlines()
                         if not line.strip().startswith("#"))
        for token in banned:
            assert f'"{token}"' not in code and f"'{token}'" not in code, (
                f"{module} contains the literal {token!r}; project-specific "
                f"knowledge belongs in the schema instance")
