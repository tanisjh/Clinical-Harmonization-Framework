"""Synthetic-only tests for registered field-rule confidence authority."""
from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

from lib import rule_registry
from lib.confidence_policy import (
    AUTHORITY_DECLARED_NO_VALUE,
    AUTHORITY_RULE,
    OUTCOME_NO_VALUE,
    OUTCOME_VALUE,
    InvalidRuleConfidence,
    UnsupportedConfidencePolicy,
    finalize_registered_rule_confidence,
)
from lib.replay import _EventOutcome, replay_cell
from lib.replay_contract import ReplayStatus
from lib.replay_orchestration import (
    ReplayEvidenceCollector,
    ReplayOrchestrationError,
    ReplayRunContext,
)
from lib.rule_registry import RegisteredRule
from lib.schema_model import contract_for, load as load_schema
from lib.spec_executor import ExecutionError, execute
from lib.spec_model import load as load_spec
from tests import test_replay_stage3 as stage3


@pytest.fixture(autouse=True)
def _registered_test_rules(monkeypatch):
    monkeypatch.setitem(
        rule_registry.REGISTRY,
        ("synthetic.adjusted.v1", "1"),
        RegisteredRule(
            "synthetic.adjusted.v1", "1", rule_registry.KIND_FIELD,
            stage3._adjusted, arity=2, params=("scale",)),
    )
    monkeypatch.setitem(
        rule_registry.REGISTRY,
        ("synthetic.identity.v1", "1"),
        RegisteredRule(
            "synthetic.identity.v1", "1", rule_registry.KIND_FIELD,
            stage3._identity, arity=1),
    )


@pytest.mark.parametrize(
    "semantic,rule_confidence,no_value,expected,outcome,authority",
    [
        ("found", 0.8, None, 0.8, OUTCOME_VALUE, AUTHORITY_RULE),
        (None, 0.8, 0.25, 0.25, OUTCOME_NO_VALUE,
         AUTHORITY_DECLARED_NO_VALUE),
        (float("nan"), 0.8, 0.25, 0.25, OUTCOME_NO_VALUE,
         AUTHORITY_DECLARED_NO_VALUE),
        (None, 0.8, None, 0.8, OUTCOME_NO_VALUE, AUTHORITY_RULE),
        ("", 0.8, 0.25, 0.8, OUTCOME_VALUE, AUTHORITY_RULE),
        (0, 0, 0.25, 0.0, OUTCOME_VALUE, AUTHORITY_RULE),
        (0.0, 1, None, 1.0, OUTCOME_VALUE, AUTHORITY_RULE),
        (False, 0.4, 0.25, 0.4, OUTCOME_VALUE, AUTHORITY_RULE),
    ],
)
def test_pure_registered_rule_confidence_policy(
        semantic, rule_confidence, no_value, expected, outcome, authority):
    actual = finalize_registered_rule_confidence(
        semantic, rule_confidence, declared_value_policy="from_rule",
        declared_no_value_confidence=no_value)
    assert (actual.value, actual.outcome, actual.authority) == (
        expected, outcome, authority)
    assert actual == finalize_registered_rule_confidence(
        semantic, rule_confidence, declared_value_policy="from_rule",
        declared_no_value_confidence=no_value)


@pytest.mark.parametrize(
    "invalid",
    [None, "0.8", True, -0.01, 1.01, float("nan"), float("inf"),
     float("-inf")],
)
def test_malformed_rule_confidence_is_not_rescued_by_no_value_override(invalid):
    with pytest.raises(InvalidRuleConfidence):
        finalize_registered_rule_confidence(
            None, invalid, declared_value_policy="from_rule",
            declared_no_value_confidence=0.0)


def test_pure_policy_rejects_unsupported_declaration_and_has_no_output_inputs():
    with pytest.raises(UnsupportedConfidencePolicy):
        finalize_registered_rule_confidence(
            1, 0.8, declared_value_policy=0.8,
            declared_no_value_confidence=None)
    names = set(inspect.signature(
        finalize_registered_rule_confidence).parameters)
    assert names == {
        "semantic_value", "rule_confidence", "declared_value_policy",
        "declared_no_value_confidence",
    }
    assert not names & {
        "cell", "recorded_confidence", "delivery", "provenance_value",
        "derivation", "contribution", "source",
    }


@pytest.mark.parametrize("invalid", [True, float("nan"), -0.1, 1.1])
def test_registry_rejects_malformed_confidence_before_normalization(invalid):
    rule = RegisteredRule(
        "synthetic.invalid.v1", "1", rule_registry.KIND_FIELD,
        lambda: (None, invalid, "bad"), arity=0)
    with pytest.raises(rule_registry.RuleReturnContractMismatch):
        rule_registry.invoke(rule, [], {})


def test_registered_rule_tuple_contract_is_preserved():
    rule = RegisteredRule(
        "synthetic.valid.v1", "1", rule_registry.KIND_FIELD,
        lambda: (7, 1, "note"), arity=0)
    assert rule_registry.invoke(rule, [], {}) == (7, 1.0, "note")


def test_non_null_and_no_value_override_replay_pass(tmp_path):
    present = stage3._bundle(tmp_path / "present")
    parent = next(c for c in present.cells
                  if c.mapping_id == "custom.adjusted")
    assert (parent.value, parent.confidence) == (23.0, 0.9)
    assert replay_cell(
        stage3.ROOT, present.manifest, present.request).status is ReplayStatus.PASS

    missing = stage3._bundle(
        tmp_path / "missing", source_updates={"base_measurement": ""})
    parent = next(c for c in missing.cells
                  if c.mapping_id == "custom.adjusted")
    assert (parent.value, parent.confidence) == (None, 0.25)
    assert replay_cell(
        stage3.ROOT, missing.manifest, missing.request).status is ReplayStatus.PASS


@pytest.mark.parametrize("recorded", ["0", "0.0", "0.00"])
def test_recorded_zero_confidence_uses_exact_numeric_equality(tmp_path, recorded):
    def zero_override(doc):
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "custom.adjusted")
        mapping["confidence"]["no_value"] = 0.0

    case = stage3._bundle(
        tmp_path, source_updates={"base_measurement": ""},
        spec_mutator=zero_override)
    stage3._mutate_provenance(case, lambda rows: stage3._event(
        rows, "custom.adjusted").update(confidence=recorded))
    assert replay_cell(
        stage3.ROOT, case.manifest, case.request).status is ReplayStatus.PASS


def test_equivalent_recorded_decimal_spelling_passes(tmp_path):
    case = stage3._bundle(tmp_path)
    stage3._mutate_provenance(case, lambda rows: stage3._event(
        rows, "custom.adjusted").update(confidence="0.900"))
    assert replay_cell(
        stage3.ROOT, case.manifest, case.request).status is ReplayStatus.PASS


@pytest.mark.parametrize(
    "recorded,status,reason",
    [
        ("", ReplayStatus.NOT_REPLAYABLE, "MANIFEST_REQUIRED_FIELD_MISSING"),
        ("nope", ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH"),
        ("NaN", ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH"),
        ("Infinity", ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH"),
        ("-0.1", ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH"),
        ("1.1", ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH"),
        ("0.1", ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH"),
    ],
)
def test_recorded_confidence_is_comparison_only(
        tmp_path, recorded, status, reason):
    case = stage3._bundle(tmp_path)
    stage3._mutate_provenance(case, lambda rows: stage3._event(
        rows, "custom.adjusted").update(confidence=recorded))
    result = replay_cell(stage3.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (status, reason)


@pytest.mark.parametrize(
    "invalid", [None, "bad", True, -0.1, 1.1, float("nan"),
                float("inf"), float("-inf")],
)
def test_replay_rejects_malformed_rule_confidence_even_with_override(
        tmp_path, monkeypatch, invalid):
    case = stage3._bundle(
        tmp_path, source_updates={"base_measurement": ""})

    def malformed(_left, _right, *, scale):
        return None, invalid, f"scale={scale}"

    monkeypatch.setitem(
        rule_registry.REGISTRY,
        ("synthetic.adjusted.v1", "1"),
        RegisteredRule(
            "synthetic.adjusted.v1", "1", rule_registry.KIND_FIELD,
            malformed, arity=2, params=("scale",)),
    )
    result = replay_cell(stage3.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "REGISTERED_RULE_EXECUTION_FAILED")


def test_executor_rejects_malformed_registered_rule_confidence(
        tmp_path, monkeypatch):
    def malformed(_left, _right, *, scale):
        return None, True, f"scale={scale}"

    monkeypatch.setitem(
        rule_registry.REGISTRY,
        ("synthetic.adjusted.v1", "1"),
        RegisteredRule(
            "synthetic.adjusted.v1", "1", rule_registry.KIND_FIELD,
            malformed, arity=2, params=("scale",)),
    )
    with pytest.raises(ExecutionError, match="invalid registered-rule confidence"):
        stage3._bundle(
            tmp_path, source_updates={"base_measurement": ""})


def test_publication_recomputes_confidence_from_runtime_dependencies(tmp_path):
    case = stage3._bundle(tmp_path)
    schema = load_schema(stage3.SCHEMA_FIXTURE)
    spec = load_spec(
        case.spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        stage3.ROOT, schema, (spec,), run_id=stage3.RUN_ID,
        application="synthetic-stage20-test")
    cells, diagnostics = execute(
        spec, stage3.ROOT, stage3._anchor(),
        contract=contract_for(schema), run_id=stage3.RUN_ID)
    parent = next(cell for cell in cells
                  if cell.mapping_id == "custom.adjusted")
    parent.confidence = 0.1
    with pytest.raises(ReplayOrchestrationError, match="confidence contradicts"):
        ReplayEvidenceCollector(context).collect(
            spec, diagnostics.sources, cells)


def test_lower_dependency_failure_precedes_recorded_confidence_mismatch(tmp_path):
    case = stage3._bundle(tmp_path)
    stage3._mutate_provenance(case, lambda rows: stage3._event(
        rows, "custom.adjusted").update(confidence="0.1"))
    case.source_path.write_text(
        case.source_path.read_text().replace(",10,3", ",11,3"))
    result = replay_cell(stage3.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_REPLAY_FAILED")


def test_rule_identity_precedes_recorded_confidence_mismatch(tmp_path):
    case = stage3._bundle(tmp_path)
    stage3._mutate_provenance(case, lambda rows: stage3._event(
        rows, "custom.adjusted").update(
            rule_version="999", confidence="0.1"))
    result = replay_cell(stage3.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RULE_IDENTITY_MISMATCH")


def test_recursive_parent_requires_upstream_confidence_pass(tmp_path):
    case = stage3._bundle(tmp_path, target="final_measurement")
    assert replay_cell(
        stage3.ROOT, case.manifest, case.request).status is ReplayStatus.PASS
    stage3._mutate_provenance(case, lambda rows: stage3._event(
        rows, "custom.adjusted").update(confidence="0.1"))
    result = replay_cell(stage3.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_REPLAY_FAILED")
    assert "CONFIDENCE_CONTEXT_MISMATCH" in result.message


def test_confidence_is_not_propagated_in_recursive_outcomes(tmp_path):
    case = stage3._bundle(tmp_path, target="final_measurement")
    parent = next(c for c in case.cells if c.mapping_id == "custom.final")
    snapshots = stage3.contributions_from_json(parent.contributions)
    assert all("confidence" not in snapshot for snapshot in snapshots)
    assert "confidence" not in {field.name for field in fields(_EventOutcome)}


def test_fixed_confidence_custom_rule_remains_executor_unsupported(tmp_path):
    def fixed(doc):
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "custom.adjusted")
        mapping["confidence"] = {"value": 0.8, "no_value": 0.25}

    with pytest.raises(ExecutionError, match="must declare confidence 'from_rule'"):
        stage3._bundle(tmp_path, spec_mutator=fixed)
