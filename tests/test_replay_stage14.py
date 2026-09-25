"""Synthetic tests for closed structured-derivation replay (Stage 14)."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

import lib.replay as replay_module
from lib import rule_registry
from lib.derivation import (
    DerivationError,
    MissingDerivationFact,
    canonical_derivation_json,
    derivation_references,
    is_stage14_derivation,
    parse_canonical_derivation_json,
    render_structured_derivation,
)
from lib.replay import replay_cell
from lib.replay_contract import REASON_CODES, REASON_CODES_VERSION, ReplayStatus
from lib.rule_registry import RegisteredRule
from lib.spec_model import DerivationVariant, load as load_spec
from tests.test_replay_stage1 import (
    ROOT,
    SPEC_FIXTURE as STAGE1_SPEC,
    _case as direct_case,
)
from tests.test_replay_stage6 import (
    _bundle as constant_bundle,
    _mutate_provenance as mutate_constant_provenance,
)


def _pair(left, right):
    if left is None or right is None:
        return None, 0.2, "missing"
    return f"{left}:{right}", 0.88, "synthetic pair"


@pytest.fixture(autouse=True)
def _stage14_rules(monkeypatch):
    monkeypatch.setitem(
        rule_registry.REGISTRY,
        ("synthetic.constant_pair.v1", "1"),
        RegisteredRule(
            "synthetic.constant_pair.v1", "1", rule_registry.KIND_FIELD,
            _pair, arity=2),
    )


def _variant(record, *, when=""):
    return DerivationVariant(record=record, when=when)


def _direct_record():
    return {
        "kind": "synthetic_direct_context",
        "final_value": {"$ref": "value"},
        "source": {"column": {"$ref": "source_column"}},
        "nested": ["first", {"again": {"$ref": "value"}}],
        "unicode": "λ",
        "whitespace": "  exact text  ",
    }


def _direct_bundle(tmp_path, *, record=None, payload=None,
                   delivered_value=..., provenance_value=..., raw_override=...,
                   provenance_updates=None):
    record = record or _direct_record()

    def mutate(doc):
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "direct.reading_text")
        mapping["derivation"] = [{"record": record}]

    if payload is None:
        payload = render_structured_derivation(
            (_variant(record),),
            {"value": "001", "source_column": "reading_text"}, {})
    updates = {"derivation": payload}
    updates.update(provenance_updates or {})
    return direct_case(
        tmp_path, "direct.reading_text", spec_mutator=mutate,
        provenance_updates=updates,
        delivered_value=delivered_value, provenance_value=provenance_value,
        raw_override=raw_override)


def test_pure_renderer_nested_literals_types_and_repeated_reference():
    record = {
        "z": [{"$ref": "value"}, {"again": {"$ref": "value"}}],
        "a": [None, True, 7, 2.5, "λ", "  exact  "],
        "column": {"$ref": "source_column"},
    }
    rendered = render_structured_derivation(
        (_variant(record),),
        {"value": "001", "source_column": "reading_text", "extra": 99}, {})
    assert rendered == (
        '{"a":[null,true,7,2.5,"\\u03bb","  exact  "],'
        '"column":"reading_text","z":["001",{"again":"001"}]}')
    assert rendered == render_structured_derivation(
        (_variant(record),),
        {"value": "001", "source_column": "reading_text"}, {})


def test_missing_fact_differs_from_present_null():
    variants = (_variant({"value": {"$ref": "value"}}),)
    with pytest.raises(MissingDerivationFact, match="value"):
        render_structured_derivation(variants, {}, {})
    assert render_structured_derivation(variants, {"value": None}, {}) == \
        '{"value":null}'


def test_variant_selection_matches_executor_but_is_not_stage14_eligible():
    variants = (
        _variant({"choice": "yes"}, when="selected"),
        _variant({"choice": "no"}),
    )
    assert render_structured_derivation(variants, {}, {"selected": True}) == \
        '{"choice":"yes"}'
    assert render_structured_derivation(variants, {}, {"selected": False}) == \
        '{"choice":"no"}'
    assert not is_stage14_derivation(variants)
    assert render_structured_derivation(
        (_variant({"unused": True}, when="selected"),), {}, {}) == ""


@pytest.mark.parametrize("record", [
    {"bad": {"$ref": "value", "other": 1}},
    {"bad": {"$ref": ""}},
    {"bad": {"$ref": 1}},
])
def test_malformed_reference_shape_is_refused(record):
    with pytest.raises(DerivationError):
        derivation_references(record)


@pytest.mark.parametrize("ref", [
    "raw_value", "source_label", "anchor_group_size",
    "anchor_selected_row", "read.input", "condition.selected",
    "group_member.g", "group_value.g", "group_label.g", "unknown",
])
def test_stage14_fact_boundary_rejects_every_other_reference(ref):
    assert not is_stage14_derivation(
        (_variant({"fact": {"$ref": ref}}),))


def test_mapping_and_contribution_conditions_are_ineligible():
    mapping = next(item for item in load_spec(STAGE1_SPEC).mappings
                   if item.mapping_id == "direct.reading_text")
    mapping = replace(mapping, derivation=(
        _variant({"value": {"$ref": "value"}}),))
    assert replay_module._is_stage14_derivation_mapping(mapping)
    assert not replay_module._is_stage14_derivation_mapping(
        replace(mapping, when="condition"))
    assert not replay_module._is_stage14_derivation_mapping(
        replace(mapping, derivation=(
            _variant({"value": {"$ref": "value"}}, when="condition"),)))
    assert not replay_module._is_stage14_derivation_mapping(
        replace(mapping, contributions=replace(
            mapping.contributions, conditions=("condition",))))


@pytest.mark.parametrize("payload", [
    '{"a":1',
    '{"a":1,"a":1}',
    '{"a":NaN}',
    '{"a":Infinity}',
])
def test_recorded_json_is_strict(payload):
    with pytest.raises(DerivationError):
        parse_canonical_derivation_json(payload)


def test_canonical_json_preserves_types_order_unicode_and_whitespace():
    assert parse_canonical_derivation_json('{"b":2,"a":1}') == \
        '{"a":1,"b":2}'
    assert canonical_derivation_json({"x": [1, 2]}) != \
        canonical_derivation_json({"x": [2, 1]})
    assert canonical_derivation_json({"x": True}) != \
        canonical_derivation_json({"x": 1})
    assert canonical_derivation_json({"x": 1}) != \
        canonical_derivation_json({"x": 1.0})
    assert parse_canonical_derivation_json(
        json.dumps({"x": " λ ", "s": "  exact  "}, ensure_ascii=False)) == \
        canonical_derivation_json({"x": " λ ", "s": "  exact  "})


def test_direct_value_and_source_column_derivation_passes(tmp_path, monkeypatch):
    called = {"count": 0}
    original = replay_module.recompute_direct

    def spy(raw, *, coerce):
        called["count"] += 1
        return original(raw, coerce=coerce)

    monkeypatch.setattr(replay_module, "recompute_direct", spy)
    case = _direct_bundle(tmp_path)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "001")
    assert called["count"] == 1


def test_semantic_object_key_order_passes(tmp_path):
    record = {"kind": "ordered", "value": {"$ref": "value"}}
    case = _direct_bundle(
        tmp_path, record=record, payload='{"value":"001","kind":"ordered"}')
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


@pytest.mark.parametrize("payload", [
    '{"final_value":"forged","kind":"synthetic_direct_context",'
    '"nested":["first",{"again":"001"}],'
    '"source":{"column":"reading_text"},"unicode":"\\u03bb",'
    '"whitespace":"  exact text  "}',
    '{"final_value":"001","kind":"synthetic_direct_context",'
    '"nested":[{"again":"001"},"first"],'
    '"source":{"column":"reading_text"},"unicode":"\\u03bb",'
    '"whitespace":"  exact text  "}',
    '{"final_value":1,"kind":"synthetic_direct_context",'
    '"nested":["first",{"again":"001"}],'
    '"source":{"column":"reading_text"},"unicode":"\\u03bb",'
    '"whitespace":"  exact text  "}',
])
def test_changed_recorded_derivation_fails(payload, tmp_path):
    case = _direct_bundle(tmp_path, payload=payload)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DERIVATION_MISMATCH")


@pytest.mark.parametrize("payload", [
    '{"x":', '{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}',
])
def test_malformed_or_non_strict_recorded_derivation_fails(payload, tmp_path):
    case = _direct_bundle(tmp_path, payload=payload)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DERIVATION_MISMATCH")


def test_missing_recorded_derivation_is_historical_incompleteness(tmp_path):
    case = _direct_bundle(tmp_path, payload="")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "DERIVATION_EVIDENCE_MISSING")


def test_delivery_and_provenance_output_cannot_populate_value_fact(tmp_path):
    case = _direct_bundle(
        tmp_path, delivered_value="forged", provenance_value="forged",
        payload=canonical_derivation_json({
            "kind": "synthetic_direct_context", "final_value": "forged",
            "source": {"column": "reading_text"},
            "nested": ["first", {"again": "forged"}],
            "unicode": "λ", "whitespace": "  exact text  ",
        }))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DERIVATION_MISMATCH")


def test_source_column_comes_from_pinned_declaration(tmp_path):
    record = {"column": {"$ref": "source_column"}}
    case = _direct_bundle(
        tmp_path, record=record, payload='{"column":"forged"}')
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DERIVATION_MISMATCH")


def test_raw_mismatch_retains_precedence(tmp_path):
    case = _direct_bundle(tmp_path, raw_override="forged")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_source_hash_mismatch_retains_precedence(tmp_path):
    case = _direct_bundle(tmp_path)
    case.source_path.write_bytes(case.source_path.read_bytes() + b"\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")


def test_recorded_provisional_cannot_override_pinned_default_state(tmp_path):
    case = _direct_bundle(
        tmp_path, provenance_updates={"value_state": "provisional"})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VALUE_STATE_CONTEXT_MISMATCH")


def test_literal_and_value_constant_reuse_existing_helper(tmp_path, monkeypatch):
    called = {"count": 0}
    original = replay_module.recompute_constant

    def spy(*, declared_value):
        called["count"] += 1
        return original(declared_value=declared_value)

    monkeypatch.setattr(replay_module, "recompute_constant", spy)
    case = constant_bundle(tmp_path, target="constant_derived")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "derived-context")
    assert called["count"] == 1


def test_recursive_dependent_unlocks_through_existing_stage3(tmp_path):
    case = constant_bundle(tmp_path, target="downstream_blocked")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "other:derived-context")


def test_constant_missing_or_changed_derivation_has_new_reasons(tmp_path):
    missing = constant_bundle(tmp_path / "missing", target="constant_derived")
    mutate_constant_provenance(missing, derivation="")
    result = replay_cell(ROOT, missing.manifest, missing.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "DERIVATION_EVIDENCE_MISSING")

    changed = constant_bundle(tmp_path / "changed", target="constant_derived")
    mutate_constant_provenance(changed, derivation='{"kind":"forged"}')
    result = replay_cell(ROOT, changed.manifest, changed.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DERIVATION_MISMATCH")


def test_reason_contract_includes_stage14_reasons_after_v10():
    assert REASON_CODES_VERSION == 10
    assert "DERIVATION_EVIDENCE_MISSING" in \
        REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
    assert "DERIVATION_MISMATCH" in REASON_CODES[ReplayStatus.FAIL]
