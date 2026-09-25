"""Stage 21 tests for closed same-source/same-row inline fallback replay."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import test_replay_stage4 as stage4
from lib.fallback_policy import (
    MAPPING_MATCHED,
    MAPPING_UNMAPPED_NULL,
    apply_inline_fallback_map,
    select_ordered_fallback,
)
from lib.replay import _is_closed_same_row_inline_fallback, replay_cell
from lib.replay_contract import ReplayStatus, encode_typed_raw
from lib.schema_model import load as load_schema
from lib.spec_model import load as load_spec


MAPPING_ID = "fallback.raw"
FIRST_COLUMN = "candidate_value"
SECOND_COLUMN = "candidate_two"
NO_VALUE_COLUMN = f"{FIRST_COLUMN}+{SECOND_COLUMN}"
DEFAULT_MAP = {
    "5": "mapped-first",
    "20": "mapped-second",
    "": "mapped-blank",
    "0": "mapped-zero",
    "False": "mapped-false",
    "Alpha Beta": "mapped-normalized",
}


def _same_row_bundle(tmp_path, *, first="5", second="20",
                     mapping_table=None, on_unmapped="null",
                     source_id="primary", spec_extra=None,
                     duplicate=False):
    def source_mutate(paths):
        for name, path in paths.items():
            fields, rows = stage4._read_rows(path)
            fields = [*fields, SECOND_COLUMN]
            for row in rows:
                row[SECOND_COLUMN] = "unused"
                if row["device_id"] == "DEVICE-A" and name == "primary":
                    row[FIRST_COLUMN] = "" if first is None else str(first)
                if row["device_id"] == "DEVICE-A" and name == source_id:
                    row[SECOND_COLUMN] = "" if second is None else str(second)
            if duplicate and name == "primary":
                rows.append(dict(rows[0]))
            stage4._write_rows(path, fields, rows)

    def spec_mutate(document):
        mapping = next(item for item in document["mappings"]
                       if item["mapping_id"] == MAPPING_ID)
        mapping["candidates"] = [
            {"source": "primary", "column": FIRST_COLUMN},
            {"source": source_id, "column": SECOND_COLUMN},
        ]
        mapping["map"] = dict(DEFAULT_MAP if mapping_table is None
                              else mapping_table)
        mapping["on_unmapped"] = on_unmapped
        mapping["contributions"] = {
            "record": "none",
            "after_decision": "not_evaluated",
            "no_value_column": NO_VALUE_COLUMN,
        }
        if duplicate:
            document["record"]["duplicates"] = {
                "policy": "first", "by": ["device_id"]}
        if spec_extra is not None:
            spec_extra(document, mapping)

    return stage4._bundle(
        tmp_path, device="DEVICE-A", target="preferred_raw",
        source_mutate=source_mutate, spec_mutate=spec_mutate)


def _result(case):
    return replay_cell(stage4.ROOT, case.manifest, case.request)


def _event(rows):
    return stage4._event(rows, MAPPING_ID)


@pytest.mark.parametrize("values,expected_rank,expected_raw", [
    (["first", "second"], 0, "first"),
    ([None, "second"], 1, "second"),
    ([None, float("nan")], None, None),
    (["", "second"], 0, ""),
    ([0, "second"], 0, 0),
    ([0.0, "second"], 0, 0.0),
    ([False, "second"], 0, False),
    (["arbitrary", "second"], 0, "arbitrary"),
])
def test_pure_selection_uses_exact_executor_missing_predicate(
        values, expected_rank, expected_raw):
    selected = select_ordered_fallback(values)
    assert selected.selected_rank == expected_rank
    assert selected.raw_selected == expected_raw
    assert selected.exhausted is (expected_rank is None)


def test_pure_inline_map_preserves_selection_before_mapping():
    selection = select_ordered_fallback(["unknown", "known"])
    mapped = apply_inline_fallback_map(
        selection.raw_selected,
        mapping_table={"known": "later-value"}, on_unmapped="null")
    assert selection.selected_rank == 0
    assert mapped.value is None
    assert mapped.outcome == MAPPING_UNMAPPED_NULL


def test_pure_inline_map_uses_executor_normalization():
    selection = select_ordered_fallback(["  Alpha   Beta "])
    mapped = apply_inline_fallback_map(
        selection.raw_selected,
        mapping_table={"Alpha Beta": "normalized"}, on_unmapped="null")
    assert mapped.value == "normalized"
    assert mapped.outcome == MAPPING_MATCHED


@pytest.mark.corpus_acceptance
def test_authorized_10104_race_declaration_is_exactly_closed_route_shape(
        corpus_root):
    schema = load_schema(
        corpus_root / "schemas/cimac_clinical_v1.yaml")
    spec = load_spec(
        corpus_root / "specs/cimac_10104_v2_0_1.yaml",
        target_schema=schema, require_executable=True,
        vocabulary_dir=corpus_root / "vocabularies")
    mapping = next(item for item in spec.mappings
                   if item.mapping_id == "race")
    assert _is_closed_same_row_inline_fallback(mapping)
    assert [item.source for item in mapping.candidates] == [
        "demographics", "demographics"]
    assert [item.column for item in mapping.candidates] == [
        "PT_RACE_CD_1", "PT_RACE_CD_2"]


@pytest.mark.parametrize(
    "first,second,expected,expected_column,expected_raw,expected_rule", [
        ("5", "20", "mapped-first", FIRST_COLUMN, "5",
         "fallback.first_present.v1"),
        (None, "20", "mapped-second", SECOND_COLUMN, "20",
         "fallback.first_present.v1"),
        (None, None, None, NO_VALUE_COLUMN, None, "fallback.no_value.v1"),
        ("   ", "20", "mapped-blank", FIRST_COLUMN, "   ",
         "fallback.first_present.v1"),
        ("0", "20", "mapped-zero", FIRST_COLUMN, "0",
         "fallback.first_present.v1"),
        ("False", "20", "mapped-false", FIRST_COLUMN, "False",
         "fallback.first_present.v1"),
        ("  Alpha   Beta ", "20", "mapped-normalized", FIRST_COLUMN,
         "  Alpha   Beta ", "fallback.first_present.v1"),
        ("unknown", "20", None, FIRST_COLUMN, "unknown",
         "fallback.no_value.v1"),
    ])
def test_closed_route_reconstructs_winner_exhaustion_and_mapping(
        tmp_path, first, second, expected, expected_column, expected_raw,
        expected_rule):
    case = _same_row_bundle(tmp_path, first=first, second=second)
    result = _result(case)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected)
    parent = next(cell for cell in case.cells if cell.mapping_id == MAPPING_ID)
    assert parent.contributions == ""
    assert parent.source_column == expected_column
    assert parent.raw_value == ("" if expected_raw is None else expected_raw)
    assert parent.rule_id == expected_rule
    if expected_raw is None:
        assert parent.raw_value_json == ""
    else:
        assert json.loads(parent.raw_value_json)["value"] == expected_raw
    usage = next(item for item in case.manifest.mapping_usages
                 if item.mapping_id == MAPPING_ID)
    assert len(usage.source_ref_ids) == 1


def test_selected_unmapped_null_does_not_resume_at_second_candidate(tmp_path):
    case = _same_row_bundle(
        tmp_path, first="unknown", second="20",
        mapping_table={"20": "would-be-second"})
    assert _result(case).status is ReplayStatus.PASS
    parent = next(cell for cell in case.cells if cell.mapping_id == MAPPING_ID)
    assert parent.value is None
    assert parent.source_column == FIRST_COLUMN
    assert parent.raw_value == "unknown"
    assert parent.rule_id == "fallback.no_value.v1"
    assert parent.confidence == 0.2


def test_record_none_rejects_fabricated_candidate_copies(tmp_path):
    case = _same_row_bundle(tmp_path)
    stage4._mutate_provenance(case, lambda rows: _event(rows).update({
        "contributions": "[]"}))
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "FALLBACK_CANDIDATE_CONTEXT_MISMATCH")


def test_missing_event_scoped_usage_link_cannot_be_repaired_run_wide(tmp_path):
    case = _same_row_bundle(tmp_path)
    usage = next(item for item in case.manifest.mapping_usages
                 if item.mapping_id == MAPPING_ID)
    case.manifest = replace(case.manifest, mapping_usages=tuple(
        replace(item, source_ref_ids=()) if item == usage else item
        for item in case.manifest.mapping_usages))
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_LINK_MISSING")


def test_missing_parent_source_link_is_not_repairable_by_output(tmp_path):
    case = _same_row_bundle(tmp_path)
    stage4._mutate_provenance(
        case, lambda rows: _event(rows).update({"source_ref_id": ""}))
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_LINK_MISSING")


def test_missing_parent_row_locator_is_not_repairable_by_output(tmp_path):
    case = _same_row_bundle(tmp_path)
    stage4._mutate_provenance(
        case, lambda rows: _event(rows).update({"source_row_number": ""}))
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_LOCATOR_INCOMPLETE")


def test_ambiguous_parent_record_identity_is_refused(tmp_path):
    case = _same_row_bundle(tmp_path, duplicate=True)
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_RECORD_KEY_MISMATCH")


def test_replacement_at_bound_path_cannot_repair_history(tmp_path):
    case = _same_row_bundle(tmp_path)
    case.source_paths["primary"].write_bytes(
        case.source_paths["primary"].read_bytes() + b"\n")
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")


def test_missing_bound_historical_artifact_remains_a_causal_failure(tmp_path):
    case = _same_row_bundle(tmp_path)
    case.source_paths["primary"].unlink()
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_FILE_MISSING")


def test_wrong_reader_flags_remain_not_replayable(tmp_path):
    case = _same_row_bundle(tmp_path)
    sources = tuple(
        replace(source, reader=replace(
            source.reader,
            effective_flags=tuple(
                (name, 7 if name == "header" else value)
                for name, value in source.reader.effective_flags)))
        if source.source_id == "primary" else source
        for source in case.manifest.sources)
    case.manifest = replace(case.manifest, sources=sources)
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "READER_FLAGS_UNSUPPORTED")


@pytest.mark.parametrize("field,value,reason", [
    ("source_column", SECOND_COLUMN, "FALLBACK_SELECTION_MISMATCH"),
    ("rule_id", "fallback.no_value.v1", "RULE_IDENTITY_MISMATCH"),
    ("confidence", "0.2", "CONFIDENCE_CONTEXT_MISMATCH"),
])
def test_recorded_outcome_fields_are_comparison_only(
        tmp_path, field, value, reason):
    case = _same_row_bundle(tmp_path)
    stage4._mutate_provenance(
        case, lambda rows: _event(rows).update({field: value}))
    result = _result(case)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, reason)


def test_recorded_parent_raw_cannot_select_the_winner(tmp_path):
    case = _same_row_bundle(tmp_path)
    stage4._mutate_provenance(case, lambda rows: _event(rows).update({
        "raw_value": "20",
        "raw_value_json": encode_typed_raw("20").to_json(),
    }))
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_recorded_output_and_delivery_cannot_select_the_winner(tmp_path):
    case = _same_row_bundle(tmp_path)
    stage4._mutate_delivered(
        case, lambda rows: rows[0].update({"preferred_raw": "mapped-second"}))
    stage4._mutate_provenance(
        case, lambda rows: _event(rows).update({"value": "mapped-second"}))
    result = _result(case)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH", "mapped-first")


def test_missing_source_authority_precedes_matching_output_fields(tmp_path):
    case = _same_row_bundle(tmp_path)
    usage = next(item for item in case.manifest.mapping_usages
                 if item.mapping_id == MAPPING_ID)
    case.manifest = replace(case.manifest, mapping_usages=tuple(
        replace(item, source_ref_ids=()) if item == usage else item
        for item in case.manifest.mapping_usages))
    stage4._mutate_provenance(case, lambda rows: _event(rows).update({
        "source_column": SECOND_COLUMN,
        "raw_value_json": encode_typed_raw("20").to_json(),
        "value": "mapped-second",
    }))
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_LINK_MISSING")


def test_multi_source_record_none_inline_fallback_remains_unsupported(tmp_path):
    case = _same_row_bundle(tmp_path, source_id="secondary")
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "FALLBACK_CANDIDATE_UNSUPPORTED")


def test_unsupported_inline_map_mode_does_not_enter_closed_route(tmp_path):
    case = _same_row_bundle(tmp_path, on_unmapped="error")
    result = _result(case)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "FALLBACK_CANDIDATE_UNSUPPORTED")


def test_candidate_gate_does_not_enter_closed_route(tmp_path):
    def add_gate(_document, mapping):
        mapping["candidates"][0]["when"] = {
            "source": "primary", "column": FIRST_COLUMN,
            "predicate": "not_null"}

    case = _same_row_bundle(tmp_path, spec_extra=add_gate)
    result = _result(case)
    assert result.status is ReplayStatus.NOT_REPLAYABLE
    assert result.reason_code == "FALLBACK_CANDIDATE_UNSUPPORTED"


def test_mapping_gate_does_not_enter_closed_route(tmp_path):
    def add_mapping_gate(document, mapping):
        document["reads"] = [{
            "id": "gate_input", "source": "primary", "column": FIRST_COLUMN}]
        document["conditions"] = [{
            "id": "use_fallback", "read": "gate_input",
            "predicate": "not_null"}]
        mapping["when"] = "use_fallback"

    case = _same_row_bundle(tmp_path, spec_extra=add_mapping_gate)
    result = _result(case)
    assert result.status is ReplayStatus.NOT_REPLAYABLE
    assert result.reason_code == "NAMED_READ_EVIDENCE_MISSING"
