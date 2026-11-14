"""Fabricated compatibility cases derived from the reviewed pivot reference."""
from __future__ import annotations

from dataclasses import asdict, replace

import pandas as pd
import pytest

from lib.pivot import pivot_cells_to_wide, threshold_for
from lib.provenance import Cell, FlagRow
from lib.schema_model import TargetContract, contract_for, parse


IDENTITY = ("trial", "cimac_part_id", "Cimac.id", "Collection_Event")


@pytest.fixture
def contract():
    return TargetContract("synthetic/pivot@1", "new-delivery.csv",
                          (*IDENTITY, "measurement", "note"), ("measurement",), IDENTITY)


def cell(**changes):
    original = Cell(
        trial="LAB", cimac_part_id="R1", Cimac_id="S1", Collection_Event="visit",
        harmonized_field="measurement", value="first", confidence=.8,
        source_file="observations.csv", source_column="reading", source_row_idx=0,
        extraction_method="fabricated", notes="observed note", rule_id="first.rule",
    )
    return replace(original, **changes)


@pytest.mark.parametrize(("config", "expected"), [
    ({}, .7), ({"default": .8}, .8), ({"default": "0.8"}, .8),
    ({"default": 0}, 0), ({"per_field": {"other": .1}}, .7),
    ({"default": .1, "per_field": {"measurement": .9}}, .9),
    ({"default": .9, "per_field": {"measurement": 0}}, 0),
    ({"per_field": {"measurement": "0.6"}}, .6),
])
def test_threshold_lookup(config, expected):
    assert threshold_for("measurement", config) == expected
    assert type(threshold_for("measurement", config)) is float


@pytest.mark.parametrize(("config", "error"), [
    ({"default": None}, TypeError), ({"default": "bad"}, ValueError),
    ({"per_field": {"measurement": None}}, TypeError),
    ({"per_field": {"measurement": ""}}, ValueError),
    ({"per_field": None}, AttributeError), (None, AttributeError),
])
def test_invalid_threshold_preserves_failure(config, error):
    with pytest.raises(error):
        threshold_for("measurement", config)


def test_rows_fields_and_replacement_keep_first_occurrence_order(contract):
    cells = [cell(cimac_part_id="B", confidence=.1),
             cell(cimac_part_id="A", harmonized_field="note", value="second row"),
             cell(cimac_part_id="B", value="winner", confidence=.9),
             cell(cimac_part_id="B", harmonized_field="note", value="first row")]
    wide, flags, suppressed = pivot_cells_to_wide(cells, {}, contract)
    assert list(wide.cimac_part_id) == ["B", "A"]
    assert list(wide.columns) == list(contract.columns)
    assert wide.loc[0, ["measurement", "note"]].tolist() == ["winner", "first row"]
    assert wide.loc[1, "note"] == "second row"
    assert not flags and not suppressed


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_confidence_first_wins_without_metadata_priority(contract, reverse):
    cells = [cell(), cell(value="second", source_file="preferred.csv",
                          source_column="exact", rule_id="priority.rule", mapping_id="first")]
    if reverse:
        cells.reverse()
    wide, flags, suppressed = pivot_cells_to_wide(cells, {}, contract)
    assert wide.at[0, "measurement"] == ("second" if reverse else "first")
    assert not flags and not suppressed


def test_higher_confidence_none_masks_lower_value(contract):
    wide, flags, suppressed = pivot_cells_to_wide(
        [cell(), cell(value=None, confidence=.9)], {}, contract)
    assert pd.isna(wide.at[0, "measurement"])
    assert len(flags) == 1 and not suppressed
    assert flags[0].reason_low_confidence == "value_NA_at_extraction (threshold=0.70)"
    assert flags[0].proposed_mapping == ""


@pytest.mark.parametrize("value", [0, False, float("nan"), pd.NA])
def test_only_none_is_absent(contract, value):
    wide, flags, suppressed = pivot_cells_to_wide([cell(value=value)], {}, contract)
    assert not flags and not suppressed
    if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        assert pd.isna(wide.at[0, "measurement"])
    else:
        assert wide.at[0, "measurement"] == value


def test_raw_snapshots_do_not_influence_selection_or_mutate_inputs(contract):
    cells = [cell(raw_value="raw", raw_value_json='{"kind":"string","value":"raw"}'),
             cell(value="second", raw_value=None, raw_value_json='{"kind":"null"}')]
    before = [asdict(item) for item in cells]
    wide, _, _ = pivot_cells_to_wide(cells, {}, contract)
    assert wide.at[0, "measurement"] == "first"
    assert [asdict(item) for item in cells] == before


@pytest.mark.parametrize("nan_first", [False, True])
def test_nan_confidence_order_sensitivity(contract, nan_first):
    cells = [cell(confidence=float("nan"), value="nan winner"), cell()]
    if not nan_first:
        cells.reverse()
    wide, flags, suppressed = pivot_cells_to_wide(cells, {}, contract)
    if nan_first:
        assert pd.isna(wide.at[0, "measurement"])
        assert len(flags) == len(suppressed) == 1
        assert pd.isna(suppressed[0]["confidence"])
        assert suppressed[0]["proposed"] == "nan winner"
    else:
        assert wide.at[0, "measurement"] == "first"
        assert not flags and not suppressed


def test_none_confidence_preserves_type_error(contract):
    with pytest.raises(TypeError):
        pivot_cells_to_wide([cell(confidence=None)], {}, contract)


@pytest.mark.parametrize(("confidence", "accepted"), [
    (.8, True), (.7, True), (.7 - 5e-10, True), (.7 - 2e-9, False), (.2, False),
])
def test_acceptance_epsilon(contract, confidence, accepted):
    wide, flags, suppressed = pivot_cells_to_wide([cell(confidence=confidence)], {}, contract)
    assert bool(flags) == bool(suppressed) == (not accepted)
    assert (wide.at[0, "measurement"] == "first") if accepted else pd.isna(wide.at[0, "measurement"])


def test_exact_diagnostic_and_withheld_payload_and_duplicates(contract):
    candidate = cell(confidence=.6)
    wide, flags, suppressed = pivot_cells_to_wide([candidate, replace(candidate)], {}, contract)
    assert len(wide) == len(flags) == len(suppressed) == 1
    assert pd.isna(wide.at[0, "measurement"])
    assert suppressed == [{
        "trial": "LAB", "cimac_part_id": "R1", "Cimac.id": "S1",
        "Collection_Event": "visit", "harmonized_field": "measurement",
        "confidence": .6, "threshold": .7, "proposed": "first", "input_rule_id": "first.rule",
    }]
    assert type(suppressed[0]["confidence"]) is type(suppressed[0]["threshold"]) is float
    assert flags == [FlagRow(
        trial="LAB", cimac_part_id="R1", Cimac_id="S1", Collection_Event="visit",
        harmonized_field="measurement", source_files="observations.csv",
        candidate_source_variables="reading", observed_source_values="observed note",
        proposed_mapping="first", confidence_score=.6,
        reason_low_confidence="confidence_below_threshold (threshold=0.70)",
        question_for_reviewer=("Is the proposed mapping for measurement ('first') correct given "
                               "source=observations.csv:reading? extraction_method=fabricated"),
    )]


def test_losers_produce_no_flags_and_decision_order_follows_group_insertion(contract):
    cells = [cell(cimac_part_id="B", confidence=.1),
             cell(cimac_part_id="A", confidence=.2),
             cell(cimac_part_id="B", value="replacement", confidence=.3)]
    _, flags, suppressed = pivot_cells_to_wide(cells, {}, contract)
    assert [item.cimac_part_id for item in flags] == ["B", "A"]
    assert [item["proposed"] for item in suppressed] == ["replacement", "first"]
    assert len(flags) == 2


@pytest.mark.parametrize("value", [None, "low"])
def test_all_absent_or_withheld_rows_remain(contract, value):
    wide, flags, suppressed = pivot_cells_to_wide([cell(value=value, confidence=.1)], {}, contract)
    assert len(wide) == len(flags) == 1
    assert pd.isna(wide.at[0, "measurement"])
    assert bool(suppressed) == (value is not None)


def test_unknown_target_is_projected_only_after_diagnostics(contract):
    wide, flags, suppressed = pivot_cells_to_wide(
        [cell(harmonized_field="unknown", confidence=.1)], {}, contract)
    assert list(wide.columns) == list(contract.columns) and len(wide) == 1
    assert flags[0].harmonized_field == suppressed[0]["harmonized_field"] == "unknown"


def test_empty_and_missing_columns_have_contract_order_and_presence(contract):
    wide, flags, suppressed = pivot_cells_to_wide([], {}, contract)
    assert wide.empty and list(wide.columns) == list(contract.columns)
    assert not flags and not suppressed
    wide, _, _ = pivot_cells_to_wide([cell(harmonized_field="note")], {}, contract)
    assert wide.at[0, "measurement"] is pd.NA  # required is presence, not non-null
    with pytest.raises(ValueError, match="requires columns"):
        pivot_cells_to_wide([], {}, replace(contract, required=("not_declared",)))


def test_schema_projection_and_no_datatype_normalization():
    schema = parse({
        "schema_id": "synthetic/projection", "schema_version": "1.0.0",
        "column_order": ["extra", "measurement"], "artifacts": ["new-delivery.csv", "other.csv"],
        "fields": [{"name": "measurement", "type": "integer"},
                   {"name": "extra", "artifacts": ["other.csv"]}],
    })
    for artifact, columns in (("new-delivery.csv", ["measurement"]),
                              ("other.csv", ["extra", "measurement"])):
        wide, _, _ = pivot_cells_to_wide([cell(value="12")], {}, contract_for(schema, artifact))
        assert list(wide.columns) == columns
        assert wide.at[0, "measurement"] == "12"


@pytest.mark.parametrize("authority", ["(none)", "different.csv"])
def test_authority_value_state_and_artifact_do_not_filter_candidates(contract, authority):
    wide, flags, suppressed = pivot_cells_to_wide(
        [cell(), cell(value="higher", confidence=.9, authoritative_for=authority,
                      value_state="suppressed")], {}, contract)
    assert wide.at[0, "measurement"] == "higher"
    assert not flags and not suppressed


def test_row_key_never_remaps_identity(contract):
    renamed = replace(contract, columns=("record_id", "measurement"), row_key=("record_id",))
    wide, _, _ = pivot_cells_to_wide([cell(), cell(Cimac_id="other")], {}, renamed)
    assert len(wide) == 2 and wide.record_id.isna().all()
    assert list(wide.measurement) == ["first", "first"]


def test_identity_field_withholding_keeps_seeded_metadata(contract):
    wide, flags, suppressed = pivot_cells_to_wide([
        cell(harmonized_field="cimac_part_id", value="different", confidence=.1),
    ], {}, contract)
    assert wide.at[0, "cimac_part_id"] == "R1"
    assert len(flags) == len(suppressed) == 1
