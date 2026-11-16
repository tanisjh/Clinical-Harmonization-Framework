"""Fabricated Cell-v1 construction parity and explicit identity boundary tests."""
from __future__ import annotations

from dataclasses import fields
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from lib.extractor_base import BaseExtractor
from lib.provenance import Cell
from lib.source_ref import SourceRef


class SyntheticExtractor(BaseExtractor):
    trial_dir_name = "synthetic"
    template_trial_name = "synthetic-trial"


@pytest.fixture
def base(tmp_path):
    return SyntheticExtractor(tmp_path, pd.DataFrame(), {}, {},
                              run_id="run-1", spec_ref="spec@1")


ROW = {"cimac_part_id": "SYNTH-001", "Cimac.id": "SAMPLE-001",
       "Collection_Event": "event-1"}
FIELD_ORDER = (
    "trial", "cimac_part_id", "Cimac_id", "Collection_Event", "harmonized_field",
    "value", "confidence", "source_file", "source_column", "source_row_idx",
    "extraction_method", "notes", "source_logical_label", "source_resolved_path",
    "source_sha256", "source_sheet", "source_row_number", "source_record_key",
    "raw_value", "rule_id", "rule_version", "contributions", "authoritative_for",
    "derivation", "value_state", "run_id", "spec_ref", "mapping_id",
    "source_ref_id", "raw_value_json", "evidence_refs",
)


def make_ref(sheet=None):
    # Fabricated reader observation; no clinical file or source I/O required.
    return SourceRef(config_key="data", logical_label="stable.csv",
                     resolved_path="inputs/records.csv", basename="records.csv",
                     sha256="a" * 64, file_type="csv", input_root="synthetic",
                     reader_id="cidc_csv@v1", sheet=sheet, source_ref_id="src-synthetic")


def build(base, row=None, source="CONFIG:constant", **kwargs):
    return base.cell(ROW if row is None else row, "measurement", 42, "0.75",
                     source, "value", "legacy-index", "direct", "notes", **kwargs)


@pytest.mark.parametrize("value,expected", [
    (None, ""), (float("nan"), ""), (np.float64("nan"), ""),
    (pd.NA, "<NA>"), (pd.NaT, "NaT"), (np.float32("nan"), "nan"),
    (Decimal("NaN"), "NaN"), (0, "0"), (False, "False"), ("None", "None"),
])
def test_anchor_string_exact_legacy_null_detection(base, value, expected):
    assert base._anchor_str(value) == expected
    row = dict.fromkeys(ROW, value)
    cell = build(base, row)
    assert (cell.cimac_part_id, cell.Cimac_id, cell.Collection_Event) == (expected,) * 3


@pytest.mark.parametrize("key", list(ROW))
def test_missing_identity_has_no_alias_fallback_and_counts_failure(base, key):
    row = {**ROW, "participant": "alternative", "sample": "alternative",
           "collection_event": "alternative", "trial": "row-trial"}
    del row[key]
    with pytest.raises(KeyError) as exc:
        build(base, row)
    assert exc.value.args == (key,)
    assert base._legacy_label_uses == 1


def test_identity_uses_exact_keys_and_verbatim_current_trial(base):
    row = {**ROW, "participant": "wrong", "sample": "wrong",
           "collection_event": "wrong", "trial": "wrong"}
    base.template_trial_name = 17
    cell = build(base, row)
    assert cell.trial == 17
    assert (cell.cimac_part_id, cell.Cimac_id, cell.Collection_Event) == tuple(ROW.values())


@pytest.mark.parametrize("source", ["records.csv", "CONFIG:constant", "DERIVED:value", "", None, 0])
def test_legacy_labels_are_verbatim_with_blank_structured_fields(base, source):
    cell = build(base, source=source)
    assert cell.source_file is source and cell.source_logical_label is source
    assert (cell.source_resolved_path, cell.source_sha256,
            cell.source_sheet, cell.source_ref_id) == ("",) * 4
    assert base._legacy_label_uses == 1


@pytest.mark.parametrize("sheet", [None, "first"])
def test_source_ref_metadata_and_no_legacy_accounting(base, sheet):
    ref = make_ref(sheet)
    cell = build(base, source=ref)
    assert cell.source_file == cell.source_logical_label == ref.logical_label
    assert cell.source_resolved_path == ref.resolved_path
    assert cell.source_sha256 == ref.sha256 and cell.source_ref_id == ref.source_ref_id
    assert cell.source_sheet == (sheet or "")
    assert base._legacy_label_uses == 0


@pytest.mark.parametrize("kwargs,legacy,encoded", [
    ({}, "", ""), ({"raw_value": None}, "", '{"kind":"null"}'),
    ({"raw_value": 0}, 0, '{"kind":"integer","value":"0"}'),
    ({"raw_value": False}, False, '{"kind":"boolean","value":false}'),
    ({"raw_value": "0"}, "0", '{"kind":"string","value":"0"}'),
    ({"raw_value": 2.5}, 2.5, '{"kind":"number","value":"2.5"}'),
    ({"raw_value": None, "raw_evidence_recorded": False}, "", ""),
    ({"raw_value": 0, "raw_evidence_recorded": False}, 0, ""),
])
def test_raw_absence_null_and_falsey_values(base, kwargs, legacy, encoded):
    cell = build(base, **kwargs)
    assert type(cell.raw_value) is type(legacy) and cell.raw_value == legacy
    assert cell.raw_value_json == encoded


@pytest.mark.parametrize("source,is_legacy", [("legacy", True), (make_ref(), False)])
@pytest.mark.parametrize("failure", ["identity", "confidence", "raw"])
def test_counter_precedes_later_failures(base, source, is_legacy, failure):
    args = {"source": source}
    error = ValueError
    if failure == "identity":
        args["row"] = {}
        error = KeyError
    elif failure == "confidence":
        # Call directly to supply invalid confidence.
        with pytest.raises(ValueError):
            base.cell(ROW, "measurement", 42, "invalid", source, "value", 0, "direct")
        assert base._legacy_label_uses == int(is_legacy)
        return
    else:
        args["raw_value"] = {"unsupported": True}
    with pytest.raises(error):
        build(base, **args)
    assert base._legacy_label_uses == int(is_legacy)


def test_unsupported_raw_is_retained_when_evidence_disabled(base):
    raw = {"unsupported": True}
    cell = build(base, raw_value=raw, raw_evidence_recorded=False)
    assert cell.raw_value is raw and cell.raw_value_json == ""


@pytest.mark.parametrize("confidence", [-2, 3, True, float("inf")])
def test_confidence_has_float_conversion_without_new_range_checks(base, confidence):
    cell = base.cell(ROW, "measurement", 42, confidence, "legacy", "value", 0, "direct")
    assert cell.confidence == float(confidence) and type(cell.confidence) is float


@pytest.mark.parametrize("row_number", [None, 0, False, "row-1"])
@pytest.mark.parametrize("record_key", [None, "", 0, False, "record=SYNTH-001"])
def test_locator_none_and_falsey_behavior(base, row_number, record_key):
    cell = build(base, source_row_number=row_number, source_record_key=record_key)
    expected = "" if row_number is None else row_number
    assert type(cell.source_row_number) is type(expected) and cell.source_row_number == expected
    assert cell.source_record_key == (record_key or "")
    assert cell.source_row_idx == "legacy-index" and cell.source_column == "value"


def test_exact_cell_fields_record_order_and_provenance_passthrough(base):
    options = dict(mapping_id="map-1", rule_id="rule-1", rule_version="1",
                   contributions='[{"order":2},{"order":1}]', derivation='{"z":0,"a":1}',
                   value_state="provisional", authoritative_for="z;a",
                   evidence_refs="evidence-z;evidence-a", raw_value=False,
                   source_row_number=0, source_record_key="record=SYNTH-001")
    cell = build(base, source=make_ref("first"), **options)
    assert isinstance(cell, Cell)
    assert tuple(f.name for f in fields(cell)) == FIELD_ORDER
    record = cell.as_record()
    assert tuple(record) == tuple(k for k in FIELD_ORDER if k != "Cimac_id") + ("Cimac.id",)
    assert record["Cimac.id"] == ROW["Cimac.id"] and "Cimac_id" not in record
    for key, value in options.items():
        assert getattr(cell, key) == value
    assert cell.run_id == "run-1" and cell.spec_ref == "spec@1"
    assert cell.harmonized_field == "measurement" and cell.value == 42
    assert cell.confidence == 0.75 and cell.notes == "notes"
    assert cell.extraction_method == "direct"


@pytest.mark.parametrize("source", ["CONFIG:constant", make_ref("first")])
@pytest.mark.parametrize("options", [
    {}, {"raw_value": None}, {"raw_value": 0}, {"raw_value": False},
    {"raw_value": "text"}, {"raw_value": 2.5},
    {"raw_value": None, "raw_evidence_recorded": False},
    {"raw_value": 0, "raw_evidence_recorded": False},
])
def test_explicit_factory_matches_characterized_facade(base, source, options):
    from lib.cell_factory import CellFactory, CellIdentity

    identity = CellIdentity(base.template_trial_name, *ROW.values())
    factory = CellFactory()
    kwargs = dict(source_row_number=0, source_record_key="record=SYNTH-001",
                  mapping_id="map-1", rule_id="rule-1", rule_version="1",
                  contributions='[{"z":0,"a":1}]', derivation='{"z":0,"a":1}',
                  authoritative_for="z;a", value_state="provisional", evidence_refs="z;a",
                  **options)
    expected = build(base, source=source, **kwargs)
    actual = factory.build(identity, "measurement", 42, "0.75", source, "value",
                           "legacy-index", "direct", "notes", run_id="run-1",
                           spec_ref="spec@1", **kwargs)
    assert actual == expected
    assert tuple(actual.as_record().items()) == tuple(expected.as_record().items())
    assert factory.legacy_label_uses == base._legacy_label_uses


def test_identity_is_explicit_frozen_and_separate_from_legacy_adaptation():
    from dataclasses import FrozenInstanceError

    from lib.cell_factory import CellFactory, CellIdentity, compatibility_identity

    identity = compatibility_identity("trial", {
        "cimac_part_id": None, "Cimac.id": False, "Collection_Event": pd.NA,
        "participant": "ignored", "trial": "ignored"})
    assert identity == CellIdentity("trial", "", "False", "<NA>")
    assert tuple(f.name for f in fields(identity)) == ("trial", "participant", "sample", "collection_event")
    with pytest.raises(FrozenInstanceError):
        identity.trial = "changed"
    # The factory consumes explicit values; it does not consult a row/schema/profile.
    cell = CellFactory().build(CellIdentity("t", "p", "s", "e"), "f", 0, 1,
                               "legacy", "column", 0, "direct")
    assert (cell.trial, cell.cimac_part_id, cell.Cimac_id, cell.Collection_Event) == ("t", "p", "s", "e")


@pytest.mark.parametrize("options,error", [
    ({"confidence": "invalid"}, ValueError),
    ({"confidence": None}, TypeError),
    ({"raw_value": {"unsupported": True}}, ValueError),
])
@pytest.mark.parametrize("source,expected_count", [("legacy", 1), (make_ref(), 0)])
def test_direct_factory_failure_counter(source, expected_count, options, error):
    from lib.cell_factory import CellFactory, CellIdentity

    factory = CellFactory()
    confidence = options.get("confidence", 1)
    kwargs = {k: v for k, v in options.items() if k != "confidence"}
    with pytest.raises(error):
        factory.build(CellIdentity("t", "p", "s", "e"), "f", 1, confidence,
                      source, "column", 0, "direct", **kwargs)
    assert factory.legacy_label_uses == expected_count


def test_raw_sentinel_singleton_and_defaults(base):
    import inspect

    from lib import cell_factory, extractor_base

    sentinel = cell_factory.RAW_VALUE_ABSENT
    assert extractor_base._RAW_VALUE_ABSENT is sentinel
    for method in (BaseExtractor.cell, cell_factory.CellFactory.build,
                   cell_factory.CellFactory._build_prepared):
        assert inspect.signature(method).parameters["raw_value"].default is sentinel
    assert build(base, raw_value=sentinel).raw_value_json == ""
    factory = cell_factory.CellFactory()
    cell = factory.build(cell_factory.CellIdentity("t", "p", "s", "e"),
                         "f", 1, 1, "legacy", "column", 0, "direct", raw_value=sentinel)
    assert cell.raw_value == cell.raw_value_json == ""


def test_factory_has_no_source_io_or_policy_dependencies(tmp_path, monkeypatch):
    import ast
    from pathlib import Path

    from lib import cell_factory

    tree = ast.parse(Path(cell_factory.__file__).read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imports == {"__future__", "dataclasses", "typing", "provenance", "replay_contract", "source_ref"}

    def forbidden(*args, **kwargs):
        pytest.fail("CellFactory performed source I/O")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        factory = cell_factory.CellFactory()
        cell = factory.build(cell_factory.CellIdentity("t", "p", "s", "e"),
                             "f", 1, 1, make_ref(), "column", 0, "direct")
    assert cell.source_ref_id == "src-synthetic" and factory.legacy_label_uses == 0
