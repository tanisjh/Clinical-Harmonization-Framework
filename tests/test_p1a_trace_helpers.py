"""Synthetic coverage for the P1A trace-helper extraction."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from lib import migrated_trials
from lib.record_keys import parse_record_key
from lib.trace_io import read_source


@pytest.mark.parametrize(
    ("serialized", "expected"),
    [
        ("record_id=SYNTH-001", {"record_id": "SYNTH-001"}),
        (
            "record_id=SYNTH-001;group=GROUP-1",
            {"record_id": "SYNTH-001", "group": "GROUP-1"},
        ),
        ("record_id=SYNTH=001", {"record_id": "SYNTH=001"}),
        (" record_id = SYNTH-001 ", {" record_id ": " SYNTH-001 "}),
        ("=", {"": ""}),
        ("record_id=", {"record_id": ""}),
        ("record_id=SYNTH-001;record_id=RECORD-A", {"record_id": "RECORD-A"}),
    ],
)
def test_parse_record_key_preserves_supported_forms(serialized, expected):
    assert parse_record_key(serialized) == expected
    assert list(parse_record_key(serialized)) == list(expected)


@pytest.mark.parametrize(
    "serialized",
    [
        None,
        "",
        "config:example",
        "adapter:example",
        "provenance:example",
        "RECORD-A",
        "record_id=SYNTH-001;malformed",
        "record_id=SYNTH-001;",
    ],
)
def test_parse_record_key_preserves_empty_sentinel_and_malformed_behavior(serialized):
    assert parse_record_key(serialized) == {}


def test_parse_record_key_preserves_duplicate_order():
    parsed = parse_record_key("record_id=SYNTH-001;group=GROUP-1;record_id=RECORD-A")
    assert parsed == {"record_id": "RECORD-A", "group": "GROUP-1"}
    assert list(parsed) == ["record_id", "group"]


def test_read_source_reads_synthetic_csv_and_resets_index(tmp_path):
    source = tmp_path / "records.data"
    expected = pd.DataFrame(
        {"record_id": ["SYNTH-001", "SYNTH-002"], "measurement_a": [123.45, 67.0]}
    )
    expected.to_csv(source, index=False)

    actual = read_source(tmp_path, source.name)

    assert isinstance(actual, pd.DataFrame)
    assert list(actual.index) == [0, 1]
    pd.testing.assert_frame_equal(actual, expected)


def test_read_source_reads_default_and_explicit_xlsx_sheets(tmp_path):
    source = tmp_path / "records.xlsx"
    first = pd.DataFrame({"record_id": ["SYNTH-001"], "category_x": ["alpha"]})
    second = pd.DataFrame({"record_id": ["SYNTH-002"], "category_x": ["beta"]})
    with pd.ExcelWriter(source) as writer:
        first.to_excel(writer, sheet_name="first", index=False)
        second.to_excel(writer, sheet_name="second", index=False)

    pd.testing.assert_frame_equal(read_source(tmp_path, source.name), first)
    pd.testing.assert_frame_equal(read_source(tmp_path, source.name, "second"), second)


def test_read_source_missing_path_preserves_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_source(tmp_path, "absent.csv")


def test_migrated_trials_reexports_the_same_callables():
    assert migrated_trials.parse_record_key is parse_record_key
    assert migrated_trials.read_source is read_source


def test_trace_value_imports_neutral_helpers_not_migrated_registry():
    path = Path(__file__).resolve().parents[1] / "scripts" / "trace_value.py"
    tree = ast.parse(path.read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "lib.record_keys" in modules
    assert "lib.trace_io" in modules
    assert "lib.migrated_trials" not in modules
