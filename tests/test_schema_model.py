"""Schema loading and rejection.

The schema document is a contract, so loading it is where mistakes have to be
caught. A schema that silently ignores a misspelled key is worse than no schema:
it reports success for a constraint nobody is enforcing. Most tests here are
therefore negative — they prove a malformed schema is refused, not merely that a
good one loads.

Run:  python -m pytest tests/test_schema_model.py -v
"""
from __future__ import annotations

import textwrap

import pytest

from lib.schema_model import (
    ERROR,
    INFO,
    WARNING,
    SchemaError,
    load,
    parse,
)

MINIMAL = {
    "schema_id": "test/minimal",
    "schema_version": "1.0.0",
    "fields": [{"name": "a", "type": "string"}],
}


def doc(**overrides) -> dict:
    d = {k: (v.copy() if isinstance(v, (dict, list)) else v)
         for k, v in MINIMAL.items()}
    d.update(overrides)
    return d


# ── valid loading ───────────────────────────────────────────────────────────

def test_minimal_schema_loads():
    s = parse(doc())
    assert s.schema_id == "test/minimal"
    assert s.schema_version == "1.0.0"
    assert s.ref == "test/minimal@1.0.0"
    assert [f.name for f in s.fields] == ["a"]


def test_field_defaults_are_permissive_but_explicit():
    """An under-specified field must not silently acquire constraints."""
    f = parse(doc()).field("a")
    assert f.type == "string"
    assert f.required_column is True
    assert f.nullable is True
    assert f.vocabulary is None
    assert f.bounds is None
    assert f.pattern == ""
    assert f.value_states == ()
    assert f.open_questions == ()
    assert f.deprecated is False


def test_version_tuple_parses():
    assert parse(doc(schema_version="2.10.3")).version_tuple() == (2, 10, 3)


# ── version validation ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["1.0", "v1.0.0", "1.0.0-rc1", "", "one.0.0", 1.0])
def test_rejects_bad_schema_version(bad):
    with pytest.raises(SchemaError, match="schema_version"):
        parse(doc(schema_version=bad))


def test_rejects_missing_schema_id():
    d = doc()
    del d["schema_id"]
    with pytest.raises(SchemaError, match="schema_id"):
        parse(d)


# ── malformed documents ─────────────────────────────────────────────────────

def test_rejects_non_mapping_document():
    with pytest.raises(SchemaError, match="mapping"):
        parse(["not", "a", "mapping"])


def test_rejects_empty_fields():
    with pytest.raises(SchemaError, match="fields"):
        parse(doc(fields=[]))


def test_rejects_unknown_document_property():
    with pytest.raises(SchemaError, match="unknown propert"):
        parse(doc(colour="blue"))


def test_rejects_unknown_field_property():
    with pytest.raises(SchemaError, match="unknown propert"):
        parse(doc(fields=[{"name": "a", "typo_type": "string"}]))


def test_rejects_duplicate_field_definitions():
    with pytest.raises(SchemaError, match="duplicate field"):
        parse(doc(fields=[{"name": "a"}, {"name": "b"}, {"name": "a"}]))


def test_rejects_unnamed_field():
    with pytest.raises(SchemaError, match="name is required"):
        parse(doc(fields=[{"type": "string"}]))


def test_rejects_unknown_type():
    with pytest.raises(SchemaError, match="type must be one of"):
        parse(doc(fields=[{"name": "a", "type": "timestamp"}]))


def test_rejects_unknown_role():
    with pytest.raises(SchemaError, match="role must be one of"):
        parse(doc(fields=[{"name": "a", "role": "biomarker"}]))


def test_rejects_unknown_severity():
    with pytest.raises(SchemaError, match="severity must be one of"):
        parse(doc(column_order_severity="fatal"))


def test_rejects_non_boolean_flag():
    with pytest.raises(SchemaError, match="must be a boolean"):
        parse(doc(fields=[{"name": "a", "nullable": "yes"}]))


# ── vocabulary ──────────────────────────────────────────────────────────────

def test_rejects_empty_vocabulary():
    with pytest.raises(SchemaError, match="non-empty list"):
        parse(doc(fields=[{"name": "a", "vocabulary": {"values": []}}]))


def test_rejects_duplicate_vocabulary_values():
    with pytest.raises(SchemaError, match="duplicates"):
        parse(doc(fields=[{"name": "a", "vocabulary": {"values": ["x", "x"]}}]))


def test_rejects_canonical_value_outside_vocabulary():
    with pytest.raises(SchemaError, match="not in values"):
        parse(doc(fields=[{"name": "a",
                           "vocabulary": {"values": ["x"], "canonical": ["y"]}}]))


def test_rejects_vocabulary_on_numeric_field():
    with pytest.raises(SchemaError, match="only meaningful on a string"):
        parse(doc(fields=[{"name": "a", "type": "integer",
                           "vocabulary": {"values": ["x"]}}]))


def test_non_canonical_is_derived_not_declared():
    s = parse(doc(fields=[{"name": "a", "vocabulary": {
        "values": ["x", "Y", "z"], "canonical": ["x", "z"]}}]))
    assert s.field("a").vocabulary.non_canonical == ("Y",)


# ── bounds ──────────────────────────────────────────────────────────────────

def test_rejects_bounds_on_string_field():
    with pytest.raises(SchemaError, match="only meaningful on a numeric"):
        parse(doc(fields=[{"name": "a", "type": "string", "bounds": {"min": 0}}]))


def test_rejects_empty_bounds():
    with pytest.raises(SchemaError, match="at least one of min/max"):
        parse(doc(fields=[{"name": "a", "type": "integer", "bounds": {}}]))


def test_rejects_inverted_bounds():
    with pytest.raises(SchemaError, match="min exceeds"):
        parse(doc(fields=[{"name": "a", "type": "integer",
                           "bounds": {"min": 10, "max": 1}}]))


def test_rejects_non_numeric_bound():
    with pytest.raises(SchemaError, match="must be numeric"):
        parse(doc(fields=[{"name": "a", "type": "integer",
                           "bounds": {"min": "low"}}]))


# ── patterns ────────────────────────────────────────────────────────────────

def test_rejects_invalid_regex():
    with pytest.raises(SchemaError, match="not a valid regex"):
        parse(doc(fields=[{"name": "a", "pattern": "([unclosed"}]))


# ── structural references ───────────────────────────────────────────────────

def test_rejects_row_key_naming_undeclared_field():
    with pytest.raises(SchemaError, match="row_key names undeclared"):
        parse(doc(row_key=["a", "ghost"]))


def test_rejects_column_order_omitting_a_declared_field():
    with pytest.raises(SchemaError, match="column_order omits"):
        parse(doc(fields=[{"name": "a"}, {"name": "b"}], column_order=["a"]))


def test_rejects_column_order_naming_undeclared_field():
    with pytest.raises(SchemaError, match="column_order names undeclared"):
        parse(doc(column_order=["a", "ghost"]))


# ── cross-field constraints ─────────────────────────────────────────────────

def test_rejects_constraint_on_undeclared_field():
    with pytest.raises(SchemaError, match="not a declared field"):
        parse(doc(constraints=[{"id": "c1", "kind": "requires_non_null",
                                "field": "a", "other": "ghost"}]))


def test_rejects_unknown_constraint_kind():
    with pytest.raises(SchemaError, match="kind must be one of"):
        parse(doc(fields=[{"name": "a"}, {"name": "b"}],
                  constraints=[{"id": "c1", "kind": "correlates_with",
                                "field": "a", "other": "b"}]))


def test_rejects_duplicate_constraint_ids():
    with pytest.raises(SchemaError, match="duplicate constraint ids"):
        parse(doc(fields=[{"name": "a"}, {"name": "b"}],
                  constraints=[
                      {"id": "c1", "kind": "requires_non_null", "field": "a", "other": "b"},
                      {"id": "c1", "kind": "requires_non_null", "field": "b", "other": "a"}]))


# ── file loading ────────────────────────────────────────────────────────────

def test_load_missing_file_raises_schema_error(tmp_path):
    with pytest.raises(SchemaError, match="not found"):
        load(tmp_path / "absent.yaml")


def test_load_invalid_yaml_raises_schema_error(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("schema_id: [unclosed\n")
    with pytest.raises(SchemaError, match="not valid YAML"):
        load(p)


def test_load_records_sha256(tmp_path):
    p = tmp_path / "ok.yaml"
    p.write_text(textwrap.dedent("""\
        schema_id: test/file
        schema_version: 0.1.0
        fields:
          - name: a
        """))
    s = load(p)
    assert len(s.sha256) == 64
    assert s.source_path == str(p)


def test_severity_constants_are_distinct():
    assert len({ERROR, WARNING, INFO}) == 3
