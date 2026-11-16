"""Fabricated canonical assembly keys, independent of Cell identity adaptation."""
from types import SimpleNamespace

import pandas as pd
import pytest

from lib import canonical
from lib.schema_model import TargetContract, contract_for, parse


IDENTITY = ("trial", "cimac_part_id", "Cimac.id", "Collection_Event")


def make_schema(columns, row_key=None, identifier=""):
    document = {
        "schema_id": "synthetic/canonical-keys",
        "schema_version": "1.0.0",
        "column_order": list(columns),
        "fields": [{"name": name, "role": "identifier" if name == identifier else "annotation"}
                   for name in columns],
    }
    if row_key is not None:
        document["row_key"] = list(row_key)
    return parse(document)


def make_trial(frame):
    return canonical.CanonicalTrial("synthetic/keys@1.0.0", "LAB", frame)


def test_explicit_key_preserves_order_over_legacy_columns():
    columns = (*IDENTITY, "record_a", "record_b")
    contract = contract_for(make_schema(columns, ("record_b", "record_a")))
    trial = make_trial(pd.DataFrame(columns=columns))
    assert isinstance(contract, TargetContract)
    assert canonical._record_key_columns(trial, contract) == ["record_b", "record_a"]


def test_distinct_explicit_keys_accept_identical_legacy_identity():
    frame = pd.DataFrame([
        ["LAB", "P1", "S1", "visit", "A"],
        ["LAB", "P1", "S1", "visit", "B"],
    ], columns=(*IDENTITY, "record_id"))
    result = canonical.assemble([make_trial(frame)], make_schema(frame.columns, ("record_id",)))
    pd.testing.assert_frame_equal(result, frame.astype(object))


def test_duplicate_explicit_keys_refused_despite_distinct_legacy_identity():
    frame = pd.DataFrame([
        ["LAB", "P1", "S1", "visit", "A"],
        ["LAB", "P2", "S2", "later", "A"],
    ], columns=(*IDENTITY, "record_id"))
    with pytest.raises(canonical.CanonicalError, match=r"2 rows sharing a record key \['record_id'\]"):
        canonical.assemble([make_trial(frame)], make_schema(frame.columns, ("record_id",)))


def test_duplicate_explicit_composite_key_reports_declared_order():
    frame = pd.DataFrame([["A", "B", 1], ["A", "B", 2]], columns=("first", "second", "value"))
    with pytest.raises(canonical.CanonicalError) as caught:
        canonical.assemble([make_trial(frame)], make_schema(frame.columns, ("second", "first")))
    assert "key ['second', 'first']" in str(caught.value)
    assert "First: ['B | A']" in str(caught.value)


@pytest.mark.parametrize("columns,expected", [
    (("value", "Collection_Event", "Cimac.id", "cimac_part_id", "trial"), list(IDENTITY)),
    (("Cimac.id", "value", "trial"), ["trial", "Cimac.id"]),
    (("Collection_Event", "cimac_part_id"), ["cimac_part_id", "Collection_Event"]),
    (("value",), []),
])
@pytest.mark.parametrize("row_key", [None, ()], ids=["omitted", "empty"])
def test_no_key_uses_only_present_legacy_columns_in_fixed_order(columns, expected, row_key):
    trial = make_trial(pd.DataFrame(columns=columns))
    contract = contract_for(make_schema(columns, row_key))
    assert contract.row_key == ()
    assert canonical._record_key_columns(trial, contract) == expected


@pytest.mark.parametrize("row_key", [None, ()], ids=["omitted", "empty"])
def test_no_key_duplicate_legacy_identity_refused_even_when_values_differ(row_key):
    frame = pd.DataFrame([
        ["LAB", "P1", "S1", "visit", 1],
        ["LAB", "P1", "S1", "visit", 2],
    ], columns=(*IDENTITY, "value"))
    with pytest.raises(canonical.CanonicalError) as caught:
        canonical.assemble([make_trial(frame)], make_schema(frame.columns, row_key))
    assert f"2 rows sharing a record key {list(IDENTITY)}" in str(caught.value)


def test_no_key_distinct_samples_and_events_remain_in_input_order():
    frame = pd.DataFrame([
        ["LAB", "P1", "S2", "visit", 2],
        ["LAB", "P1", "S1", "visit", 1],
        ["LAB", "P1", "S1", "later", 3],
    ], columns=(*IDENTITY, "value"))
    schema = make_schema(frame.columns)
    trials = [make_trial(frame.iloc[:1]), make_trial(frame.iloc[1:])]
    result = canonical.assemble(trials, schema)
    pd.testing.assert_frame_equal(result, frame.astype(object))


def test_no_key_partial_legacy_identity_still_detects_duplicates():
    frame = pd.DataFrame([[1, "P1", "LAB"], [2, "P1", "LAB"]],
                         columns=("value", "cimac_part_id", "trial"))
    with pytest.raises(canonical.CanonicalError, match=r"key \['trial', 'cimac_part_id'\]"):
        canonical.assemble([make_trial(frame)], make_schema(frame.columns))


def test_real_contract_does_not_derive_keys_from_schema_identifier_roles():
    frame = pd.DataFrame([["LAB", "A"], ["LAB", "B"]], columns=("trial", "role_id"))
    schema = make_schema(frame.columns, identifier="role_id")
    contract = contract_for(schema)
    assert isinstance(contract, TargetContract)
    assert not hasattr(contract, "identifier_fields")
    assert canonical._record_key_columns(make_trial(frame), contract) == ["trial"]
    with pytest.raises(canonical.CanonicalError, match=r"key \['trial'\]"):
        canonical.assemble([make_trial(frame)], schema)


def test_contract_without_row_key_attribute_keeps_attribute_error():
    trial = make_trial(pd.DataFrame(columns=IDENTITY))
    with pytest.raises(AttributeError, match="row_key"):
        canonical._record_key_columns(trial, SimpleNamespace(identifier_fields=[]))


def test_private_role_duck_without_declared_key_uses_fixed_fallback():
    trial = make_trial(pd.DataFrame(columns=("role_id", "Cimac.id", "trial")))
    contract = SimpleNamespace(row_key=(), identifier_fields=[SimpleNamespace(name="role_id")])
    assert canonical._record_key_columns(trial, contract) == ["trial", "Cimac.id"]
