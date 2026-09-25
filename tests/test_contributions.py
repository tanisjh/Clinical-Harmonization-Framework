"""Unit tests for lib/contributions.py — structured multi-input provenance.

Run:  python -m pytest tests/test_contributions.py -v
"""
from __future__ import annotations

import json

import pytest

from lib.contributions import (
    STATUS_NOT_EVALUATED,
    STATUS_REJECTED,
    STATUS_SELECTED,
    STATUS_UNAVAILABLE,
    Contribution,
    from_json,
    from_source_ref,
    selected_of,
    to_json,
)
from lib.source_ref import SourceRef

SYNTHETIC_SOURCE = "synthetic_measurements.csv"
SYNTHETIC_LABEL = "measurements.csv"


@pytest.fixture
def ref(tmp_path) -> SourceRef:
    source = tmp_path / SYNTHETIC_SOURCE
    source.write_text(
        "participant_key,measurement_a,measurement_b\n"
        "SYNTH-001,alpha,beta\n"
    )
    return SourceRef.build(
        config_key="measurements", path=source, project_root=tmp_path,
        file_type="csv", reader_id="cidc_csv@v1",
        logical_label=SYNTHETIC_LABEL,
    )


def test_status_vocabulary_is_enforced():
    with pytest.raises(ValueError):
        Contribution(role="x", status="maybe", source_column="C")


@pytest.mark.parametrize("status", [STATUS_SELECTED, STATUS_UNAVAILABLE,
                                    STATUS_REJECTED, STATUS_NOT_EVALUATED])
def test_valid_statuses_accepted(status):
    c = Contribution(role="candidate", status=status, source_column="C")
    assert c.status == status


def test_from_source_ref_copies_source_identity(ref):
    c = from_source_ref(
        ref, role="measurement_candidate", status=STATUS_SELECTED,
        source_column="measurement_a", raw_value="alpha", rank=0)
    assert c.source_resolved_path == SYNTHETIC_SOURCE
    assert c.source_sha256 == ref.sha256
    assert c.source_logical_label == SYNTHETIC_LABEL
    assert c.source_sheet == ""


def test_roundtrip_json(ref):
    cs = [
        from_source_ref(
            ref, role="measurement_candidate", status=STATUS_UNAVAILABLE,
            source_column="measurement_a", raw_value="", rank=0),
        from_source_ref(
            ref, role="measurement_candidate", status=STATUS_SELECTED,
            source_column="measurement_b", raw_value="alpha", rank=1),
    ]
    payload = to_json(cs)
    parsed = from_json(payload)
    assert len(parsed) == 2
    assert parsed[0]["status"] == STATUS_UNAVAILABLE
    assert parsed[1]["source_column"] == "measurement_b"
    assert selected_of(payload)["source_column"] == "measurement_b"


def test_empty_contributions_serialize_to_empty_string():
    assert to_json([]) == ""
    assert from_json("") == []
    assert selected_of("") is None


def test_json_is_deterministic(ref):
    cs = [from_source_ref(
        ref, role="r", status=STATUS_SELECTED,
        source_column="measurement_a", raw_value="alpha", rank=0)]
    assert to_json(cs) == to_json(cs)


def test_nan_raw_value_becomes_empty_string():
    c = Contribution(role="r", status=STATUS_UNAVAILABLE, source_column="C",
                     raw_value=float("nan"))
    assert c.as_record()["raw_value"] == ""


def test_none_raw_value_becomes_empty_string():
    c = Contribution(role="r", status=STATUS_UNAVAILABLE, source_column="C",
                     raw_value=None)
    assert c.as_record()["raw_value"] == ""


def test_non_scalar_raw_value_is_stringified():
    import numpy as np
    c = Contribution(role="r", status=STATUS_SELECTED, source_column="C",
                     raw_value=np.int64(7))
    rec = c.as_record()
    assert rec["raw_value"] == "7"
    json.dumps(rec)          # must be serializable


def test_selected_of_returns_none_when_nothing_selected(ref):
    cs = [from_source_ref(ref, role="r", status=STATUS_UNAVAILABLE,
                          source_column="measurement_a", rank=0),
          from_source_ref(ref, role="r", status=STATUS_UNAVAILABLE,
                          source_column="measurement_b", rank=1)]
    assert selected_of(to_json(cs)) is None


def test_contribution_is_immutable():
    c = Contribution(role="r", status=STATUS_SELECTED, source_column="C")
    with pytest.raises(Exception):
        c.status = STATUS_REJECTED
