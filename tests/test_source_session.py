"""Fabricated source/session behavior with independent observation expectations."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from lib import source_ref, xlsx_reader
from lib.source_ref import PathContainmentError, SourceRef, assert_within_roots
from lib.source_session import SourceSession
from lib.xlsx_reader import (
    XLSX_READER_FLAGS, XLSX_READER_ID, XLSX_SHEET_CANONICALIZATION_ID,
    XlsxReaderError, xlsx_sheet_sha256,
)


@pytest.fixture
def make_subject(tmp_path):
    def make(sources=None):
        sources = sources or {"data": "records.csv"}
        session = SourceSession(
            tmp_path,
            resolve_source=lambda key: assert_within_roots(
                session.project_root / sources[key], session.project_root),
            context_name=lambda: "synthetic")
        return session
    return make


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("record_id,value\nSYNTH-001,12\nSYNTH-002,34\n")
    return path


@pytest.fixture
def workbook(tmp_path):
    path = tmp_path / "records.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"record_id": ["SYNTH-001"], "value": [12]}).to_excel(
            writer, sheet_name="first", index=False)
        pd.DataFrame({"record_id": ["SYNTH-002"], "value": [34]}).to_excel(
            writer, sheet_name="second", index=False)
    return path


@pytest.mark.parametrize("preamble", [False, True])
@pytest.mark.parametrize("encoding", ["utf-8", "latin-1"])
def test_csv_values_and_exact_ref(make_subject, tmp_path, preamble, encoding):
    path = tmp_path / "records.csv"
    payload = (("version,2026-01-01\n" if preamble else "")
               + "record_id,value\nSYNTH-001,café\n").encode(encoding)
    path.write_bytes(payload)
    subject = make_subject()
    frame = subject.load_csv("data", logical_label="stable.csv")
    assert frame.to_dict("records") == [{"record_id": "SYNTH-001", "value": "café"}]
    flags = {"encoding": encoding, "header": 0, "low_memory": False,
             "skiprows": int(preamble)}
    expected = SourceRef.build(
        config_key="data", path=path, project_root=tmp_path,
        file_type="csv", reader_id="cidc_csv@v1", logical_label="stable.csv",
        reader_flags=f"preamble_skipped={preamble};encoding={encoding}",
        reader_effective_flags=flags, n_rows=1)
    assert subject.ref("data") == expected
    assert expected.sha256 == hashlib.sha256(payload).hexdigest()
    assert expected.resolved_path == path.name
    assert expected.sheet is None
    assert expected.parsed_columns == ()


def test_csv_cache_mutation_labels_and_io(make_subject, csv_path, monkeypatch):
    reads, hashes = [], []
    original_read, original_hash = pd.read_csv, source_ref.sha256_file

    def read(*args, **kwargs):
        reads.append((args, kwargs))
        return original_read(*args, **kwargs)

    def digest(path, *args, **kwargs):
        hashes.append(path)
        return original_hash(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read)
    monkeypatch.setattr(source_ref, "sha256_file", digest)
    subject = make_subject({"data": csv_path.name, "alias": csv_path.name,
                            "unlabeled": csv_path.name})
    first = subject.load_csv("data")
    ref = subject.ref("data")
    assert ref.logical_label == csv_path.name
    first.loc[0, "value"] = 99
    assert subject.load_csv("data", "ignored.csv") is first
    assert subject.ref("data") is ref
    assert first.loc[0, "value"] == 99
    assert subject._cache == {str(csv_path): first}
    assert subject.load_csv("unlabeled") is first
    assert "unlabeled" not in subject.source_refs()
    assert subject.load_csv("alias", "alias.csv") is first
    alias = subject.ref("alias")
    assert alias.config_key == "alias" and alias.logical_label == "alias.csv"
    assert alias.source_ref_id != ref.source_ref_id
    assert alias.reader_effective_flags == ref.reader_effective_flags
    assert alias.sha256 == ref.sha256
    assert len(reads) == 1 and len(hashes) == 2
    assert reads[0][1] == {"skiprows": 0, "encoding": "utf-8", "low_memory": False}
    # An existing unlabeled cache alias gains a ref only when a truthy label arrives.
    subject.load_csv("unlabeled", "")
    assert "unlabeled" not in subject.source_refs()
    subject.load_csv("unlabeled", "now-labeled.csv")
    assert subject.ref("unlabeled").logical_label == "now-labeled.csv"
    assert len(reads) == 1 and len(hashes) == 3


def test_csv_labeled_alias_rehashes_current_bytes_with_cached_flags(
        make_subject, csv_path):
    subject = make_subject({"data": csv_path.name, "alias": csv_path.name})
    frame = subject.load_csv("data")
    first = subject.ref("data")
    csv_path.write_text("version,2026-01-01\nrecord_id,value\nSYNTH-NEW,90\n")
    assert subject.load_csv("alias", "alias.csv") is frame
    alias = subject.ref("alias")
    assert alias.sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert alias.sha256 != first.sha256
    assert alias.n_rows == 2
    assert alias.reader_effective_flags == first.reader_effective_flags


@pytest.mark.parametrize("payload,note", [
    (b"", "empty_file"),
    (b'a,b\n1,"unterminated\n', "ParserError:"),
])
def test_csv_warning_and_empty_frame(make_subject, tmp_path, caplog, payload, note):
    (tmp_path / "records.csv").write_bytes(payload)
    subject = make_subject()
    with caplog.at_level("WARNING", logger="extractor"):
        frame = subject.load_csv("data")
    assert frame.empty and subject.ref("data").n_rows == 0
    assert len(caplog.records) == 1
    assert caplog.records[0].name == "extractor"
    assert f"CSV read note for records.csv: {note}" in caplog.text
    subject.load_csv("data")
    assert len(caplog.records) == 1


def test_missing_file_has_no_ref_or_cache(make_subject):
    subject = make_subject()
    with pytest.raises(FileNotFoundError):
        subject.load_csv("data")
    assert subject.source_refs() == {} and subject._cache == {}


@pytest.mark.parametrize("kind", ["parent", "symlink", "outputs"])
def test_containment_before_io(make_subject, tmp_path, kind):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("record_id\nSYNTH-001\n")
    if kind == "parent":
        relative = "../outside.csv"
    elif kind == "symlink":
        (root / "link.csv").symlink_to(outside)
        relative = "link.csv"
    else:
        relative = "harmonization_outputs/records.csv"
    subject = make_subject({"data": relative})
    subject.project_root = root
    # The caller-supplied resolver consults the live session root.
    with pytest.raises(PathContainmentError):
        subject.load_csv("data")
    assert subject.source_refs() == {}


def test_contained_symlink_uses_resolved_identity(make_subject, csv_path, tmp_path):
    (tmp_path / "link.csv").symlink_to(csv_path)
    subject = make_subject({"data": "link.csv"})
    subject.load_csv("data")
    assert subject.ref("data").resolved_path == csv_path.name
    assert subject.ref("data").basename == csv_path.name


@pytest.mark.parametrize("sheet,expected", [(None, 12), ("first", 12), (0, 12),
                                           ("second", 34), (1, 34)])
def test_legacy_excel_sheets_cache_and_ref(
        make_subject, workbook, tmp_path, monkeypatch, sheet, expected):
    calls = []
    original = pd.read_excel

    def read(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_excel", read)
    subject = make_subject({"data": workbook.name, "alias": workbook.name})
    frame = subject.load_xlsx("data", sheet, "stable.xlsx")
    assert frame.loc[0, "value"] == expected
    frame.loc[0, "value"] = 99
    assert subject.load_xlsx("data", sheet) is frame
    assert subject.load_xlsx("alias", sheet, "alias.xlsx") is frame
    assert "alias" not in subject.source_refs()
    assert subject._cache[f"{workbook}::{sheet}"] is frame
    assert len(calls) == 1
    assert calls[0][1] == ({} if sheet is None else {"sheet_name": sheet})
    assert subject.ref("data") == SourceRef.build(
        config_key="data", path=workbook, project_root=tmp_path,
        file_type="xlsx", reader_id="excel@v1", logical_label="stable.xlsx",
        sheet=str(sheet) if sheet is not None else None, n_rows=1)
    assert subject.load_xlsx("data", sheet).loc[0, "value"] == 99


def test_legacy_excel_none_name_and_index_have_distinct_cache_keys(make_subject, workbook):
    subject = make_subject({"data": workbook.name})
    default = subject.load_xlsx("data")
    default_ref = subject.ref("data")
    named = subject.load_xlsx("data", "first")
    assert named is not default
    assert subject.ref("data").sheet == "first"
    indexed = subject.load_xlsx("data", 0, reader_id="legacy-unknown")
    assert indexed is not named and indexed is not default
    assert subject.ref("data").reader_id == "excel@v1"
    assert subject.ref("data").sheet == "0"
    assert subject.load_xlsx("data") is default
    # A cache hit does not replace the most recently created ref.
    assert subject.ref("data").sheet == "0" and default_ref.sheet is None


@pytest.mark.parametrize("sheet", [None, "", 0, 1])
def test_trusted_xlsx_requires_exact_sheet(make_subject, workbook, sheet):
    subject = make_subject({"data": workbook.name})
    with pytest.raises(ValueError, match="requires a non-empty exact sheet name"):
        subject.load_xlsx("data", sheet, reader_id=XLSX_READER_ID)
    assert subject.source_refs() == {} and subject._xlsx_observations == {}


def test_trusted_xlsx_observation_copies_aliases_and_io(
        make_subject, workbook, monkeypatch):
    reads, hashes, parses = [], [], []
    original_bytes, original_hash = Path.read_bytes, source_ref.sha256_file
    original_parse = xlsx_reader.parse_xlsx_bytes

    def parse(*args, **kwargs):
        parses.append(kwargs)
        return original_parse(*args, **kwargs)

    def read_bytes(path):
        if path == workbook:
            reads.append(path)
        return original_bytes(path)

    def digest(path, *args, **kwargs):
        hashes.append(path)
        return original_hash(path, *args, **kwargs)

    payload = workbook.read_bytes()
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(source_ref, "sha256_file", digest)
    monkeypatch.setattr(xlsx_reader, "parse_xlsx_bytes", parse)
    subject = make_subject({"data": workbook.name, "alias": workbook.name})
    frame = subject.load_xlsx("data", "first", "stable.xlsx", reader_id=XLSX_READER_ID)
    observation = subject._xlsx_observations["data"]
    ref = subject.ref("data")
    assert observation.source_ref is ref
    assert observation._payload == payload
    assert observation._frame is not frame
    assert ref.sha256 == hashlib.sha256(payload).hexdigest()
    assert ref.reader_id == XLSX_READER_ID and ref.sheet == "first"
    assert ref.logical_label == "stable.xlsx"
    assert ref.resolved_path == workbook.name
    assert dict(ref.reader_effective_flags) == XLSX_READER_FLAGS
    assert ref.parsed_canonicalization == XLSX_SHEET_CANONICALIZATION_ID
    assert ref.parsed_columns == ("record_id", "value") and ref.n_rows == 1
    assert ref.parsed_sheet_sha256 == xlsx_sheet_sha256(frame, sheet_name="first")
    frame.loc[0, "value"] = 99
    again = subject.load_xlsx("data", "first", "ignored", reader_id=XLSX_READER_ID)
    assert again is not frame and again.loc[0, "value"] == 12
    assert subject._xlsx_observations["data"] is observation
    assert subject.ref("data") is ref
    with pytest.raises(ValueError, match="was already observed at sheet 'first', not 'second'"):
        subject.load_xlsx("data", "second", reader_id=XLSX_READER_ID)
    assert reads == [workbook] and hashes == []
    alias = subject.load_xlsx("alias", "first", reader_id=XLSX_READER_ID)
    assert alias.loc[0, "value"] == 12
    assert subject._xlsx_observations["alias"] is not observation
    assert subject.ref("alias").source_ref_id != ref.source_ref_id
    assert subject.ref("alias").sha256 == ref.sha256
    assert reads == [workbook, workbook] and hashes == []
    assert parses == [{"sheet_name": "first"}, {"sheet_name": "first"}]
    assert subject._cache == {}


def test_trusted_xlsx_retains_observation_after_path_change(make_subject, workbook, tmp_path):
    subject = make_subject({"data": workbook.name})
    first = subject.load_xlsx("data", "first", reader_id=XLSX_READER_ID)
    ref = subject.ref("data")
    workbook.rename(tmp_path / "retained.xlsx")
    # Resolution still occurs, but a missing current file cannot replace an observation.
    pd.testing.assert_frame_equal(
        subject.load_xlsx("data", "first", reader_id=XLSX_READER_ID), first)
    assert subject.ref("data") is ref


def test_trusted_xlsx_detects_retained_frame_tampering(make_subject, workbook):
    subject = make_subject({"data": workbook.name})
    subject.load_xlsx("data", "first", reader_id=XLSX_READER_ID)
    subject._xlsx_observations["data"]._frame.loc[0, "value"] = 99
    with pytest.raises(XlsxReaderError, match="no longer matches its observation"):
        subject.load_xlsx("data", "first", reader_id=XLSX_READER_ID)


def test_refs_are_observed_only_and_reporting_is_shallow(make_subject, csv_path):
    subject = make_subject()
    with pytest.raises(KeyError) as exc:
        subject.ref("data")
    assert exc.value.args == (
        "No SourceRef for 'data' in synthetic; call load_csv()/load_xlsx() first",)
    subject.load_csv("data")
    refs = subject.source_refs()
    assert refs is not subject._refs and refs["data"] is subject.ref("data")
    refs.clear()
    assert subject.ref("data") is subject._refs["data"]


def test_sessions_are_isolated(make_subject, csv_path):
    first, second = make_subject(), make_subject()
    frame = first.load_csv("data")
    frame.loc[0, "value"] = 99
    assert second.source_refs() == {}
    assert second.load_csv("data").loc[0, "value"] == 12


def test_session_constructor_does_no_io_or_resolution(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("SourceSession construction performed source I/O or resolution")

    with monkeypatch.context() as patch:
        for name in ("open", "read_bytes", "resolve", "iterdir", "glob", "rglob"):
            patch.setattr(Path, name, forbidden)
        session = SourceSession(tmp_path / "absent", resolve_source=forbidden,
                                context_name=forbidden)
    assert session.source_refs() == {} and session._cache == {}
    assert session._xlsx_observations == {}


def test_session_callbacks_are_invoked_only_when_needed(tmp_path):
    calls = []
    current = {"path": tmp_path / "first.csv", "context": "first"}

    def resolve(key):
        calls.append(key)
        return current["path"]

    session = SourceSession(tmp_path, resolve_source=resolve,
                            context_name=lambda: current["context"])
    assert session.resolve("data") == tmp_path / "first.csv"
    current["path"] = tmp_path / "second.csv"
    assert session.resolve("data") == tmp_path / "second.csv"
    current["context"] = "changed"
    with pytest.raises(KeyError, match="in changed;"):
        session.ref("data")
    assert calls == ["data", "data"]


def test_source_state_replacement_preserves_behavior(make_subject, tmp_path):
    (tmp_path / "records.csv").write_text("value\n12\n")
    session = make_subject()
    original = session.load_csv("data")
    ref = session.ref("data")
    replacement = pd.DataFrame({"value": [99]})
    session._cache = {str(tmp_path / "records.csv"): replacement}
    assert session.load_csv("data") is replacement
    session._refs = {"other": ref}
    assert session.ref("other") is ref
    with pytest.raises(KeyError):
        session.ref("data")
    session._cache = {}
    assert session.load_csv("data") is not original
    assert session.ref("data") == ref


def test_trusted_observation_state_replacement_is_live(make_subject, tmp_path):
    from lib.xlsx_reader import XLSX_READER_ID

    path = tmp_path / "records.xlsx"
    pd.DataFrame({"value": [12]}).to_excel(path, sheet_name="first", index=False)
    session = make_subject({"data": path.name})
    session.load_xlsx("data", "first", reader_id=XLSX_READER_ID)
    first = session._xlsx_observations["data"]
    session._xlsx_observations = {}
    session.load_xlsx("data", "first", reader_id=XLSX_READER_ID)
    second = session._xlsx_observations["data"]
    assert second is not first
    assert session.ref("data") is second.source_ref
    session._xlsx_observations = {"data": first}
    session.load_xlsx("data", "first", reader_id=XLSX_READER_ID)
    assert session._xlsx_observations["data"] is first
    # Replacing observations alone historically does not repopulate the ref dict.
    assert session.ref("data") is second.source_ref
