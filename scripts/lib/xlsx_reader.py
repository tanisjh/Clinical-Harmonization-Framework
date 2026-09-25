"""Closed exact-byte XLSX reader used by execution, preflight, and replay.

``xlsx_openpyxl@v1`` observes one immutable workbook payload, validates the
selected worksheet with formulas visible, and parses a new ``BytesIO`` view of
that same payload.  The path is never reopened between hashing and parsing.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd

from .replay_contract import encode_typed_raw
from .source_ref import SourceRef, assert_within_roots


XLSX_READER_ID = "xlsx_openpyxl@v1"
XLSX_SHEET_CANONICALIZATION_ID = "xlsx_sheet_frame@v1"
XLSX_SHEET_CANONICALIZATION_VERSION = 1

# Flat, JSON-native values keep SourceRef and manifest identities stable.  The
# dtype backend sentinel cannot be supplied through pandas' public API, so v1
# binds the current no-default behavior together with exact pandas identity.
XLSX_READER_FLAGS = {
    "io": "bytesio",
    "sheet_selector": "exact_name",
    "engine": "openpyxl",
    "header": 0,
    "names": None,
    "index_col": None,
    "usecols": None,
    "dtype": None,
    "converters": None,
    "true_values": None,
    "false_values": None,
    "skiprows": None,
    "nrows": None,
    "na_values": None,
    "keep_default_na": True,
    "na_filter": True,
    "parse_dates": False,
    "date_format": None,
    "thousands": None,
    "decimal": ".",
    "comment": None,
    "skipfooter": 0,
    "storage_options": None,
    "dtype_backend": "pandas_no_default",
    "engine_read_only": True,
    "engine_data_only": True,
    "engine_keep_links": False,
}
XLSX_READER_DEPENDENCIES = ("numpy", "openpyxl", "pandas")
_TRUSTED_XLSX_TOKEN = object()


class XlsxReaderError(ValueError):
    """The workbook is outside the deliberately closed v1 reader shape."""


def _read_excel_kwargs() -> dict[str, Any]:
    """Translate the pinned contract to the installed pandas API."""
    return {
        "engine": "openpyxl",
        "header": 0,
        "names": None,
        "index_col": None,
        "usecols": None,
        "dtype": None,
        "converters": None,
        "true_values": None,
        "false_values": None,
        "skiprows": None,
        "nrows": None,
        "na_values": None,
        "keep_default_na": True,
        "na_filter": True,
        "parse_dates": False,
        "date_format": None,
        "thousands": None,
        "decimal": ".",
        "comment": None,
        "skipfooter": 0,
        "storage_options": None,
        "engine_kwargs": {
            "read_only": True,
            "data_only": True,
            "keep_links": False,
        },
    }


def _trimmed_width(values: tuple[Any, ...]) -> int:
    width = len(values)
    while width and values[width - 1] in (None, ""):
        width -= 1
    return width


def _validate_sheet_shape(payload: bytes, sheet_name: str) -> tuple[str, ...]:
    """Inspect formulas, merges, and raw headings from the captured bytes."""
    try:
        workbook = openpyxl.load_workbook(
            io.BytesIO(payload), read_only=False, data_only=False,
            keep_links=False)
    except Exception as exc:
        raise XlsxReaderError(f"workbook is not a supported OOXML XLSX: {exc}") from exc
    try:
        if sheet_name not in workbook.sheetnames:
            raise XlsxReaderError(f"workbook has no exact sheet {sheet_name!r}")
        sheet = workbook[sheet_name]
        if sheet.merged_cells.ranges:
            raise XlsxReaderError("xlsx_openpyxl@v1 refuses merged cells")

        rows: list[tuple[Any, ...]] = []
        width = 0
        for cells in sheet.iter_rows():
            values = tuple(cell.value for cell in cells)
            rows.append(values)
            width = max(width, _trimmed_width(values))
            if any(cell.data_type == "f" for cell in cells):
                raise XlsxReaderError("xlsx_openpyxl@v1 refuses formula cells")
        if not rows or width == 0:
            raise XlsxReaderError("selected worksheet has no parsed header")
        raw_header = tuple(rows[0][i] if i < len(rows[0]) else None
                           for i in range(width))
        if any(not isinstance(name, str) or not name for name in raw_header):
            raise XlsxReaderError(
                "xlsx_openpyxl@v1 requires non-empty string headings")
        duplicates = sorted({name for name in raw_header
                             if raw_header.count(name) > 1})
        if duplicates:
            raise XlsxReaderError(f"duplicate XLSX headings {duplicates}")
        return raw_header
    finally:
        workbook.close()


def parse_xlsx_bytes(payload: bytes, *, sheet_name: str) -> pd.DataFrame:
    """Validate and parse one exact captured XLSX payload under v1."""
    if not isinstance(payload, bytes):
        raise TypeError("XLSX payload must be immutable bytes")
    if not isinstance(sheet_name, str) or not sheet_name:
        raise XlsxReaderError("xlsx_openpyxl@v1 requires an exact sheet name")
    raw_header = _validate_sheet_shape(payload, sheet_name)
    try:
        frame = pd.read_excel(
            io.BytesIO(payload), sheet_name=sheet_name, **_read_excel_kwargs())
    except Exception as exc:
        raise XlsxReaderError(f"XLSX parse failed: {exc}") from exc
    if not isinstance(frame, pd.DataFrame):
        raise XlsxReaderError("xlsx_openpyxl@v1 requires exactly one parsed table")
    if tuple(frame.columns) != raw_header:
        raise XlsxReaderError(
            "parsed XLSX headings differ from the validated worksheet header")
    if (not isinstance(frame.index, pd.RangeIndex)
            or tuple(frame.index) != tuple(range(len(frame)))):
        raise XlsxReaderError(
            "parsed XLSX table must have a zero-based RangeIndex")
    # Canonicalization is part of acceptance: unsupported sheet values fail at
    # observation time even if no current mapping consumes their column.
    canonical_xlsx_sheet(frame, sheet_name=sheet_name)
    return frame


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NaT:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _temporal_cell(value: Any) -> dict[str, Any] | None:
    if isinstance(value, pd.Period) or isinstance(value, (dt.timedelta, pd.Timedelta)):
        raise XlsxReaderError("XLSX periods and timedeltas are unsupported")
    if isinstance(value, pd.Timestamp):
        if value is pd.NaT:
            return {"kind": "null"}
        if value.tz is not None:
            raise XlsxReaderError("timezone-aware XLSX timestamps are unsupported")
        if value.nanosecond:
            raise XlsxReaderError("sub-microsecond XLSX timestamps are unsupported")
        value = value.to_pydatetime()
    if isinstance(value, dt.datetime):
        if value.utcoffset() is not None:
            raise XlsxReaderError("timezone-aware XLSX datetimes are unsupported")
        return {"kind": "datetime", "value": value.isoformat(timespec="microseconds")}
    if isinstance(value, dt.time):
        if value.utcoffset() is not None:
            raise XlsxReaderError("timezone-aware XLSX times are unsupported")
        return {"kind": "time", "value": value.isoformat(timespec="microseconds")}
    if isinstance(value, dt.date):
        return {"kind": "date", "value": value.isoformat()}
    return None


def _canonical_cell(value: Any) -> dict[str, Any]:
    if _is_missing(value):
        return {"kind": "null"}
    temporal = _temporal_cell(value)
    if temporal is not None:
        return temporal
    try:
        return json.loads(encode_typed_raw(value).to_json())
    except ValueError as exc:
        raise XlsxReaderError(str(exc)) from exc


def canonical_xlsx_sheet(frame: pd.DataFrame, *, sheet_name: str) -> bytes:
    """Return deterministic ``xlsx_sheet_frame@v1`` UTF-8 JSON bytes."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("canonical XLSX sheet input must be a DataFrame")
    if not isinstance(sheet_name, str) or not sheet_name:
        raise XlsxReaderError("canonical XLSX sheet requires an exact name")
    columns = list(frame.columns)
    if any(not isinstance(name, str) or not name for name in columns):
        raise XlsxReaderError("XLSX columns must be non-empty strings")
    duplicates = sorted({name for name in columns if columns.count(name) > 1})
    if duplicates:
        raise XlsxReaderError(f"duplicate XLSX columns {duplicates}")
    document = {
        "canonicalization": XLSX_SHEET_CANONICALIZATION_ID,
        "canonicalization_version": XLSX_SHEET_CANONICALIZATION_VERSION,
        "sheet": sheet_name,
        "columns": columns,
        "rows": [[_canonical_cell(value) for value in row]
                 for row in frame.itertuples(index=False, name=None)],
    }
    return json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        allow_nan=False).encode("utf-8")


def xlsx_sheet_sha256(frame: pd.DataFrame, *, sheet_name: str) -> str:
    return hashlib.sha256(
        canonical_xlsx_sheet(frame, sheet_name=sheet_name)).hexdigest()


@dataclass(frozen=True)
class TrustedXlsxObservation:
    """Private immutable-payload observation with a protected parsed frame."""

    source_ref: SourceRef
    sheet_sha256: str
    _payload: bytes = field(repr=False, compare=False)
    _frame: pd.DataFrame = field(repr=False, compare=False)
    _factory_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _TRUSTED_XLSX_TOKEN:
            raise ValueError("trusted XLSX observations must use load_xlsx_source")

    def frame_copy(self) -> pd.DataFrame:
        """Detect retained-frame mutation, then return an isolated deep copy."""
        ref = self.source_ref
        digest = xlsx_sheet_sha256(self._frame, sheet_name=ref.sheet or "")
        if (digest != self.sheet_sha256
                or digest != ref.parsed_sheet_sha256
                or tuple(self._frame.columns) != ref.parsed_columns
                or len(self._frame) != ref.n_rows):
            raise XlsxReaderError(
                "trusted XLSX parsed frame no longer matches its observation")
        return self._frame.copy(deep=True)


def load_xlsx_source(
        path: str | Path, *, project_root: str | Path, config_key: str,
        sheet_name: str, logical_label: str | None = None
        ) -> TrustedXlsxObservation:
    """Read once, hash and parse the same bytes, and bind the parsed sheet."""
    root = Path(project_root).resolve()
    resolved = assert_within_roots(path, root)
    if resolved.suffix.lower() != ".xlsx":
        raise XlsxReaderError("xlsx_openpyxl@v1 accepts only .xlsx files")
    if not resolved.is_file():
        raise FileNotFoundError(f"XLSX source file does not exist: {resolved}")
    payload = resolved.read_bytes()
    frame = parse_xlsx_bytes(payload, sheet_name=sheet_name)
    sheet_digest = xlsx_sheet_sha256(frame, sheet_name=sheet_name)
    source_ref = SourceRef.from_observed_bytes(
        config_key=config_key, path=resolved, project_root=root,
        payload=payload, file_type="xlsx", reader_id=XLSX_READER_ID,
        logical_label=logical_label, sheet=sheet_name,
        reader_flags=json.dumps(
            XLSX_READER_FLAGS, ensure_ascii=False, separators=(",", ":"),
            sort_keys=True),
        reader_effective_flags=XLSX_READER_FLAGS, n_rows=len(frame),
        parsed_columns=tuple(frame.columns),
        parsed_canonicalization=XLSX_SHEET_CANONICALIZATION_ID,
        parsed_sheet_sha256=sheet_digest,
    )
    return TrustedXlsxObservation(
        source_ref=source_ref, sheet_sha256=sheet_digest, _payload=payload,
        _frame=frame.copy(deep=True), _factory_token=_TRUSTED_XLSX_TOKEN)
