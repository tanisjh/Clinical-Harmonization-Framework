"""Pure grouped-record construction shared by execution and strict replay.

This module implements policy only.  Callers acquire and verify source data;
the helpers here perform no I/O and consume no recorded provenance or output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from .derivation import derivation_references
from .spec_model import (
    GROUPING_SELECTIONS,
    REF_ANCHOR_GROUP_SIZE,
    REF_ANCHOR_SELECTED_ROW,
)


ORIGIN_OBSERVED = "observed"
ORIGIN_SYNTHESIZED = "synthesized"
_ANCHOR_FACTS = frozenset({
    REF_ANCHOR_GROUP_SIZE,
    REF_ANCHOR_SELECTED_ROW,
})


class RecordConstructionError(ValueError):
    """Pinned record construction cannot be applied to verified frames."""


@dataclass(frozen=True)
class ConstructedRecord:
    """One deterministic logical record, before replay trust projection."""

    anchor_items: tuple[tuple[str, Any], ...]
    origin: str
    group_key: tuple[tuple[str, str], ...] = ()
    selected_source_row: int | None = None
    group_size: int | None = None
    declaration_id: str = ""
    source_items: tuple[tuple[str, Any], ...] = field(
        default=(), repr=False, compare=False)

    @property
    def anchor_row(self) -> dict[str, Any]:
        return dict(self.anchor_items)

    @property
    def source_row(self) -> pd.Series | None:
        return pd.Series(dict(self.source_items)) if self.source_items else None


@dataclass(frozen=True)
class ConstructedRecordSet:
    """Pure construction output and per-declaration diagnostics."""

    records: tuple[ConstructedRecord, ...]
    additional_counts: tuple[tuple[str, int], ...]


def _join_column(spec, source_id: str) -> str:
    source = spec.source(source_id)
    return (
        source.participant_column
        if source is not None and source.participant_column
        else spec.record.participant_key
    )


def prepare_participant_frame(spec, source_id: str,
                              frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the executor's exact participant indexing/duplicate policy."""
    record = spec.record
    participant_key = record.participant_key
    prepared = frame.copy()
    prepared["_spec_src_row"] = prepared.index
    join_column = _join_column(spec, source_id)
    if join_column not in prepared.columns:
        named = (
            ""
            if join_column == participant_key
            else " (declared as this source's participant_column for "
                 f"{participant_key!r})"
        )
        raise RecordConstructionError(
            f"source {source_id!r} has no participant key column "
            f"{join_column!r}{named}")
    prepared[join_column] = prepared[join_column].astype(str)
    duplicate_columns = [
        join_column if name == participant_key else name
        for name in record.duplicates.by
    ] or [join_column]
    missing = [name for name in duplicate_columns
               if name not in prepared.columns]
    if missing:
        raise RecordConstructionError(
            f"source {source_id!r} has no duplicate-key column(s) {missing}")
    policy = record.duplicates.policy
    if policy in ("first", "last"):
        prepared = prepared.drop_duplicates(
            subset=duplicate_columns, keep=policy)
    elif policy == "error":
        if prepared.duplicated(subset=duplicate_columns).any():
            raise RecordConstructionError(
                f"source {source_id!r} has duplicate records on "
                f"{duplicate_columns} and the duplicate policy is 'error'")
    else:
        raise RecordConstructionError(
            f"duplicate policy {policy!r} is not implemented")
    return prepared.set_index(join_column)


def construct_grouping_records(
        spec, anchor_frame: pd.DataFrame,
        participant_frames: Mapping[str, pd.DataFrame]
        ) -> ConstructedRecordSet:
    """Reproduce grouping plus declared missing-participant records exactly."""
    record = spec.record
    grouping = record.grouping
    if record.anchor != "grouping" or grouping is None:
        raise RecordConstructionError(
            "grouped construction requires a grouping record anchor")
    if grouping.select != "first_row":
        raise RecordConstructionError(
            f"grouping select {grouping.select!r} is not implemented; "
            f"supported selections are {GROUPING_SELECTIONS}")

    frame = anchor_frame.copy()
    frame["_spec_src_row"] = frame.index
    missing = [name for name in grouping.key_columns
               if name not in frame.columns]
    if missing:
        raise RecordConstructionError(
            f"grouping anchor source {record.anchor_source!r} has no key "
            f"column(s) {missing}; it has {list(frame.columns)}")
    for name in grouping.key_columns:
        frame[name] = frame[name].astype(str)

    sizes = frame.groupby(
        list(grouping.key_columns), sort=False).size().to_dict()
    selected = frame.drop_duplicates(
        subset=list(grouping.key_columns), keep="first")
    declared_fields = tuple(dict.fromkeys(
        name for name in (
            record.participant_key,
            record.sample_key,
            record.collection_event_key,
        ) if name
    ))
    records: list[ConstructedRecord] = []
    for _, row in selected.iterrows():
        key_values = tuple(str(row[name]) for name in grouping.key_columns)
        key = tuple(zip(grouping.key_columns, key_values))
        anchor_row = {
            name: (str(row[column]) if column else None)
            for name, column in grouping.anchor_fields.items()
        }
        for name in declared_fields:
            if name not in anchor_row:
                anchor_row[name] = None
        size_key: Any = key_values if len(key_values) > 1 else key_values[0]
        records.append(ConstructedRecord(
            anchor_items=tuple(anchor_row.items()),
            origin=ORIGIN_OBSERVED,
            group_key=key,
            selected_source_row=int(row["_spec_src_row"]),
            group_size=int(sizes[size_key]),
            source_items=tuple(row.items()),
        ))

    additional_counts: list[tuple[str, int]] = []
    for declaration in record.additional_records:
        if declaration.when != "no_record_for_participant":
            raise RecordConstructionError(
                f"additional record {declaration.id!r} uses unsupported "
                f"condition {declaration.when!r}")
        source_id = declaration.participants_from
        source_frame = participant_frames.get(source_id)
        if source_frame is None:
            raise RecordConstructionError(
                f"additional record {declaration.id!r} has no verified "
                f"participant frame {source_id!r}")
        indexed = prepare_participant_frame(spec, source_id, source_frame)
        have = {
            str(item.anchor_row[record.participant_key])
            for item in records
        }
        added = 0
        for participant in (str(value) for value in indexed.index):
            if participant in have:
                continue
            anchor_row = {name: None for name in declared_fields}
            anchor_row.update(declaration.values)
            anchor_row[record.participant_key] = participant
            if participant in have:
                raise RecordConstructionError(
                    f"additional record {declaration.id!r} would add a "
                    f"second record for participant {participant!r}")
            records.append(ConstructedRecord(
                anchor_items=tuple(anchor_row.items()),
                origin=ORIGIN_SYNTHESIZED,
                declaration_id=declaration.id,
            ))
            have.add(participant)
            added += 1
        additional_counts.append((declaration.id, added))

    return ConstructedRecordSet(tuple(records), tuple(additional_counts))


def mapping_uses_grouping_facts(mapping) -> bool:
    """Whether any declared derivation variant consumes an anchor fact."""
    return any(
        bool(derivation_references(variant.record) & _ANCHOR_FACTS)
        for variant in mapping.derivation
    )


def mapping_uses_record_origin(spec, mapping) -> bool:
    """Whether mapping/derivation selection consumes record origin."""
    conditions = {condition.id: condition for condition in spec.conditions}
    ids = {mapping.when} if mapping.when else set()
    ids.update(variant.when for variant in mapping.derivation if variant.when)
    for variant in mapping.derivation:
        ids.update(
            ref.partition(".")[2]
            for ref in derivation_references(variant.record)
            if ref.startswith("condition.")
        )
    return any(
        condition_id in conditions
        and bool(conditions[condition_id].record_origin)
        for condition_id in ids
    )


def mapping_uses_record_construction(spec, mapping) -> bool:
    """Generic declaration-shape trigger for Stage 19 causal linkage."""
    return (
        spec.record.anchor == "grouping"
        and (mapping_uses_grouping_facts(mapping)
             or mapping_uses_record_origin(spec, mapping))
    )
