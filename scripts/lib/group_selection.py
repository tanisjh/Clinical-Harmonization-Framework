"""Pure declaration semantics for ordered source-group selection.

This module deliberately performs no I/O and consumes no provenance output.
Execution and replay may share the pinned ``first_containing_record`` decision
without sharing source reads or evidence verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contributions import (
    STATUS_CONDITION,
    STATUS_NOT_EVALUATED,
    STATUS_SELECTED,
    STATUS_UNAVAILABLE,
)
from .spec_model import (
    GROUP_REF_SEP,
    OP_DIRECT,
    OP_FILE_MEMBERSHIP,
    OP_VALUE_MAP,
)


@dataclass(frozen=True)
class GroupSelectionDecision:
    """One deterministic decision from independently observed membership."""

    selected_member_id: str
    selected_rank: int | None
    statuses: tuple[str, ...]


def select_first_containing_record(
        ordered_member_ids: tuple[str, ...],
        presence_by_member: Mapping[str, bool], *,
        membership_status: str = STATUS_SELECTED) -> GroupSelectionDecision:
    """Select the first present member and derive the complete status chain.

    ``presence_by_member`` must contain exactly one boolean for every declared
    member.  The function receives only independently reconstructed facts; it
    cannot consult a Contribution, delivered value, provenance value, or raw
    snapshot.
    """
    if not ordered_member_ids or len(set(ordered_member_ids)) != len(
            ordered_member_ids):
        raise ValueError("ordered source-group members must be non-empty and unique")
    if set(presence_by_member) != set(ordered_member_ids):
        raise ValueError("presence facts must exactly match ordered group members")
    if any(type(presence_by_member[name]) is not bool
           for name in ordered_member_ids):
        raise ValueError("source-group presence facts must be booleans")
    if membership_status not in (STATUS_SELECTED, STATUS_CONDITION):
        raise ValueError(
            "source-group winner status must be 'selected' or 'condition'")

    selected_rank = next(
        (rank for rank, name in enumerate(ordered_member_ids)
         if presence_by_member[name]), None)
    statuses = []
    for rank, _name in enumerate(ordered_member_ids):
        if selected_rank is None or rank < selected_rank:
            statuses.append(STATUS_UNAVAILABLE)
        elif rank == selected_rank:
            statuses.append(membership_status)
        else:
            statuses.append(STATUS_NOT_EVALUATED)
    return GroupSelectionDecision(
        selected_member_id=(ordered_member_ids[selected_rank]
                            if selected_rank is not None else ""),
        selected_rank=selected_rank,
        statuses=tuple(statuses),
    )


def source_group_id_for_mapping(mapping) -> str:
    """Return the one declared group a mapping consumes, if identifiable."""
    if mapping.group:
        return mapping.group
    if mapping.provenance.contributions_from:
        return mapping.provenance.contributions_from
    for ref in (mapping.source, mapping.provenance.row_from):
        if ref and GROUP_REF_SEP in ref:
            return ref.partition(GROUP_REF_SEP)[0]
    return ""


def source_group_for_mapping(spec, mapping):
    group_id = source_group_id_for_mapping(mapping)
    return spec.group(group_id) if group_id else None


def scalar_role_for_mapping(mapping, group) -> str:
    """Return the member role that supplies the semantic scalar."""
    if mapping.op == OP_FILE_MEMBERSHIP:
        return group.decided_by
    group_id, separator, role = mapping.source.partition(GROUP_REF_SEP)
    return role if separator and group_id == group.id else ""


def membership_role(spec, group) -> str:
    """Reproduce the executor's existing membership Contribution role."""
    for mapping in spec.mappings:
        if mapping.op == OP_FILE_MEMBERSHIP and mapping.group == group.id:
            return f"{mapping.target}_file_membership"
    return f"{group.id}_file_membership"


def group_rule_id(spec, group) -> str:
    """Reproduce the executor's existing group-decision rule identity."""
    if group.rule is not None:
        return group.rule.id
    for mapping in spec.mappings:
        if mapping.op == OP_FILE_MEMBERSHIP and mapping.group == group.id:
            return mapping.rule.id
    return ""


def is_closed_source_group_mapping(spec, mapping) -> bool:
    """Whether ``mapping`` is inside the approved Stage 12 boundary."""
    group = source_group_for_mapping(spec, mapping)
    if group is None or group.select != "first_containing_record":
        return False
    if mapping.op not in (OP_FILE_MEMBERSHIP, OP_DIRECT, OP_VALUE_MAP):
        return False
    if (mapping.when or mapping.candidates or mapping.clauses
            or mapping.derivation or mapping.inputs
            or mapping.params
            or mapping.provenance.raw_from
            or mapping.contributions.conditions):
        return False
    if mapping.provenance.contributions_from not in ("", group.id):
        return False
    if mapping.op == OP_FILE_MEMBERSHIP:
        if mapping.group != group.id or mapping.source or mapping.members:
            return False
    else:
        role = scalar_role_for_mapping(mapping, group)
        if not role or role not in group.roles:
            return False
        if mapping.group and mapping.group != group.id:
            return False
        if mapping.provenance.row_from not in ("", mapping.source):
            return False
        if mapping.provenance.source_ref not in ("", mapping.source):
            return False
    if mapping.value_state not in (
            "", "delivered", "transformed", "provisional"):
        return False
    if mapping.op == OP_DIRECT and mapping.coerce not in (
            "none", "text", "numeric"):
        return False
    if mapping.op == OP_VALUE_MAP and mapping.coerce != "none":
        return False
    if mapping.fallback_rules and not mapping.fallback_vocabulary:
        return False

    roles = {group.decided_by, scalar_role_for_mapping(mapping, group)}
    for member in group.members:
        for role in roles:
            source = spec.source(member.sources.get(role, ""))
            if (source is None or not source.path or source.path_pattern
                    or source.file_type != "csv"
                    or source.reader != "cidc_csv@v1" or source.sheet):
                return False
    return True


def is_stage17_source_group_mapping(spec, mapping) -> bool:
    """Whether the scalar/group route is Stage 12-closed before raw policy.

    Unlike the Stage 12 predicate, this permits mapping conditions, structured
    derivation, and ``raw_from`` because Stages 15–17 verify those independent
    layers.  It does not broaden the underlying group operation grammar.
    """
    group = source_group_for_mapping(spec, mapping)
    if group is None or group.select != "first_containing_record":
        return False
    if mapping.op not in (OP_FILE_MEMBERSHIP, OP_DIRECT, OP_VALUE_MAP):
        return False
    if (mapping.candidates or mapping.clauses or mapping.inputs
            or mapping.params):
        return False
    if mapping.provenance.contributions_from not in ("", group.id):
        return False
    if mapping.op == OP_FILE_MEMBERSHIP:
        if mapping.group != group.id or mapping.source or mapping.members:
            return False
    else:
        role = scalar_role_for_mapping(mapping, group)
        if not role or role not in group.roles:
            return False
        if mapping.group and mapping.group != group.id:
            return False
        if mapping.provenance.row_from not in ("", mapping.source):
            return False
        if mapping.provenance.source_ref not in ("", mapping.source):
            return False
    if mapping.value_state not in (
            "", "delivered", "transformed", "provisional"):
        return False
    if mapping.op == OP_DIRECT and mapping.coerce not in (
            "none", "text", "numeric"):
        return False
    if mapping.op == OP_VALUE_MAP and mapping.coerce != "none":
        return False
    if mapping.fallback_rules and not mapping.fallback_vocabulary:
        return False

    roles = {group.decided_by, scalar_role_for_mapping(mapping, group)}
    for member in group.members:
        for role in roles:
            source = spec.source(member.sources.get(role, ""))
            if (source is None or not source.path or source.path_pattern
                    or source.file_type != "csv"
                    or source.reader != "cidc_csv@v1" or source.sheet):
                return False
    return True
