"""Read-only structural lint for declarative specification mapping identities.

The linter reads only the YAML paths explicitly supplied by its caller.  It
never resolves or opens source paths declared inside a specification and never
mutates a specification.
"""
from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

MAPPING_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_MAPPING_ID_RE = re.compile(MAPPING_ID_PATTERN)


class SpecIdentityLintError(ValueError):
    """A requested lint input is malformed or cannot be represented."""


def target_slug(value: Any) -> str:
    """Return the reviewed production target normalization form."""
    result = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    if not result:
        return "field"
    return result if result[0].isalpha() else f"field_{result}"


def load_declarative_yaml(path: str | Path) -> dict[str, Any]:
    """Load exactly one supplied YAML file, without following declared paths."""
    requested = Path(path)
    if not requested.is_file():
        raise SpecIdentityLintError(f"specification does not exist: {requested}")
    try:
        document = yaml.safe_load(requested.read_text())
    except yaml.YAMLError as exc:
        raise SpecIdentityLintError(
            f"invalid YAML in {requested}: {exc}") from exc
    if not isinstance(document, dict):
        raise SpecIdentityLintError(
            f"specification must be a YAML mapping: {requested}")
    return document


def semantic_structure(document: dict[str, Any]) -> dict[str, Any]:
    """Copy a spec while removing only version and per-mapping identity."""
    normalized = copy.deepcopy(document)
    normalized.pop("spec_version", None)
    mappings = normalized.get("mappings")
    if isinstance(mappings, list):
        for mapping in mappings:
            if isinstance(mapping, dict):
                mapping.pop("mapping_id", None)
    return normalized


def compare_semantic_structure(old_path: str | Path,
                               new_path: str | Path) -> dict[str, Any]:
    """Compare two declarative specs after only the approved normalization."""
    old_doc = load_declarative_yaml(old_path)
    new_doc = load_declarative_yaml(new_path)
    old_mappings = old_doc.get("mappings") or []
    new_mappings = new_doc.get("mappings") or []
    return {
        "old_path": str(Path(old_path)),
        "new_path": str(Path(new_path)),
        "old_spec_ref": f"{old_doc.get('spec_id')}@{old_doc.get('spec_version')}",
        "new_spec_ref": f"{new_doc.get('spec_id')}@{new_doc.get('spec_version')}",
        "old_status": old_doc.get("status", "draft"),
        "new_status": new_doc.get("status", "draft"),
        "old_mapping_count": len(old_mappings),
        "new_mapping_count": len(new_mappings),
        "equivalent": semantic_structure(old_doc) == semantic_structure(new_doc),
    }


def _mapping_key(mapping: dict[str, Any]) -> tuple[str, str, str]:
    return (str(mapping.get("target") or ""),
            str(mapping.get("op") or ""),
            str(mapping.get("when") or ""))


def _plan_index(plan_path: str | Path | None
                ) -> dict[tuple[str, str], dict[tuple[str, str, str], list[str]]]:
    if plan_path is None:
        return {}
    plan = load_declarative_yaml(plan_path)
    out: dict[tuple[str, str], dict[tuple[str, str, str], list[str]]] = {}
    for entry in plan.get("mappings") or []:
        if not isinstance(entry, dict):
            raise SpecIdentityLintError("mapping-ID plan contains a non-object entry")
        old_ref = str(entry.get("spec_ref") or "")
        spec_id = old_ref.rsplit("@", 1)[0]
        proposed_version = str(entry.get("proposed_spec_version") or "")
        spec_key = (spec_id, proposed_version)
        by_mapping = out.setdefault(spec_key, defaultdict(list))
        by_mapping[(str(entry.get("target") or ""),
                    str(entry.get("operation") or ""),
                    str(entry.get("mapping_when") or ""))].append(
                        str(entry.get("proposed_mapping_id") or ""))
    return out


def lint_spec(path: str | Path, *,
              plan_index: dict | None = None) -> dict[str, Any]:
    """Lint one explicit specification path and return JSON-safe findings."""
    document = load_declarative_yaml(path)
    spec_id = str(document.get("spec_id") or "")
    version = str(document.get("spec_version") or "")
    mappings = document.get("mappings")
    if not isinstance(mappings, list):
        raise SpecIdentityLintError(f"{path}: mappings must be a list")

    labels = ["|".join(_mapping_key(m)) if isinstance(m, dict) else "<non-object>"
              for m in mappings]
    ids = [str(m.get("mapping_id") or "") if isinstance(m, dict) else ""
           for m in mappings]
    missing = [label for label, mapping_id in zip(labels, ids) if not mapping_id]
    invalid = [mapping_id for mapping_id in ids
               if mapping_id and not _MAPPING_ID_RE.fullmatch(mapping_id)]
    duplicates = sorted(mapping_id for mapping_id, count in Counter(ids).items()
                        if mapping_id and count > 1)

    by_target: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for label, mapping, mapping_id in zip(labels, mappings, ids):
        if isinstance(mapping, dict):
            by_target[str(mapping.get("target") or "")].append(
                (label, mapping_id))
    branch_issues = []
    for target, values in sorted(by_target.items()):
        if len(values) < 2:
            continue
        prefix = f"{target_slug(target)}."
        for label, mapping_id in values:
            if not mapping_id or not mapping_id.startswith(prefix) \
                    or not mapping_id[len(prefix):]:
                branch_issues.append(label)

    targets_by_slug: dict[str, set[str]] = defaultdict(set)
    for target in by_target:
        targets_by_slug[target_slug(target)].add(target)
    normalization_collisions = [
        {"slug": slug, "targets": sorted(targets)}
        for slug, targets in sorted(targets_by_slug.items()) if len(targets) > 1
    ]

    plan_status = "not_checked"
    plan_drift: list[str] = []
    if plan_index is not None:
        expected = plan_index.get((spec_id, version))
        if expected is None:
            plan_status = "spec_not_in_plan"
            plan_drift.append(f"no plan entry for {spec_id}@{version}")
        else:
            plan_status = "pass"
            seen = Counter()
            for label, mapping, mapping_id in zip(labels, mappings, ids):
                if not isinstance(mapping, dict):
                    plan_drift.append(f"non-object mapping: {label}")
                    continue
                key = _mapping_key(mapping)
                seen[key] += 1
                planned = expected.get(key, [])
                if len(planned) != 1:
                    plan_drift.append(
                        f"{label}: expected one plan identity, found {len(planned)}")
                elif mapping_id != planned[0]:
                    plan_drift.append(
                        f"{label}: {mapping_id!r} != planned {planned[0]!r}")
            for key, planned in expected.items():
                if seen[key] != len(planned):
                    plan_drift.append(
                        f"{'|'.join(key)}: spec count {seen[key]} != plan count {len(planned)}")
            if plan_drift:
                plan_status = "drift"

    ready = not any((missing, invalid, duplicates, branch_issues,
                     normalization_collisions, plan_drift))
    return {
        "path": str(Path(path)),
        "spec_ref": f"{spec_id}@{version}",
        "spec_version": version,
        "status": str(document.get("status", "draft")),
        "mapping_count": len(mappings),
        "present_id_count": sum(bool(mapping_id) for mapping_id in ids),
        "missing_ids": missing,
        "invalid_ids": sorted(set(invalid)),
        "duplicate_ids": duplicates,
        "duplicate_target_branch_issues": branch_issues,
        "normalization_collisions": normalization_collisions,
        "plan_status": plan_status,
        "plan_drift": plan_drift,
        "identity_ready": ready,
    }


def lint_paths(paths: Iterable[str | Path], *,
               plan_path: str | Path | None = None) -> dict[str, Any]:
    """Lint explicit paths in deterministic order."""
    requested = sorted({Path(path) for path in paths}, key=lambda p: str(p))
    if not requested:
        raise SpecIdentityLintError("at least one explicit spec path is required")
    plan_index = _plan_index(plan_path) if plan_path is not None else None
    specs = [lint_spec(path, plan_index=plan_index) for path in requested]
    return {
        "spec_count": len(specs),
        "mapping_count": sum(spec["mapping_count"] for spec in specs),
        "present_id_count": sum(spec["present_id_count"] for spec in specs),
        "identity_ready": all(spec["identity_ready"] for spec in specs),
        "approved_count": sum(spec["status"] == "approved" for spec in specs),
        "specs": specs,
    }
