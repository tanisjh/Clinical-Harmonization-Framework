"""Strict mutable project settings for current tooling.

A :class:`ProjectProfile` names project-level references and convenience
defaults.  It is deliberately *not* harmonization policy and is never
historical replay authority.  Mapping specifications, the target schema,
registered rules, and pinned replay evidence retain those responsibilities.

This module only loads or validates a profile when a caller explicitly asks it
to.  Existing harmonization, tracing, artifact, and replay paths do not import
it.  A later P1D change may adopt selected defaults with the precedence rule
``explicit CLI argument > profile default``.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field as dc_field, replace
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.nodes import MappingNode


PROFILE_FORMAT_VERSION = 1

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9._-]*$")
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})

_DOC_KEYS = {
    "profile_id", "profile_format_version", "profile_version",
    "target_schema", "specifications", "tooling", "artifacts",
}
_SPECIFICATION_KEYS = {"approved", "roots"}
_TOOLING_KEYS = {
    "external_anchor", "confidence_config", "output_directory",
    "provenance_filename", "report_filename",
}
_ARTIFACT_KEYS = {"roles", "aliases", "required", "optional"}


class ProjectProfileError(ValueError):
    """A project profile or one of its declared references is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProjectProfileError(message)


def _check_keys(value: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(
        (key for key in value if not isinstance(key, str) or key not in allowed),
        key=repr,
    )
    _require(
        not unknown,
        f"{where}: unknown propert{'y' if len(unknown) == 1 else 'ies'} "
        f"{unknown} (allowed: {sorted(allowed)})",
    )


def _text(value: Any, where: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()),
             f"{where}: must be a non-empty string")
    _require(value == value.strip(),
             f"{where}: must not have leading or trailing whitespace")
    return value


def _validate_identifier(value: Any, where: str) -> str:
    text = _text(value, where)
    _require(bool(_IDENTIFIER_RE.fullmatch(text)),
             f"{where}: must match {_IDENTIFIER_RE.pattern!r}, got {text!r}")
    return text


def _validate_reference(value: Any, where: str, *, yaml_file: bool = False) -> str:
    """Validate one portable project-root-relative path without opening it."""
    text = _text(value, where)
    _require("\\" not in text, f"{where}: use portable '/' path separators")
    _require(":" not in text, f"{where}: URI and drive-qualified paths are not allowed")
    _require(not text.startswith("~"), f"{where}: home-relative paths are not allowed")
    _require(not any(char in text for char in "*?[]{}"),
             f"{where}: wildcard or recursive discovery syntax is not allowed")
    _require(not text.startswith("/"), f"{where}: absolute paths are not allowed")
    raw_parts = text.split("/")
    _require(all(part not in ("", ".", "..") for part in raw_parts),
             f"{where}: empty, current-directory, and parent-directory segments "
             "are not allowed")
    path = PurePosixPath(text)
    _require(not path.is_absolute(), f"{where}: absolute paths are not allowed")
    if yaml_file:
        _require(path.suffix.lower() in _YAML_SUFFIXES,
                 f"{where}: must reference a .yaml or .yml file")
    return text


def _validate_filename(value: Any, where: str) -> str:
    text = _validate_reference(value, where)
    _require(len(PurePosixPath(text).parts) == 1,
             f"{where}: must be a filename, not a path")
    return text


def _string_list(value: Any, where: str, *, paths: bool = False,
                 yaml_files: bool = False) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{where}: must be a list")
    items: list[str] = []
    for index, item in enumerate(value):
        if paths:
            items.append(_validate_reference(
                item, f"{where}[{index}]", yaml_file=yaml_files))
        else:
            items.append(_text(item, f"{where}[{index}]"))
    duplicates = sorted({item for item in items if items.count(item) > 1})
    _require(not duplicates, f"{where}: contains duplicates {duplicates}")
    return tuple(items)


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _StrictSafeLoader, node: MappingNode,
                              deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    result: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ProjectProfileError(
                "profile YAML mapping keys must be scalar/hashable") from exc
        if duplicate:
            raise ProjectProfileError(f"profile YAML contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class SpecificationSettings:
    """Explicit files and non-recursive directories containing approved specs."""

    approved: tuple[str, ...] = ()
    roots: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolingDefaults:
    """Current convenience defaults; none is historical or replay evidence."""

    external_anchor: str = ""
    confidence_config: str = ""
    output_directory: str = ""
    provenance_filename: str = ""
    report_filename: str = ""


@dataclass(frozen=True)
class ArtifactSettings:
    """Project names associated with artifact IDs declared by TargetSchema."""

    roles: tuple[tuple[str, str], ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()

    def for_role(self, role: str) -> str | None:
        return dict(self.roles).get(role)

    def for_alias(self, alias: str) -> str | None:
        return dict(self.aliases).get(alias)


@dataclass(frozen=True)
class ProjectProfile:
    """Loaded current-tooling settings with exact file-byte audit identity."""

    profile_id: str
    profile_format_version: int
    profile_version: str
    target_schema: str
    specifications: SpecificationSettings = dc_field(
        default_factory=SpecificationSettings)
    tooling: ToolingDefaults = dc_field(default_factory=ToolingDefaults)
    artifacts: ArtifactSettings = dc_field(default_factory=ArtifactSettings)
    sha256: str = ""
    source_path: str = ""
    project_root: str = ""
    target_schema_identity: str = ""
    validated_specifications: tuple[str, ...] = ()

    @property
    def ref(self) -> str:
        return f"{self.profile_id}@{self.profile_version}"

    def resolve(self, reference: str) -> Path:
        """Resolve a validated reference inside this profile's project root."""
        _require(bool(self.project_root),
                 "profile was parsed without a project root; load it to resolve paths")
        return _resolve_under_root(Path(self.project_root), reference,
                                   "profile reference")


def _parse_specifications(raw: Any) -> SpecificationSettings:
    if raw is None:
        return SpecificationSettings()
    _require(isinstance(raw, dict), "profile.specifications: must be a mapping")
    _check_keys(raw, _SPECIFICATION_KEYS, "profile.specifications")
    approved = _string_list(
        raw.get("approved", []), "profile.specifications.approved",
        paths=True, yaml_files=True)
    roots = _string_list(
        raw.get("roots", []), "profile.specifications.roots", paths=True)
    return SpecificationSettings(approved=approved, roots=roots)


def _parse_tooling(raw: Any) -> ToolingDefaults:
    if raw is None:
        return ToolingDefaults()
    _require(isinstance(raw, dict), "profile.tooling: must be a mapping")
    _check_keys(raw, _TOOLING_KEYS, "profile.tooling")

    def reference(key: str) -> str:
        value = raw.get(key, "")
        return (_validate_reference(value, f"profile.tooling.{key}")
                if value != "" else "")

    def filename(key: str) -> str:
        value = raw.get(key, "")
        return (_validate_filename(value, f"profile.tooling.{key}")
                if value != "" else "")

    return ToolingDefaults(
        external_anchor=reference("external_anchor"),
        confidence_config=reference("confidence_config"),
        output_directory=reference("output_directory"),
        provenance_filename=filename("provenance_filename"),
        report_filename=filename("report_filename"),
    )


def _named_artifacts(raw: Any, where: str) -> tuple[tuple[str, str], ...]:
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    pairs: list[tuple[str, str]] = []
    for name, artifact_id in raw.items():
        key = _validate_identifier(name, f"{where} key")
        pairs.append((key, _text(artifact_id, f"{where}.{key}")))
    return tuple(pairs)


def _parse_artifacts(raw: Any) -> ArtifactSettings:
    if raw is None:
        return ArtifactSettings()
    _require(isinstance(raw, dict), "profile.artifacts: must be a mapping")
    _check_keys(raw, _ARTIFACT_KEYS, "profile.artifacts")
    roles = _named_artifacts(raw.get("roles", {}), "profile.artifacts.roles")
    aliases = _named_artifacts(raw.get("aliases", {}), "profile.artifacts.aliases")
    overlap = sorted(set(dict(roles)) & set(dict(aliases)))
    _require(not overlap,
             f"profile.artifacts: names cannot be both roles and aliases: {overlap}")
    required = _string_list(raw.get("required", []), "profile.artifacts.required")
    optional = _string_list(raw.get("optional", []), "profile.artifacts.optional")
    both = sorted(set(required) & set(optional))
    _require(not both,
             f"profile.artifacts: artifact IDs cannot be both required and optional: {both}")
    return ArtifactSettings(
        roles=roles, aliases=aliases, required=required, optional=optional)


def parse(doc: Any, *, sha256: str = "", source_path: str = "") -> ProjectProfile:
    """Build a structurally validated profile from a parsed YAML document."""
    _require(isinstance(doc, dict), "profile document must be a mapping")
    _check_keys(doc, _DOC_KEYS, "profile")
    profile_id = _validate_identifier(doc.get("profile_id"), "profile.profile_id")
    format_version = doc.get("profile_format_version")
    _require(type(format_version) is int,  # bool is intentionally not an int here
             "profile.profile_format_version: must be an integer")
    _require(format_version == PROFILE_FORMAT_VERSION,
             f"profile.profile_format_version: unsupported version {format_version!r}; "
             f"expected {PROFILE_FORMAT_VERSION}")
    profile_version = _text(doc.get("profile_version"), "profile.profile_version")
    _require(bool(_VERSION_RE.fullmatch(profile_version)),
             "profile.profile_version: must be MAJOR.MINOR.PATCH")
    target_schema = _validate_reference(
        doc.get("target_schema"), "profile.target_schema", yaml_file=True)
    return ProjectProfile(
        profile_id=profile_id,
        profile_format_version=format_version,
        profile_version=profile_version,
        target_schema=target_schema,
        specifications=_parse_specifications(doc.get("specifications")),
        tooling=_parse_tooling(doc.get("tooling")),
        artifacts=_parse_artifacts(doc.get("artifacts")),
        sha256=sha256,
        source_path=source_path,
    )


def _resolve_under_root(root: Path, reference: str, where: str) -> Path:
    validated = _validate_reference(reference, where)
    resolved = (root / Path(*PurePosixPath(validated).parts)).resolve()
    _require(resolved.is_relative_to(root),
             f"{where}: resolves outside the explicit project root")
    return resolved


def _profile_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    _require(resolved.is_relative_to(root),
             "profile file resolves outside the explicit project root")
    _require(resolved.is_file(), f"profile file not found: {resolved}")
    return resolved


def _specification_paths(profile: ProjectProfile, root: Path) -> tuple[Path, ...]:
    found: list[Path] = []
    for index, reference in enumerate(profile.specifications.approved):
        path = _resolve_under_root(
            root, reference, f"profile.specifications.approved[{index}]")
        _require(path.is_file(), f"approved specification file not found: {reference}")
        found.append(path)
    for index, reference in enumerate(profile.specifications.roots):
        directory = _resolve_under_root(
            root, reference, f"profile.specifications.roots[{index}]")
        _require(directory.is_dir(), f"specification root not found: {reference}")
        immediate = sorted(
            (path.resolve() for path in directory.iterdir()
             if path.is_file() and path.suffix.lower() in _YAML_SUFFIXES),
            key=lambda path: path.name,
        )
        _require(immediate, f"specification root contains no YAML files: {reference}")
        for path in immediate:
            _require(path.is_relative_to(root),
                     f"specification in {reference!r} resolves outside project root")
        found.extend(immediate)
    duplicates = sorted({str(path) for path in found if found.count(path) > 1})
    _require(not duplicates,
             "profile specifications resolve to duplicate files: "
             f"{[Path(path).name for path in duplicates]}")
    return tuple(found)


def _validate_artifacts(profile: ProjectProfile, target_schema) -> None:
    settings = profile.artifacts
    referenced = (
        [artifact for _, artifact in settings.roles]
        + [artifact for _, artifact in settings.aliases]
        + list(settings.required)
        + list(settings.optional)
    )
    if not referenced:
        return
    declared = tuple(target_schema.artifacts)
    _require(bool(declared),
             f"target schema {target_schema.ref} declares no artifact IDs; "
             "profile roles/aliases cannot create a parallel artifact model")
    duplicate_schema_ids = sorted({item for item in declared
                                   if declared.count(item) > 1})
    _require(not duplicate_schema_ids,
             f"target schema {target_schema.ref} declares duplicate artifact IDs "
             f"{duplicate_schema_ids}")
    unknown = sorted(set(referenced) - set(declared))
    _require(not unknown,
             f"profile artifacts reference IDs not declared by target schema "
             f"{target_schema.ref}: {unknown}")


def load(path: str | Path, *, project_root: str | Path) -> ProjectProfile:
    """Load and fully validate one profile under an explicit project root.

    The exact profile bytes define ``sha256``.  Reference validation loads the
    target schema and parses explicitly selected specs without following their
    source paths or resolving external vocabularies.  Nothing in this function
    runs harmonization or replay.
    """
    from .schema_model import SchemaError, load as load_schema
    from .spec_model import SpecError, load as load_spec

    root = Path(project_root).resolve()
    _require(root.is_dir(), f"project root not found: {root}")
    profile_path = _profile_path(path, root)
    try:
        payload = profile_path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProjectProfileError(f"profile file could not be read as UTF-8: {exc}") from exc
    try:
        document = yaml.load(text, Loader=_StrictSafeLoader)
    except ProjectProfileError:
        raise
    except yaml.YAMLError as exc:
        raise ProjectProfileError(f"profile file is not valid YAML: {exc}") from exc

    profile = parse(
        document,
        sha256=hashlib.sha256(payload).hexdigest(),
        source_path=str(profile_path),
    )
    schema_path = _resolve_under_root(root, profile.target_schema,
                                      "profile.target_schema")
    try:
        target_schema = load_schema(schema_path)
    except (OSError, SchemaError) as exc:
        raise ProjectProfileError(f"target schema reference is invalid: {exc}") from exc
    _validate_artifacts(profile, target_schema)

    for label, reference in (
        ("external_anchor", profile.tooling.external_anchor),
        ("confidence_config", profile.tooling.confidence_config),
    ):
        if reference:
            resolved = _resolve_under_root(root, reference, f"profile.tooling.{label}")
            _require(resolved.is_file(),
                     f"profile.tooling.{label}: referenced file not found: {reference}")
    if profile.tooling.output_directory:
        _resolve_under_root(
            root,
            profile.tooling.output_directory,
            "profile.tooling.output_directory",
        )

    identities: list[str] = []
    for spec_path in _specification_paths(profile, root):
        try:
            spec = load_spec(
                spec_path,
                target_schema=target_schema,
                require_executable=True,
                resolve_vocabularies=False,
            )
        except (OSError, SpecError) as exc:
            relative = spec_path.relative_to(root)
            raise ProjectProfileError(
                f"approved specification {relative} is invalid: {exc}") from exc
        identities.append(spec.ref)
    duplicate_identities = sorted({ref for ref in identities
                                   if identities.count(ref) > 1})
    _require(not duplicate_identities,
             f"profile specifications declare duplicate identities "
             f"{duplicate_identities}")

    return replace(
        profile,
        project_root=str(root),
        target_schema_identity=target_schema.ref,
        validated_specifications=tuple(identities),
    )
