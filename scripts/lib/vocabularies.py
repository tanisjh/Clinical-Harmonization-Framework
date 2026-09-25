"""vocabularies.py — resolve a shared vocabulary reference to its table.

Phase 2c, Pilot 2. Some harmonization knowledge is cross-trial rather than
per-trial: the response-code buckets that `derived.bor_binary.v1` matches
against are the same table for every trial that uses that rule. Copying it into
each specification would duplicate it eleven times and let the copies drift.

A specification therefore names a vocabulary by a stable `id@version` and this
module resolves it. The distinction the rest of the architecture rests on is
preserved:

  * the **vocabulary file** is declarative clinical data, versioned and
    reviewable, and it is where the terms live;
  * **Python implements matching, never the vocabulary itself.**

Resolution is deterministic and closed: files are discovered once from a fixed
directory, an unknown id or version is an error rather than a fallback, and no
vocabulary is constructed in code.
"""
from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOCABULARY_DIR = PROJECT_ROOT / "vocabularies"

_DOC_KEYS = {"vocabulary_id", "version", "kind", "description", "evidence",
             "terms"}

#: A vocabulary is either a keyed table (a term -> its members) or an ordered
#: list of match rules where the first match wins. Closed: these are the only
#: two shapes the harmonization actually uses.
KIND_BUCKETS = "buckets"
KIND_ORDERED_RULES = "ordered_rules"
KINDS = (KIND_BUCKETS, KIND_ORDERED_RULES)

#: Match kinds an ordered rule may declare. Exactly the set the pipeline's
#: existing normalization implements — no more, so a vocabulary can never ask
#: for matching behaviour that does not exist.
MATCH_KINDS = ("exact", "iexact", "prefix", "contains", "regex")

_TERM_KEYS = {"match", "kind", "value", "confidence", "rule", "note"}


class VocabularyError(ValueError):
    """A vocabulary cannot be loaded or resolved."""


@dataclass(frozen=True)
class MatchRule:
    """One ordered normalization rule: what it matches and what it delivers."""

    match: str
    kind: str = "exact"
    value: Any = None
    confidence: float = 0.7
    rule_id: str = ""
    rule_version: str = "1"
    note: str = ""


@dataclass(frozen=True)
class Vocabulary:
    vocabulary_id: str
    version: str
    #: A keyed table for `buckets`; an ordered tuple of MatchRule for
    #: `ordered_rules`.
    terms: Any
    kind: str = KIND_BUCKETS
    description: str = ""
    evidence: str = ""
    source_path: str = ""
    source_sha256: str = ""
    parser_id: str = "pyyaml.safe_load"
    parser_version: str = ""

    @property
    def ref(self) -> str:
        return f"{self.vocabulary_id}@{self.version}"

    def contained_path(self, project_root: str | Path) -> str:
        """Return the exact loaded path relative to *project_root*.

        The runtime object, not a later directory lookup, is the authority.
        Resolving the already-retained path is safe; opening or searching for a
        replacement is deliberately outside this method.
        """
        root = Path(project_root).resolve()
        path = Path(self.source_path).resolve()
        if not path.is_relative_to(root):
            raise VocabularyError(
                f"loaded vocabulary path escapes project root: {path}")
        return str(path.relative_to(root))

    def evidence_ref_id(self, project_root: str | Path) -> str:
        """Deterministic identity for this exact runtime-loaded binding."""
        identity = {
            "kind": "vocabulary",
            "path": self.contained_path(project_root),
            "reference": self.ref,
            "sha256": self.source_sha256,
            "vocabulary_kind": self.kind,
        }
        encoded = json.dumps(
            identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return f"vocab-{hashlib.sha256(encoded).hexdigest()}"

    def evidence_details(self) -> dict[str, str]:
        """Manifest details that reproduce the parsed vocabulary identity."""
        return {
            "vocabulary_id": self.vocabulary_id,
            "version": self.version,
            "vocabulary_kind": self.kind,
            "description": self.description,
            "evidence": self.evidence,
            "parser_id": self.parser_id,
            "parser_version": self.parser_version,
        }


def parse_ref(ref: str) -> tuple[str, str]:
    """Split `id@version`, refusing anything that is not exactly that."""
    if not isinstance(ref, str) or ref.count("@") != 1:
        raise VocabularyError(
            f"vocabulary reference must be 'id@version', got {ref!r}")
    vid, version = ref.split("@")
    if not vid or not version:
        raise VocabularyError(
            f"vocabulary reference must be 'id@version', got {ref!r}")
    return vid, version


def load_bytes(payload: bytes, *, source_path: str | Path) -> Vocabulary:
    """Parse one exact byte payload and retain its source identity."""
    p = Path(source_path).resolve()
    try:
        doc = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise VocabularyError(f"vocabulary is not valid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise VocabularyError(f"{p}: a vocabulary must be a mapping")
    unknown = sorted(set(doc) - _DOC_KEYS)
    if unknown:
        raise VocabularyError(f"{p}: unknown key(s) {unknown}")
    for key in ("vocabulary_id", "version", "terms"):
        if not doc.get(key):
            raise VocabularyError(f"{p}: {key} is required")
    kind = doc.get("kind", KIND_BUCKETS)
    if kind not in KINDS:
        raise VocabularyError(f"{p}: kind must be one of {list(KINDS)}, got {kind!r}")
    raw_terms = doc["terms"]

    if kind == KIND_BUCKETS:
        if not isinstance(raw_terms, dict) or not raw_terms:
            raise VocabularyError(f"{p}: buckets terms must be a non-empty mapping")
        terms: Any = raw_terms
    else:
        if not isinstance(raw_terms, list) or not raw_terms:
            raise VocabularyError(
                f"{p}: ordered_rules terms must be a non-empty list — order is "
                f"part of the meaning, so a mapping cannot express them")
        rules = []
        for i, t in enumerate(raw_terms):
            where = f"{p}: terms[{i}]"
            if not isinstance(t, dict):
                raise VocabularyError(f"{where}: must be a mapping")
            unknown = sorted(set(t) - _TERM_KEYS)
            if unknown:
                raise VocabularyError(f"{where}: unknown key(s) {unknown}")
            if not t.get("match"):
                raise VocabularyError(f"{where}: match is required")
            mkind = t.get("kind", "exact")
            if mkind not in MATCH_KINDS:
                raise VocabularyError(
                    f"{where}: kind must be one of {list(MATCH_KINDS)}, "
                    f"got {mkind!r}")
            conf = t.get("confidence")
            if not isinstance(conf, (int, float)) or isinstance(conf, bool) \
                    or not 0.0 <= float(conf) <= 1.0:
                raise VocabularyError(
                    f"{where}: confidence must be a number in [0, 1]; a matched "
                    f"entry states what its delivery is worth")
            rule = t.get("rule") or {}
            if not isinstance(rule, dict) or not rule.get("id"):
                raise VocabularyError(
                    f"{where}: rule.id is required — the provenance identity of a "
                    f"match is stated, never inferred from its confidence")
            rules.append(MatchRule(
                match=str(t["match"]), kind=mkind, value=t.get("value"),
                confidence=float(conf), rule_id=str(rule["id"]),
                rule_version=str(rule.get("version", "1")),
                note=t.get("note", "") or ""))
        terms = tuple(rules)

    return Vocabulary(
        vocabulary_id=str(doc["vocabulary_id"]), version=str(doc["version"]),
        terms=terms, kind=kind, description=doc.get("description", "") or "",
        evidence=doc.get("evidence", "") or "", source_path=str(p),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        parser_version=str(getattr(yaml, "__version__", "")))


def load_file(path: str | Path) -> Vocabulary:
    p = Path(path).resolve()
    if not p.is_file():
        raise VocabularyError(f"vocabulary file not found: {p}")
    payload = p.read_bytes()
    # The digest and parsed document intentionally come from the same byte
    # payload. No later reopen can silently bind different bytes.
    return load_bytes(payload, source_path=p)


@functools.lru_cache(maxsize=1)
def _index(directory: str) -> dict[tuple[str, str], Vocabulary]:
    out: dict[tuple[str, str], Vocabulary] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        vocab = load_file(path)
        key = (vocab.vocabulary_id, vocab.version)
        if key in out:
            raise VocabularyError(
                f"two files declare vocabulary {vocab.ref}: "
                f"{out[key].source_path} and {vocab.source_path}")
        out[key] = vocab
    return out


def resolve(ref: str, *, directory: str | Path = VOCABULARY_DIR) -> Vocabulary:
    """Resolve `id@version` to its vocabulary, or fail saying what exists."""
    vid, version = parse_ref(ref)
    index = _index(str(directory))
    found = index.get((vid, version))
    if found is not None:
        return found
    versions = sorted(v for (i, v) in index if i == vid)
    if versions:
        raise VocabularyError(
            f"vocabulary {vid!r} exists at version(s) {versions}, not {version!r}")
    raise VocabularyError(
        f"vocabulary {vid!r} is not defined; known vocabularies: "
        f"{sorted({i for i, _ in index})}")


def match(value: Any, rules) -> "MatchRule | None":
    """First ordered rule whose pattern matches `value`, or None.

    Mirrors the matching the pipeline's `lib.normalize.normalize` already
    performs, so a vocabulary describes behaviour that exists rather than asking
    for behaviour that does not. Order decides: the first match wins and later
    rules are not consulted.

    Returns None for an absent value and for one no rule recognises; the caller
    distinguishes those two, because they are different facts.
    """
    import re

    if value is None or (isinstance(value, float) and value != value):
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "na", "n/a"):
        return None
    for rule in rules:
        m = str(rule.match)
        try:
            hit = (
                rule.kind == "exact" and s == m
                or rule.kind == "iexact" and s.lower() == m.lower()
                or rule.kind == "prefix" and s.lower().startswith(m.lower())
                or rule.kind == "contains" and m.lower() in s.lower()
                or rule.kind == "regex" and bool(re.search(m, s))
            )
        except re.error:
            hit = False
        if hit:
            return rule
    return None


def is_absent(value: Any) -> bool:
    """True when a source value is missing rather than merely unrecognised."""
    if value is None or (isinstance(value, float) and value != value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in ("nan", "na", "n/a")


def available(*, directory: str | Path = VOCABULARY_DIR) -> list[str]:
    return sorted(v.ref for v in _index(str(directory)).values())
