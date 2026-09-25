"""rule_registry.py — resolve a rule identity to a vetted implementation.

Phase 2c. A specification names a rule by a stable `(rule_id, version)` and says
nothing about Python. This module is the only place that knows which callable
that pair means. The split is the point:

  * the **specification** carries the clinical content — what the rule is for,
    its parameters, the evidence behind it, who approved it, what remains an
    open question;
  * the **registry** carries implementation facts only — the callable, how many
    inputs it takes, which keyword parameters it accepts, what it returns.

A registry entry therefore has no clinical meaning in it. Moving a landmark
threshold or a response vocabulary in here would hide a clinical decision behind
an identifier, which is exactly what the identifier exists to prevent.

There is deliberately **no** dynamic import and no `implementation:` path read
from YAML. A specification that could name `module:function` could name any
function; the registry is a closed set, and an unknown id is an error rather
than an import.

This is not a plugin framework. It is a dict with a lookup that checks four
things — that the id exists, that the version matches, that the kind is the one
the caller is in a position to use, and that the call the specification
describes is one the implementation can accept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from lib import normalize, post_rules
from lib.confidence_policy import (
    InvalidRuleConfidence,
    validate_registered_rule_confidence,
)

#: A rule that produces one target field for one record, from values the
#: pipeline has already produced. The only kind implemented so far.
KIND_FIELD = "field"

#: A rule applied to a whole artifact after the individual fields exist. It
#: reads an assembled table and returns events describing what it would change;
#: it never mutates anything itself.
KIND_POST = "post"

KINDS = (KIND_FIELD, KIND_POST)


class RuleError(RuntimeError):
    """A rule reference cannot be honoured."""


class RuleNotRegistered(RuleError):
    pass


class RuleVersionMismatch(RuleError):
    pass


class RuleKindMismatch(RuleError):
    pass


class RuleContractMismatch(RuleError):
    pass


class RuleReturnContractMismatch(RuleError):
    pass


@dataclass(frozen=True)
class RegisteredRule:
    """Implementation facts about one rule. No clinical content."""

    rule_id: str
    version: str
    kind: str
    func: Callable[..., Any]
    #: How many positional inputs the callable takes. Deliberately a count and
    #: not a list of field names: naming CIMAC fields here would put domain
    #: knowledge into the registry, and the specification already declares which
    #: fields feed the rule, in order.
    arity: int
    #: Keyword parameters the callable accepts, supplied from the
    #: specification's `params` block.
    params: tuple[str, ...] = ()
    #: True when the callable returns `(value, confidence, note)` rather than a
    #: bare value, so the confidence of the result comes from the rule.
    returns_confidence: bool = True
    #: True when the third return element is a human-readable note.
    returns_note: bool = True

    @property
    def ref(self) -> str:
        return f"{self.rule_id}@{self.version}"


def _register(*rules: RegisteredRule) -> dict[tuple[str, str], RegisteredRule]:
    out: dict[tuple[str, str], RegisteredRule] = {}
    for r in rules:
        key = (r.rule_id, r.version)
        if key in out:
            raise RuleError(f"duplicate registry entry for {r.ref}")
        if r.kind not in KINDS:
            raise RuleError(f"{r.ref}: unknown rule kind {r.kind!r}")
        out[key] = r
    return out


#: The registered set. Only the rules an executed specification currently needs.
#: Each callable already existed and is unchanged; registering it does not alter
#: any behaviour, it only gives the rule id something to resolve to.
REGISTRY: dict[tuple[str, str], RegisteredRule] = _register(
    RegisteredRule(
        rule_id="derived.bor_binary.v1", version="1", kind=KIND_FIELD,
        func=normalize.derive_bor_binary, arity=1, params=("rules",),
    ),
    RegisteredRule(
        rule_id="derived.bor_bin_120d.v1", version="1", kind=KIND_FIELD,
        func=normalize.derive_bor_bin, arity=2, params=("landmark_days",),
    ),
    RegisteredRule(
        rule_id="derived.pfs_bin_120d.v1", version="1", kind=KIND_FIELD,
        func=normalize.derive_pfs_bin, arity=2, params=("landmark_days",),
    ),
    RegisteredRule(
        rule_id="derived.clinical_benefit_from_pfs_120d.v1", version="1",
        kind=KIND_POST, func=post_rules.landmark_split, arity=1,
        params=("input_field", "landmark_days", "at_or_above", "below"),
        returns_confidence=False, returns_note=False,
    ),
)


def resolve(rule_id: str, version: str, *, kind: str) -> RegisteredRule:
    """Look up a rule, refusing every mismatch rather than approximating one."""
    versions = sorted(v for (rid, v) in REGISTRY if rid == rule_id)
    if not versions:
        raise RuleNotRegistered(
            f"rule {rule_id!r} is not registered; known rules: "
            f"{sorted({rid for rid, _ in REGISTRY})}")
    rule = REGISTRY.get((rule_id, version))
    if rule is None:
        raise RuleVersionMismatch(
            f"rule {rule_id!r} is registered at version(s) {versions}, "
            f"not {version!r}")
    if rule.kind != kind:
        raise RuleKindMismatch(
            f"rule {rule.ref} is a {rule.kind!r} rule and cannot be used where a "
            f"{kind!r} rule is expected")
    return rule


def check_contract(rule: RegisteredRule, inputs: tuple[str, ...],
                   params: dict[str, Any]) -> None:
    """Verify the call a specification describes is one the rule can accept."""
    if len(inputs) != rule.arity:
        raise RuleContractMismatch(
            f"rule {rule.ref} takes {rule.arity} input(s); the specification "
            f"declares {len(inputs)}: {list(inputs)}")
    unknown = sorted(set(params) - set(rule.params))
    if unknown:
        raise RuleContractMismatch(
            f"rule {rule.ref} accepts parameter(s) {list(rule.params)}; the "
            f"specification supplies unknown parameter(s) {unknown}")
    missing = sorted(set(rule.params) - set(params))
    if missing:
        raise RuleContractMismatch(
            f"rule {rule.ref} requires parameter(s) {missing}, which the "
            f"specification does not supply — a rule parameter is a declared "
            f"decision, not a Python default")


def invoke(rule: RegisteredRule, values: list[Any],
           params: dict[str, Any]) -> tuple[Any, float | None, str]:
    """Call a field rule and normalise its return into (value, confidence, note)."""
    result = rule.func(*values, **params)
    if not rule.returns_confidence:
        return result, None, ""
    value, confidence, note = result
    try:
        confidence = validate_registered_rule_confidence(confidence)
    except InvalidRuleConfidence as exc:
        raise RuleReturnContractMismatch(
            f"rule {rule.ref} returned invalid confidence: {exc}") from exc
    return value, confidence, str(note or "")
