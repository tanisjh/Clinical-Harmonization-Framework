#!/usr/bin/env python
"""view_provenance.py — read one Cell's recorded lineage from a run's provenance.

Read-only, and **explicit-file only**. It shows what a harmonization run
recorded for one delivered value: which file, row and column the value came
from, which approved rule produced it, and what the source actually said.

    python scripts/view_provenance.py --provenance out/demo_cells.csv \
        --participant-id SYNTH-PARTICIPANT-001 --field example_field \
        --collection-event example-event

WHAT THIS IS
    A viewer. It reads the provenance CSV that `scripts/harmonize_spec.py`
    writes with `--provenance`, and prints one Cell.

WHAT THIS IS NOT
    It is **not** an independent verifier. It does not re-read the source, does
    not re-apply the rule, and does not recompute anything. It reports what the
    run recorded. The truthful claim is: *the provenance trace shows exactly
    which recorded source evidence and approved rule produced a harmonized
    value* — not that the value has been independently confirmed.

    It is also not `trace_value.py`, and deliberately does not behave like it.
    That tool resolves a closed allow-list of approved production artifacts and
    re-reads the source to confirm the recorded raw value. Its refusals are a
    safety property: a basename-resolved trace once opened the wrong file and
    returned a confident wrong answer for another patient. **Nothing here
    discovers, resolves or searches for a file.** The only provenance this reads
    is the exact path the caller names, which is why it can be pointed at a
    fresh run without weakening anything.

AMBIGUITY
    A participant and a field do not identify one Cell — a participant
    contributes one record per collection event. Rather than silently show the
    first match, an ambiguous query is refused and the distinguishing values are
    printed so the query can be narrowed.

Exit codes
    0   one Cell shown (or several, with --all)
    1   no Cell matched
    2   usage error
    3   the provenance file could not be read or is missing required columns
    4   the query is ambiguous — narrow it, or pass --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXIT_OK = 0
EXIT_NO_MATCH = 1
EXIT_USAGE = 2
EXIT_BAD_PROVENANCE = 3
EXIT_AMBIGUOUS = 4

#: The columns a Cell view needs. A file missing any of these is not a
#: provenance CSV from this pipeline, and guessing would be worse than refusing.
REQUIRED_COLUMNS = (
    "trial", "cimac_part_id", "Cimac.id", "Collection_Event",
    "harmonized_field", "value", "confidence", "rule_id", "rule_version",
    "source_file", "source_resolved_path", "source_column",
    "source_row_number", "source_record_key", "raw_value",
)

#: Columns that distinguish otherwise-identical matches, in the order a person
#: would use to narrow a query.
DISCRIMINATORS = ("trial", "cimac_part_id", "Cimac.id", "Collection_Event",
                  "harmonized_field")

WIDTH = 74


def load_provenance(path: Path) -> pd.DataFrame:
    """Read the named provenance CSV, or refuse with a reason."""
    if not path.is_file():
        raise FileNotFoundError(f"provenance file not found: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"{path}: not readable as CSV: {e}") from e
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path}: not a harmonization provenance CSV; missing column(s) "
            f"{missing}. Generate one with: scripts/harmonize_spec.py "
            f"<spec> --output <csv> --provenance <csv>")
    return frame


def select(frame: pd.DataFrame, args) -> pd.DataFrame:
    """Rows matching every filter the caller supplied."""
    mask = pd.Series(True, index=frame.index)
    for column, wanted in (("cimac_part_id", args.participant_id),
                           ("harmonized_field", args.field),
                           ("Collection_Event", args.collection_event),
                           ("Cimac.id", args.cimac_id),
                           ("trial", args.trial),
                           ("source_record_key", args.record_key)):
        if wanted:
            mask &= frame[column] == wanted
    return frame[mask]


def _decode(text: str):
    """Decode a structured provenance field, tolerating anything unexpected.

    `json.loads` only — never `eval`. A field that will not decode is shown as
    the text it is rather than hidden, because a reader who came here to see the
    evidence should not be shown less than the run recorded.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def _line(label: str, value) -> str:
    return f"  {label:<12}{value}"


def render(row: pd.Series) -> str:
    """One Cell, as a person would want to read it."""
    L: list[str] = ["─" * WIDTH]
    L.append(f"PROVENANCE VIEW  {row['trial']} | {row['cimac_part_id']} | "
             f"{row['harmonized_field']}")
    L.append("─" * WIDTH)

    L.append("")
    L.append("HARMONIZED")
    L.append(_line("field", row["harmonized_field"]))
    L.append(_line("value", repr(row["value"]) if row["value"] != "" else "(none)"))
    L.append(_line("confidence", row["confidence"]))
    rule = row["rule_id"] + (f"  v{row['rule_version']}"
                             if row["rule_version"] else "")
    L.append(_line("rule", rule))
    if row.get("value_state"):
        L.append(_line("state", row["value_state"]))
    if row.get("extraction_method"):
        L.append(_line("method", row["extraction_method"]))
    if row.get("authoritative_for"):
        L.append(_line("authoritative", row["authoritative_for"]))

    L.append("")
    L.append("RECORD")
    L.append(_line("participant", row["cimac_part_id"]))
    if row.get("Cimac.id"):
        L.append(_line("sample", row["Cimac.id"]))
    if row.get("Collection_Event"):
        L.append(_line("event", row["Collection_Event"]))
    if row["source_record_key"]:
        L.append(_line("record key", row["source_record_key"]))

    L.append("")
    if row["source_resolved_path"]:
        L.append("SOURCE")
        L.append(_line("file", row["source_resolved_path"]))
        if row.get("source_logical_label"):
            L.append(_line("label", row["source_logical_label"]))
        if row.get("source_sheet"):
            L.append(_line("sheet", row["source_sheet"]))
        if row["source_column"]:
            L.append(_line("column", row["source_column"]))
        if row["source_row_number"] != "":
            L.append(_line("row", row["source_row_number"]))
        if row["raw_value"] != "":
            L.append(_line("raw", repr(row["raw_value"])))
        if row.get("source_sha256"):
            L.append(_line("sha256", row["source_sha256"]))
    else:
        # A constant, an anchor value or a derived value reads no file. Saying
        # so is more useful than printing a block of empty fields.
        L.append("SOURCE")
        L.append(_line("origin", f"{row['source_file']}  (not a source file)"))
        if row["source_column"]:
            L.append(_line("inputs", row["source_column"]))
        if row["raw_value"] != "":
            L.append(_line("raw", repr(row["raw_value"])))

    contributions = _decode(row.get("contributions", ""))
    if contributions:
        L.append("")
        L.append(f"CONTRIBUTIONS  ({len(contributions) if isinstance(contributions, list) else 1})")
        items = contributions if isinstance(contributions, list) else [contributions]
        for i, c in enumerate(items, 1):
            if not isinstance(c, dict):
                L.append(f"  [{i}] {c}")
                continue
            head = c.get("role") or c.get("status") or c.get("label") or "input"
            L.append(f"  [{i}] {head}")
            for label, key in (("column", "source_column"),
                               ("raw", "raw_value"),
                               ("status", "status"),
                               ("file", "source_logical_label"),
                               ("row", "source_row_number"),
                               ("rule", "rule_id")):
                if c.get(key) not in (None, ""):
                    L.append(f"        {label:<8}{c[key]!r}"
                             if label == "raw" else
                             f"        {label:<8}{c[key]}")

    derivation = _decode(row.get("derivation", ""))
    if derivation:
        L.append("")
        L.append("DERIVATION")
        if isinstance(derivation, dict):
            for key in sorted(derivation):
                value = derivation[key]
                if value in (None, "", [], {}):
                    continue
                if key in ("open_question_ref", "unresolved"):
                    continue
                L.append(f"  {key:<24}{value}")
            if derivation.get("open_question_ref"):
                L.append("")
                L.append(f"  OPEN QUESTION  {derivation['open_question_ref']}")
                if derivation.get("unresolved"):
                    for line in _wrap(str(derivation["unresolved"]), WIDTH - 6):
                        L.append(f"    {line}")
        else:
            L.append(f"  {derivation}")

    if row.get("notes"):
        L.append("")
        L.append("NOTES")
        for line in _wrap(str(row["notes"]), WIDTH - 4):
            L.append(f"  {line}")

    L.append("")
    L.append("─" * WIDTH)
    L.append("This is a PROVENANCE VIEW of what the run recorded. It does not")
    L.append("re-read the source or recompute the value.")
    return "\n".join(L)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) or [""]


def _describe_ambiguity(hits: pd.DataFrame, args) -> str:
    """Show what differs between the matches, so the query can be narrowed."""
    L = [f"AMBIGUOUS  {len(hits)} cells match that query.", ""]
    varying = [c for c in DISCRIMINATORS if hits[c].nunique() > 1]
    shown = varying or list(DISCRIMINATORS)
    L.append("  " + "  ".join(f"{c:<22}" for c in shown))
    for _, row in hits.head(12).iterrows():
        L.append("  " + "  ".join(f"{str(row[c]):<22}" for c in shown))
    if len(hits) > 12:
        L.append(f"  … and {len(hits) - 12} more")
    L.append("")
    flags = {"trial": "--trial", "cimac_part_id": "--participant-id",
             "Cimac.id": "--cimac-id", "Collection_Event": "--collection-event",
             "harmonized_field": "--field"}
    hint = ", ".join(flags[c] for c in varying if c in flags)
    L.append(f"  Narrow the query with {hint}" if hint else
             "  Pass --all to show every match.")
    L.append("  Or pass --all to show every match.")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="View one Cell's recorded lineage from a run's provenance CSV.")
    p.add_argument("--provenance", required=True,
                   help="explicit path to a provenance CSV from harmonize_spec.py")
    p.add_argument("--participant-id", help="cimac_part_id")
    p.add_argument("--field", help="harmonized field name")
    p.add_argument("--collection-event", help="timepoint, when a participant has several")
    p.add_argument("--cimac-id", help="sample id, when a participant has several")
    p.add_argument("--trial", help="trial, when the file holds more than one")
    p.add_argument("--record-key", help="source record key")
    p.add_argument("--all", action="store_true",
                   help="show every match instead of refusing an ambiguous query")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    if not any((args.participant_id, args.field, args.collection_event,
                args.cimac_id, args.trial, args.record_key)):
        print("ERROR  give at least one filter, e.g. --participant-id and --field",
              file=sys.stderr)
        return EXIT_USAGE

    try:
        frame = load_provenance(Path(args.provenance))
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR  {e}", file=sys.stderr)
        return EXIT_BAD_PROVENANCE

    hits = select(frame, args)
    if hits.empty:
        print("NO MATCH  no cell in that provenance matches the query.",
              file=sys.stderr)
        if args.field and args.field not in set(frame["harmonized_field"]):
            known = sorted(set(frame["harmonized_field"]))
            print(f"          unknown field {args.field!r}; the file holds: "
                  f"{known}", file=sys.stderr)
        elif args.participant_id and \
                args.participant_id not in set(frame["cimac_part_id"]):
            print(f"          unknown participant {args.participant_id!r}",
                  file=sys.stderr)
        return EXIT_NO_MATCH

    if len(hits) > 1 and not args.all:
        if args.json:
            print(json.dumps({"status": "AMBIGUOUS", "matches": len(hits),
                              "cells": hits[list(DISCRIMINATORS)]
                              .to_dict("records")}, indent=2))
        else:
            print(_describe_ambiguity(hits, args), file=sys.stderr)
        return EXIT_AMBIGUOUS

    if args.json:
        print(json.dumps({"status": "OK", "matches": len(hits),
                          "cells": [
                              {**{k: v for k, v in row.items()},
                               "contributions": _decode(row.get("contributions", "")),
                               "derivation": _decode(row.get("derivation", ""))}
                              for _, row in hits.iterrows()]},
                         indent=2, default=str))
        return EXIT_OK

    for i, (_, row) in enumerate(hits.iterrows()):
        if i:
            print()
        print(render(row))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
