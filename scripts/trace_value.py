"""trace_value.py — Read-only lineage for one delivered harmonized value.

Answers "where did this number come from?" by joining a delivered cell to its
authoritative provenance, then RE-READING the source to confirm the recorded raw
value is still true.

    python scripts/trace_value.py --artifact harmonized_11trials \\
        --trial SYNTHETIC-TRIAL --participant-id SYNTH-PARTICIPANT-001 \\
        --field example_field

The row anchor is (trial, cimac_part_id, Cimac.id, Collection_Event). A
participant id alone is often enough, but where it is not — a participant with
several samples or timepoints — the tool REFUSES and lists the candidates rather
than picking one.

Refusal is the design centre. Phase 1a found provenance citing files the pipeline
never opened, with identically-named files sitting on disk ready to be read by
mistake. A trace that guessed would have returned a confident wrong answer, which
in a clinical audit is worse than no answer. So this command verifies rather than
trusts, and every unverifiable condition has its own exit code:

    0  VERIFIED       value traced and re-read from source
    2  usage error
    3  NOT_FOUND      no such artifact row, or no provenance for the cell
    4  AMBIGUOUS_ROW  the anchor matches more than one delivered row
    5  AMBIGUOUS_AUTH more than one authoritative provenance event
    6  INCOMPLETE     provenance lacks what a trace needs
    7  UNRESOLVABLE   source path missing or outside approved roots
    8  DRIFTED        source hash no longer matches what was recorded
    9  MISMATCH       raw value could not be reproduced from source
   10  UNKNOWN_ARTIFACT

There is deliberately no basename fallback: if the recorded path does not
resolve, the answer is UNRESOLVABLE, never a same-named file somewhere else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import artifacts, run_manifest, value_states          # noqa: E402
from lib.contributions import from_json                        # noqa: E402
from lib.record_keys import parse_record_key                    # noqa: E402
from lib.source_ref import (                                    # noqa: E402
    PathContainmentError,
    assert_within_roots,
    sha256_file,
)
from lib.trace_io import read_source                            # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3
EXIT_AMBIGUOUS_ROW = 4
EXIT_AMBIGUOUS_AUTH = 5
EXIT_INCOMPLETE = 6
EXIT_UNRESOLVABLE = 7
EXIT_DRIFTED = 8
EXIT_MISMATCH = 9
EXIT_UNKNOWN_ARTIFACT = 10

ANCHOR_COLS = ["trial", "cimac_part_id", "Cimac.id", "Collection_Event"]

#: Short names accepted on the command line.
ARTIFACT_ALIASES = {
    "harmonized_11trials": artifacts.ELEVEN,
    "harmonized_9trials_gold": artifacts.GOLD9,
    "harmonized_9trials_reproduced": artifacts.REPRODUCED,
}


class TraceRefusal(Exception):
    """A condition under which the trace must not answer."""

    def __init__(self, code: int, status: str, message: str, detail=None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message
        self.detail = detail or {}


def _norm(v) -> str:
    s = str(v).strip()
    if s.lower() in ("", "nan", "none", "<na>"):
        return ""
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def resolve_artifact(name: str) -> str:
    if name in ARTIFACT_ALIASES:
        return ARTIFACT_ALIASES[name]
    if name in artifacts.ALL_ARTIFACTS:
        return name
    raise TraceRefusal(
        EXIT_UNKNOWN_ARTIFACT, "UNKNOWN_ARTIFACT",
        f"unknown artifact {name!r}",
        {"known": sorted(ARTIFACT_ALIASES) + list(artifacts.ALL_ARTIFACTS)})


def _exclusion_for(prov: pd.DataFrame, trial: str | None, pid: str) -> list[dict]:
    m = ((prov["extraction_method"] == "row_excluded")
         & (prov["cimac_part_id"] == pid))
    if trial:
        m &= prov["trial"] == trial
    out = []
    for _, r in prov[m].iterrows():
        d = json.loads(r["derivation"]) if r["derivation"] else {}
        out.append({"trial": r["trial"], "rule_id": r["rule_id"],
                    "reason": d.get("reason", ""),
                    "absent_from": d.get("absent_from", []),
                    "present_in": d.get("present_in", []),
                    "vacuous": d.get("vacuous")})
    return out


def find_row(h: pd.DataFrame, args, prov: pd.DataFrame | None = None) -> pd.Series:
    """Locate exactly one delivered row, or refuse."""
    m = pd.Series(True, index=h.index)
    if args.trial:
        m &= h["trial"].astype(str) == args.trial
    m &= h["cimac_part_id"].astype(str) == args.participant_id
    if args.cimac_id:
        m &= h["Cimac.id"].astype(str) == args.cimac_id
    if args.collection_event:
        m &= h["Collection_Event"].astype(str) == args.collection_event

    hits = h[m]
    if hits.empty:
        # An absent row is not necessarily an unknown one. If an exclusion event
        # explains the absence, say so — that IS the trace for an excluded
        # record, and it is the difference between "no such patient" and
        # "deliberately removed, here is why".
        detail = {"participant_id": args.participant_id, "trial": args.trial}
        if prov is not None:
            ex = _exclusion_for(prov, args.trial, args.participant_id)
            if ex:
                raise TraceRefusal(
                    EXIT_NOT_FOUND, "ROW_EXCLUDED",
                    "the row is absent because the participant was excluded",
                    {**detail, "exclusion_events": ex})
        raise TraceRefusal(EXIT_NOT_FOUND, "NOT_FOUND",
                           "no delivered row matches that anchor", detail)
    if len(hits) > 1:
        cands = [{c: str(r[c]) for c in ANCHOR_COLS} for _, r in hits.head(10).iterrows()]
        raise TraceRefusal(
            EXIT_AMBIGUOUS_ROW, "AMBIGUOUS_ROW",
            f"{len(hits)} delivered rows match; specify --cimac-id and/or "
            "--collection-event to identify one",
            {"candidates": cands, "n_matches": int(len(hits))})
    return hits.iloc[0]


def authoritative_event(prov: pd.DataFrame, row: pd.Series, field: str,
                        artifact: str) -> pd.Series:
    """The one provenance row that speaks for this cell in this artifact."""
    m = ((prov["trial"] == str(row["trial"]))
         & (prov["cimac_part_id"] == str(row["cimac_part_id"]))
         & (prov["Cimac.id"] == str(row["Cimac.id"]))
         & (prov["Collection_Event"] == str(row["Collection_Event"]))
         & (prov["harmonized_field"] == field))
    cell_rows = prov[m]
    if cell_rows.empty:
        raise TraceRefusal(EXIT_NOT_FOUND, "NOT_FOUND",
                           f"no provenance rows for field {field!r} on this anchor")

    claiming = cell_rows[cell_rows["authoritative_for"].map(
        lambda v: artifacts.is_authoritative_for(v, artifact))]
    if claiming.empty:
        # Every candidate was demoted: the value is absent by design. Explain why.
        states = sorted({r["value_state"] for _, r in cell_rows.iterrows() if r["value_state"]})
        raise TraceRefusal(
            EXIT_NOT_FOUND, "NOT_FOUND",
            f"no provenance row is authoritative for {field!r} in {artifact}",
            {"value_states_seen": states,
             "hint": ("the value may be suppressed or the row excluded; "
                      "trace the __row_exclusion__ field for this participant")})

    if len(claiming) > 1:
        # Ordinary candidate competition resolves by confidence; a tie with
        # disagreeing values does not resolve at all.
        c = claiming.copy()
        c["_c"] = pd.to_numeric(c["confidence"], errors="coerce").fillna(-1)
        top = c[c["_c"] == c["_c"].max()]
        if len(top) > 1 and len({_norm(v) for v in top["value"]}) > 1:
            raise TraceRefusal(
                EXIT_AMBIGUOUS_AUTH, "AMBIGUOUS_AUTHORITY",
                f"{len(top)} provenance events tie for authority with different values",
                {"events": [{"rule_id": r["rule_id"], "value": r["value"],
                             "method": r["extraction_method"]} for _, r in top.iterrows()]})
        return top.sort_values("_c", ascending=False).iloc[0]
    return claiming.iloc[0]


def verify_source(root: Path, ev: pd.Series) -> dict:
    """Re-read the recorded location and confirm the recorded raw value.

    Returns a verification record. Raises on any condition that makes the
    recorded lineage unverifiable — never falls back to another file.
    """
    path = ev["source_resolved_path"]
    if not path:
        # Not a file-backed value (config constant, template anchor, derivation).
        return {"kind": "not_file_backed", "verified": True,
                "note": "value does not come from a source file"}

    try:
        abs_path = assert_within_roots(root / path, root)
    except PathContainmentError as e:
        raise TraceRefusal(EXIT_UNRESOLVABLE, "UNRESOLVABLE",
                           "recorded source path escapes approved input roots",
                           {"path": path, "error": str(e)[:200]})
    if not abs_path.is_file():
        raise TraceRefusal(
            EXIT_UNRESOLVABLE, "UNRESOLVABLE",
            "recorded source file does not exist",
            {"path": path,
             "note": "no basename fallback is attempted, by design"})

    recorded_hash = ev["source_sha256"]
    actual_hash = sha256_file(abs_path)
    if recorded_hash and actual_hash != recorded_hash:
        raise TraceRefusal(
            EXIT_DRIFTED, "DRIFTED",
            "source file has changed since the run that produced this value",
            {"path": path, "recorded_sha256": recorded_hash,
             "actual_sha256": actual_hash})

    rowno = ev["source_row_number"]
    col = ev["source_column"]
    if rowno == "" or not col:
        return {"kind": "no_cell_locator", "verified": False,
                "path": path, "sha256": actual_hash,
                "note": ("no row/column locator recorded; the value is derived "
                         "or read from a group — see contributions")}

    df = read_source(root, path, ev["source_sheet"])
    rowno = int(rowno)
    if col not in df.columns:
        return {"kind": "descriptive_column", "verified": False,
                "path": path, "sha256": actual_hash, "column": col,
                "note": ("source_column describes a derivation rather than naming "
                         "a column; inputs are in contributions")}
    if rowno >= len(df):
        raise TraceRefusal(EXIT_INCOMPLETE, "INCOMPLETE",
                           "recorded row number is out of range for the source",
                           {"path": path, "row": rowno, "n_rows": int(len(df))})

    actual_raw = df.iloc[rowno][col]
    if _norm(actual_raw) != _norm(ev["raw_value"]):
        raise TraceRefusal(
            EXIT_MISMATCH, "MISMATCH",
            "recorded raw value could not be reproduced from the source",
            {"path": path, "row": rowno, "column": col,
             "recorded": ev["raw_value"], "source": str(actual_raw)})

    # The record key must also still describe the row it points at.
    key_ok = True
    for kcol, kval in parse_record_key(ev["source_record_key"]).items():
        if kcol in df.columns and str(df.iloc[rowno][kcol]).strip() != kval:
            key_ok = False
    if not key_ok:
        raise TraceRefusal(
            EXIT_MISMATCH, "MISMATCH",
            "recorded row no longer holds the recorded record key",
            {"path": path, "row": rowno, "record_key": ev["source_record_key"]})

    return {"kind": "cell", "verified": True, "path": path, "sha256": actual_hash,
            "sheet": ev["source_sheet"], "row": rowno, "column": col,
            "raw_value": str(actual_raw), "record_key": ev["source_record_key"]}


def related_events(prov: pd.DataFrame, row: pd.Series, field: str) -> dict:
    """Superseded events, exclusions and suppressions touching this cell."""
    m = ((prov["trial"] == str(row["trial"]))
         & (prov["cimac_part_id"] == str(row["cimac_part_id"])))
    part = prov[m]
    cell = part[(part["Cimac.id"] == str(row["Cimac.id"]))
                & (part["Collection_Event"] == str(row["Collection_Event"]))
                & (part["harmonized_field"] == field)]
    return {
        "superseded": [
            {"rule_id": r["rule_id"], "value": r["value"],
             "authoritative_for": r["authoritative_for"],
             "method": r["extraction_method"]}
            for _, r in cell.iterrows()
            if artifacts.parse(r["authoritative_for"]) == () or
            (r["authoritative_for"] and r["extraction_method"] != "post_pivot_override")
        ],
        "exclusion_events": [
            {"reason": json.loads(r["derivation"]).get("reason", ""),
             "absent_from": json.loads(r["derivation"]).get("absent_from", [])}
            for _, r in part[part["extraction_method"] == "row_excluded"].iterrows()
            if r["derivation"]
        ],
        "suppression_events": [
            {"withheld_value": json.loads(r["derivation"]).get("withheld_value", ""),
             "threshold": json.loads(r["derivation"]).get("threshold")}
            for _, r in cell[cell["extraction_method"] == "value_suppressed"].iterrows()
            if r["derivation"]
        ],
    }


def build_trace(root: Path, args) -> dict:
    artifact = resolve_artifact(args.artifact)
    outdir = root / "harmonization_outputs"

    apath = outdir / artifact
    if not apath.is_file():
        raise TraceRefusal(EXIT_UNKNOWN_ARTIFACT, "UNKNOWN_ARTIFACT",
                           f"artifact file not present: {artifact}")
    h = pd.read_csv(apath, dtype=str, keep_default_na=False)
    if args.field not in h.columns:
        raise TraceRefusal(EXIT_USAGE, "USAGE",
                           f"unknown field {args.field!r}",
                           {"known": [c for c in h.columns if c not in ANCHOR_COLS]})

    prov = pd.read_csv(outdir / "provenance_long.csv", dtype=str, keep_default_na=False)
    row = find_row(h, args, prov)
    delivered = row[args.field]
    ev = authoritative_event(prov, row, args.field, artifact)

    if _norm(ev["value"]) != _norm(delivered):
        raise TraceRefusal(
            EXIT_INCOMPLETE, "INCOMPLETE",
            "authoritative provenance disagrees with the delivered value",
            {"delivered": delivered, "provenance": ev["value"],
             "rule_id": ev["rule_id"]})

    verification = verify_source(root, ev)

    manifest_id = None
    try:
        manifest_id = run_manifest.load(outdir).get("run_id")
    except (OSError, ValueError):
        pass

    derivation = json.loads(ev["derivation"]) if ev["derivation"] else {}
    return {
        "status": "VERIFIED" if verification.get("verified") else "PARTIAL",
        "artifact": artifact,
        "anchor": {c: str(row[c]) for c in ANCHOR_COLS},
        "field": args.field,
        "delivered_value": delivered,
        "value_state": value_states.normalize(ev["value_state"]),
        "rule": {"id": ev["rule_id"], "version": ev["rule_version"],
                 "extraction_method": ev["extraction_method"]},
        "authoritative_for": list(artifacts.parse(ev["authoritative_for"])),
        "source": {
            "logical_label": ev["source_logical_label"] or ev["source_file"],
            "resolved_path": ev["source_resolved_path"],
            "sha256": ev["source_sha256"],
            "sheet": ev["source_sheet"],
            "record_key": ev["source_record_key"],
            "row_number": ev["source_row_number"],
            "column": ev["source_column"],
            "raw_value": ev["raw_value"],
        },
        "verification": verification,
        "contributions": from_json(ev["contributions"]),
        "derivation": derivation,
        "approval_ref": derivation.get("approval_ref", ""),
        "open_question_ref": derivation.get("open_question_ref", ""),
        "unresolved": derivation.get("unresolved", ""),
        "related": related_events(prov, row, args.field),
        "notes": ev["notes"],
        "run_manifest_id": manifest_id,
    }


def render(t: dict) -> str:
    L: list[str] = []
    a = t["anchor"]
    L.append("─" * 74)
    L.append(f"TRACE  {a['trial']} | {a['cimac_part_id']} | {a['Cimac.id']} | "
             f"{a['Collection_Event']} | {t['field']}")
    L.append("─" * 74)
    L.append(f"DELIVERED   {t['artifact']}")
    L.append(f"            value: {t['delivered_value']!r}   state: {t['value_state']}")
    v = t["verification"]
    mark = "✓" if v.get("verified") else "!"
    L.append(f"STATUS      {t['status']}  {mark} {v.get('note', 'source re-read and consistent')}")
    L.append("")
    L.append(f"RULE        {t['rule']['id']}  v{t['rule']['version']}  "
             f"({t['rule']['extraction_method']})")
    L.append(f"            authoritative for: {', '.join(t['authoritative_for']) or '(none)'}")
    if t["approval_ref"]:
        L.append(f"            approval: {t['approval_ref']}")
    if t["open_question_ref"]:
        L.append(f"            OPEN QUESTION: {t['open_question_ref']}")
        if t["unresolved"]:
            for line in _wrap(t["unresolved"], 58):
                L.append(f"              {line}")
    L.append("")

    s = t["source"]
    if s["resolved_path"]:
        L.append(f"SOURCE      {s['resolved_path']}")
        L.append(f"            sha256 {s['sha256'][:16]}…"
                 + (f"   sheet {s['sheet']}" if s["sheet"] else ""))
        if s["record_key"]:
            L.append(f"            record  {s['record_key']}")
        if s["column"]:
            L.append(f"            column  {s['column']}"
                     + (f"   row {s['row_number']}" if s["row_number"] != "" else ""))
        if s["raw_value"] != "":
            L.append(f"            raw     {s['raw_value']!r}")
    else:
        L.append(f"SOURCE      {s['logical_label']}  (not a source file)")
        if s["record_key"]:
            L.append(f"            record  {s['record_key']}")
    L.append("")

    if t["contributions"]:
        L.append("CONTRIBUTIONS")
        for c in sorted(t["contributions"], key=lambda x: x.get("rank", 0)):
            col = c.get("source_column") or "(file membership)"
            L.append(f"   [{c['status']:<13}] {col:<34} {str(c.get('raw_value', ''))[:18]!r}")
            L.append(f"                    role={c.get('role', '')}"
                     + (f"  file={Path(c['source_resolved_path']).name[:44]}"
                        if c.get("source_resolved_path") else "")
                     + (f"  ref={c['provenance_ref']}" if c.get("provenance_ref") else ""))
        L.append("")

    d = t["derivation"]
    if d:
        L.append(f"DERIVATION  kind={d.get('kind', '')}")
        for k in ("predicate", "branch", "landmark_days", "final_value", "matched",
                  "days_to_crossover", "randomized_arm", "initial_treatment",
                  "crossover_treatment", "n_source_rows", "selection",
                  "anchor_as_stated_by_source_column", "source_file_role",
                  "withheld_value", "threshold", "reason", "absent_from",
                  "present_in", "vacuous", "pending"):
            if k in d and d[k] not in (None, "", [], {}):
                L.append(f"            {k}: {d[k]}")
        if d.get("supersedes"):
            L.append(f"            supersedes: {d['supersedes']}")
        L.append("")

    rel = t["related"]
    if rel["exclusion_events"]:
        L.append("EXCLUSIONS")
        for e in rel["exclusion_events"]:
            L.append(f"   absent from {', '.join(e['absent_from'])}: {e['reason'][:60]}")
        L.append("")
    if rel["suppression_events"]:
        L.append("SUPPRESSION")
        for e in rel["suppression_events"]:
            L.append(f"   withheld {e['withheld_value']!r} (threshold {e['threshold']})")
        L.append("")

    L.append(f"RUN         {t['run_manifest_id'] or '(no manifest)'}")
    return "\n".join(L)


def _wrap(text: str, width: int) -> list[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", default="harmonized_11trials",
                    help="harmonized_11trials | harmonized_9trials_gold | "
                         "harmonized_9trials_reproduced")
    ap.add_argument("--trial")
    ap.add_argument("--participant-id", required=True)
    ap.add_argument("--cimac-id", help="sample id, when the participant has several")
    ap.add_argument("--collection-event", help="timepoint, when the participant has several")
    ap.add_argument("--field", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        trace = build_trace(PROJECT_ROOT, args)
    except TraceRefusal as r:
        payload = {"status": r.status, "error": r.message, **r.detail}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"REFUSED [{r.status}] {r.message}", file=sys.stderr)
            for k, v in r.detail.items():
                print(f"   {k}: {json.dumps(v)[:600]}", file=sys.stderr)
        return r.code

    print(json.dumps(trace, indent=2, sort_keys=True) if args.json else render(trace))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
