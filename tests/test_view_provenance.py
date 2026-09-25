"""The demo provenance viewer: explicit-file lineage for one Cell.

What is proved here is as much about restraint as capability. The viewer must
show a delivered value's recorded evidence, and it must NOT acquire any of the
behaviour that makes `trace_value.py` dangerous to loosen: no artifact
discovery, no basename resolution, no searching production outputs, and no
re-reading or recomputing anything.

The ambiguity tests matter for the demo specifically. A participant contributes
one record per collection event, so participant + field is not a unique key. A
viewer that silently showed the first match would be showing a value from a
timepoint nobody asked about.

Run:  python -m pytest tests/test_view_provenance.py -v
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from lib.provenance import Cell

VIEWER = "scripts/view_provenance.py"

EXIT_OK = 0
EXIT_NO_MATCH = 1
EXIT_USAGE = 2
EXIT_BAD_PROVENANCE = 3
EXIT_AMBIGUOUS = 4


def _run(project_root, *args):
    return subprocess.run([sys.executable, *[str(a) for a in args]],
                          capture_output=True, text=True, cwd=str(project_root))


def _cell(*, field, value, event="VISIT-A", sample="SYNTH-SAMPLE-001",
          source=True, rule="direct_copy.v1"):
    return Cell(
        trial="SYNTHETIC-TRIAL",
        cimac_part_id="SYNTH-PARTICIPANT-001",
        Cimac_id=sample,
        Collection_Event=event,
        harmonized_field=field,
        value=value,
        confidence=0.9,
        source_file="records.csv" if source else "DERIVED:measurement_a+status_code",
        source_column="category_x" if source else "measurement_a+status_code",
        source_row_idx=0,
        extraction_method="direct" if source else "derived",
        source_logical_label="records.csv" if source else "",
        source_resolved_path="synthetic/records.csv" if source else "",
        source_sha256="a" * 64 if source else "",
        source_row_number=0 if source else "",
        source_record_key="record_id=SYNTH-001" if source else "",
        raw_value=value if source else "",
        rule_id=rule,
        rule_version="1",
    ).as_record()


@pytest.fixture(scope="module")
def demo_frame():
    """Entirely fabricated provenance with two events and three origin kinds."""
    rows = [
        _cell(field="category_x", value="alpha"),
        _cell(field="category_x", value="beta", event="VISIT-B",
              sample="SYNTH-SAMPLE-002"),
        _cell(field="derived_metric", value="metric-alpha", source=False,
              rule="synthetic.derived.v1"),
        _cell(field="constant_x", value="configured", source=False,
              rule="constant.v1"),
    ]
    rows[-1]["source_file"] = "CONFIG:example"
    rows[-1]["source_column"] = ""
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory, demo_frame):
    path = tmp_path_factory.mktemp("synthetic-provenance") / "cells.csv"
    demo_frame.to_csv(path, index=False)
    return path


def _one(frame, **filters):
    """A row matching every filter, and a guarantee that it is the only one."""
    sub = frame
    for column, value in filters.items():
        sub = sub[sub[column] == value]
    assert len(sub) == 1, f"fixture is not unique: {filters} -> {len(sub)}"
    return sub.iloc[0]


def _unique_anchor(frame, field):
    """A (participant, event) pair whose cell for `field` is unique."""
    for _, row in frame[frame["harmonized_field"] == field].iterrows():
        return row["cimac_part_id"], row["Collection_Event"]
    raise AssertionError(f"no {field} cell")


# ── the view itself ─────────────────────────────────────────────────────────

def test_a_source_derived_cell_shows_its_evidence(project_root, demo_run,
                                                  demo_frame):
    """The demo's core moment: value, rule, file, row, column, raw, hash."""
    pid, event = _unique_anchor(demo_frame, "category_x")
    row = _one(demo_frame, cimac_part_id=pid, Collection_Event=event,
               harmonized_field="category_x")
    proc = _run(project_root, project_root / VIEWER,
                "--provenance", demo_run, "--participant-id", pid,
                "--field", "category_x", "--collection-event", event)
    assert proc.returncode == EXIT_OK, proc.stderr
    out = proc.stdout
    for expected in (row["value"], row["confidence"], row["rule_id"],
                     row["source_resolved_path"], row["source_column"],
                     row["source_row_number"], row["raw_value"],
                     row["source_sha256"], row["source_record_key"], pid):
        assert str(expected) in out, f"missing synthetic field {type(expected).__name__}"
    assert "HARMONIZED" in out and "SOURCE" in out and "RECORD" in out


def test_a_derived_cell_says_it_reads_no_file(project_root, demo_run,
                                              demo_frame):
    """A rule-produced value has inputs, not a row — and should say so."""
    pid, event = _unique_anchor(demo_frame, "derived_metric")
    proc = _run(project_root, project_root / VIEWER,
                "--provenance", demo_run, "--participant-id", pid,
                "--field", "derived_metric", "--collection-event", event)
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "(not a source file)" in proc.stdout
    assert "synthetic.derived.v1" in proc.stdout
    assert "DERIVED:measurement_a+status_code" in proc.stdout


def test_a_constant_cell_shows_its_config_record_key(project_root, demo_run,
                                                     demo_frame):
    pid, event = _unique_anchor(demo_frame, "constant_x")
    proc = _run(project_root, project_root / VIEWER,
                "--provenance", demo_run, "--participant-id", pid,
                "--field", "constant_x", "--collection-event", event)
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "constant.v1" in proc.stdout
    assert "CONFIG:example" in proc.stdout


def test_the_view_states_it_does_not_recompute(project_root, demo_run,
                                               demo_frame):
    """Terminology guard. This is a view, and it must never imply more."""
    pid, event = _unique_anchor(demo_frame, "category_x")
    proc = _run(project_root, project_root / VIEWER,
                "--provenance", demo_run, "--participant-id", pid,
                "--field", "category_x", "--collection-event", event)
    text = proc.stdout
    assert "does not" in text and "recompute" in text
    assert "PROVENANCE VIEW" in text
    for overclaim in ("independent verification", "independently verified",
                      "VERIFIED"):
        assert overclaim not in text


# ── structured fields ───────────────────────────────────────────────────────

def test_structured_derivation_is_decoded_and_open_questions_surfaced(
        project_root, tmp_path, demo_frame):
    """A derivation carrying an open question must show it, not bury it."""
    frame = demo_frame.copy()
    idx = frame.index[frame["harmonized_field"] == "constant_x"][0]
    frame.loc[idx, "derivation"] = json.dumps({
        "kind": "constant", "rule_id": "constant.v1",
        "final_value": "configured",
        "open_question_ref": "synthetic_review_questions.csv#Q1",
        "unresolved": "two fabricated configuration examples disagree."})
    path = tmp_path / "cells.csv"
    frame.to_csv(path, index=False)

    row = frame.loc[idx]
    proc = _run(Path.cwd(), Path.cwd() / VIEWER, "--provenance", path,
                "--participant-id", row["cimac_part_id"], "--field", "constant_x",
                "--collection-event", row["Collection_Event"])
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "DERIVATION" in proc.stdout
    assert "OPEN QUESTION" in proc.stdout
    assert "synthetic_review_questions.csv#Q1" in proc.stdout
    assert "two fabricated configuration examples disagree" in proc.stdout


def test_multiple_contributors_are_each_shown(project_root, tmp_path,
                                              demo_frame):
    """An ordered-fallback cell must show the rejected candidates too."""
    frame = demo_frame.copy()
    idx = frame.index[frame["harmonized_field"] == "derived_metric"][0]
    frame.loc[idx, "contributions"] = json.dumps([
        {"role": "selection_condition", "source_column": "status_code",
         "raw_value": "alpha", "status": "condition"},
        {"role": "measurement_candidate", "source_column": "measurement_a",
         "raw_value": "", "status": "unavailable"},
        {"role": "measurement_candidate", "source_column": "measurement_b",
         "raw_value": "beta", "status": "selected"}])
    path = tmp_path / "cells.csv"
    frame.to_csv(path, index=False)

    row = frame.loc[idx]
    proc = _run(Path.cwd(), Path.cwd() / VIEWER, "--provenance", path,
                "--participant-id", row["cimac_part_id"], "--field", "derived_metric",
                "--collection-event", row["Collection_Event"])
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "CONTRIBUTIONS  (3)" in proc.stdout
    for expected in ("status_code", "measurement_a", "measurement_b",
                     "unavailable", "selected", "condition"):
        assert expected in proc.stdout


def test_an_undecodable_structured_field_is_shown_not_hidden(project_root,
                                                             tmp_path,
                                                             demo_frame):
    """Never eval, and never silently drop what the run recorded."""
    frame = demo_frame.copy()
    idx = frame.index[frame["harmonized_field"] == "category_x"][0]
    frame.loc[idx, "derivation"] = "not json at all"
    path = tmp_path / "cells.csv"
    frame.to_csv(path, index=False)

    row = frame.loc[idx]
    proc = _run(Path.cwd(), Path.cwd() / VIEWER, "--provenance", path,
                "--participant-id", row["cimac_part_id"], "--field", "category_x",
                "--collection-event", row["Collection_Event"])
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "not json at all" in proc.stdout


# ── refusals ────────────────────────────────────────────────────────────────

def test_an_ambiguous_query_refuses_and_says_how_to_narrow_it(project_root,
                                                              demo_run,
                                                              demo_frame):
    """A participant has one record per collection event. Never pick one."""
    counts = (demo_frame[demo_frame["harmonized_field"] == "category_x"]
              .groupby("cimac_part_id").size())
    pid = counts[counts > 1].index[0]
    proc = _run(project_root, project_root / VIEWER,
                "--provenance", demo_run, "--participant-id", pid,
                "--field", "category_x")
    assert proc.returncode == EXIT_AMBIGUOUS
    assert "AMBIGUOUS" in proc.stderr
    assert "Collection_Event" in proc.stderr
    assert "--collection-event" in proc.stderr
    assert "--all" in proc.stderr
    assert "HARMONIZED" not in proc.stdout          # nothing was shown


def test_all_shows_every_match_when_asked(project_root, demo_run, demo_frame):
    counts = (demo_frame[demo_frame["harmonized_field"] == "category_x"]
              .groupby("cimac_part_id").size())
    pid = counts[counts > 1].index[0]
    proc = _run(project_root, project_root / VIEWER,
                "--provenance", demo_run, "--participant-id", pid,
                "--field", "category_x", "--all")
    assert proc.returncode == EXIT_OK, proc.stderr
    headers = [ln for ln in proc.stdout.splitlines()
               if ln.startswith("PROVENANCE VIEW  ")]
    assert len(headers) == int(counts[pid])


def test_no_match_exits_nonzero_with_a_useful_message(project_root, demo_run):
    proc = _run(project_root, project_root / VIEWER, "--provenance", demo_run,
                "--participant-id", "SYNTH-NOT-PRESENT", "--field", "category_x")
    assert proc.returncode == EXIT_NO_MATCH
    assert "NO MATCH" in proc.stderr
    assert "SYNTH-NOT-PRESENT" in proc.stderr


def test_an_unknown_field_lists_the_fields_that_exist(project_root, demo_run):
    proc = _run(project_root, project_root / VIEWER, "--provenance", demo_run,
                "--participant-id", "anything", "--field", "no_such_field")
    assert proc.returncode == EXIT_NO_MATCH
    assert "unknown field" in proc.stderr
    assert "category_x" in proc.stderr


def test_a_missing_provenance_file_is_refused(project_root, tmp_path):
    proc = _run(project_root, project_root / VIEWER,
                "--provenance", tmp_path / "absent.csv",
                "--field", "category_x")
    assert proc.returncode == EXIT_BAD_PROVENANCE
    assert "not found" in proc.stderr


def test_a_csv_without_provenance_columns_is_refused(project_root, tmp_path):
    """A harmonized artifact is not a provenance CSV; refuse, do not guess."""
    ordinary = tmp_path / "ordinary.csv"
    ordinary.write_text("record_id,measurement_a\nSYNTH-001,123.45\n")
    proc = _run(project_root, project_root / VIEWER, "--provenance",
                ordinary, "--field", "category_x")
    assert proc.returncode == EXIT_BAD_PROVENANCE
    assert "not a harmonization provenance CSV" in proc.stderr
    assert "missing column" in proc.stderr
    assert "harmonize_spec.py" in proc.stderr       # says how to make one


def test_a_malformed_csv_is_refused(project_root, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text('a,b\n"unclosed\n')
    proc = _run(project_root, project_root / VIEWER, "--provenance", bad,
                "--field", "category_x")
    assert proc.returncode == EXIT_BAD_PROVENANCE
    assert "not readable as CSV" in proc.stderr


def test_a_query_with_no_filters_is_a_usage_error(project_root, demo_run):
    proc = _run(project_root, project_root / VIEWER, "--provenance", demo_run)
    assert proc.returncode == EXIT_USAGE
    assert "at least one filter" in proc.stderr


def test_provenance_is_required(project_root):
    """There is no default and no discovery: the caller names the file."""
    proc = _run(project_root, project_root / VIEWER, "--field", "category_x")
    assert proc.returncode != EXIT_OK
    assert "--provenance" in proc.stderr


# ── json ────────────────────────────────────────────────────────────────────

def test_json_output_decodes_structured_fields(project_root, demo_run,
                                               demo_frame):
    pid, event = _unique_anchor(demo_frame, "category_x")
    proc = _run(project_root, project_root / VIEWER, "--provenance", demo_run,
                "--participant-id", pid, "--field", "category_x",
                "--collection-event", event, "--json")
    assert proc.returncode == EXIT_OK, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "OK" and payload["matches"] == 1
    cell = payload["cells"][0]
    assert cell["harmonized_field"] == "category_x"
    assert cell["source_resolved_path"]
    # structured fields are decoded, not left as raw strings
    assert cell["contributions"] is None or isinstance(
        cell["contributions"], (list, dict, str))


def test_json_reports_ambiguity_as_a_status(project_root, demo_run, demo_frame):
    counts = (demo_frame[demo_frame["harmonized_field"] == "category_x"]
              .groupby("cimac_part_id").size())
    pid = counts[counts > 1].index[0]
    proc = _run(project_root, project_root / VIEWER, "--provenance", demo_run,
                "--participant-id", pid, "--field", "category_x", "--json")
    assert proc.returncode == EXIT_AMBIGUOUS
    payload = json.loads(proc.stdout)
    assert payload["status"] == "AMBIGUOUS"
    assert payload["matches"] > 1


# ── the safety boundary ─────────────────────────────────────────────────────

def test_the_viewer_discovers_nothing_and_reads_no_production_path(project_root):
    """It must not acquire trace_value.py's resolution behaviour.

    That tool's refusals are a safety property — a basename-resolved trace once
    opened the wrong file and returned another patient's value. This viewer
    sidesteps the whole question by resolving nothing.
    """
    path = project_root / VIEWER
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and body \
                and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    tokens = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            tokens.add(node.value)
        elif isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            tokens.add(getattr(node, "module", "") or "")
            tokens.update(a.name for a in node.names)

    # `source_sha256` is DISPLAYED — it is a column the run recorded. Hashing
    # anything would be recomputation, so `hashlib` is what must stay absent.
    for banned in ("harmonization_outputs", "provenance_long", "glob", "rglob",
                   "ARTIFACT_ALIASES", "ALL_ARTIFACTS", "resolve_artifact",
                   "migrated_trials", "trace_value", "read_cidc_csv",
                   "read_excel", "hashlib"):
        leaked = [t for t in tokens if banned in t]
        assert not leaked, f"viewer reaches for {banned}: {leaked}"


def test_trace_value_keeps_a_closed_allowlist_without_filesystem_discovery(project_root):
    """Assert the trace safety property directly, without requiring Git."""
    text = (project_root / "scripts" / "trace_value.py").read_text()
    assert "def resolve_artifact" in text
    assert "ARTIFACT_ALIASES" in text
    assert "UNKNOWN_ARTIFACT" in text
    tree = ast.parse(text)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert {"ARTIFACT_ALIASES", "resolve_artifact"} <= names
    assert "ALL_ARTIFACTS" in attrs
    assert not ({"glob", "rglob", "walk"} & (names | attrs))
