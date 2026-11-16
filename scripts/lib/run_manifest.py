"""run_manifest.py — Machine-readable record of one pipeline execution.

A manifest answers "what exactly produced these outputs?" without requiring the
reader to reconstruct it. It pins the code, the configuration, every input file
actually opened, and every output written, each by content hash.

The critical design point is that **inputs come from the reader instrumentation**
(`SourceSession.source_refs()`), not from a directory scan. Phase 1a found that
`inspect_trial_files.py` globbed `*-clinical/` and therefore hashed 311 legacy
files and zero of the CIDC files the pipeline actually read. A manifest built the
same way would be confidently wrong in exactly the same manner. Here, a file
appears because a reader opened it.

Schema (v2) — see docs for the field-by-field contract:

    schema_version, run_id, timestamp_utc, git{commit,dirty,branch},
    command, python{version,implementation}, dependencies{...},
    config{path,sha256}, inputs[{trial,config_key,logical_label,resolved_path,
    sha256,file_type,sheet,reader_id,n_rows}], outputs[{name,sha256,bytes,
    n_rows}], row_counts{...}, flagged_values, validation_verdict,
    provenance_gate_verdict, rule_versions{rule_id: version},
    target_schema{id,version,sha256}, schema_validation_verdict,
    schema_validation_counts{error,warning,info}

v2 (Phase 2a) adds the last three keys. A manifest previously pinned the code,
the config and every input by hash but could not say which *target contract* the
outputs were validated against — so two runs producing different columns could
both read as `validation_verdict: pass`. `target_schema.sha256` closes that: a
changed schema is as visible as a changed input.

Deliberately excluded: absolute paths, usernames, hostnames, environment
variables. Paths are repo-relative so a manifest is comparable across machines,
and nothing here should ever carry a credential.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .source_ref import sha256_file

SCHEMA_VERSION = 2
MANIFEST_NAME = "run_manifest.json"

#: Placeholder recorded by the orchestrator. The schema-validation step, which
#: runs later in the pipeline, stamps the real values — the same pattern the
#: existing verdicts already use.
NOT_RUN = "not_run"


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def git_state(root: Path) -> dict:
    """Commit, branch and whether the working tree had uncommitted changes.

    `dirty` matters more than the commit: a manifest from a dirty tree does not
    describe a reproducible state, and saying so is the point.
    """
    status = _git(root, "status", "--porcelain")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "n_dirty_paths": len([ln for ln in status.splitlines() if ln.strip()]),
    }


def dependency_snapshot(root: Path) -> dict:
    """Declared dependencies plus the resolved versions of those that matter.

    A full `pip freeze` would bind the manifest to one machine's environment;
    the declared ranges plus the resolved versions of the libraries that can
    change results is the reproducibility-relevant subset.
    """
    snapshot: dict = {}
    for name in ("requirements.txt", "requirements-dev.txt"):
        p = root / name
        if p.is_file():
            snapshot[name] = {"sha256": sha256_file(p),
                              "declared": [ln.strip() for ln in p.read_text().splitlines()
                                           if ln.strip() and not ln.startswith("#")]}
    resolved: dict[str, str] = {}
    for mod in ("pandas", "numpy", "yaml", "openpyxl"):
        try:
            m = __import__(mod)
            resolved[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            resolved[mod] = "absent"
    snapshot["resolved"] = resolved
    return snapshot


def build(root: Path, *, run_id: str, command: list[str], config_path: Path,
          source_refs: list, outputs: list[str], outputs_dir: Path,
          row_counts: dict, flagged_values: int,
          rule_versions: dict | None = None,
          validation_verdict: str = "not_run",
          provenance_gate_verdict: str = "not_run") -> dict:
    """Assemble a manifest. Pure: does no I/O beyond hashing named files."""
    inputs = []
    for trial, ref in source_refs:
        inputs.append({
            "trial": trial,
            "config_key": ref.config_key,
            "logical_label": ref.logical_label,
            "resolved_path": ref.resolved_path,
            "sha256": ref.sha256,
            "file_type": ref.file_type,
            "sheet": ref.sheet or "",
            "reader_id": ref.reader_id,
            "reader_flags": ref.reader_flags,
            "n_rows": ref.n_rows,
        })
    inputs.sort(key=lambda d: (d["trial"], d["config_key"]))

    out_records = []
    for name in outputs:
        p = outputs_dir / name
        if not p.is_file():
            continue
        rec = {"name": name, "sha256": sha256_file(p), "bytes": p.stat().st_size}
        if name in row_counts:
            rec["n_rows"] = row_counts[name]
        out_records.append(rec)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_state(root),
        "command": command,
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "dependencies": dependency_snapshot(root),
        "config": {"path": str(config_path.relative_to(root)),
                   "sha256": sha256_file(config_path)},
        "inputs": inputs,
        "n_inputs": len(inputs),
        "outputs": out_records,
        "row_counts": row_counts,
        "flagged_values": int(flagged_values),
        "validation_verdict": validation_verdict,
        "provenance_gate_verdict": provenance_gate_verdict,
        "rule_versions": dict(sorted((rule_versions or {}).items())),
        "target_schema": {"id": "", "version": "", "sha256": ""},
        "schema_validation_verdict": NOT_RUN,
        "schema_validation_counts": {"error": 0, "warning": 0, "info": 0},
    }


def stamp_schema_validation(outputs_dir: Path, *, schema_id: str, schema_version: str,
                            schema_sha256: str, verdict: str,
                            counts: dict) -> Path | None:
    """Record which target schema validated this run, and the outcome.

    Loads and rewrites rather than reconstructing, so verdicts stamped by other
    pipeline steps survive. Returns None when there is no manifest to stamp,
    which is the normal case for a standalone validation run.
    """
    path = outputs_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    manifest = load(outputs_dir)
    manifest["target_schema"] = {"id": schema_id, "version": schema_version,
                                 "sha256": schema_sha256}
    manifest["schema_validation_verdict"] = verdict
    manifest["schema_validation_counts"] = {
        "error": int(counts.get("error", 0)),
        "warning": int(counts.get("warning", 0)),
        "info": int(counts.get("info", 0)),
    }
    return write(manifest, outputs_dir)


def write(manifest: dict, outputs_dir: Path) -> Path:
    path = outputs_dir / MANIFEST_NAME
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load(outputs_dir: Path) -> dict:
    with (outputs_dir / MANIFEST_NAME).open() as f:
        return json.load(f)


def make_run_id(git_commit: str) -> str:
    """Timestamp plus short commit — sortable, and it names the code that ran."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{(git_commit or 'nogit')[:8]}"
