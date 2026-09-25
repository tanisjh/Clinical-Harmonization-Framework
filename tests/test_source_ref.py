"""Unit tests for lib/source_ref.py — reader-emitted source identity.

Covers SourceRef construction, path containment (R3), sha256 correctness, and
the explicit absence of basename-fallback resolution (defect D2).

Run:  python -m pytest tests/test_source_ref.py -v
"""
from __future__ import annotations

import hashlib

import pytest

from lib.source_ref import (
    PathContainmentError,
    SourceRef,
    approved_input_roots,
    assert_within_roots,
    sha256_file,
)

BOUND_SOURCE = "inputs/bound/records.csv"
LEGACY_SOURCE = "inputs/legacy/records.csv"
SOURCE_LABEL = "records.csv"


@pytest.fixture
def source_tree(tmp_path):
    """Two same-basename, different-content files under a synthetic root."""
    root = tmp_path / "project"
    bound = root / BOUND_SOURCE
    legacy = root / LEGACY_SOURCE
    bound.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    bound.write_text("record_id,measurement_a\nSYNTH-001,123.45\n")
    legacy.write_text("record_id,measurement_a\nSYNTH-LEGACY,67.0\n")
    return root, bound, legacy


# ── sha256 ──────────────────────────────────────────────────────────────────

def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"a,b\n1,2\n")
    assert sha256_file(p) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()


def test_sha256_file_differs_for_different_content(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert sha256_file(a) != sha256_file(b)


# ── path containment (R3) ───────────────────────────────────────────────────

def test_approved_roots_is_project_root(source_tree):
    root, _, _ = source_tree
    assert approved_input_roots(root) == [root.resolve()]


def test_within_roots_accepts_configured_source(source_tree):
    root, bound, _ = source_tree
    resolved = assert_within_roots(bound, root)
    assert resolved.is_file()
    assert resolved.is_relative_to(root.resolve())


def test_within_roots_rejects_outside_path(source_tree, tmp_path):
    root, _, _ = source_tree
    outside = tmp_path / "outside.csv"
    outside.write_text("record_id\nSYNTH-OUTSIDE\n")
    with pytest.raises(PathContainmentError):
        assert_within_roots(outside, root)


def test_within_roots_rejects_parent_escape(source_tree):
    root, _, _ = source_tree
    with pytest.raises(PathContainmentError):
        assert_within_roots(root / ".." / "outside.csv", root)


def test_within_roots_rejects_generated_outputs(source_tree):
    """harmonization_outputs/ holds generated artifacts, never source inputs."""
    root, _, _ = source_tree
    with pytest.raises(PathContainmentError):
        assert_within_roots(
            root / "harmonization_outputs" / "synthetic_delivery.csv",
            root,
        )


def test_within_roots_rejects_symlink_escape(source_tree, tmp_path):
    """A symlink inside the repo pointing outside it must not be followed."""
    root, _, _ = source_tree
    link = root / "escape.csv"
    outside = tmp_path / "outside.csv"
    outside.write_text("record_id\nSYNTH-OUTSIDE\n")
    try:
        link.symlink_to(outside)
        with pytest.raises(PathContainmentError):
            assert_within_roots(link, root)
    finally:
        if link.is_symlink():
            link.unlink()


# ── SourceRef.build ─────────────────────────────────────────────────────────

def test_build_records_the_file_actually_opened(source_tree):
    root, bound, _ = source_tree
    ref = SourceRef.build(
        config_key="measurements",
        path=bound,
        project_root=root,
        file_type="csv",
        reader_id="cidc_csv@v1",
        logical_label=SOURCE_LABEL,
    )
    assert ref.config_key == "measurements"
    assert ref.logical_label == SOURCE_LABEL
    assert ref.resolved_path == BOUND_SOURCE
    assert ref.basename == SOURCE_LABEL
    assert ref.file_type == "csv"
    assert ref.reader_id == "cidc_csv@v1"
    assert ref.sheet is None
    assert len(ref.sha256) == 64
    assert ref.sha256 == sha256_file(bound)


def test_logical_label_defaults_to_real_basename(source_tree):
    root, bound, _ = source_tree
    ref = SourceRef.build(
        config_key="measurements", path=bound,
        project_root=root, file_type="csv", reader_id="cidc_csv@v1",
    )
    assert ref.logical_label == ref.basename


def test_build_rejects_missing_file(source_tree):
    root, _, _ = source_tree
    with pytest.raises(FileNotFoundError):
        SourceRef.build(
            config_key="missing", path=root / "does_not_exist.csv",
            project_root=root, file_type="csv", reader_id="cidc_csv@v1",
        )


def test_build_rejects_out_of_root(source_tree, tmp_path):
    root, _, _ = source_tree
    outside = tmp_path / "outside.csv"
    outside.write_text("record_id\nSYNTH-OUTSIDE\n")
    with pytest.raises(PathContainmentError):
        SourceRef.build(
            config_key="bad", path=outside, project_root=root,
            file_type="csv", reader_id="cidc_csv@v1",
        )


def test_ref_is_immutable(source_tree):
    root, bound, _ = source_tree
    ref = SourceRef.build(
        config_key="measurements", path=bound,
        project_root=root, file_type="csv", reader_id="cidc_csv@v1",
    )
    with pytest.raises(Exception):
        ref.sha256 = "tampered"  # frozen dataclass


# ── the defect this design exists to prevent ────────────────────────────────

def test_same_basename_files_can_exist_and_differ(source_tree):
    """A stale same-basename path may resolve to a different file.

    This is why a basename-based trace is unsafe and why SourceRef records a
    resolved path plus a hash instead of a filename.
    """
    _, bound, legacy = source_tree
    assert legacy.is_file()
    assert bound.is_file()
    assert legacy.name == bound.name == SOURCE_LABEL
    assert sha256_file(legacy) != sha256_file(bound), (
        "synthetic same-basename files unexpectedly match — the hazard this test "
        "documents would no longer be demonstrable"
    )


def test_verify_detects_hash_drift(source_tree):
    root, bound, _ = source_tree
    ref = SourceRef.build(
        config_key="measurements", path=bound,
        project_root=root, file_type="csv", reader_id="cidc_csv@v1",
    )
    ok, detail = ref.verify(root)
    assert ok, detail

    drifted = SourceRef(
        config_key=ref.config_key, logical_label=ref.logical_label,
        resolved_path=ref.resolved_path, basename=ref.basename,
        sha256="0" * 64, file_type=ref.file_type, input_root=ref.input_root,
        reader_id=ref.reader_id,
    )
    ok, detail = drifted.verify(root)
    assert not ok and "drift" in detail


def test_verify_reports_missing_file(source_tree):
    root, _, _ = source_tree
    ghost = SourceRef(
        config_key="x", logical_label="x.csv", resolved_path="no/such/file.csv",
        basename="file.csv", sha256="0" * 64, file_type="csv",
        input_root="harmonization", reader_id="cidc_csv@v1",
    )
    ok, detail = ghost.verify(root)
    assert not ok and "missing" in detail
