"""The writer's output contract comes from the target schema, and nowhere else.

Phase 2a left two definitions of the target: the schema, and `TEMPLATE_COLUMNS`
in the orchestrator. A test pinned them equal, but two sources of truth held in
step by a test are still two sources of truth — the next person to add a column
has to know to change both. Phase 2b deletes the list.

These tests prove the remaining definition is load-bearing: change the schema and
the writer's contract changes with it, in the direction you would expect.

Run:  python -m pytest tests/test_schema_driven_writer.py -v
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pandas as pd
import pytest

from lib.provenance import Cell
from lib.schema_model import contract_for, load, parse

INSTANCE = "schemas/cimac_clinical_v1.yaml"
ARTIFACTS = ("harmonized_11trials.csv", "harmonized_9trials_gold.csv",
             "harmonized_9trials_reproduced.csv")


@pytest.fixture(scope="module")
def production_schema(project_root):
    return load(project_root / INSTANCE)


@pytest.fixture
def legacy_writer():
    """Compatibility writer, absent from the portable engine checkout."""
    try:
        return importlib.import_module("extract_harmonized_clinical")
    except ModuleNotFoundError:
        pytest.skip("legacy production writer module is not present")


def toy_schema(names, **overrides):
    doc = {
        "schema_id": "test/writer",
        "schema_version": "1.0.0",
        "column_order": list(names),
        "fields": [{"name": n} for n in names],
    }
    doc.update(overrides)
    return parse(doc)


def one_cell(field: str, value: str) -> Cell:
    return Cell(trial="SYNTHETIC-TRIAL",
                cimac_part_id="SYNTH-PARTICIPANT-001",
                Cimac_id="SYNTH-SAMPLE-001",
                Collection_Event="VISIT-A", harmonized_field=field, value=value,
                confidence=1.0, source_file="f.csv", source_column="c",
                source_row_idx=0, extraction_method="direct")


def pivot(writer, schema, cells, artifact=""):
    wide, _, _ = writer.pivot_cells_to_wide(
        cells, {}, contract_for(schema, artifact))
    return wide


# ── the writer has no column list of its own ────────────────────────────────

@pytest.mark.production_integration
def test_template_columns_constant_is_gone(legacy_writer):
    assert not hasattr(legacy_writer, "TEMPLATE_COLUMNS"), (
        "TEMPLATE_COLUMNS still exists; the target is defined twice again")


@pytest.mark.production_integration
def test_orchestrator_source_declares_no_target_column_list(
        project_root, legacy_writer):
    """A literal list of target column names anywhere in the writer is a relapse.

    Parsed rather than grepped, so a mention in a comment or docstring does not
    trip it — only a real list of string literals matching target field names.
    """
    schema = load(project_root / INSTANCE)
    target_names = {f.name for f in schema.fields}
    src = Path(legacy_writer.__file__).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            literals = {e.value for e in node.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            overlap = literals & target_names
            assert len(overlap) < 5, (
                f"line {node.lineno}: a literal collection names {len(overlap)} "
                f"target fields ({sorted(overlap)[:6]}) — the column contract "
                f"must come from the schema")


@pytest.mark.production_integration
def test_pivot_requires_a_contract(legacy_writer):
    """The signature itself forces a caller to supply the target shape."""
    with pytest.raises(TypeError):
        legacy_writer.pivot_cells_to_wide([], {})


# ── contract derivation ─────────────────────────────────────────────────────

def test_contract_uses_schema_column_order():
    schema = toy_schema(["gamma", "alpha", "beta"])
    contract = contract_for(schema)
    assert list(contract.columns) == list(schema.column_order)
    assert contract.schema_ref == "test/writer@1.0.0"


def test_contract_falls_back_to_declaration_order():
    schema = parse({"schema_id": "t/x", "schema_version": "1.0.0",
                    "fields": [{"name": "b"}, {"name": "a"}]})
    assert list(contract_for(schema).columns) == ["b", "a"]


def test_contract_required_reflects_required_column():
    schema = parse({"schema_id": "t/x", "schema_version": "1.0.0",
                    "fields": [{"name": "a"},
                               {"name": "b", "required_column": False}]})
    contract = contract_for(schema)
    assert contract.columns == ("a", "b")
    assert contract.required == ("a",)


def test_contract_respects_artifact_applicability():
    schema = parse({"schema_id": "t/x", "schema_version": "1.0.0",
                    "column_order": ["a", "b"],
                    "fields": [{"name": "a"},
                               {"name": "b", "artifacts": ["other.csv"]}]})
    assert contract_for(schema, "mine.csv").columns == ("a",)
    assert contract_for(schema, "other.csv").columns == ("a", "b")


# ── schema changes reach the writer ─────────────────────────────────────────

@pytest.mark.production_integration
def test_writer_emits_columns_in_schema_order(legacy_writer):
    schema = toy_schema(["gamma", "alpha", "beta"])
    wide = pivot(legacy_writer, schema, [one_cell("alpha", "1")])
    assert list(wide.columns) == ["gamma", "alpha", "beta"]


@pytest.mark.production_integration
def test_reordering_the_schema_reorders_the_output(legacy_writer):
    cells = [one_cell("alpha", "1")]
    first = pivot(legacy_writer, toy_schema(["alpha", "beta"]), cells)
    second = pivot(legacy_writer, toy_schema(["beta", "alpha"]), cells)
    assert list(first.columns) == ["alpha", "beta"]
    assert list(second.columns) == ["beta", "alpha"]


@pytest.mark.production_integration
def test_adding_a_schema_field_adds_an_empty_column(legacy_writer):
    wide = pivot(
        legacy_writer, toy_schema(["alpha", "beta", "delta"]),
        [one_cell("alpha", "1")])
    assert "delta" in wide.columns
    assert wide["delta"].isna().all()


@pytest.mark.production_integration
def test_removing_a_schema_field_drops_the_column_even_if_extracted(legacy_writer):
    """A value with no declared field must not leak into the delivered table."""
    cells = [one_cell("alpha", "1"), one_cell("beta", "2")]
    wide = pivot(legacy_writer, toy_schema(["alpha"]), cells)
    assert list(wide.columns) == ["alpha"]
    assert "beta" not in wide.columns


@pytest.mark.production_integration
def test_writer_honours_artifact_scoped_fields(legacy_writer):
    schema = parse({"schema_id": "t/x", "schema_version": "1.0.0",
                    "column_order": ["a", "b"],
                    "fields": [{"name": "a"},
                               {"name": "b", "artifacts": ["only_here.csv"]}]})
    assert list(pivot(
        legacy_writer, schema, [one_cell("a", "1")],
        "elsewhere.csv").columns) == ["a"]
    assert list(pivot(
        legacy_writer, schema, [one_cell("a", "1")],
        "only_here.csv").columns) == ["a", "b"]


# ── the delivered artifacts still match the contract ────────────────────────

@pytest.mark.production_integration
@pytest.mark.parametrize("name", ARTIFACTS)
def test_delivered_column_order_matches_the_contract(
        production_schema, production_outputs_dir, name):
    delivered = list(pd.read_csv(production_outputs_dir / name, nrows=0).columns)
    assert delivered == list(contract_for(production_schema).columns)


@pytest.mark.production_integration
@pytest.mark.parametrize("name", ARTIFACTS)
def test_delivered_column_order_is_the_historical_one(
        production_outputs_dir, name):
    """Independent of the schema: the exact 19 columns Phase 1 delivered.

    Written out so a schema edit that silently reshaped the deliverable would
    fail here rather than agreeing with itself.
    """
    expected = ["cimac_part_id", "Cimac.id", "Collection_Event", "race", "sex",
                "age", "arm", "os_time", "os_stat", "pfs_time", "pfs_stat",
                "clinical_benefit", "treatment", "phase", "trial",
                "clinical_benefit.binary", "Collection_Event_alt", "pfs_bin",
                "bor_bin"]
    assert list(pd.read_csv(
        production_outputs_dir / name, nrows=0).columns) == expected


# ── schema location is configuration, not code ──────────────────────────────

@pytest.mark.production_integration
def test_schema_path_comes_from_config(
        project_root, config_path, legacy_writer):
    import yaml
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["target_schema"]["path"] == INSTANCE
    resolved = legacy_writer.resolve_target_schema_path(cfg, None, project_root)
    assert resolved == project_root / INSTANCE


@pytest.mark.production_integration
def test_cli_override_wins_over_config(project_root, tmp_path, legacy_writer):
    cfg = {"target_schema": {"path": "schemas/cimac_clinical_v1.yaml"}}
    other = tmp_path / "elsewhere.yaml"
    assert legacy_writer.resolve_target_schema_path(
        cfg, str(other), project_root) == other


@pytest.mark.production_integration
def test_missing_config_key_falls_back_to_the_default(project_root, legacy_writer):
    resolved = legacy_writer.resolve_target_schema_path({}, None, project_root)
    assert resolved == project_root / legacy_writer.DEFAULT_TARGET_SCHEMA


@pytest.mark.production_integration
def test_relative_paths_resolve_against_the_project_root(
        project_root, legacy_writer):
    resolved = legacy_writer.resolve_target_schema_path(
        {"target_schema": {"path": "schemas/other.yaml"}}, None, project_root)
    assert resolved == project_root / "schemas" / "other.yaml"
