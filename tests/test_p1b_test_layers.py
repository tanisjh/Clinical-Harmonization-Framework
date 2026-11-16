"""Guards for the portable test and publication-safety boundaries."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


def _decorator_names(function: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for decorator in function.decorator_list:
        node = decorator.func if isinstance(decorator, ast.Call) else decorator
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        names.add(".".join(reversed(parts)))
    return names


def test_default_pytest_selection_is_portable(repo_root: Path):
    config = (repo_root / "pytest.ini").read_text()
    assert 'not production_integration and not corpus_acceptance' in config
    assert "production_integration:" in config
    assert "corpus_acceptance:" in config


def test_test_sources_do_not_embed_private_cluster_roots(repo_root: Path):
    # Assemble the sentinel patterns so this guard does not itself embed one.
    forbidden = ("/" + "vf/users/", "/" + "data/nextgen2/")
    offenders = []
    for path in sorted((repo_root / "tests").glob("*.py")):
        text = path.read_text()
        if any(token in text for token in forbidden):
            offenders.append(path.name)
    assert not offenders, f"private cluster roots in test sources: {offenders}"


def test_production_fixtures_are_used_only_by_marked_tests(repo_root: Path):
    fixture_names = {
        "production_outputs_dir",
        "production_source_root",
        "production_expectations",
        "legacy_writer",
    }
    offenders = []
    for path in sorted((repo_root / "tests").glob("test_*.py")):
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        module_marked = "pytestmark = pytest.mark.production_integration" in text
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            arguments = {arg.arg for arg in node.args.args}
            if arguments & fixture_names:
                marked = "pytest.mark.production_integration" in _decorator_names(node)
                if not (module_marked or marked):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"production fixtures used by default tests: {offenders}"


def test_publication_ignore_rules_cover_generated_private_surfaces(repo_root: Path):
    rules = set((repo_root / ".gitignore").read_text().splitlines())
    assert {".scratch/", "harmonization_outputs/", "*-clinical/"} <= rules
    assert {".venv/", "__pycache__/", ".pytest_cache/"} <= rules


def test_p1a_trace_dependency_remains_registry_independent(repo_root: Path):
    tree = ast.parse((repo_root / "scripts" / "trace_value.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "lib.record_keys" in imported
    assert "lib.trace_io" in imported
    assert "lib.migrated_trials" not in imported


def test_runtime_has_no_retired_registry_and_keeps_neutral_helper_owners(repo_root: Path):
    assert not (repo_root / "scripts/lib/migrated_trials.py").exists()
    imports = []
    owners = {"parse_record_key": set(), "read_source": set()}
    for path in sorted((repo_root / "scripts").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            else:
                names = []
            if any("migrated_trials" in name.split(".") for name in names):
                imports.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in owners:
                owners[node.name].add(str(path.relative_to(repo_root)))
    assert not imports, f"runtime imports retired registry: {imports}"
    assert owners == {
        "parse_record_key": {"scripts/lib/record_keys.py"},
        "read_source": {"scripts/lib/trace_io.py"},
    }


def test_tools_and_execution_import_without_retired_registry_in_fresh_process(repo_root: Path):
    code = '''
import importlib.abc
import sys
class BlockRegistry(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if "migrated_trials" in fullname.split("."):
            raise AssertionError("runtime imported retired registry: " + fullname)
sys.meta_path.insert(0, BlockRegistry())
sys.path.insert(0, "scripts")
from pathlib import Path
import pandas as pd
import harmonize_spec
import trace_value
import view_provenance
from lib import replay
from lib.record_keys import parse_record_key
from lib.trace_io import read_source
from lib.spec_executor import SpecExecutor
from lib.spec_model import load
assert trace_value.parse_record_key is parse_record_key
assert trace_value.read_source is read_source
assert parse_record_key.__module__ == "lib.record_keys"
assert read_source.__module__ == "lib.trace_io"
spec = load("tests/fixtures/synthetic_replay/spec.yaml", require_executable=True)
row = {"device_id": "DEVICE-A", "reading_id": "READ-001", "location": "bench-west",
       "cimac_part_id": "DEVICE-A", "Cimac.id": "READ-001", "Collection_Event": "bench-west"}
executor = SpecExecutor(spec, Path.cwd(), pd.DataFrame([row]), run_id="registry-independent")
assert len(list(executor.extract())) == 4
assert replay.recompute_direct("001", coerce="none") == "001"
assert not any("migrated_trials" in name.split(".") for name in sys.modules)
'''
    subprocess.run([sys.executable, "-B", "-c", code], cwd=repo_root, check=True)
