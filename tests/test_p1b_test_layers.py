"""Guards for the portable test and publication-safety boundaries."""
from __future__ import annotations

import ast
from pathlib import Path


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
