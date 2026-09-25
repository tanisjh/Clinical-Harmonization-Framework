"""Central, portable pytest configuration for the harmonization repository."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    group = parser.getgroup("harmonization test layers")
    group.addoption(
        "--run-production-integration",
        action="store_true",
        default=False,
        help="enable tests that inspect explicitly supplied production artifacts",
    )
    group.addoption(
        "--production-outputs-dir",
        default="",
        metavar="PATH",
        help="external production output directory (never auto-discovered)",
    )
    group.addoption(
        "--production-source-root",
        default="",
        metavar="PATH",
        help="external production source root (never auto-discovered)",
    )
    group.addoption(
        "--production-expectations",
        default="",
        metavar="PATH",
        help="external JSON expectations for production-only assertions",
    )
    group.addoption(
        "--corpus-root",
        default="",
        metavar="PATH",
        help="approved declarative-only corpus root containing specs/schemas/vocabularies",
    )
    group.addoption(
        "--corpus-plan",
        default="",
        metavar="PATH",
        help="explicit declarative identity plan for corpus-acceptance checks",
    )


def pytest_collection_modifyitems(config, items):
    """Keep explicitly selected production tests safe without their opt-in flag."""
    if config.getoption("--run-production-integration"):
        return
    marker = pytest.mark.skip(
        reason="requires --run-production-integration and explicit external paths"
    )
    for item in items:
        if item.get_closest_marker("production_integration"):
            item.add_marker(marker)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Root of this source checkout; never a clinical-data root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def project_root(repo_root: Path) -> Path:
    """Compatibility alias for tests that mean the repository root."""
    return repo_root


@pytest.fixture(scope="session")
def config_path(tmp_path_factory) -> Path:
    """Small synthetic config used by portable configuration tests."""
    path = tmp_path_factory.mktemp("portable-config") / "config.yaml"
    path.write_text("target_schema:\n  path: schemas/cimac_clinical_v1.yaml\n")
    return path


def _explicit_directory(request, option: str, label: str) -> Path:
    if not request.config.getoption("--run-production-integration"):
        pytest.skip("production integration was not explicitly enabled")
    raw = request.config.getoption(option)
    if not raw:
        pytest.skip(f"{label} requires explicit {option}")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        pytest.skip(f"configured {label} is not an existing directory")
    return path


@pytest.fixture(scope="session")
def production_outputs_dir(request) -> Path:
    """Explicit external production outputs; never guessed or auto-discovered."""
    return _explicit_directory(
        request, "--production-outputs-dir", "production output directory"
    )


@pytest.fixture(scope="session")
def production_source_root(request) -> Path:
    """Explicit external production sources; never guessed or auto-discovered."""
    return _explicit_directory(
        request, "--production-source-root", "production source root"
    )


@pytest.fixture(scope="session")
def production_expectations(request) -> dict:
    """Explicit external expectations; no production values live in tests."""
    if not request.config.getoption("--run-production-integration"):
        pytest.skip("production integration was not explicitly enabled")
    raw = request.config.getoption("--production-expectations")
    if not raw:
        pytest.skip("production expectations require --production-expectations")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        pytest.skip("configured production expectations file does not exist")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        pytest.skip("configured production expectations must be a JSON object")
    return payload


@pytest.fixture(scope="session")
def corpus_root(request) -> Path:
    """Explicit source-free declarative corpus root."""
    raw = request.config.getoption("--corpus-root")
    if not raw:
        pytest.skip("corpus acceptance requires explicit --corpus-root")
    path = Path(raw).expanduser().resolve()
    required = (path / "specs", path / "schemas", path / "vocabularies")
    if not all(item.is_dir() for item in required):
        pytest.skip("configured corpus root lacks specs/schemas/vocabularies")
    return path


@pytest.fixture(scope="session")
def corpus_plan(request) -> Path:
    """Explicit source-free declarative identity plan."""
    raw = request.config.getoption("--corpus-plan")
    if not raw:
        pytest.skip("corpus acceptance requires explicit --corpus-plan")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        pytest.skip("configured corpus identity plan is not a file")
    return path
