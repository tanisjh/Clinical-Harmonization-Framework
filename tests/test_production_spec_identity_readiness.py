"""Read-only identity/equivalence checks for approved production candidates.

No source path declared by a spec is opened by these tests.
"""
from __future__ import annotations

import pytest
import yaml

from lib.schema_model import load as load_schema
from lib.spec_identity_lint import compare_semantic_structure, lint_paths
from lib.spec_model import load as load_spec
from lib.replay_orchestration import ReplayRunContext

pytestmark = pytest.mark.corpus_acceptance


def _plan(corpus_plan):
    return yaml.safe_load(corpus_plan.read_text())


def test_all_production_candidates_match_identity_plan(corpus_root, corpus_plan):
    plan = _plan(corpus_plan)
    spec_dir = corpus_root / "specs"
    candidates = [spec_dir / spec["candidate_spec_file"]
                  for spec in plan["specs"]]
    assert len(candidates) == 11
    assert all(path.is_file() for path in candidates)
    report = lint_paths(candidates, plan_path=corpus_plan)
    assert report["spec_count"] == 11
    assert report["mapping_count"] == 204
    assert report["present_id_count"] == 204
    assert report["approved_count"] == 11
    assert report["identity_ready"] is True
    assert all(spec["status"] == "approved" for spec in report["specs"])
    assert all(spec["plan_status"] == "pass" for spec in report["specs"])


def test_production_candidates_parse_as_executable_declarations(
        corpus_root, corpus_plan):
    plan = _plan(corpus_plan)
    spec_dir = corpus_root / "specs"
    schema = load_schema(corpus_root / "schemas/cimac_clinical_v1.yaml")
    loaded = [load_spec(
        spec_dir / spec["candidate_spec_file"],
        target_schema=schema, require_executable=True,
        vocabulary_dir=corpus_root / "vocabularies") for spec in plan["specs"]]
    assert len(loaded) == 11
    assert sum(len(spec.mappings) for spec in loaded) == 204
    assert all(spec.status == "approved" for spec in loaded)
    assert all(mapping.mapping_id for spec in loaded for mapping in spec.mappings)

    context = ReplayRunContext.create(
        corpus_root, schema, loaded,
        run_id="run-production-readiness-approved")
    assert len(context.specification_bindings) == 11


def test_production_candidates_are_structurally_equivalent_to_originals(
        corpus_root, corpus_plan):
    plan = _plan(corpus_plan)
    spec_dir = corpus_root / "specs"
    checks = []
    for spec in plan["specs"]:
        old = spec_dir / spec["spec_file"]
        new = spec_dir / spec["candidate_spec_file"]
        checks.append(compare_semantic_structure(old, new))
    assert len(checks) == 11
    assert sum(check["old_mapping_count"] for check in checks) == 204
    assert all(not check["equivalent"] for check in checks)
    assert all(check["old_status"] == "draft" for check in checks)
    assert all(check["new_status"] == "approved" for check in checks)

    for spec in plan["specs"]:
        old = yaml.safe_load((spec_dir / spec["spec_file"]).read_text())
        new = yaml.safe_load(
            (spec_dir / spec["candidate_spec_file"]).read_text())
        old.pop("spec_version")
        old.pop("status")
        new.pop("spec_version")
        new.pop("status")
        for mapping in old["mappings"]:
            mapping.pop("mapping_id", None)
        for mapping in new["mappings"]:
            mapping.pop("mapping_id", None)
        assert old == new
