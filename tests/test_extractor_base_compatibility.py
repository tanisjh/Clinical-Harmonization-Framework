"""Source-facing compatibility and late binding of the BaseExtractor facade."""
from __future__ import annotations

import pandas as pd
import pytest

from lib.extractor_base import BaseExtractor


class SyntheticExtractor(BaseExtractor):
    trial_dir_name = "synthetic"
    template_trial_name = "synthetic"


def make_base(root, sources=None):
    return SyntheticExtractor(root, pd.DataFrame(),
                              {"source_dir": ".", "sources": sources or {}}, {})


def test_constructor_does_not_resolve_or_open_sources(tmp_path):
    class NoIOExtractor(SyntheticExtractor):
        def src_path(self, key):
            pytest.fail("constructor resolved a source")
    base = NoIOExtractor(tmp_path / "absent", pd.DataFrame(), {}, {})
    assert base.source_refs() == {} and base._cache == {}


def test_resolver_replacement_and_cache_resolution_timing(tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    first.write_text("value\n12\n")
    second.write_text("value\n34\n")
    base = make_base(tmp_path)
    calls = []

    def resolve_first(key):
        calls.append(key)
        return first

    base.src_path = resolve_first
    frame = base.load_csv("data")
    assert base.load_csv("data") is frame and calls == ["data", "data"]
    base.src_path = lambda key: second
    assert base.load_csv("data").loc[0, "value"] == 34
    assert base.ref("data").resolved_path == "second.csv"


def test_context_and_config_remain_late_bound(tmp_path):
    base = make_base(tmp_path)
    base.trial_dir_name = "changed-context"
    with pytest.raises(KeyError, match="in changed-context;"):
        base.ref("missing")
    with pytest.raises(KeyError, match="configured for trial changed-context"):
        base.load_csv("missing")
    (tmp_path / "records.csv").write_text("value\n12\n")
    base.cfg = {"source_dir": ".", "sources": {"data": "records.csv"}}
    assert base.load_csv("data").loc[0, "value"] == 12


def test_source_state_replacement_preserves_behavior(tmp_path):
    (tmp_path / "records.csv").write_text("value\n12\n")
    base = make_base(tmp_path, {"data": "records.csv"})
    original = base.load_csv("data")
    ref = base.ref("data")
    replacement = pd.DataFrame({"value": [99]})
    base._cache = {str(tmp_path / "records.csv"): replacement}
    assert base.load_csv("data") is replacement
    base._refs = {"other": ref}
    assert base.ref("other") is ref
    with pytest.raises(KeyError):
        base.ref("data")
    base._cache = {}
    assert base.load_csv("data") is not original
    assert base.ref("data") == ref


def test_project_root_reassignment_keeps_original_trial_dir_semantics(tmp_path):
    base = make_base(tmp_path, {"data": "records.csv"})
    trial_dir = base.trial_dir
    other = tmp_path / "other"
    other.mkdir()
    (other / "records.csv").write_text("value\n34\n")
    base.project_root = other
    assert base.trial_dir == trial_dir
    assert base.load_csv("data").loc[0, "value"] == 34
    assert base.ref("data").resolved_path == "records.csv"


@pytest.mark.parametrize("attribute", ["_cache", "_refs", "_xlsx_observations"])
def test_source_state_views_write_through_both_directions(tmp_path, attribute):
    base = make_base(tmp_path)
    session = base._source_session
    original = getattr(base, attribute)
    assert original is getattr(session, attribute)
    replacement = {}
    setattr(base, attribute, replacement)
    assert getattr(base, attribute) is replacement
    assert getattr(session, attribute) is replacement
    replacement["sentinel"] = object()
    assert "sentinel" in getattr(base, attribute)
    next_replacement = {}
    setattr(session, attribute, next_replacement)
    assert getattr(base, attribute) is next_replacement
    assert original == {}


def test_project_root_is_one_live_owner(tmp_path):
    base = make_base(tmp_path)
    session = base._source_session
    replacement = tmp_path / "replacement"
    base.project_root = replacement
    assert session.project_root is replacement
    session.project_root = tmp_path
    assert base.project_root is tmp_path


def test_trusted_observation_state_replacement_is_live(tmp_path):
    from lib.xlsx_reader import XLSX_READER_ID

    path = tmp_path / "records.xlsx"
    pd.DataFrame({"value": [12]}).to_excel(path, sheet_name="first", index=False)
    base = make_base(tmp_path, {"data": path.name})
    base.load_xlsx("data", "first", reader_id=XLSX_READER_ID)
    first = base._xlsx_observations["data"]
    base._xlsx_observations = {}
    base.load_xlsx("data", "first", reader_id=XLSX_READER_ID)
    second = base._source_session._xlsx_observations["data"]
    assert second is not first
    assert base.ref("data") is second.source_ref
    base._source_session._xlsx_observations = {"data": first}
    base.load_xlsx("data", "first", reader_id=XLSX_READER_ID)
    assert base._xlsx_observations["data"] is first
    # Replacing observations alone historically does not repopulate the ref dict.
    assert base.ref("data") is second.source_ref


def test_external_anchor_registration_reuses_observed_ref_without_io(tmp_path, monkeypatch):
    from pathlib import Path

    from lib.anchor_evidence import EXTERNAL_ANCHOR_SOURCE_ID, load_external_anchor
    from lib.spec_executor import SpecExecutor
    from lib.spec_model import load as load_spec

    fixture = Path(__file__).parent / "fixtures" / "synthetic_replay"
    spec = load_spec(fixture / "spec_stage11.yaml", require_executable=True)
    path = tmp_path / "anchor.csv"
    path.write_text("project,cimac_part_id,Cimac.id,Collection_Event\n"
                    "alpha,SYNTH-001,READ-001,bench\n")
    anchor = load_external_anchor(
        path, project_root=tmp_path, spec_ref=spec.ref,
        filter_column="project", selector="alpha")

    def forbidden(*args, **kwargs):
        pytest.fail("executor reread or rehashed an already observed anchor")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        executor = SpecExecutor(spec, tmp_path, anchor)
    assert isinstance(executor, BaseExtractor)
    assert executor.ref(EXTERNAL_ANCHOR_SOURCE_ID) is anchor.source_ref
    assert executor._source_session.ref(EXTERNAL_ANCHOR_SOURCE_ID) is anchor.source_ref
    assert executor._anchor_evidence is anchor.evidence


@pytest.mark.parametrize("reader", ["csv", "excel@v1", "xlsx_openpyxl@v1"])
def test_all_cached_loaders_resolve_again_and_propagate_resolver_errors(tmp_path, reader):
    path = tmp_path / ("records.csv" if reader == "csv" else "records.xlsx")
    if reader == "csv":
        path.write_text("value\n12\n")
    else:
        pd.DataFrame({"value": [12]}).to_excel(path, sheet_name="first", index=False)
    base = make_base(tmp_path)
    calls = []

    def resolve(key):
        calls.append(key)
        return path

    base.src_path = resolve

    def load():
        if reader == "csv":
            return base.load_csv("data")
        return base.load_xlsx("data", "first", reader_id=reader)

    load()
    load()
    assert calls == ["data", "data"]

    def refuse(key):
        raise LookupError("current resolver refuses source")

    base.src_path = refuse
    with pytest.raises(LookupError, match="current resolver refuses source"):
        load()
