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
    assert not isinstance(executor, BaseExtractor)
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


def test_cell_evaluation_order_including_late_failure(tmp_path):
    events = []

    class OrderedExtractor(SyntheticExtractor):
        @property
        def template_trial_name(self):
            events.append("trial")
            assert self._legacy_label_uses == 1
            return "trial-verbatim"

        @property
        def run_id(self):
            events.append("run")
            return self._run

        @run_id.setter
        def run_id(self, value):
            self._run = value

        @property
        def spec_ref(self):
            events.append("spec")
            return self._spec

        @spec_ref.setter
        def spec_ref(self, value):
            self._spec = value

        @staticmethod
        def _anchor_str(value):
            events.append("normalize:" + value)
            return "normalized:" + value

    class Row:
        def __getitem__(self, key):
            events.append("get:" + key)
            return key

    class Confidence:
        def __float__(self):
            events.append("confidence")
            return 0.75

    class RecordKey:
        def __bool__(self):
            events.append("record-key")
            return False

    base = OrderedExtractor(tmp_path, pd.DataFrame(), {}, {})
    events.clear()
    with pytest.raises(ValueError, match="raw value is not a supported scalar: object"):
        base.cell(Row(), "field", 1, Confidence(), "legacy", "column", 0,
                  "direct", source_record_key=RecordKey(), raw_value=object())
    assert events == ["trial", "get:cimac_part_id", "normalize:cimac_part_id",
                      "get:Cimac.id", "normalize:Cimac.id",
                      "get:Collection_Event", "normalize:Collection_Event",
                      "confidence", "record-key", "run", "spec"]
    assert base._legacy_label_uses == 1


def test_invalid_confidence_precedes_missing_run_context(tmp_path):
    base = make_base(tmp_path)
    del base.run_id
    row = {"cimac_part_id": "SYNTH-001", "Cimac.id": "SAMPLE-001", "Collection_Event": "event-1"}
    with pytest.raises(ValueError, match="could not convert string to float"):
        base.cell(row, "field", 1, "invalid", "legacy", "column", 0, "direct")
    assert base._legacy_label_uses == 1


def test_legacy_counter_has_one_live_factory_owner(tmp_path):
    from lib.cell_factory import CellFactory

    base = make_base(tmp_path)
    other = make_base(tmp_path)
    assert base._cell_factory is not other._cell_factory
    session = base._source_session
    base._legacy_label_uses = 7
    assert base._cell_factory.legacy_label_uses == 7
    base._cell_factory.legacy_label_uses = 11
    assert base._legacy_label_uses == 11
    base._cell_factory = CellFactory()
    assert base._legacy_label_uses == other._legacy_label_uses == 0
    assert base._source_session is session


def test_cell_context_is_live_and_source_session_unchanged(tmp_path):
    base = make_base(tmp_path)
    session = base._source_session
    row = {"cimac_part_id": "SYNTH-001", "Cimac.id": "SAMPLE-001", "Collection_Event": "event-1"}
    for trial, run, spec in [("trial-1", "run-1", "spec@1"), ("trial-2", "run-2", "spec@2")]:
        base.template_trial_name, base.run_id, base.spec_ref = trial, run, spec
        cell = base.cell(row, "field", False, 1, "legacy", "column", 0, "direct")
        assert (cell.trial, cell.run_id, cell.spec_ref) == (trial, run, spec)
    assert base._legacy_label_uses == 2
    assert base._source_session is session and base.source_refs() == {}


def test_normalizer_overrides_are_resolved_before_each_identity_lookup(tmp_path):
    base = make_base(tmp_path)
    base._anchor_str = lambda value: "initial:" + value

    class Row:
        def __getitem__(self, key):
            base._anchor_str = lambda value: "next:" + value
            return key

    cell = base.cell(Row(), "field", 1, 1, "legacy", "column", 0, "direct")
    assert cell.cimac_part_id == "initial:cimac_part_id"
    assert cell.Cimac_id == "next:Cimac.id"
    assert cell.Collection_Event == "next:Collection_Event"
