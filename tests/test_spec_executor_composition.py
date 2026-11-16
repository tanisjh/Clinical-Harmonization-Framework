"""Fabricated parity and architecture checks for the composed executor."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from lib import normalize, rule_registry
from lib.anchor_evidence import RuntimeSourceRefs, load_external_anchor
from lib.cell_factory import RAW_VALUE_ABSENT, CellFactory
from lib.source_ref import PathContainmentError
from lib.source_session import SourceSession
from lib.spec_executor import ExecutionError, SpecExecutor, _READER_LOADERS, execute
from lib.spec_model import load as load_spec
from lib.spec_preflight import executable_subset, source_columns
from lib.xlsx_reader import XLSX_READER_ID, XlsxReaderError

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
READERS = [("cidc_csv@v1", "csv"), ("excel@v1", "xlsx"), ("xlsx_openpyxl@v1", "xlsx")]
ROW = {"device_id": "DEVICE-A", "reading_id": "READ-001", "location": "bench-west",
       "cimac_part_id": "DEVICE-A", "Cimac.id": "READ-001", "Collection_Event": "bench-west"}


@pytest.fixture
def case(tmp_path):
    def make(reader="cidc_csv@v1"):
        frame = pd.read_csv(FIXTURE / "sources/device_readings.csv")
        source = tmp_path / ("records.csv" if reader == "cidc_csv@v1" else "records.xlsx")
        if reader == "cidc_csv@v1":
            frame.to_csv(source, index=False)
        else:
            frame.to_excel(source, sheet_name="Readings", index=False)
        document = yaml.safe_load((FIXTURE / "spec.yaml").read_text())
        document["sources"][0].update(path=source.name, reader=reader,
                                       file_type="csv" if reader == "cidc_csv@v1" else "xlsx")
        if reader != "cidc_csv@v1":
            document["sources"][0]["sheet"] = "Readings"
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(document, sort_keys=False))
        spec = load_spec(spec_path, require_executable=True)
        anchor = pd.DataFrame([ROW], index=[17])
        return SimpleNamespace(root=tmp_path, source=source, spec=spec, anchor=anchor,
                               executor=SpecExecutor(spec, tmp_path, anchor, run_id="composition-fixed"))
    return make


def encoded(cells):
    return json.dumps([cell.as_record() for cell in cells], default=str)


@pytest.mark.parametrize("reader", [item[0] for item in READERS])
def test_reader_cell_ref_diagnostics_and_repeat_parity(case, reader):
    c = case(reader)
    ex = c.executor
    assert isinstance(ex._source_session, SourceSession) and isinstance(ex._cell_factory, CellFactory)
    assert list(ex.anchor.index) == [0] and list(c.anchor.index) == [17]
    first = list(ex.extract())
    assert [cell.harmonized_field for cell in first] == [m.target for m in c.spec.mappings]
    assert ex.diagnostics.order == tuple(cell.harmonized_field for cell in first)
    assert ex.diagnostics.cells == len(first) == 4
    assert ex.diagnostics.anchor_rows == 1
    assert isinstance(ex.diagnostics.sources, RuntimeSourceRefs)
    assert ex.diagnostics.sources.anchor_evidence == ()
    assert ex.ref("readings").reader_id == reader
    refs = ex.source_refs()
    assert refs is not ex._refs and refs["readings"] is ex.ref("readings")
    again = list(ex.extract())
    assert encoded(first) == encoded(again)
    wrapped, diagnostics = execute(c.spec, c.root, c.anchor, run_id="composition-fixed")
    assert encoded(first) == encoded(wrapped)
    assert diagnostics == ex.diagnostics
    assert ex.source_refs()["readings"] is refs["readings"]


def test_csv_legacy_cache_mutability_and_live_state(case):
    c = case()
    ex = c.executor
    frame = ex.load_csv("readings")
    frame.loc[0, "sequence_number"] = 99
    assert ex._load_source("readings") is frame
    assert ex._cache is ex._source_session._cache
    assert ex._load_source("readings").loc[0, "sequence_number"] == 99
    for name in ("_cache", "_refs", "_xlsx_observations"):
        replacement = {}
        setattr(ex, name, replacement)
        assert getattr(ex._source_session, name) is replacement
        other = {}
        setattr(ex._source_session, name, other)
        assert getattr(ex, name) is other
    ex.project_root = c.root / "other"
    assert ex._source_session.project_root == c.root / "other"
    ex._source_session.project_root = c.root
    assert ex.project_root is c.root


@pytest.mark.parametrize("sheet", [None, 0, "Readings"])
def test_legacy_excel_cache_and_sheet_behavior(case, sheet):
    c = case("excel@v1")
    frame = c.executor.load_xlsx("readings", sheet)
    frame.loc[0, "sequence_number"] = 99
    assert c.executor.load_xlsx("readings", sheet) is frame
    assert c.executor.ref("readings").sheet == (str(sheet) if sheet is not None else None)
    assert c.executor.load_xlsx("readings", sheet).loc[0, "sequence_number"] == 99


def test_trusted_xlsx_copies_and_retained_tamper_detection(case):
    c = case(XLSX_READER_ID)
    ex = c.executor
    with pytest.raises(ValueError, match="non-empty exact sheet"):
        ex.load_xlsx("readings", reader_id=XLSX_READER_ID)
    frame = ex._load_source("readings")
    retained = ex._xlsx_observations["readings"]
    frame.loc[0, "sequence_number"] = 99
    assert ex._load_source("readings").loc[0, "sequence_number"] == 1
    assert ex._xlsx_observations["readings"] is retained
    retained._frame.loc[0, "sequence_number"] = 99
    with pytest.raises(XlsxReaderError, match="no longer matches"):
        ex._load_source("readings")


def test_constructor_no_source_io_and_current_context_errors(case, monkeypatch):
    c = case()
    def forbidden(*args, **kwargs):
        pytest.fail("constructor performed source I/O")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        ex = SpecExecutor(c.spec, c.root, c.anchor)
    assert ex.source_refs() == {} and ex._cache == {}
    with pytest.raises(ExecutionError, match="no source 'missing' is declared"):
        ex._load_source("missing")
    ex.trial_dir_name = "changed"
    with pytest.raises(KeyError, match="in changed;"):
        ex.ref("readings")


def test_resolver_is_late_bound_even_on_cache_hits(case):
    c = case()
    ex = c.executor
    calls = []
    ex.src_path = lambda key: (calls.append(key), c.source)[1]
    frame = ex.load_csv("readings")
    assert ex.load_csv("readings") is frame and calls == ["readings", "readings"]
    def refuse(key):
        raise LookupError("replacement resolver")
    ex.src_path = refuse
    with pytest.raises(LookupError, match="replacement resolver"):
        ex.load_csv("readings")


def test_modern_resolver_containment_is_unchanged(case):
    c = case()
    ex = c.executor
    ex.project_root = c.root / "nested"
    # Preserve declaration lookup and containment even when source context changes.
    assert ex.src_path("readings") == (c.root / "nested/records.csv").resolve()
    with pytest.raises(FileNotFoundError):
        ex.load_csv("readings")
    ex.project_root = c.root
    c.source.unlink()
    c.source.symlink_to(c.root.parent / "outside.csv")
    with pytest.raises(PathContainmentError):
        ex.load_csv("readings")


def test_external_anchor_reuses_observed_ref_and_evidence_without_io(tmp_path, monkeypatch):
    spec = load_spec(FIXTURE / "spec_stage11.yaml", require_executable=True)
    runtime = load_external_anchor(FIXTURE / "sources/stage11_external_anchor.csv",
                                  project_root=ROOT, spec_ref=spec.ref,
                                  filter_column="project", selector="alpha")
    def forbidden(*args, **kwargs):
        pytest.fail("observed anchor reread or rehashed")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        ex = SpecExecutor(spec, ROOT, runtime)
    assert ex.ref("$anchor") is runtime.source_ref
    assert ex._source_session.ref("$anchor") is runtime.source_ref
    assert ex._anchor_evidence is runtime.evidence
    list(ex.extract())
    assert ex.diagnostics.sources.anchor_evidence == (runtime.evidence,)
    assert ex.diagnostics.sources["$anchor"] is runtime.source_ref


def test_group_order_and_runtime_evidence_remain_stable():
    spec = load_spec(FIXTURE / "spec_stage9.yaml", require_executable=True)
    ex = SpecExecutor(spec, ROOT, run_id="composition-group")
    first = list(ex.extract())
    keys = ex.anchor[["cimac_part_id", "Cimac.id", "Collection_Event"]].to_dict("records")
    evidence = ex.diagnostics.sources.anchor_evidence
    assert len(evidence) == 1
    assert evidence[0].details["record_count"] == len(keys)
    assert [key for key in ex.source_refs()] == ["readings", "participants"]
    assert encoded(list(ex.extract())) == encoded(first)
    assert ex.anchor[["cimac_part_id", "Cimac.id", "Collection_Event"]].to_dict("records") == keys
    assert ex.diagnostics.sources.anchor_evidence == evidence


@pytest.mark.parametrize("raw,expected", [(RAW_VALUE_ABSENT, ""), (None, '{"kind":"null"}'),
                                           (0, '{"kind":"integer","value":"0"}')])
def test_cell_facade_identity_raw_metadata_and_counter(case, raw, expected):
    c = case()
    ex = c.executor
    ex.run_id, ex.spec_ref, ex.template_trial_name = "current-run", "current-spec", "current-trial"
    cell = ex.cell(ROW, "field", False, "1.25", "CONFIG:constant", "column", 0, "direct",
                   raw_value=raw, source_row_number=0, mapping_id="map", contributions="z;a",
                   authoritative_for="a;z", derivation="{}", evidence_refs="e1;e2")
    assert cell.trial == "current-trial" and cell.run_id == "current-run" and cell.spec_ref == "current-spec"
    assert cell.raw_value_json == expected and cell.source_row_number == 0
    assert cell.contributions == "z;a" and cell.authoritative_for == "a;z" and cell.evidence_refs == "e1;e2"
    assert ex._legacy_label_uses == 1
    with pytest.raises(KeyError):
        ex.cell({}, "field", 1, 1, "legacy", "column", 0, "direct")
    assert ex._legacy_label_uses == 2
    ex._legacy_label_uses = 7
    assert ex._cell_factory.legacy_label_uses == 7
    ex._cell_factory.legacy_label_uses = 11
    assert ex._legacy_label_uses == 11


def test_reader_table_and_preflight_contract(case):
    assert list(_READER_LOADERS.items()) == READERS
    assert executable_subset()["readers"] == tuple(key for key, _ in READERS)
    c = case()
    assert "sequence_number" in source_columns(c.spec, "readings", c.root)


def test_empty_compatibility_context_state_is_preserved(case):
    ex = case().executor
    assert ex.cfg == ex.value_norms == ex.derived_rules == {}
    assert ex.trial_dir == ex.project_root / ex.trial_dir_name


def test_normalization_owner_still_applies_rules():
    rules = [{"match": "raw", "value": "mapped", "confidence": .9, "note": "synthetic exact"}]
    assert normalize.normalize("synthetic", " raw ", rules) == ("mapped", .9, "synthetic exact")
    assert normalize.normalize("synthetic", None, rules) == (None, 0.0, "input_is_NaN")


@pytest.mark.parametrize("rule_id,function_name,values,params,expected_value", [
    ("derived.bor_binary.v1", "derive_bor_binary", ["synthetic"], {"rules": {"R": ["synthetic"]}}, "R"),
    ("derived.bor_binary.v1", "derive_bor_binary", [None], {"rules": {"R": ["synthetic"]}}, None),
    ("derived.bor_bin_120d.v1", "derive_bor_bin", ["R", 0], {"landmark_days": 120}, 1),
    ("derived.bor_bin_120d.v1", "derive_bor_bin", ["SD", 119], {"landmark_days": 120}, 0),
    ("derived.bor_bin_120d.v1", "derive_bor_bin", ["SD", 120], {"landmark_days": 120}, 1),
    ("derived.pfs_bin_120d.v1", "derive_pfs_bin", [119, 1], {"landmark_days": 120}, 0),
    ("derived.pfs_bin_120d.v1", "derive_pfs_bin", [119, 0], {"landmark_days": 120}, None),
    ("derived.pfs_bin_120d.v1", "derive_pfs_bin", [120, 0], {"landmark_days": 120}, 1),
])
def test_registered_scientific_rules_keep_canonical_owners(
        rule_id, function_name, values, params, expected_value):
    function = getattr(normalize, function_name)
    rule = rule_registry.resolve(rule_id, "1", kind=rule_registry.KIND_FIELD)
    assert rule.func is function
    expected = function(*values, **params)
    assert expected[0] == expected_value
    assert rule_registry.invoke(rule, values, params) == expected


def test_executor_uses_registered_scientific_owners(case):
    c = case()
    document = yaml.safe_load((c.root / "spec.yaml").read_text())
    declarations = [
        ("response_class", "derived.bor_binary.v1", ["reading_text"], {"rules": {"R": ["001"]}}),
        ("response_flag", "derived.bor_bin_120d.v1", ["response_class", "sequence_number"], {"landmark_days": 120}),
        ("landmark_flag", "derived.pfs_bin_120d.v1", ["sequence_number", "reading_text"], {"landmark_days": 120}),
    ]
    for target, rule_id, inputs, params in declarations:
        document["mappings"].append({
            "mapping_id": "custom." + target, "target": target, "op": "custom_rule",
            "inputs": inputs, "params": params,
            "output_type": "string" if target == "response_class" else "number",
            "rule": {"id": rule_id, "version": "1", "review_status": "approved"},
            "confidence": {"value": "from_rule", "no_value": .25},
            "provenance": {"record_key_from": "participant"},
        })
    path = c.root / "derived.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    executor = SpecExecutor(load_spec(path, require_executable=True), c.root, c.anchor,
                            run_id="scientific-owners")
    cells = {cell.harmonized_field: cell for cell in executor.extract()}
    assert [cells[name].value for name, *_ in declarations] == ["R", 1, 0]
    for target, rule_id, *_ in declarations:
        assert cells[target].rule_id == rule_id and cells[target].rule_version == "1"
        assert cells[target].confidence == .95


def test_architecture_and_unchanged_public_signatures():
    import ast
    import inspect
    from lib import spec_executor

    assert SpecExecutor.__bases__ == (object,)
    parameters = inspect.signature(SpecExecutor.__init__).parameters
    assert list(parameters) == ["self", "spec", "project_root", "anchor_df", "contract", "targets", "run_id"]
    assert parameters["anchor_df"].default is None and parameters["run_id"].default == ""
    assert parameters["contract"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(execute).parameters) == ["spec", "project_root", "anchor_df", "contract", "targets", "run_id"]
    assert list(inspect.signature(SpecExecutor.extract).parameters) == ["self"]
    cell_parameters = inspect.signature(SpecExecutor.cell).parameters
    assert list(cell_parameters) == [
        "self", "anchor_row", "harmonized_field", "value", "confidence", "source_file",
        "source_column", "source_row_idx", "extraction_method", "notes", "raw_value",
        "source_row_number", "source_record_key", "rule_id", "rule_version", "contributions",
        "authoritative_for", "derivation", "value_state", "mapping_id", "evidence_refs",
        "raw_evidence_recorded"]
    assert {name: parameter.default for name, parameter in cell_parameters.items()
            if parameter.default is not inspect.Parameter.empty} == {
        "notes": "", "raw_value": RAW_VALUE_ABSENT, "source_row_number": None,
        "source_record_key": None, "rule_id": "", "rule_version": "", "contributions": "",
        "authoritative_for": "", "derivation": "", "value_state": "", "mapping_id": "",
        "evidence_refs": "", "raw_evidence_recorded": True}
    for index, parameter in enumerate(cell_parameters.values()):
        assert parameter.kind is (inspect.Parameter.POSITIONAL_OR_KEYWORD if index < 10
                                  else inspect.Parameter.KEYWORD_ONLY)
    tree = ast.parse(Path(spec_executor.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpecExecutor")
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "super"
                   for n in ast.walk(cls))
    retired = {"get_norm_rules", "normalize_value", "derive_BOR_binary", "derive_bor_bin", "derive_pfs_bin"}
    assert not any(hasattr(SpecExecutor, name) for name in retired)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id != "BaseExtractor"
        elif isinstance(node, ast.Attribute):
            assert node.attr != "BaseExtractor"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
            assert not any({"extractor_base", "BaseExtractor"} & set(name.split("."))
                           for name in names)


def test_fresh_modern_runtime_does_not_import_base_or_legacy_registry():
    import subprocess
    import sys

    code = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"lib.extractor_base", "lib.migrated_trials", "extract_harmonized_clinical"}:
            raise AssertionError("modern runtime imported " + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, "scripts")
from pathlib import Path
import pandas as pd
from lib.spec_model import load
from lib.spec_executor import SpecExecutor
spec = load("tests/fixtures/synthetic_replay/spec.yaml", require_executable=True)
row = {"device_id": "DEVICE-A", "reading_id": "READ-001", "location": "bench-west",
       "cimac_part_id": "DEVICE-A", "Cimac.id": "READ-001", "Collection_Event": "bench-west"}
ex = SpecExecutor(spec, Path.cwd(), pd.DataFrame([row]), run_id="composition-fresh")
assert len(list(ex.extract())) == 4
assert "lib.extractor_base" not in sys.modules
'''
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)


def test_modern_runtime_uses_shared_source_and_cell_components(case, monkeypatch):
    c = case()
    sessions, factories = [], []
    load, finish = SourceSession.load_csv, CellFactory._build_prepared
    def load_spy(session, *args, **kwargs):
        sessions.append(session)
        return load(session, *args, **kwargs)
    def cell_spy(factory, *args, **kwargs):
        factories.append(factory)
        return finish(factory, *args, **kwargs)
    monkeypatch.setattr(SourceSession, "load_csv", load_spy)
    monkeypatch.setattr(CellFactory, "_build_prepared", cell_spy)
    assert len(list(c.executor.extract())) == 4
    assert sessions and all(s is c.executor._source_session for s in sessions)
    assert len(factories) == 4 and all(f is c.executor._cell_factory for f in factories)


def test_explicit_join_and_source_mapping_order(case):
    c = case()
    c.source.with_name("secondary.csv").write_bytes(c.source.read_bytes())
    doc = yaml.safe_load((c.root / "spec.yaml").read_text())
    doc["sources"].append({**doc["sources"][0], "id": "secondary", "path": "secondary.csv"})
    doc["mappings"][1]["source"] = "secondary"
    doc["record"]["joins"] = [{"id": "paired", "left": "readings", "right": "secondary",
                                  "keys": ["device_id"], "cardinality": "one_to_one"}]
    path = c.root / "joined.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    spec = load_spec(path, require_executable=True)
    cells, diagnostics = execute(spec, c.root, c.anchor)
    assert diagnostics.joins_validated == ("paired",)
    assert diagnostics.order == tuple(m.target for m in spec.mappings)
    assert [cell.mapping_id for cell in cells] == [m.mapping_id for m in spec.mappings]
    assert list(diagnostics.sources) == ["readings", "secondary"]


def test_lookup_miss_cells_keep_source_identity_and_order(case):
    c = case()
    doc = yaml.safe_load((c.root / "spec.yaml").read_text())
    targets = [m["target"] for m in doc["mappings"]]
    doc["sources"][0]["on_missing_record"] = {
        "mode": "lookup_miss", "emit_for": targets, "confidence": .1,
        "rule": {"id": "lookup.miss.v1", "version": 1, "review_status": "approved"}}
    path = c.root / "missing.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    spec = load_spec(path, require_executable=True)
    row = {**ROW, "device_id": "DEVICE-Z", "cimac_part_id": "DEVICE-Z"}
    cells, diagnostics = execute(spec, c.root, pd.DataFrame([row]))
    assert [cell.harmonized_field for cell in cells] == targets
    assert all(cell.extraction_method == "spec:lookup_miss" for cell in cells)
    assert all(cell.source_ref_id == diagnostics.sources["readings"].source_ref_id for cell in cells)
    assert all(cell.confidence == .1 and cell.source_row_idx == -1 for cell in cells)
    assert diagnostics.lookup_misses == ("DEVICE-Z",)


def test_invalid_confidence_precedes_missing_run_context(case):
    ex = case().executor
    del ex.run_id
    row = {"cimac_part_id": "SYNTH-001", "Cimac.id": "SAMPLE-001", "Collection_Event": "event-1"}
    with pytest.raises(ValueError, match="could not convert string to float"):
        ex.cell(row, "field", 1, "invalid", "legacy", "column", 0, "direct")
    assert ex._legacy_label_uses == 1


def test_legacy_counter_has_one_live_factory_owner(case):
    from lib.cell_factory import CellFactory

    ex = case().executor
    other = case().executor
    assert ex._cell_factory is not other._cell_factory
    session = ex._source_session
    ex._legacy_label_uses = 7
    assert ex._cell_factory.legacy_label_uses == 7
    ex._cell_factory.legacy_label_uses = 11
    assert ex._legacy_label_uses == 11
    ex._cell_factory = CellFactory()
    assert ex._legacy_label_uses == other._legacy_label_uses == 0
    assert ex._source_session is session


def test_cell_context_is_live_and_source_session_unchanged(case):
    ex = case().executor
    session = ex._source_session
    row = {"cimac_part_id": "SYNTH-001", "Cimac.id": "SAMPLE-001", "Collection_Event": "event-1"}
    for trial, run, spec in [("trial-1", "run-1", "spec@1"), ("trial-2", "run-2", "spec@2")]:
        ex.template_trial_name, ex.run_id, ex.spec_ref = trial, run, spec
        cell = ex.cell(row, "field", False, 1, "legacy", "column", 0, "direct")
        assert (cell.trial, cell.run_id, cell.spec_ref) == (trial, run, spec)
    assert ex._legacy_label_uses == 2
    assert ex._source_session is session and ex.source_refs() == {}


def test_normalizer_overrides_are_resolved_before_each_identity_lookup(case):
    ex = case().executor
    ex._anchor_str = lambda value: "initial:" + value

    class Row:
        def __getitem__(self, key):
            ex._anchor_str = lambda value: "next:" + value
            return key

    cell = ex.cell(Row(), "field", 1, 1, "legacy", "column", 0, "direct")
    assert cell.cimac_part_id == "initial:cimac_part_id"
    assert cell.Cimac_id == "next:Cimac.id"
    assert cell.Collection_Event == "next:Collection_Event"


@pytest.mark.parametrize("reader", [item[0] for item in READERS])
def test_all_cached_readers_reconsult_current_resolver(case, reader):
    c = case(reader)
    ex = c.executor
    calls = []

    def resolve(key):
        calls.append(key)
        return c.source

    ex.src_path = resolve
    ex._load_source("readings")
    ex._load_source("readings")
    assert calls == ["readings", "readings"]

    def refuse(key):
        raise LookupError("current resolver refuses source")

    ex.src_path = refuse
    with pytest.raises(LookupError, match="current resolver refuses source"):
        ex._load_source("readings")


def test_replaced_resolver_can_select_a_different_observed_file(case):
    c = case()
    ex = c.executor
    original = ex.load_csv("readings")
    second = c.source.with_name("second.csv")
    second.write_text("value\n34\n")
    ex.src_path = lambda key: second
    assert ex.load_csv("readings").loc[0, "value"] == 34
    assert ex.ref("readings").resolved_path == "second.csv"
    ex.src_path = lambda key: c.source
    assert ex.load_csv("readings") is original


@pytest.mark.parametrize("source_kind", ["legacy", "observed"])
@pytest.mark.parametrize("failure", ["identity", "confidence", "raw"])
def test_cell_facade_accounts_before_later_failures(case, source_kind, failure):
    c = case()
    ex = c.executor
    source = "legacy"
    if source_kind == "observed":
        ex.load_csv("readings")
        source = ex.ref("readings")
    options = {"raw_value": object()} if failure == "raw" else {}
    with pytest.raises(KeyError if failure == "identity" else ValueError):
        ex.cell({} if failure == "identity" else ROW, "field", 1,
                "invalid" if failure == "confidence" else 1, source,
                "column", 0, "direct", **options)
    assert ex._legacy_label_uses == int(source_kind == "legacy")


@pytest.mark.parametrize("key", ["cimac_part_id", "Cimac.id", "Collection_Event"])
def test_cell_facade_missing_identity_has_no_alias_fallback(case, key):
    ex = case().executor
    row = {**ROW, "participant": "alternative", "sample": "alternative",
           "collection_event": "alternative", "trial": "ignored"}
    del row[key]
    with pytest.raises(KeyError) as exc:
        ex.cell(row, "field", 1, 1, "legacy", "column", 0, "direct")
    assert exc.value.args == (key,)
    assert ex._legacy_label_uses == 1


def test_cell_evaluation_order_including_late_failure(case, monkeypatch):
    ex = case().executor
    events = []

    def trial(instance):
        events.append("trial")
        assert instance._legacy_label_uses == 1
        return "trial-verbatim"

    def run(instance):
        events.append("run")
        return "run-verbatim"

    def spec(instance):
        events.append("spec")
        return "spec-verbatim"

    def normalize(value):
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

    # Instrument the real modern facade after initialization; no replacement extractor.
    monkeypatch.setattr(SpecExecutor, "template_trial_name", property(trial), raising=False)
    monkeypatch.setattr(SpecExecutor, "run_id", property(run), raising=False)
    monkeypatch.setattr(SpecExecutor, "spec_ref", property(spec), raising=False)
    ex._anchor_str = normalize
    with pytest.raises(ValueError, match="raw value is not a supported scalar: object"):
        ex.cell(Row(), "field", 1, Confidence(), "legacy", "column", 0,
                "direct", source_record_key=RecordKey(), raw_value=object())
    assert events == ["trial", "get:cimac_part_id", "normalize:cimac_part_id",
                      "get:Cimac.id", "normalize:Cimac.id",
                      "get:Collection_Event", "normalize:Collection_Event",
                      "confidence", "record-key", "run", "spec"]
    assert ex._legacy_label_uses == 1


def test_retired_base_absent_and_runtime_has_only_shared_component_owners():
    import ast

    assert not (ROOT / "scripts/lib/extractor_base.py").exists()
    owners = {"SourceSession": [], "CellFactory": []}
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.name != "BaseExtractor", path
                if node.name in owners:
                    owners[node.name].append(path.relative_to(ROOT).as_posix())
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                if isinstance(node, ast.ImportFrom):
                    names.append(node.module or "")
                assert not any({"extractor_base", "BaseExtractor"} & set(name.split("."))
                               for name in names), path
            elif isinstance(node, ast.Name):
                assert node.id != "BaseExtractor", path
            elif isinstance(node, ast.Attribute):
                assert node.attr != "BaseExtractor", path
    assert owners == {"SourceSession": ["scripts/lib/source_session.py"],
                      "CellFactory": ["scripts/lib/cell_factory.py"]}
