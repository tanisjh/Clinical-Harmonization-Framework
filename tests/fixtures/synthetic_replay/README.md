# Synthetic replay fixture

This non-clinical fixture describes calibrated device readings. It exists only
to exercise the generic Stage 0 evidence contract. Generated delivered,
provenance, and manifest artifacts are created under the test scratch directory
and are not checked in.

`schema_stage3.yaml`, `spec_stage3.yaml`, and `sources/stage3_readings.csv`
exercise recursive registered field-rule replay with non-clinical measurements.

`schema_stage5.yaml`, `spec_stage5.yaml`, `sources/stage5_readings.csv`, and the
two files under `vocabularies/` exercise exact runtime vocabulary evidence,
ordered vocabulary-backed value maps, and bucket-backed registered rules.

`schema_stage6.yaml` and `spec_stage6.yaml` exercise source-independent scalar
constants, exact null/no-value policy, raw-snapshot policy, and closed refusal
of condition, derivation, anchor, source-attributed declared-absence,
collection, and non-finite shapes.

`schema_stage7.yaml` and `spec_stage7.yaml` exercise source-independent
declared absence with exact `not_applicable` versus `source_missing` semantics,
Cell-state serialization policy, typed-null/omitted raw evidence, source-free
origin and record identity, recursive field-rule use, and closed refusal of
source-attributed or otherwise unsupported absence shapes.

`schema_stage8.yaml`, `schema_stage8_group.yaml`, `spec_stage8.yaml`,
`spec_stage8_group.yaml`, and
`sources/stage8_attribution.csv` exercise exact runtime source-file
attribution for declared absence. The declared absence columns intentionally
do not exist in the source: replay verifies file identity and never reads a
source row or scalar.

`schema_stage9.yaml`, `spec_stage9.yaml`, `sources/stage9_grouping.csv`, and
`sources/stage9_participants.csv` exercise exact runtime grouping-anchor
evidence. They cover read-order-preserving first-row selection, projected
anchor fields, a declared `no_record_for_participant` synthesized route,
usage-scoped anchor Contributions, and strict replay of grouping-backed
`no_mapping_configured`.

`schema_stage10.yaml`, `spec_stage10.yaml`,
`sources/stage10_grouping.csv`, `sources/stage10_participants.csv`, and
`sources/stage10_attribution.csv` exercise the generic verified-anchor-scalar
adapter for grouping-anchor direct mappings. They cover grouped and supported
synthesized routes, `none`/`text`/`numeric` direct coercion, and independent
no-row source attribution. Grouping-anchor value maps remain unsupported.

`schema_stage11.yaml`, `spec_stage11.yaml`,
`sources/stage11_external_anchor.csv`, and
`sources/stage11_attribution.csv` exercise trusted external-anchor runtime
evidence. They cover exact-byte `external_anchor_csv@v1` parsing, a generic
typed equality filter, ordered `external_anchor_frame@v1` identity, original
and filtered row locators, direct and inline value-map replay, and independent
no-row source attribution. Bare in-memory DataFrames intentionally cannot
publish strict replay evidence.

`schema_stage12.yaml`, `spec_stage12.yaml`, the four `stage12_*.csv`
sources, and `vocabularies/stage12_ordered.yaml` exercise independently
reconstructed ordered source-group selection. They cover differing deciding and
scalar roles, `first_containing_record`, duplicate-first behavior,
`column_by_member`, `file_membership`, direct and inline/vocabulary value maps,
and recursive registered-rule replay. Recorded membership Contributions are
verification targets and never choose the winning member.


`schema_stage13.yaml`, `spec_stage13.yaml`, and
`vocabularies/stage13_ordered.yaml` exercise exact-byte
`xlsx_openpyxl@v1` observation and `xlsx_sheet_frame@v1` identity. Workbooks
are generated synthetically in repository-local pytest scratch. The fixture
covers direct, inline value-map, unit-conversion, vocabulary value-map, and
recursive registered-rule replay without any clinical workbook data.
