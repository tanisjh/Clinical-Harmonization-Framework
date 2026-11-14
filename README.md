# Clinical Harmonization Framework

## Overview

The Clinical Harmonization Framework converts heterogeneous clinical files into
consistent tables while preserving a detailed technical audit trail.

Harmonization rules are written in structured configuration files, reviewed by
people, and executed by generic deterministic Python software. This separates
project-specific knowledge—source columns, category mappings, fallback order,
and target fields—from the code that validates, executes, and verifies those
rules.

The framework is intended to make clinical-data transformations easier to
inspect, reproduce, and review. It does not decide whether a clinical mapping
is scientifically correct; that responsibility remains with qualified human
reviewers.

## Why this exists

Clinical projects commonly receive files with different layouts, names,
encodings, and levels of completeness. A conventional script can hide those
differences in trial-specific branches. This framework instead makes supported
mapping decisions explicit and reviewable, then applies them through shared
execution machinery.

For each harmonized value, cell-level provenance records where the value came
from and which rule produced it. Strict replay can then independently
reconstruct a result from the pinned rule and source evidence and compare that
result with the recorded provenance and delivered output.

## How it works

```text
heterogeneous clinical files
            |
            v
reviewed structured harmonization rules
            |
            v
deterministic validation and execution
            |
            +----> harmonized tables
            |
            +----> cell-level provenance
                         |
                         v
             strict replay verification
```

The main responsibilities are deliberately separated:

- **Structured specifications** identify sources, target fields, operations,
  parameters, mappings, fallback order, and evidence policy.
- **The target schema** defines the machine-readable output contract.
- **The executor** validates supported rules and applies them deterministically.
- **Provenance** records source, rule, confidence, raw-value, and contribution
  evidence for each output cell as applicable.
- **Strict replay** verifies source and rule authority, recomputes the expected
  result without using the recorded output as an input, and compares delivery
  last.

Recorded provenance is a verification target, not proof by itself. A matching
replay result establishes technical reproducibility under the recorded
evidence; it does not establish clinical correctness.

## Human and AI roles

AI tools may help collect mapping information, draft structured specifications,
or identify questions for review. Clinically or semantically important choices
still require human approval. Deterministic Python executes only supported,
reviewed rules, while provenance and replay provide technical traceability and
reproducibility checks.

Acceptance by the specification validator or replay engine must not be treated
as independent clinical validation of an AI- or human-authored mapping.

## Repository structure

```text
scripts/      harmonization, validation, provenance, tracing, and replay tools
scripts/lib/  shared execution, evidence, schema, and replay modules
profiles/     mutable project-tooling settings loaded only when requested
schemas/      machine-readable target data contract
tests/        portable synthetic tests and source-free acceptance checks
```

Normal automated tests use fabricated data. Production-data integration tests
are opt-in and require explicitly supplied external files. Clinical source data
and generated production artifacts are not bundled with this repository.

The checkout contains the engine, a current target schema, and synthetic test
fixtures. A real harmonization run also requires a reviewed project
specification and its separately managed source files.

Canonical execution uses the shared `scripts/lib/pivot.py` and does not require
the historical project orchestrator. Its identity handling retains the existing
Cell-v1 compatibility shape.

## Project profiles

A `ProjectProfile` is a small, strict settings document for current tooling. It
may name the authoritative target schema, explicitly approved specification
files or non-recursive specification roots, current configuration references,
output defaults, and roles or aliases for artifact IDs already declared by the
target schema. Unknown fields, non-portable references, ambiguous aliases, and
references inconsistent with the schema or specifications fail validation.

The profile does not contain target fields, mappings, clinical values,
normalization rules, scientific logic, or replay evidence. The target schema
defines output structure; mapping specifications and registered rules define
harmonization behavior; pinned manifests and evidence define historical replay
authority. Strict replay never consults today's mutable profile to reinterpret
a historical event.

The public-safe current profile is
`profiles/clinical_harmonization.yaml`. Validate it without running
harmonization or replay:

```bash
./.venv/bin/python scripts/validate_project_profile.py \
  profiles/clinical_harmonization.yaml
```

Profile adoption has begun only for explicitly supported tooling:
`validate_harmonized_schema.py` accepts `--project-profile PATH` for current
validation defaults; `validate_spec.py` and `harmonize_spec.py` accept it for
the target-schema default only. There is no automatic or global profile discovery.
Explicit CLI arguments override profile defaults independently; omitted values
retain the command's existing defaults or requirements when no profile is
supplied. Broader command adoption remains separately scoped future work.
Harmonization semantics and historical replay authority remain independent of
ProjectProfile. A profile does not establish specification approval.

## Installation

The project currently runs directly from a checkout; it is not packaged for
installation as a library. Use a project-local Python 3.10-or-newer virtual
environment. Install runtime dependencies with:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

For development and testing, install the additional requirements:

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
```

## Usage

The examples below use placeholders and synthetic identifiers. They do not
refer to production clinical data.

### Validate a harmonization specification

Validate grammar and registered-rule references without opening declared source
files:

```bash
./.venv/bin/python scripts/validate_spec.py \
  --no-source-check path/to/approved-spec.yaml
```

Omit `--no-source-check` when the explicitly configured source files are
available and should also be validated. Use `--schema` to select a target schema
other than `schemas/cimac_clinical_v1.yaml`.

An explicitly supplied profile can provide only the target-schema default for
this command; explicit `--schema` wins:

```bash
./.venv/bin/python scripts/validate_spec.py \
  --project-profile profiles/clinical_harmonization.yaml \
  --no-source-check path/to/approved-spec.yaml
```

Specification paths remain required and explicit; profile-listed specs are not
selected as command inputs. A supplied profile still undergoes full strict,
source-free validation of its declared references. Source checking remains
enabled by default and is disabled only by `--no-source-check`.

### Execute a reviewed specification

```bash
./.venv/bin/python scripts/harmonize_spec.py \
  path/to/approved-spec.yaml \
  --schema path/to/target-schema.yaml \
  --output path/to/harmonized.csv \
  --provenance path/to/provenance.csv \
  --preflight
```

Strict replay publication additionally requires an explicit run directory,
run identity, provenance output, and artifact identity; see
`scripts/harmonize_spec.py --help` for the supported options.

For `harmonize_spec.py`, an explicit `--project-profile PATH` supplies only the
target-schema location when `--schema` is omitted. Explicit `--schema` wins;
the specification path remains required. The full strict profile loader may
validate other declared references, but anchor/config inputs, output/provenance
destinations and replay/identity controls keep their existing CLI behavior.
They are not profile-defaulted. The loaded schema and pinned evidence remain
authoritative; the current profile is not historical replay authority. Broader
execution-input adoption remains separately scoped future work.

### Validate a harmonized table

```bash
./.venv/bin/python scripts/validate_harmonized_schema.py \
  --schema path/to/target-schema.yaml \
  --artifact path/to/harmonized.csv
```

This read-only validator is the first tool to support explicit profile-backed
defaults. Supplying `--project-profile` may provide the target schema, output
directory, provenance filename, and report filename when the corresponding CLI
argument is omitted:

```bash
./.venv/bin/python scripts/validate_harmonized_schema.py \
  --project-profile profiles/clinical_harmonization.yaml \
  --all
```

Explicit CLI values always override profile defaults. Without
`--project-profile`, behavior is unchanged; the command never discovers or
loads a profile automatically.
Profile provenance and report filenames resolve beneath the selected output
directory, including an explicit `--outputs-dir` override. Use `--provenance ''`
or `--out ''` to disable the corresponding profile default. `--all` always uses
the artifact IDs declared by the target schema; profile roles and aliases do
not select artifacts in this slice.

### Inspect recorded provenance

```bash
./.venv/bin/python scripts/view_provenance.py \
  --provenance path/to/provenance.csv \
  --participant-id SYNTH-PARTICIPANT-001 \
  --field example_field
```

The project-oriented trace command can also verify a delivered value against
its recorded lineage and current bound source:

```bash
./.venv/bin/python scripts/trace_value.py \
  --artifact harmonized_11trials \
  --trial SYNTHETIC-TRIAL \
  --participant-id SYNTH-PARTICIPANT-001 \
  --field example_field
```

Tracing is read-only inspection tooling and is distinct from strict historical
replay.

## Testing

The test suite has three layers.

### Portable synthetic tests

The default test command uses repository code and fabricated fixtures only:

```bash
./.venv/bin/python -m pytest
```

### Source-free corpus acceptance

These checks inspect explicitly supplied schemas and specification structure but
do not follow clinical source paths:

```bash
./.venv/bin/python -m pytest -c pytest.ini tests -m corpus_acceptance \
  --corpus-root /path/to/declaratives \
  --corpus-plan /path/to/mapping-id-plan.yaml
```

### Opt-in production integration

Production integration tests require an enable flag and explicit external
locations. They are excluded from the default suite:

```bash
./.venv/bin/python -m pytest -c pytest.ini tests -m production_integration \
  --run-production-integration \
  --production-outputs-dir /path/to/external/outputs \
  --production-expectations /path/to/external/expectations.json
```

Production paths are never inferred from the checkout. See
[`tests/README.md`](tests/README.md) for the maintained commands.

## Data and privacy boundary

This repository does not include the clinical source datasets used in
production harmonization. It may include field names, schemas, mapping
structure, controlled vocabularies, project or trial metadata, and clearly
synthetic examples.

The public repository should not contain participant-level source records,
production harmonized outputs containing those records, or production
provenance containing clinical values. External source data and generated
production artifacts must remain outside version control.

## Reproducibility and limitations

- Technical replay does not prove that a mapping is clinically correct.
- Clinical semantics and scientific policy require qualified human review.
- Historical replay can pass only when the required pinned specification,
  source bindings, hashes, record identity, and other causal evidence are
  available. An evidence-incomplete event may correctly be reported as not
  replayable.
- Some named and versioned scientific rules are implemented in Python. Replay
  relies on stable rule identifiers, closed registry resolution, and controlled
  software versions; it does not currently hash registered-rule source code.
- Some compatibility and project-oriented tooling remains and is being
  generalized. Supporting a genuinely new operation, reader, or scientific rule
  may require a reviewed code change.
- Clinical source data, approved project specifications, and production replay
  bundles are managed outside this public repository.

## Development status

The strict replay engine has an implemented route for all **204 currently
approved mapping declaration shapes** in the present inventory. This is a
structural capability result. It does **not** mean that every historical event
has complete evidence, that production replay has been performed for every
event, that every mapping is clinically correct, or that every possible future
mapping construct is supported.

The current validation inventory grew out of clinical-trial harmonization work
in a CIMAC context. Those project identifiers and field definitions are
metadata; the framework itself is designed around reusable operation shapes,
explicit configuration, and generic execution rather than trial-specific
runtime branches.
