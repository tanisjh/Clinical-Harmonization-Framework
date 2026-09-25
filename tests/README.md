# Test layers

The default suite is portable and uses only synthetic data plus repository
source:

    ./.venv/bin/python -m pytest

Source-free checks of an approved declarative corpus are separate and require
explicit metadata locations. They never follow source paths declared by specs:

    ./.venv/bin/python -m pytest -c pytest.ini tests -m corpus_acceptance \
      --corpus-root /path/to/declaratives \
      --corpus-plan /path/to/mapping-id-plan.yaml

Tests that inspect real delivery artifacts are opt-in and require both an
enable flag and explicit external paths:

    ./.venv/bin/python -m pytest -c pytest.ini tests -m production_integration \
      --run-production-integration \
      --production-outputs-dir /path/to/external/outputs \
      --production-expectations /path/to/external/expectations.json

Production paths and expectations are never inferred from the checkout. The
expectations JSON holds project-reviewed assertions such as trial row counts;
it is not a repository fixture. Do not place clinical files, generated
harmonized outputs, production provenance, or production expectation values in
this repository.
