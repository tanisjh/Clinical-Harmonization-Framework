#!/usr/bin/env python
"""Read-only validation for mutable current-tooling project profiles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.project_profile import ProjectProfileError, load  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate mutable project-tooling settings without running "
            "harmonization or replay."
        )
    )
    parser.add_argument("profile", help="project-root-relative profile YAML")
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="explicit root for profile references (default: repository root)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    try:
        profile = load(args.profile, project_root=args.project_root)
    except ProjectProfileError as exc:
        print(f"PROFILE ERROR  {exc}", file=sys.stderr)
        return 1

    result = {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_sha256": profile.sha256,
        "target_schema_reference": profile.target_schema,
        "target_schema_identity": profile.target_schema_identity,
        "validated_specification_count": len(profile.validated_specifications),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"PASS  {profile.ref}")
        print(f"  profile SHA-256  {profile.sha256}")
        print(f"  target schema    {profile.target_schema}")
        print(f"  schema identity  {profile.target_schema_identity}")
        print(f"  approved specs   {len(profile.validated_specifications)}")
        print("  authority        current tooling only; never historical replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
