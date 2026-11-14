"""Shared Cell-v1 pivot with preserved candidate and threshold semantics.

Identity names remain the Cell-v1 compatibility shape. Schema columns control
projection; they do not infer identity adapters or artifact authority.
"""
from __future__ import annotations

import pandas as pd

from lib.provenance import Cell, FlagRow
from lib.schema_model import TargetContract


def threshold_for(field: str, thresholds_cfg: dict) -> float:
    return float(thresholds_cfg.get("per_field", {}).get(field, thresholds_cfg.get("default", 0.7)))


def pivot_cells_to_wide(cells: list[Cell], thresholds_cfg: dict,
                        contract: TargetContract) -> tuple[pd.DataFrame, list[FlagRow], list[dict]]:
    """
    Pivot long Cell stream to wide target-shaped DataFrame.
    For each (trial, cimac_part_id, Cimac.id, Collection_Event, harmonized_field):
      - if multiple Cells, take the one with highest confidence
      - if confidence < threshold, write NA and emit a FlagRow

    `contract` carries the output columns and their order, read from the target
    schema. The writer has no column list of its own.
    """
    rows: dict[tuple, dict] = {}
    best_by_key: dict[tuple, Cell] = {}
    for c in cells:
        key = (c.trial, c.cimac_part_id, c.Cimac_id, c.Collection_Event, c.harmonized_field)
        if key not in best_by_key or c.confidence > best_by_key[key].confidence:
            best_by_key[key] = c

    flags: list[FlagRow] = []
    suppressed: list[dict] = []
    for key, c in best_by_key.items():
        anchor = (c.trial, c.cimac_part_id, c.Cimac_id, c.Collection_Event)
        if anchor not in rows:
            rows[anchor] = {
                "cimac_part_id": c.cimac_part_id,
                "Cimac.id": c.Cimac_id,
                "Collection_Event": c.Collection_Event,
                "trial": c.trial,
            }
        thr = threshold_for(c.harmonized_field, thresholds_cfg)
        if c.confidence + 1e-9 >= thr and c.value is not None:
            rows[anchor][c.harmonized_field] = c.value
        else:
            rows[anchor].setdefault(c.harmonized_field, None)
            # A value below threshold is WITHHELD, not absent. Record which,
            # unless the extractor found nothing in the first place.
            if c.value is not None:
                suppressed.append({
                    "trial": c.trial, "cimac_part_id": c.cimac_part_id,
                    "Cimac.id": c.Cimac_id, "Collection_Event": c.Collection_Event,
                    "harmonized_field": c.harmonized_field,
                    "confidence": float(c.confidence), "threshold": float(thr),
                    "proposed": c.value, "input_rule_id": c.rule_id,
                })
            flags.append(FlagRow(
                trial=c.trial,
                cimac_part_id=c.cimac_part_id,
                Cimac_id=c.Cimac_id,
                Collection_Event=c.Collection_Event,
                harmonized_field=c.harmonized_field,
                source_files=c.source_file,
                candidate_source_variables=c.source_column,
                observed_source_values=c.notes,
                proposed_mapping=str(c.value) if c.value is not None else "",
                confidence_score=c.confidence,
                reason_low_confidence=(
                    "confidence_below_threshold"
                    if c.value is not None else "value_NA_at_extraction"
                ) + f" (threshold={thr:.2f})",
                question_for_reviewer=(
                    f"Is the proposed mapping for {c.harmonized_field} ({c.value!r}) correct given "
                    f"source={c.source_file}:{c.source_column}? extraction_method={c.extraction_method}"
                ),
            ))

    wide = pd.DataFrame(list(rows.values()))
    # Ensure every column the target schema declares exists; fill missing with NaN.
    for col in contract.columns:
        if col not in wide.columns:
            wide[col] = pd.NA
    missing_required = [c for c in contract.required if c not in wide.columns]
    if missing_required:                                 # unreachable via the loop
        raise ValueError(f"target contract requires columns the writer could not "
                         f"produce: {missing_required}")
    wide = wide[list(contract.columns)]
    return wide, flags, suppressed
