"""Registry of trials migrated to reader-emitted source references.

One entry per migrated trial. This is the single source of truth for "which
trials are migrated": both scripts/audit_migrated_provenance.py and the test
suite read it, so adding a trial here extends the audit and the generic tests
together, and they cannot drift apart.

Per-trial test modules then only cover what is genuinely trial-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .record_keys import parse_record_key
from .trace_io import read_source


@dataclass(frozen=True)
class SourceSpec:
    config_key: str          # key in cfg['sources']
    logical_label: str       # stable label: config column_map key + provenance source_file
    cidc_path: str           # repo-relative path actually read
    legacy_path: str         # repo-relative same-named legacy file (the D2 hazard)
    #: Columns that together identify one record in THIS file. Empty means the
    #: trial default. Some files are participant-level and some are
    #: participant x timepoint — 10013's specimen_collection has 789 rows for 51
    #: participants, so cimac_part_id alone would not locate a record.
    record_key_columns: tuple[str, ...] = ()
    #: False when the key columns identify a GROUP of source rows rather than
    #: one row — 10013's specimen_collection is one row per specimen, so
    #: (cimac_part_id, M7) matches several. Such a source must record the
    #: multiplicity and which row was selected, rather than silently taking the
    #: first and presenting it as the record.
    record_key_is_unique: bool = True
    #: Excel only: the sheet the extractor reads. The config key may differ from
    #: the workbook identity (E4412's column_map is keyed "baseline_outcomes::
    #: Sheet1" while the provenance label is "baseline_outcomes.xlsx"), so both
    #: are recorded rather than conflated.
    sheet: str = ""

    def key_columns(self, trial_default: str) -> tuple[str, ...]:
        return self.record_key_columns or (trial_default,)


@dataclass(frozen=True)
class TrialSpec:
    trial: str                       # template trial name, as in provenance
    extractor: str                   # module basename
    sources: tuple[SourceSpec, ...]
    record_key_column: str = "cimac_part_id"
    # True when the legacy file holds a DIFFERENT participant at the same row
    # index, so a basename-resolved trace returns another patient's value.
    legacy_rows_misaligned: bool = True
    #: False when the trial has NOT been repointed to the CIDC bucket, so the
    #: file it reads IS the file in the legacy directory. There is no stale-label
    #: hazard for such a trial — and no CIDC copy to drift from — but a future
    #: repoint would reintroduce exactly defect D1, so the un-repointed state is
    #: pinned by a test rather than assumed.
    repointed: bool = True
    notes: str = ""


MIGRATED: tuple[TrialSpec, ...] = (
    TrialSpec(
        trial="CIMAC-s1400i",
        extractor="s1400i",
        sources=(
            SourceSpec(
                config_key="clinical",
                logical_label="Clinical Dataset 2023_03_14.csv",
                cidc_path="cimac-cidc-x001/S1400I/xt_S1400I_clinical_Clinical Dataset 2023_03_14.csv",
                legacy_path="S1400I-clinical/Clinical Dataset 2023_03_14.csv",
            ),
        ),
        legacy_rows_misaligned=False,
        notes=("legacy file is a 252x24 superset of the 160x18 CIDC file whose "
               "overlapping rows coincide; the hazard here is out-of-range row "
               "numbers and extra columns, not wrong-patient reads"),
    ),
    TrialSpec(
        trial="10026",
        extractor="nci_10026",
        sources=(
            SourceSpec(
                config_key="demographics",
                logical_label="demographics_04282024.csv",
                cidc_path="cimac-cidc-x001/10026/xt_10026_clinical_demographics_04282024.csv",
                legacy_path="10026-clinical/demographics_04282024.csv",
            ),
            SourceSpec(
                config_key="response",
                logical_label="response_04282024.csv",
                cidc_path="cimac-cidc-x001/10026/xt_10026_clinical_response_04282024.csv",
                legacy_path="10026-clinical/response_04282024.csv",
            ),
        ),
        notes=("45/48 demographics rows and 41/42 response rows hold a different "
               "participant at the same row index in the legacy file"),
    ),
    TrialSpec(
        trial="CIMAC-10021",
        extractor="nci_10021",
        sources=(
            SourceSpec(
                config_key="all_patient",
                logical_label="10021_AllPatientData.2023-06-20.csv",
                cidc_path="cimac-cidc-x001/10021/xt_10021_clinical_10021_AllPatientData.2023-06-20.csv",
                legacy_path="10021-clinical/10021_AllPatientData.2023-06-20.csv",
            ),
        ),
        notes="28/94 row indices hold a different participant in the legacy copy",
    ),
    TrialSpec(
        trial="CIMAC-gu16257",
        extractor="gu16_257",
        sources=(
            SourceSpec(
                config_key="demographics",
                logical_label="patient_demographics.2023-01-04.csv",
                cidc_path="cimac-cidc-x001/GU16-257/xt_GU16-257_clinical_patient_demographics.2023-01-04.csv",
                legacy_path="GU16-257-clinical/patient_demographics.2023-01-04.csv",
            ),
            SourceSpec(
                config_key="response",
                logical_label="response.2023-01-04.csv",
                cidc_path="cimac-cidc-x001/GU16-257/xt_GU16-257_clinical_response.2023-01-04.csv",
                legacy_path="GU16-257-clinical/response.2023-01-04.csv",
            ),
        ),
        notes="first multi-candidate fallback case (pfs_time: DFSTIM -> DRFSTIM)",
    ),
    TrialSpec(
        trial="ABTC1603",
        extractor="abtc1603",
        sources=(
            SourceSpec(
                config_key="demographics",
                logical_label="abtc_1603_demographic.demographics_2024-04-17.csv",
                cidc_path="cimac-cidc-x001/ABTC-1603/xt_ABTC1603_clinical_abtc_1603_demographic.demographics_2024-04-17.csv",
                legacy_path="ABTC1603-clinical/abtc_1603_demographic.demographics_2024-04-17.csv",
            ),
            SourceSpec(
                config_key="response",
                logical_label="abtc_1603_treatmentresponse_03042024_2024-04-17.csv",
                cidc_path="cimac-cidc-x001/ABTC-1603/xt_ABTC1603_clinical_abtc_1603_treatmentresponse_03042024_2024-04-17.csv",
                legacy_path="ABTC1603-clinical/abtc_1603_treatmentresponse_03042024_2024-04-17.csv",
            ),
        ),
        legacy_rows_misaligned=False,
        notes=("legacy copies are row-aligned (41 rows, same participant order) but "
               "carry extra columns (11 vs 6, 12 vs 11); the hazard is column drift "
               "rather than wrong-patient reads. First trial with conditional "
               "composites and a post-pivot override."),
    ),
    TrialSpec(
        trial="CIMAC-9204",
        extractor="nci_9204",
        sources=(
            SourceSpec(
                config_key="demographics_ipi",
                logical_label="demographics_dose_level.ipilimumab_2024-05-01.csv",
                cidc_path="cimac-cidc-x001/9204/xt_9204_clinical_demographics_dose_level.ipilimumab_2024-05-01.csv",
                legacy_path="9204-clinical/demographics_dose_level.ipilimumab_2024-05-01.csv",
            ),
            SourceSpec(
                config_key="demographics_nivo",
                logical_label="demographics_dose_level.nivolumab_2024-05-01.csv",
                cidc_path="cimac-cidc-x001/9204/xt_9204_clinical_demographics_dose_level.nivolumab_2024-05-01.csv",
                legacy_path="9204-clinical/demographics_dose_level.nivolumab_2024-05-01.csv",
            ),
            SourceSpec(
                config_key="response_ipi",
                logical_label="best_response.ipi_2024-05-01.csv",
                cidc_path="cimac-cidc-x001/9204/xt_9204_clinical_best_response.ipi_2024-05-01.csv",
                legacy_path="9204-clinical/best_response.ipi_2024-05-01.csv",
            ),
            SourceSpec(
                config_key="response_nivo",
                logical_label="best_response.nivo_2024-05-01.csv",
                cidc_path="cimac-cidc-x001/9204/xt_9204_clinical_best_response.nivo_2024-05-01.csv",
                legacy_path="9204-clinical/best_response.nivo_2024-05-01.csv",
            ),
        ),
        legacy_rows_misaligned=False,
        notes=("four per-arm files; `treatment` is decided by WHICH FILE the "
               "participant appears in, so provenance records membership evidence "
               "rather than a fabricated source column"),
    ),
    TrialSpec(
        trial="10013",
        extractor="nci_10013",
        sources=(
            SourceSpec(
                config_key="demographics",
                logical_label="demographics_2023-09-13.csv",
                cidc_path="cimac-cidc-x001/10013/xt_10013_clinical_demographics_2023-09-13.csv",
                legacy_path="10013-clinical/demographics_2023-09-13.csv",
            ),
            SourceSpec(
                config_key="sample_coll",
                logical_label="specimen_collection_2023-09-13.csv",
                cidc_path="cimac-cidc-x001/10013/xt_10013_clinical_specimen_collection_2023-09-13.csv",
                legacy_path="10013-clinical/specimen_collection_2023-09-13.csv",
                # one row per SPECIMEN: 789 rows, 196 (participant, timepoint)
                # groups, so the anchor key matches several source rows
                record_key_columns=("cimac_part_id", "M7"),
                record_key_is_unique=False,
            ),
            SourceSpec(
                config_key="response",
                logical_label="response_updated_2024-11-07.csv",
                cidc_path="cimac-cidc-x001/10013/xt_10013_clinical_response_updated_2024-11-07.csv",
                legacy_path="10013-clinical/response_updated_2024-11-07.csv",
            ),
            SourceSpec(
                config_key="treatment",
                logical_label="treatment_2023-09-13.csv",
                cidc_path="cimac-cidc-x001/10013/xt_10013_clinical_treatment_2023-09-13.csv",
                legacy_path="10013-clinical/treatment_2023-09-13.csv",
            ),
        ),
        legacy_rows_misaligned=False,
        notes=("anchor-constructing trial: rows are built from specimen_collection "
               "M7 timepoints rather than template anchors. Four-file join; "
               "clinical_benefit comes from the per-patient pCR column (RAE C6)."),
    ),
    TrialSpec(
        trial="14C0059G",
        extractor="nih_14c0059g",
        sources=(
            SourceSpec(
                config_key="demographics",
                logical_label="patient_demographics_all.csv",
                cidc_path="cimac-cidc-x001/14C0059G/xt_14C0059G_clinical_patient_demographics_all.csv",
                legacy_path="14C0059G-clinical/patient_demographics_all.csv",
            ),
            SourceSpec(
                config_key="sample_coll",
                logical_label="research_sample_collection_apheresis.csv",
                cidc_path="cimac-cidc-x001/14C0059G/xt_14C0059G_clinical_research_sample_collection_apheresis.csv",
                legacy_path="14C0059G-clinical/research_sample_collection_apheresis.csv",
                record_key_columns=("cimac_part_id", "Visit"),
            ),
            SourceSpec(
                config_key="survival_death",
                logical_label="response_off_treatment_date_of_death.csv",
                cidc_path="cimac-cidc-x001/14C0059G/xt_14C0059G_clinical_response_off_treatment_date_of_death.csv",
                legacy_path="14C0059G-clinical/response_off_treatment_date_of_death.csv",
            ),
            SourceSpec(
                config_key="treatment",
                logical_label="treatment.csv",
                cidc_path="cimac-cidc-x001/14C0059G/xt_14C0059G_clinical_treatment.csv",
                legacy_path="14C0059G-clinical/treatment.csv",
            ),
        ),
        legacy_rows_misaligned=False,
        notes=("anchor-constructing, with SYNTHESIZED baseline anchors for "
               "participants absent from sample collection. Survival is "
               "PROVISIONAL pending trial-team delivery (RAE C11); age and phase "
               "have no source column at all."),
    ),
    TrialSpec(
        trial="CIMAC-e4412",
        extractor="e4412",
        sources=(
            SourceSpec(
                config_key="baseline_outcomes",
                logical_label="baseline_outcomes.xlsx",
                # NOT repointed: the CIDC copy is structurally incomplete
                # (RAE C1), so the file read IS the one in the trial directory.
                cidc_path="E4412-clinical/baseline_outcomes.xlsx",
                legacy_path="E4412-clinical/baseline_outcomes.xlsx",
                record_key_columns=("CIMAC ID",),
                sheet="Sheet1",
            ),
        ),
        record_key_column="CIMAC ID",
        legacy_rows_misaligned=False,
        repointed=False,
        notes=("the only Excel source. Config column_map is keyed "
               "'baseline_outcomes::Sheet1' while the provenance label is the "
               "workbook name; workbook identity and sheet are recorded "
               "separately rather than conflated."),
    ),
    TrialSpec(
        trial="EAY131_Z1D",
        extractor="eay131_z1d",
        sources=(
            SourceSpec(
                config_key="demographics",
                logical_label="demographics_2023-09-25.csv",
                cidc_path="EAY131-Z1D-clinical/demographics_2023-09-25.csv",
                legacy_path="EAY131-Z1D-clinical/demographics_2023-09-25.csv",
            ),
            SourceSpec(
                config_key="response",
                logical_label="response_2023-09-25.csv",
                cidc_path="EAY131-Z1D-clinical/response_2023-09-25.csv",
                legacy_path="EAY131-Z1D-clinical/response_2023-09-25.csv",
            ),
        ),
        legacy_rows_misaligned=False,
        repointed=False,
        notes=("INERT CONTROL: not repointed because its CIDC files have not "
               "been delivered (RAE C2, open item Q4). The logical label already "
               "named the file being read, so migration adds structure without "
               "correcting a stale reference. Treatment identity is unresolved "
               "(open item Q5) and is recorded, not changed."),
    ),
    TrialSpec(
        trial="10104",
        extractor="nci_10104",
        sources=(
            SourceSpec(
                config_key="demographics",
                logical_label="10104_demographics.2023-04-04.csv",
                cidc_path="cimac-cidc-x001/10104/xt_10104_clinical_10104_demographics.2023-04-04.csv",
                legacy_path="10104-clinical/10104_demographics.2023-04-04.csv",
            ),
            SourceSpec(
                config_key="response_aandb",
                logical_label="10104_armaandb_response_pfsos_treatment_update.2023-04-04.csv",
                cidc_path="cimac-cidc-x001/10104/xt_10104_clinical_10104_armaandb_response_pfsos_treatment_update.2023-04-04.csv",
                legacy_path="10104-clinical/10104_armaandb_response_pfsos_treatment_update.2023-04-04.csv",
            ),
            SourceSpec(
                config_key="response_armc",
                logical_label="10104_clinical_armc_response_pfsos_treatment_update16mar2023.03262024.csv",
                cidc_path="cimac-cidc-x001/10104/xt_10104_clinical_10104_clinical_armc_response_pfsos_treatment_update16mar2023.03262024.csv",
                legacy_path="10104-clinical/10104_clinical_armc_response_pfsos_treatment_update16mar2023.03262024.csv",
            ),
            SourceSpec(
                config_key="arm_aandb",
                logical_label="10104_armaandb_enrollment_assignment.2023-04-04.csv",
                cidc_path="cimac-cidc-x001/10104/xt_10104_clinical_10104_armaandb_enrollment_assignment.2023-04-04.csv",
                legacy_path="10104-clinical/10104_armaandb_enrollment_assignment.2023-04-04.csv",
            ),
            SourceSpec(
                config_key="arm_armc",
                logical_label="10104_clinical_armc_subgroup_updated.03262024.csv",
                cidc_path="cimac-cidc-x001/10104/xt_10104_clinical_10104_clinical_armc_subgroup_updated.03262024.csv",
                legacy_path="10104-clinical/10104_clinical_armc_subgroup_updated.03262024.csv",
            ),
        ),
        legacy_rows_misaligned=False,
        notes=("five files. Response and arm-origin routing are both file "
               "membership, and the Arm A/B file is consulted FIRST so a "
               "crossover participant keeps their randomized ARM B. Survival "
               "comes from a per-arm file whose column name states the anchor; "
               "the crossover anchor question (Q3) is resolved in favour of the "
               "Arm A/B first_cycle_first_day values, which is what is read."),
    ),
)

MIGRATED_TRIAL_NAMES = tuple(t.trial for t in MIGRATED)

# Sentinel prefixes that legitimately are not file references.
NON_FILE_PREFIXES = ("CONFIG:", "DERIVED:", "TEMPLATE_ANCHOR", "(")


def spec_for(trial: str) -> TrialSpec:
    for t in MIGRATED:
        if t.trial == trial:
            return t
    raise KeyError(f"{trial} is not a migrated trial")


def is_file_reference(source_file: str) -> bool:
    """True when a provenance source_file names a real file rather than a sentinel."""
    s = str(source_file)
    return not s.startswith(NON_FILE_PREFIXES) and "." in s
