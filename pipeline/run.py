from pipeline.config import (
    BG_OUTPUT_COL,
    BP_SBP_COL,
    DRUG_ALLERGY_OUTPUT_COL,
    MERGED_AFTER_BP_NAME,
    MERGED_AFTER_LABS_NAME,
    MERGED_AFTER_PATIENT_NAME,
    MERGED_AFTER_SAT_NAME,
    MERGED_AFTER_SMOKING_NAME,
    MERGED_AFTER_SURGERY_MEDS_NAME,
    MERGED_SURGERIES_NAME,
    ML_DATASET_NAME,
    SAT_COL,
    SMOKING_COL,
    SMOKING_FILL_UNKNOWN,
    STAGED_DIR,
    SURGERY_MEDS_OUTPUT_COL,
)
from pipeline.extract import (
    extract_background_diseases,
    extract_bmi,
    extract_bp,
    extract_chronic_meds,
    extract_drug_allergy,
    extract_labs,
    extract_other_allergy,
    extract_saturation,
    extract_smoking,
    extract_surgeries,
    extract_surgery_meds,
)
from pipeline.load import save_staged
from pipeline.transform import (
    clean_background_diseases,
    clean_bmi,
    clean_bp,
    clean_chronic_meds,
    clean_drug_allergy,
    clean_labs,
    clean_other_allergy,
    clean_saturation,
    clean_smoking,
    clean_surgeries,
    clean_surgery_meds,
    encode_merged_categoricals,
    engineer_merged_features,
    export_ml_dataset_by_department,
    finalize_english_ml_dataset,
    impute_merged_bmi_metrics,
    merge_background_diseases,
    merge_bp,
    merge_drug_allergy,
    merge_labs,
    merge_other_allergy,
    merge_chronic_meds,
    merge_saturation,
    merge_surgeries_bmi,
    merge_surgeries_smoking,
    merge_surgery_meds,
    prepare_final_dataset,
)

import pandas as pd

def run_surgeries(conn=None, save=True):
    """Full ETL for surgeries table."""
    df = extract_surgeries(conn)
    print(f"extract: {df.shape}")

    df = clean_surgeries(df)
    print(f"clean:   {df.shape}")

    if save:
        save_staged(df, "surgeries")

    return df


def run_bmi(conn=None, save=True):
    """Full ETL for BMI table."""
    df = extract_bmi(conn)
    print(f"extract: {df.shape}")

    df = clean_bmi(df)
    print(f"clean:   {df.shape}")

    if save:
        save_staged(df, "bmi")

    return df


def run_merge_surgeries_bmi(surgeries=None, bmi=None, save=True):
    """Merge cleaned BMI into surgeries anchor table."""
    if surgeries is None:
        surgeries = pd.read_pickle(STAGED_DIR / "surgeries.pkl")
    if bmi is None:
        bmi = pd.read_pickle(STAGED_DIR / "bmi.pkl")

    merged = merge_surgeries_bmi(surgeries, bmi)
    print(f"merged:  {merged.shape}")
    print(f"BMI matched: {merged['BMI'].notna().sum():,} / {len(merged):,}")

    merged = impute_merged_bmi_metrics(merged)
    print(f"after impute — height nulls: {merged['גובה'].isna().sum():,}")
    print(f"after impute — weight nulls: {merged['משקל'].isna().sum():,}")
    print(f"after impute — BMI nulls:    {merged['BMI'].isna().sum():,}")

    if save:
        save_staged(merged, MERGED_SURGERIES_NAME)

    return merged


def run_smoking(conn=None, save=True):
    """Full ETL for smoking table."""
    df = extract_smoking(conn)
    print(f"extract: {df.shape}")

    df = clean_smoking(df)
    print(f"clean:   {df.shape}")

    if save:
        save_staged(df, "smoking")

    return df


def run_merge_surgeries_smoking(merged=None, smoking=None, save=True):
    """Merge cleaned smoking into surgeries+BMI table."""
    if merged is None:
        merged = pd.read_pickle(STAGED_DIR / MERGED_SURGERIES_NAME)
    if smoking is None:
        smoking = pd.read_pickle(STAGED_DIR / "smoking.pkl")

    merged = merge_surgeries_smoking(merged, smoking)
    print(f"merged:  {merged.shape}")
    n_matched = (merged[SMOKING_COL] != SMOKING_FILL_UNKNOWN).sum()
    print(f"Smoking matched: {n_matched:,} / {len(merged):,}")

    merged = encode_merged_categoricals(merged)
    print(f"after OHE: {merged.shape}")

    if save:
        save_staged(merged, MERGED_AFTER_SMOKING_NAME)

    return merged


def _run_clean_table(extract_fn, clean_fn, table_name, conn=None, save=True):
    df = extract_fn(conn)
    print(f"extract: {df.shape}")
    df = clean_fn(df)
    print(f"clean:   {df.shape}")
    if save:
        save_staged(df, table_name)
    return df


def run_bp(conn=None, save=True):
    return _run_clean_table(extract_bp, clean_bp, "bp", conn, save)


def run_saturation(conn=None, save=True):
    return _run_clean_table(extract_saturation, clean_saturation, "saturation", conn, save)


def run_background_diseases(conn=None, save=True):
    return _run_clean_table(extract_background_diseases, clean_background_diseases, "background_diseases", conn, save)


def run_drug_allergy(conn=None, save=True):
    return _run_clean_table(extract_drug_allergy, clean_drug_allergy, "drug_allergy", conn, save)


def run_other_allergy(conn=None, save=True):
    return _run_clean_table(extract_other_allergy, clean_other_allergy, "other_allergy", conn, save)


def run_chronic_meds(conn=None, save=True):
    return _run_clean_table(extract_chronic_meds, clean_chronic_meds, "chronic_meds", conn, save)


def run_surgery_meds(conn=None, save=True):
    return _run_clean_table(extract_surgery_meds, clean_surgery_meds, "surgery_meds", conn, save)


def run_merge_bp(merged=None, bp=None, save=True):
    if merged is None:
        merged = pd.read_pickle(STAGED_DIR / MERGED_AFTER_SMOKING_NAME)
    if bp is None:
        bp = pd.read_pickle(STAGED_DIR / "bp.pkl")
    merged = merge_bp(merged, bp)
    print(f"merged: {merged.shape}")
    print(f"BP matched: {merged[BP_SBP_COL].notna().sum():,} / {len(merged):,}")
    if save:
        save_staged(merged, MERGED_AFTER_BP_NAME)
    return merged


def run_merge_saturation(merged=None, saturation=None, save=True):
    if merged is None:
        merged = pd.read_pickle(STAGED_DIR / MERGED_AFTER_BP_NAME)
    if saturation is None:
        saturation = pd.read_pickle(STAGED_DIR / "saturation.pkl")
    merged = merge_saturation(merged, saturation)
    print(f"merged: {merged.shape}")
    print(f"saturation matched: {merged[SAT_COL].notna().sum():,} / {len(merged):,}")
    if save:
        save_staged(merged, MERGED_AFTER_SAT_NAME)
    return merged


def run_merge_patient_tables(
    merged=None,
    background=None,
    drug_allergy=None,
    save=True,
):
    if merged is None:
        merged = pd.read_pickle(STAGED_DIR / MERGED_AFTER_SAT_NAME)
    if background is None:
        background = pd.read_pickle(STAGED_DIR / "background_diseases.pkl")
    if drug_allergy is None:
        drug_allergy = pd.read_pickle(STAGED_DIR / "drug_allergy.pkl")

    merged = merge_background_diseases(merged, background)
    merged = merge_drug_allergy(merged, drug_allergy)

    print(f"merged: {merged.shape}")
    print(f"{BG_OUTPUT_COL} matched: {merged[BG_OUTPUT_COL].notna().sum():,} / {len(merged):,}")
    print(f"{DRUG_ALLERGY_OUTPUT_COL} matched: {merged[DRUG_ALLERGY_OUTPUT_COL].notna().sum():,} / {len(merged):,}")

    if save:
        save_staged(merged, MERGED_AFTER_PATIENT_NAME)
    return merged


def run_merge_surgery_meds(merged=None, surgery_meds=None, save=True):
    if merged is None:
        merged = pd.read_pickle(STAGED_DIR / MERGED_AFTER_PATIENT_NAME)
    if surgery_meds is None:
        surgery_meds = pd.read_pickle(STAGED_DIR / "surgery_meds.pkl")

    merged = merge_surgery_meds(merged, surgery_meds)
    merged = engineer_merged_features(merged)
    print(f"merged: {merged.shape}")
    print(f"{SURGERY_MEDS_OUTPUT_COL} matched: {merged[SURGERY_MEDS_OUTPUT_COL].notna().sum():,} / {len(merged):,}")

    if save:
        save_staged(merged, MERGED_AFTER_SURGERY_MEDS_NAME)
    return merged


def run_labs(conn=None, save=True):
    return _run_clean_table(extract_labs, clean_labs, "labs", conn, save)


def run_merge_labs(merged=None, labs=None, save=True):
    if merged is None:
        merged = pd.read_pickle(STAGED_DIR / MERGED_AFTER_SURGERY_MEDS_NAME)
    if labs is None:
        labs = pd.read_pickle(STAGED_DIR / "labs.pkl")

    merged = merge_labs(merged, labs)
    merged = prepare_final_dataset(merged)
    merged = finalize_english_ml_dataset(merged)
    print(f"merged: {merged.shape}")

    if save:
        save_staged(merged, MERGED_AFTER_LABS_NAME)
        save_staged(merged, ML_DATASET_NAME)
    return merged

def run_full_pipeline(conn=None, save=True):
    """Run the complete ETL pipeline and export department CSVs."""
    run_surgeries(conn=conn, save=save)
    run_bmi(conn=conn, save=save)
    merged = run_merge_surgeries_bmi(save=save)
    run_smoking(conn=conn, save=save)
    merged = run_merge_surgeries_smoking(merged=merged, save=save)
    run_bp(conn=conn, save=save)
    run_saturation(conn=conn, save=save)
    merged = run_merge_bp(merged=merged, save=save)
    merged = run_merge_saturation(merged=merged, save=save)
    run_background_diseases(conn=conn, save=save)
    run_drug_allergy(conn=conn, save=save)
    merged = run_merge_patient_tables(merged=merged, save=save)
    run_surgery_meds(conn=conn, save=save)
    merged = run_merge_surgery_meds(merged=merged, save=save)
    run_labs(conn=conn, save=save)
    merged = run_merge_labs(merged=merged, save=save)
    export_ml_dataset_by_department(merged)
    return merged
