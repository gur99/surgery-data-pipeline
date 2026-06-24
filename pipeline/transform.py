from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd

from pipeline.config import *

def map_parent_department(dept_series):
    """אורטופדיה א/ב -> אורטופדיה, כירורגיה א/ב -> כירורגיה, אף אוזן גרון נשאר."""
    dept = dept_series.str.strip()
    return dept.str.replace(r" [אב]$", "", regex=True)


def join_if_many(series):
    """ערך יחיד -> נשמר כמו שהוא. כמה ערכים -> מחרוזת מופרדת בפסיק."""
    values = series.dropna()
    if len(values) == 0:
        return pd.NA
    unique = values.unique()
    if len(unique) == 1:
        return unique[0]
    return ",".join(str(v) for v in unique)


def group_to_one_row_per_surgery(df, group_keys):
    """שורה אחת לכל (מספר מקרה, Medical Record)."""
    other_cols = [c for c in df.columns if c not in group_keys]
    agg = {col: join_if_many for col in other_cols}
    return df.groupby(group_keys, as_index=False).agg(agg)


def build_datetime(date_series, time_series):
    """Combine date + time into one pandas datetime."""
    date_part = pd.to_datetime(date_series, errors="coerce")
    time_part = pd.to_timedelta(time_series.astype(str), errors="coerce")
    return date_part + time_part


def _is_consistent_with_others(ts, others, max_gap):
    """True if ts is not null and within max_gap of every other non-null timestamp."""
    if pd.isna(ts):
        return False
    for other in others:
        if pd.isna(other):
            continue
        if abs(ts - other) > max_gap:
            return False
    return True


def choose_consistent_start(entry, anes_start, surg_start, max_gap_hours=2):
    """Pick earliest valid OR start timestamp with cross-check against other starts."""
    max_gap = pd.Timedelta(hours=max_gap_hours)
    selected = pd.Series(pd.NaT, index=entry.index, dtype="datetime64[ns]")
    source = pd.Series("missing_start", index=entry.index, dtype=object)

    candidates = [
        ("entry_to_or", entry),
        ("anesthesia_start", anes_start),
        ("surgery_start", surg_start),
    ]

    for i in entry.index:
        row_vals = [(name, candidates[j][1].loc[i]) for j, (name, _) in enumerate(candidates)]
        if not any(not pd.isna(ts) for _, ts in row_vals):
            continue

        picked = False
        for name, ts in row_vals:
            others = [t for n, t in row_vals if n != name]
            if _is_consistent_with_others(ts, others, max_gap):
                selected.loc[i] = ts
                source.loc[i] = name
                picked = True
                break

        if not picked:
            source.loc[i] = "invalid_start"

    return selected, source


def choose_consistent_end(exit_or, anes_end, surg_end, discharge, max_gap_hours=2, discharge_close_minutes=30):
    """Pick latest valid OR end timestamp; discharge only if close to selected end."""
    max_gap = pd.Timedelta(hours=max_gap_hours)
    close_window = pd.Timedelta(minutes=discharge_close_minutes)
    selected = pd.Series(pd.NaT, index=exit_or.index, dtype="datetime64[ns]")
    source = pd.Series("missing_end", index=exit_or.index, dtype=object)

    candidates = [
        ("exit_from_or", exit_or),
        ("anesthesia_end", anes_end),
        ("surgery_end", surg_end),
    ]

    for i in exit_or.index:
        row_vals = [(name, candidates[j][1].loc[i]) for j, (name, _) in enumerate(candidates)]
        disc = discharge.loc[i]

        if not any(not pd.isna(ts) for _, ts in row_vals) and pd.isna(disc):
            continue

        base_ts = pd.NaT
        base_src = "missing_end"
        for name, ts in row_vals:
            others = [t for n, t in row_vals if n != name]
            if _is_consistent_with_others(ts, others, max_gap):
                base_ts = ts
                base_src = name
                break

        if pd.isna(base_ts):
            if any(not pd.isna(ts) for _, ts in row_vals):
                source.loc[i] = "invalid_end"
            continue

        if not pd.isna(disc) and abs(disc - base_ts) <= close_window:
            selected.loc[i] = disc
            source.loc[i] = "discharge_close_to_end"
        else:
            selected.loc[i] = base_ts
            source.loc[i] = base_src

    return selected, source


def _classify_duration(start_dt, end_dt, duration_min):
    if pd.isna(start_dt):
        return "missing_start"
    if pd.isna(end_dt):
        return "missing_end"
    if pd.isna(duration_min) or duration_min <= 0:
        return "negative_or_zero_duration"
    if duration_min > 24 * 60:
        return "over_24h"
    if duration_min > 12 * 60:
        return "suspicious_over_12h"
    return "valid"


def calculate_total_surgery_time(df):
    """OR occupancy time in minutes + quality flags. Does not drop rows."""
    df = df.copy()

    # 1. build all datetime columns
    df["entry_to_or_datetime"] = build_datetime(df[ENTRY_DATE_COL], df[ENTRY_TIME_COL])
    df["anesthesia_start_datetime"] = build_datetime(df[ANES_START_DATE], df[ANES_START_TIME])
    df["surgery_start_datetime"] = build_datetime(df[SURG_START_DATE], df[SURG_START_TIME])
    df["exit_from_or_datetime"] = build_datetime(df[EXIT_DATE_COL], df[EXIT_TIME_COL])
    df["anesthesia_end_datetime"] = build_datetime(df[ANES_END_DATE], df[ANES_END_TIME])
    df["surgery_end_datetime"] = build_datetime(df[SURG_END_DATE], df[SURG_END_TIME])
    df["discharge_datetime"] = build_datetime(df[DISC_DATE_COL], df[DISC_TIME_COL])

    # 2. pick consistent start / end
    start_dt, start_src = choose_consistent_start(
        df["entry_to_or_datetime"],
        df["anesthesia_start_datetime"],
        df["surgery_start_datetime"],
        max_gap_hours=MAX_START_GAP_HOURS,
    )
    end_dt, end_src = choose_consistent_end(
        df["exit_from_or_datetime"],
        df["anesthesia_end_datetime"],
        df["surgery_end_datetime"],
        df["discharge_datetime"],
        max_gap_hours=MAX_END_GAP_HOURS,
        discharge_close_minutes=DISCHARGE_CLOSE_MINUTES,
    )

    df["selected_start_datetime"] = start_dt
    df["selected_end_datetime"] = end_dt
    df["start_time_source"] = start_src
    df["end_time_source"] = end_src

    # 3. duration in minutes (midnight crossing handled by full datetimes)
    duration_min = (end_dt - start_dt).dt.total_seconds() / 60

    quality = []
    total = []
    for s, e, d in zip(start_dt, end_dt, duration_min):
        q = _classify_duration(s, e, d)
        quality.append(q)
        if q in ("missing_start", "missing_end", "negative_or_zero_duration", "over_24h"):
            total.append(pd.NA)
        else:
            total.append(d)

    df["duration_quality"] = quality
    df[TARGET_COL] = pd.to_numeric(total, errors="coerce")

    return df


def handle_surgeries_missing_values(df):
    """Fill categoricals; drop rows missing target or age."""
    df = df.copy()

    for col in SURGERIES_FILL_UNKNOWN:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    for col in SURGERIES_FILL_NON:
        if col in df.columns:
            df[col] = df[col].fillna("NON")

    for col in SURGERIES_DROP_IF_MISSING:
        if col in df.columns:
            df = df[df[col].notna()]

    return df.reset_index(drop=True)


def encode_procedure_side_value(val):
    """0.5 = Right/Left only, 1 = BOTH or Right+Left, else 0."""
    if pd.isna(val):
        return 0.0
    text = str(val).strip().upper()
    if not text or text == "NON":
        return 0.0
    if text == "BOTH":
        return 1.0

    parts = [p.strip() for p in text.split(",") if p.strip()]
    sides = set()
    for part in parts:
        if part == "BOTH":
            return 1.0
        if part in ("RIGHT", "LEFT"):
            sides.add(part)

    if "RIGHT" in sides and "LEFT" in sides:
        return 1.0
    if sides == {"RIGHT"} or sides == {"LEFT"}:
        return 0.5
    return 0.0


def encode_procedure_side(series):
    """Encode procedure side column to 0 / 0.5 / 1."""
    return series.apply(encode_procedure_side_value).astype("float64")


def drop_surgeries_timestamp_cols(df):
    """Remove all timestamp columns except surgery date."""
    cols_to_drop = [
        c for c in SURGERIES_RAW_TIMESTAMP_COLS + SURGERIES_COMPUTED_DATETIME_COLS
        if c in df.columns and c != SURGERY_DATE_COL
    ]
    return df.drop(columns=cols_to_drop)


def cast_surgeries_int_cols(df):
    """Convert key numeric columns to int."""
    df = df.copy()
    for col in SURGERIES_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("int64")
    return df


def clean_surgeries(df):
    """ניקוי טבלת ניתוחים."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # 1. הסרת עמודות
    cols_to_drop = [c for c in SURGERIES_COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. מיפוי מחלקה מנתחת ל-3 מחלקות אב
    df["מחלקה מנתחת"] = map_parent_department(df["מחלקה מנתחת"])

    # 3. זמן כולל בחדר ניתוח (לפני קיבוץ)
    df = calculate_total_surgery_time(df)

    # 4. שורה אחת לכל ניתוח
    df = group_to_one_row_per_surgery(df, SURGERIES_GROUP_KEYS)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # 5. ערכים חסרים + הסרת שורות ללא target / גיל
    df = handle_surgeries_missing_values(df)

    # 5b. קידוד צד פרוצדורה: Right/Left=0.5, BOTH/שני צדדים=1, אחרת=0
    if PROCEDURE_SIDE_COL in df.columns:
        df[PROCEDURE_SIDE_COL] = encode_procedure_side(df[PROCEDURE_SIDE_COL])

    # 6. הסרת חותמות זמן (מלבד תאריך ניתוח)
    df = drop_surgeries_timestamp_cols(df)

    # 7. drop חדר + המרה ל-int
    extra_drop = [c for c in SURGERIES_COLS_TO_DROP_EXTRA if c in df.columns]
    df = df.drop(columns=extra_drop)

    df = cast_surgeries_int_cols(df)

    return df


def clean_bmi(df):
    """ניקוי טבלת BMI."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Patient" in df.columns:
        df = df.rename(columns={"Patient": "patient"})

    cols_to_drop = [c for c in BMI_COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df["מספר מקרה"] = pd.to_numeric(df["מספר מקרה"], errors="coerce")
    df["תאריך ניתוח"] = pd.to_datetime(df["תאריך ניתוח"], errors="coerce")

    # drop rows without merge keys
    df = df[df["מספר מקרה"].notna() & df["תאריך ניתוח"].notna()]

    # keep last row per (מספר מקרה, תאריך ניתוח)
    df = df.drop_duplicates(subset=BMI_GROUP_KEYS, keep="last")
    df["מספר מקרה"] = df["מספר מקרה"].round().astype("int64")

    return df.reset_index(drop=True)


def merge_surgeries_bmi(surgeries, bmi):
    """Left join BMI onto surgeries by (מספר מקרה, תאריך ניתוח)."""
    surgeries = surgeries.copy()
    bmi = bmi.copy()

    surgeries["תאריך ניתוח"] = pd.to_datetime(surgeries["תאריך ניתוח"], errors="coerce")

    bmi_cols = [c for c in bmi.columns if c not in BMI_GROUP_KEYS and c != "patient"]
    bmi_side = bmi[BMI_GROUP_KEYS + bmi_cols]

    return surgeries.merge(bmi_side, on=BMI_GROUP_KEYS, how="left")


def clean_smoking(df):
    """ניקוי טבלת עישון."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Patient" in df.columns:
        df = df.rename(columns={"Patient": "patient"})

    df["מספר מקרה"] = pd.to_numeric(df["מספר מקרה"], errors="coerce")
    df["מועד ניתוח"] = pd.to_datetime(df["מועד ניתוח"], errors="coerce").dt.normalize()

    # remove rows without case number
    df = df[df["מספר מקרה"].notna()]

    # keep last per (מספר מקרה, מועד ניתוח)
    df = df.drop_duplicates(subset=SMOKING_GROUP_KEYS, keep="last")
    df["מספר מקרה"] = df["מספר מקרה"].round().astype("int64")

    if SMOKING_COL in df.columns:
        df[SMOKING_COL] = df[SMOKING_COL].fillna(SMOKING_FILL_UNKNOWN)

    return df.reset_index(drop=True)


def merge_surgeries_smoking(merged, smoking):
    """Left join smoking onto surgeries anchor (תאריך ניתוח ↔ מועד ניתוח)."""
    merged = merged.copy()
    smoking = smoking.copy()

    merged[SURGERY_DATE_COL] = pd.to_datetime(merged[SURGERY_DATE_COL], errors="coerce").dt.normalize()
    smoking["מועד ניתוח"] = pd.to_datetime(smoking["מועד ניתוח"], errors="coerce").dt.normalize()

    smoking_cols = [
        c for c in smoking.columns
        if c not in SMOKING_GROUP_KEYS and c != "patient"
    ]
    smoking_side = smoking[SMOKING_GROUP_KEYS + smoking_cols]

    merged = merged.merge(
        smoking_side,
        left_on=["מספר מקרה", SURGERY_DATE_COL],
        right_on=["מספר מקרה", "מועד ניתוח"],
        how="left",
    )
    merged = merged.drop(columns=["מועד ניתוח"], errors="ignore")
    merged[SMOKING_COL] = merged[SMOKING_COL].fillna(SMOKING_FILL_UNKNOWN)

    return merged


def prepare_case_date_table(df, case_col_raw):
    """Rename case column, parse keys, drop rows without מספר מקרה."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Patient" in df.columns:
        df = df.rename(columns={"Patient": "patient"})

    if case_col_raw in df.columns and case_col_raw != "מספר מקרה":
        df = df.rename(columns={case_col_raw: "מספר מקרה"})

    df["מספר מקרה"] = pd.to_numeric(df["מספר מקרה"], errors="coerce")
    df["תאריך ניתוח"] = pd.to_datetime(df["תאריך ניתוח"], errors="coerce").dt.normalize()
    df = df[df["מספר מקרה"].notna() & df["תאריך ניתוח"].notna()]
    df["מספר מקרה"] = df["מספר מקרה"].round().astype("int64")
    return df.reset_index(drop=True)


def merge_case_date_table(merged, side, group_keys, exclude_cols=None):
    """Left join a case+date table onto the surgeries anchor."""
    merged = merged.copy()
    side = side.copy()
    exclude_cols = exclude_cols or [PATIENT_KEY]

    merged[SURGERY_DATE_COL] = pd.to_datetime(merged[SURGERY_DATE_COL], errors="coerce").dt.normalize()
    side["תאריך ניתוח"] = pd.to_datetime(side["תאריך ניתוח"], errors="coerce").dt.normalize()

    side_cols = [c for c in side.columns if c not in group_keys + exclude_cols]
    return merged.merge(side[group_keys + side_cols], on=group_keys, how="left")


def aggregate_values(df, group_key, value_col, output_col):
    """One row per key with comma-separated unique values."""
    return (
        df.groupby(group_key, as_index=False)[value_col]
        .agg(join_if_many)
        .rename(columns={value_col: output_col})
    )


def prepare_patient_table(df):
    """Rename Patient -> patient and drop rows without patient id."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "Patient" in df.columns:
        df = df.rename(columns={"Patient": PATIENT_KEY})
    df[PATIENT_KEY] = pd.to_numeric(df[PATIENT_KEY], errors="coerce")
    df = df[df[PATIENT_KEY].notna()]
    df[PATIENT_KEY] = df[PATIENT_KEY].round().astype("int64")
    return df.reset_index(drop=True)


def merge_patient_table(merged, patient_side):
    """Left join a patient-level aggregated table."""
    return merged.merge(patient_side, on=PATIENT_KEY, how="left")


def clean_bp(df):
    """ניקוי טבלת לחץ דם."""
    df = prepare_case_date_table(df, BP_CASE_COL_RAW)
    df = df.drop_duplicates(subset=BP_GROUP_KEYS, keep="last")

    sbp = pd.to_numeric(df[BP_SBP_COL], errors="coerce")
    dbp = pd.to_numeric(df[BP_DBP_COL], errors="coerce")
    df.loc[(sbp <= 30) | (sbp >= 350), BP_SBP_COL] = np.nan
    df.loc[(dbp <= 20) | (dbp >= 250), BP_DBP_COL] = np.nan

    keep_cols = BP_GROUP_KEYS + [BP_SBP_COL, BP_DBP_COL]
    return df[keep_cols].reset_index(drop=True)


def clean_saturation(df):
    """ניקוי טבלת סטורציה."""
    df = prepare_case_date_table(df, SAT_CASE_COL_RAW)
    df = df.drop_duplicates(subset=SAT_GROUP_KEYS, keep="last")

    sat = pd.to_numeric(df[SAT_COL], errors="coerce")
    df.loc[(sat < 50) | (sat > 100), SAT_COL] = np.nan

    return df[SAT_GROUP_KEYS + [SAT_COL]].reset_index(drop=True)


def clean_background_diseases(df):
    """מחלות רקע — שורה אחת לכל patient עם רשימת ICD9."""
    df = prepare_patient_table(df)
    return aggregate_values(df, PATIENT_KEY, "ICD9", BG_OUTPUT_COL)


def clean_drug_allergy(df):
    """רגישות לתרופות — שורה אחת לכל patient."""
    df = prepare_patient_table(df)
    return aggregate_values(df, PATIENT_KEY, "Drug_Name", DRUG_ALLERGY_OUTPUT_COL)


def clean_other_allergy(df):
    """רגישות אחרת — שורה אחת לכל patient."""
    df = prepare_patient_table(df)
    return aggregate_values(df, PATIENT_KEY, "רגישות", OTHER_ALLERGY_OUTPUT_COL)


def clean_chronic_meds(df):
    """תרופות קבועות — שורה אחת לכל patient."""
    df = prepare_patient_table(df)
    return aggregate_values(df, PATIENT_KEY, "Drug_Name", CHRONIC_MEDS_OUTPUT_COL)


def clean_surgery_meds(df):
    """תרופות בניתוח — שורה אחת לכל מספר מקרה עם רשימת תרופות."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Patient" in df.columns:
        df = df.rename(columns={"Patient": PATIENT_KEY})

    cols_to_drop = [c for c in SURGERY_MEDS_DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df[SURGERY_MEDS_KEY] = pd.to_numeric(df[SURGERY_MEDS_KEY], errors="coerce")
    df = df[df[SURGERY_MEDS_KEY].notna()]
    df = df.drop_duplicates(subset=[SURGERY_MEDS_KEY, SURGERY_MEDS_VALUE_COL], keep="last")
    df[SURGERY_MEDS_KEY] = df[SURGERY_MEDS_KEY].round().astype("int64")

    return aggregate_values(df, SURGERY_MEDS_KEY, SURGERY_MEDS_VALUE_COL, SURGERY_MEDS_OUTPUT_COL)


def merge_bp(merged, bp):
    return merge_case_date_table(merged, bp, BP_GROUP_KEYS)


def merge_saturation(merged, saturation):
    return merge_case_date_table(merged, saturation, SAT_GROUP_KEYS)


def merge_background_diseases(merged, background):
    return merge_patient_table(merged, background)


def merge_drug_allergy(merged, drug_allergy):
    return merge_patient_table(merged, drug_allergy)


def merge_other_allergy(merged, other_allergy):
    return merge_patient_table(merged, other_allergy)


def merge_chronic_meds(merged, chronic_meds):
    return merge_patient_table(merged, chronic_meds)


def merge_surgery_meds(merged, surgery_meds):
    return merged.merge(surgery_meds, on=SURGERY_MEDS_KEY, how="left")


def clean_labs(df):
    """ניקוי טבלת בדיקות מעבדה."""
    df = df.copy()
    if len(df.columns) == len(LAB_COLUMNS):
        df.columns = LAB_COLUMNS

    if "Patient" in df.columns:
        df = df.rename(columns={"Patient": "patient"})

    df["מספר מקרה"] = pd.to_numeric(df["מספר מקרה"], errors="coerce")
    df = df[df["מספר מקרה"].notna()]
    df[LAB_DATE_COL] = pd.to_datetime(df[LAB_DATE_COL], errors="coerce")
    df["מספר מקרה"] = df["מספר מקרה"].round().astype("int64")
    df[LAB_VALUE_COL] = pd.to_numeric(df[LAB_VALUE_COL], errors="coerce")
    df = df[df[LAB_TEST_COL].isin(LAB_SELECTED_TESTS)]
    return df.reset_index(drop=True)


def _pick_closest_lab_dates(surgery_keys, labs):
    """For each (מספר מקרה, תאריך ניתוח), pick closest תאריך בדיקה."""
    lab_dates = labs[["מספר מקרה", LAB_DATE_COL]].drop_duplicates()
    pairs = surgery_keys.merge(lab_dates, on="מספר מקרה", how="left")
    pairs["diff"] = (pairs[LAB_DATE_COL] - pairs[SURGERY_DATE_COL]).abs()
    closest = (
        pairs.sort_values("diff")
        .drop_duplicates(subset=["מספר מקרה", SURGERY_DATE_COL], keep="first")
    )
    return closest.dropna(subset=[LAB_DATE_COL])


def merge_labs(merged, labs):
    """Pivot selected lab tests wide; closest lab date per surgery row; missing=0."""
    merged = merged.copy()
    labs = labs.copy()

    merged[SURGERY_DATE_COL] = pd.to_datetime(
        merged[SURGERY_DATE_COL], errors="coerce"
    ).dt.normalize()

    surgery_keys = merged[["מספר מקרה", SURGERY_DATE_COL]].drop_duplicates()
    closest = _pick_closest_lab_dates(surgery_keys, labs)

    labs_match = labs.merge(
        closest,
        on=["מספר מקרה", LAB_DATE_COL],
        how="inner",
    )
    labs_match = labs_match.drop_duplicates(
        subset=["מספר מקרה", SURGERY_DATE_COL, LAB_TEST_COL],
        keep="last",
    )

    wide = labs_match.pivot(
        index=["מספר מקרה", SURGERY_DATE_COL],
        columns=LAB_TEST_COL,
        values=LAB_VALUE_COL,
    ).reset_index()
    wide.columns.name = None
    wide[SURGERY_DATE_COL] = pd.to_datetime(wide[SURGERY_DATE_COL]).dt.normalize()

    for col in LAB_SELECTED_TESTS:
        if col not in wide.columns:
            wide[col] = np.nan

    merged = merged.merge(
        wide[["מספר מקרה", SURGERY_DATE_COL] + LAB_SELECTED_TESTS],
        on=["מספר מקרה", SURGERY_DATE_COL],
        how="left",
    )

    for col in LAB_SELECTED_TESTS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(LABS_FILL_VALUE)

    n_with_labs = (merged[LAB_SELECTED_TESTS] != LABS_FILL_VALUE).any(axis=1).sum()
    print(f"lab columns: {len(LAB_SELECTED_TESTS)}")
    print(f"rows with lab data: {n_with_labs:,} / {len(merged):,}")

    return merged


def map_smoking_binary(series):
    """Map Smoking codes: 1=מעשן, 2/3/empty/UNKNOWN=לא מעשן."""
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(NON_SMOKER_LABEL, index=series.index, dtype="string")
    result[numeric == 1] = SMOKER_LABEL
    return result


def split_combined_category(value):
    """Split A+B / A,B / A + B into separate category labels."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "<na>") or text.upper() == "UNKNOWN":
        return []
    text = re.sub(r"\s*[,+/|&]\s*", "+", text)
    text = re.sub(r"\s*\+\s*", "+", text)
    return [p.strip() for p in text.split("+") if p.strip()]


def standardize_anesthesia_type(value):
    """Unify combined anesthesia labels (A+B, A,B, ...) to A_B before OHE."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "<na>"):
        return text
    if text.upper() == "UNKNOWN":
        return text
    parts = split_combined_category(value)
    if len(parts) <= 1:
        return parts[0] if parts else text
    return "_".join(parts)


def encode_multilabel_ohe(df, col, prefix):
    """Multi-label OHE: A+B marks both prefix_A and prefix_B as 1."""
    df = df.copy()
    parts_series = df[col].apply(split_combined_category)
    all_parts = sorted({p for parts in parts_series for p in parts})
    if not all_parts:
        return df.drop(columns=[col])
    for part in all_parts:
        col_name = f"{prefix}_{part}"
        df[col_name] = parts_series.apply(lambda ps, p=part: int(p in ps))
    return df.drop(columns=[col])


def encode_merged_categoricals(df):
    """One-hot encode smoking, gender, and surgery categorical columns."""
    df = df.copy()

    if SMOKING_COL in df.columns:
        smoking_bin = map_smoking_binary(df[SMOKING_COL])
        smoking_dummies = pd.get_dummies(
            smoking_bin, prefix=SMOKING_OHE_PREFIX, dtype=int
        )
        df = df.drop(columns=[SMOKING_COL])
        df = pd.concat([df, smoking_dummies], axis=1)

    for col in OHE_CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if col == SURGERY_TYPE_COL:
            df = encode_multilabel_ohe(df, col, col)
            continue
        values = df[col].astype(str)
        if col == ANESTHESIA_TYPE_COL:
            values = values.apply(standardize_anesthesia_type)
        dummies = pd.get_dummies(values, prefix=col, dtype=int)
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)

    drop_unknown = [c for c in OHE_DROP_UNKNOWN_COLS if c in df.columns]
    if drop_unknown:
        df = df.drop(columns=drop_unknown)

    return df


def count_comma_values(value):
    """Count non-empty comma-separated items."""
    if pd.isna(value):
        return 0
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return len(parts)


DRUG_FORM_WORDS = {
    "TAB", "TABLET", "TABLETS", "CAPLET", "CAPLETS", "CAP", "CAPS", "CAPSULE", "CAPSULES",
    "INJ", "INJECTION", "INJECTIONS", "AMP", "AMPOULE", "AMPOULES", "SOLUTION", "SOLUTIONS",
    "SYRUP", "CREAM", "OINTMENT", "GEL", "PATCH", "SPRAY", "DROPS", "SUSPENSION", "POWDER",
    "GRANULES", "SUPPOSITORY", "OVULE", "OVULES", "PREFILLED", "SYRINGE", "VIAL", "VIALS",
    "INFUSION", "INOVAMED", "PRESERVATIVE", "FREE", "AND", "IN", "PLASTIC", "CONTAINER",
}

_DRUG_INDEX_CACHE = None


def load_drug_index(path=DRUG_INDEX_PATH):
    """Load canonical drug names; return set plus longest/shortest sort orders."""
    names = (
        pd.read_csv(path, dtype=str)["DrugName"]
        .dropna()
        .str.strip()
        .str.upper()
        .unique()
    )
    index_set = set(names)
    sorted_longest = sorted(names, key=len, reverse=True)
    sorted_shortest = sorted(names, key=len)
    return index_set, sorted_longest, sorted_shortest


def get_drug_index():
    """Load drug index once and cache for repeated normalization."""
    global _DRUG_INDEX_CACHE
    if _DRUG_INDEX_CACHE is None:
        _DRUG_INDEX_CACHE = load_drug_index()
    return _DRUG_INDEX_CACHE


def normalize_drug_for_matching(raw):
    """Uppercase and strip dosage, form, and route tokens for index matching."""
    s = str(raw).strip().upper()
    if not s or s in ("NAN", "NONE", "<NA>"):
        return ""
    s = re.sub(r"\s*\*\*.*?\*\*", "", s).strip()
    s = s.replace("-", " ").replace("'", "")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(
        r"\s+\d[\d./\s%-]*(?:MG|ML|G|MCG|IU|UNITS?|µG|UG|%|MEQ|MCG/ML).*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s+\d.*$", "", s)
    tokens = [t for t in s.split() if t not in DRUG_FORM_WORDS]
    return " ".join(tokens).strip()


def _drug_word_boundary_prefix(prefix, text):
    return text == prefix or text.startswith(prefix + " ")


def match_canonical_drug(raw, index_set, sorted_longest, sorted_shortest):
    """Map one raw drug string to a canonical FDA DrugName from the index."""
    s = normalize_drug_for_matching(raw)
    if not s:
        return None
    tokens = s.split()
    for n in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:n])
        if candidate in index_set:
            return candidate
    for canonical in sorted_longest:
        c = canonical.replace("'", "")
        if _drug_word_boundary_prefix(c, s):
            return canonical
    for canonical in sorted_shortest:
        c = canonical.replace("'", "")
        if _drug_word_boundary_prefix(s, c):
            return canonical
    return None


def clean_drugs_with_index(value, drug_index=None):
    """Normalize comma-separated drugs to unique canonical names from the index."""
    if pd.isna(value):
        return pd.NA
    if drug_index is None:
        drug_index = get_drug_index()
    index_set, sorted_longest, sorted_shortest = drug_index
    found = []
    for part in str(value).split(","):
        canonical = match_canonical_drug(
            part, index_set, sorted_longest, sorted_shortest
        )
        name_to_add = canonical if canonical else normalize_drug_for_matching(part)
        if name_to_add and name_to_add not in found:
            found.append(name_to_add)
    return ",".join(found) if found else pd.NA


def add_age_group_flags(df):
    """One-hot flags: Age Group_Young / Middle-aged / Elderly."""
    df = df.copy()
    age = pd.to_numeric(df["גיל"], errors="coerce")
    df["Age Group_Young"] = ((age >= 0) & (age <= AGE_YOUNG_MAX)).astype(int)
    df["Age Group_Middle-aged"] = ((age > AGE_YOUNG_MAX) & (age < AGE_ELDERLY_MIN)).astype(int)
    df["Age Group_Elderly"] = (age >= AGE_ELDERLY_MIN).astype(int)
    return df


def add_bmi_category_flags(df):
    """One-hot flags for BMI categories."""
    df = df.copy()
    bmi = pd.to_numeric(df["BMI"], errors="coerce")
    df["BMI Category_Underweight"] = (bmi < BMI_UNDERWEIGHT_MAX).astype(int)
    df["BMI Category_Normal"] = ((bmi >= BMI_UNDERWEIGHT_MAX) & (bmi < BMI_NORMAL_MAX)).astype(int)
    df["BMI Category_Overweight"] = ((bmi >= BMI_NORMAL_MAX) & (bmi < BMI_OVERWEIGHT_MAX)).astype(int)
    df["BMI Category_Obese"] = (bmi >= BMI_OVERWEIGHT_MAX).astype(int)
    return df


def engineer_merged_features(df):
    """Drop unused cols; add age/BMI flags, drug cleaning, and counters."""
    df = df.copy()

    drop_cols = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_cols)

    df = add_age_group_flags(df)
    df = add_bmi_category_flags(df)

    if SURGERY_MEDS_OUTPUT_COL in df.columns or DRUG_ALLERGY_OUTPUT_COL in df.columns:
        drug_index = get_drug_index()
    if SURGERY_MEDS_OUTPUT_COL in df.columns:
        df[CLEANED_DRUGS_COL] = df[SURGERY_MEDS_OUTPUT_COL].apply(
            clean_drugs_with_index, drug_index=drug_index
        )
    if DRUG_ALLERGY_OUTPUT_COL in df.columns:
        df[CLEANED_DRUG_ALLERGIES_COL] = df[DRUG_ALLERGY_OUTPUT_COL].apply(
            clean_drugs_with_index, drug_index=drug_index
        )

    if BG_OUTPUT_COL in df.columns:
        df[BG_COUNT_COL] = df[BG_OUTPUT_COL].apply(count_comma_values)
    if CLEANED_DRUG_ALLERGIES_COL in df.columns:
        df[DRUG_ALLERGY_COUNT_COL] = df[CLEANED_DRUG_ALLERGIES_COL].apply(count_comma_values)
    elif DRUG_ALLERGY_OUTPUT_COL in df.columns:
        df[DRUG_ALLERGY_COUNT_COL] = df[DRUG_ALLERGY_OUTPUT_COL].apply(count_comma_values)
    if "קוד פרוצדורה" in df.columns:
        df[PROCEDURE_CODE_COUNT_COL] = df["קוד פרוצדורה"].apply(count_comma_values)

    return df


def zscore_column(series, treat_zero_as_missing=False):
    """Z-score normalization; optional: treat 0 as missing for mean/std."""
    s = pd.to_numeric(series, errors="coerce")
    stats = s.replace(0, np.nan) if treat_zero_as_missing else s
    mu = stats.mean()
    sigma = stats.std(ddof=0)
    if pd.isna(sigma) or sigma == 0:
        z = pd.Series(0.0, index=s.index)
    else:
        z = (s - mu) / sigma
    return z.fillna(0.0)


def normalize_ohe_category_suffix(suffix):
    """Standardize OHE category suffix (merge + , / | & variants)."""
    text = str(suffix).strip()
    text = re.sub(r"[,/|&]", " + ", text)
    text = re.sub(r"\s*\+\s*", " + ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if " + " in text:
        parts = sorted(p.strip() for p in text.split(" + ") if p.strip())
        return " + ".join(parts)
    return text


def explode_combined_ohe_columns(df, prefix):
    """Split legacy combined OHE columns (A + B) into atomic prefix_A, prefix_B."""
    df = df.copy()
    prefix_tag = f"{prefix}_"
    combined_cols = [
        c for c in df.columns
        if c.startswith(prefix_tag) and len(split_combined_category(c[len(prefix_tag):])) > 1
    ]
    for col in combined_cols:
        parts = split_combined_category(col[len(prefix_tag):])
        vals = df[col]
        for part in parts:
            atomic = f"{prefix_tag}{part}"
            if atomic not in df.columns:
                df[atomic] = 0
            df[atomic] = df[atomic].combine(vals, max).astype(int)
        df = df.drop(columns=[col])
    return df


def merge_ohe_prefix_columns(df, prefix):
    """Merge duplicate one-hot columns that differ only by separators."""
    df = df.copy()
    prefix_tag = f"{prefix}_"
    cols = [c for c in df.columns if c.startswith(prefix_tag)]
    if not cols:
        return df

    groups = {}
    for col in cols:
        suffix = col[len(prefix_tag):]
        canonical = prefix_tag + normalize_ohe_category_suffix(suffix)
        groups.setdefault(canonical, []).append(col)

    for canonical, members in groups.items():
        if len(members) == 1 and members[0] == canonical:
            continue
        merged_vals = df[members].max(axis=1).astype(int)
        drop_cols = [c for c in members if c != canonical]
        df[canonical] = merged_vals
        df = df.drop(columns=drop_cols)

    return df


def consolidate_smoker_column(df):
    """Convert עישון_מעשן / עישון_לא מעשן into single smoker column."""
    df = df.copy()
    smoker_col = f"{SMOKING_OHE_PREFIX}_{SMOKER_LABEL}"
    non_smoker_col = f"{SMOKING_OHE_PREFIX}_{NON_SMOKER_LABEL}"
    if smoker_col not in df.columns:
        return df

    df[SMOKER_COL] = df[smoker_col].astype(int)
    drop_cols = [c for c in [smoker_col, non_smoker_col] if c in df.columns]
    return df.drop(columns=drop_cols)


def prepare_final_dataset(df):
    """Normalization and feature cleanup for ML-ready dataset."""
    df = df.copy()

    if "BMI" in df.columns:
        df[BMI_Z_COL] = zscore_column(df["BMI"])

    if SAT_COL in df.columns:
        sat = pd.to_numeric(df[SAT_COL], errors="coerce")
        df[SAT_NORM_COL] = (sat / 100.0).fillna(0.0)

    for col in LAB_SELECTED_TESTS:
        if col in df.columns:
            df[f"{col}_Z"] = zscore_column(df[col], treat_zero_as_missing=True)

    lab_raw_drop = [c for c in LAB_SELECTED_TESTS if c in df.columns]
    if lab_raw_drop:
        df = df.drop(columns=lab_raw_drop)

    if BP_DBP_COL in df.columns:
        df[DBP_Z_COL] = zscore_column(df[BP_DBP_COL])
    if BP_SBP_COL in df.columns:
        df[SBP_Z_COL] = zscore_column(df[BP_SBP_COL])

    df = explode_combined_ohe_columns(df, SURGERY_TYPE_OHE_PREFIX)
    df = merge_ohe_prefix_columns(df, ANESTHESIA_OHE_PREFIX)
    df = consolidate_smoker_column(df)

    return df


def _safe_icd9_code(code):
    """Sanitize ICD9 code for use in a column name."""
    return re.sub(r"[^\w.]", "_", str(code).strip())


def encode_top_k_icd9_ohe(df, col=BG_OUTPUT_COL, k=BG_TOP_K_ICD9):
    """One-hot encode the top-K most frequent ICD9 codes from background diseases."""
    df = df.copy()
    if col not in df.columns:
        return df, []

    from collections import Counter

    counter = Counter()
    row_codes = {}
    for idx, val in df[col].items():
        if pd.isna(val):
            row_codes[idx] = []
            continue
        codes = [c.strip() for c in str(val).split(",") if c.strip()]
        row_codes[idx] = codes
        counter.update(codes)

    top_k_codes = [code for code, _ in counter.most_common(k)]
    for code in top_k_codes:
        ohe_col = f"{BG_ICD9_OHE_PREFIX}_{_safe_icd9_code(code)}"
        flags = pd.Series(
            [int(code in row_codes[idx]) for idx in df.index],
            index=df.index,
            dtype=int,
        )
        df[ohe_col] = flags

    return df, top_k_codes


def _build_column_rename_map():
    """Hebrew / internal names → English ML column names."""
    surgery_type_map = {
        f"{SURGERY_TYPE_OHE_PREFIX}_אלקטיבי": "ST_Elective",
        f"{SURGERY_TYPE_OHE_PREFIX}_ססיה": "ST_Sesia",
        f"{SURGERY_TYPE_OHE_PREFIX}_דחוף": "ST_Urgent",
        f"{SURGERY_TYPE_OHE_PREFIX}_בהול": "ST_Emergency",
        f"{SURGERY_TYPE_OHE_PREFIX}_פעולה לא ניתוחית": "ST_Non_Surgical_Operation",
        f"{SURGERY_TYPE_OHE_PREFIX}_קציר איברים": "ST_Organ_Procurement",
    }
    anesthesia_map = {
        f"{ANESTHESIA_OHE_PREFIX}_General": "AT_General",
        f"{ANESTHESIA_OHE_PREFIX}_General_Local": "AT_General_Local",
        f"{ANESTHESIA_OHE_PREFIX}_General_Regional": "AT_General_Regional",
        f"{ANESTHESIA_OHE_PREFIX}_Local": "AT_Local",
        f"{ANESTHESIA_OHE_PREFIX}_Local_Regional": "AT_Local_Regional",
        f"{ANESTHESIA_OHE_PREFIX}_Local_Sedation": "AT_Local_Sedation",
        f"{ANESTHESIA_OHE_PREFIX}_Other": "AT_Other",
        f"{ANESTHESIA_OHE_PREFIX}_Regional": "AT_Regional",
        f"{ANESTHESIA_OHE_PREFIX}_Regional_Sedation": "AT_Regional_Sedation",
        f"{ANESTHESIA_OHE_PREFIX}_Sedation": "AT_Sedation",
        f"{ANESTHESIA_OHE_PREFIX}_Spinal": "AT_Spinal",
        f"{ANESTHESIA_OHE_PREFIX}_Spinal_Epidural": "AT_Spinal_Epidural",
        f"{ANESTHESIA_OHE_PREFIX}_Epidural": "AT_Epidural",
    }
    weekday_map = {
        "יום ניתוח_ראשון": "Sunday",
        "יום ניתוח_שני": "Monday",
        "יום ניתוח_שלישי": "Tuesday",
        "יום ניתוח_רביעי": "Wednesday",
        "יום ניתוח_חמישי": "Thursday",
        "יום ניתוח_שישי": "Friday",
        "יום ניתוח_שבת": "Saturday",
    }
    rename = {
        PATIENT_KEY: "Patient",
        "Medical Record": "Medical_Record",
        "מספר מקרה": "Case_Number",
        "מחלקה מנתחת": "Surgical_Department",
        SURGERY_DATE_COL: "Surgery_Date",
        "גיל": "Age",
        "מין_זכר": "Male",
        "מין_נקבה": "Female",
        "BMI": "BMI",
        BMI_Z_COL: "Normalized_BMI",
        "קוד פרוצדורה": "Procedure_Code",
        "צד פרוצדורה": "Procedure_Side",
        SAT_NORM_COL: "Oxygen_Saturation",
        SBP_Z_COL: "BP_Systolic_Before_Surgery",
        DBP_Z_COL: "BP_Diastolic_Before_Surgery",
        SMOKER_COL: "Smoking",
        "סוג מקרה_אשפוז": "CT_Hospitalized",
        "סוג מקרה_אמבולטורי": "CT_Ambulatory",
        SURGERY_MEDS_OUTPUT_COL: "Drug_Names",
        DRUG_ALLERGY_OUTPUT_COL: "Allergy_Drug_Names",
        BG_OUTPUT_COL: "Disease_ICD9_Codes",
        TARGET_COL: "Total_Surgery_Time",
        ESTIMATED_SURGERY_TIME_COL: ESTIMATED_SURGERY_TIME_COL,
        CLEANED_DRUGS_COL: "Cleaned_Drug_Names",
        CLEANED_DRUG_ALLERGIES_COL: "Cleaned_Allergy_Drug_Names",
        NUM_MEDICATIONS_COL: NUM_MEDICATIONS_COL,
        NUM_ALLERGY_MEDICATIONS_COL: NUM_ALLERGY_MEDICATIONS_COL,
        SURGERY_TIME_ANOMALY_COL: SURGERY_TIME_ANOMALY_COL,
        "Age Group_Young": "Age_Group_Young",
        "Age Group_Middle-aged": "Age_Group_Middle-aged",
        "Age Group_Elderly": "Age_Group_Elderly",
        "BMI Category_Underweight": "BMI_Category_Underweight",
        "BMI Category_Normal": "BMI_Category_Normal",
        "BMI Category_Overweight": "BMI_Category_Overweight",
        "BMI Category_Obese": "BMI_Category_Obese",
    }
    rename.update(surgery_type_map)
    rename.update(anesthesia_map)
    rename.update(weekday_map)
    return rename


def _build_final_column_order(top_k_icd9_codes):
    """Return ordered English column list for the ML-ready dataset."""
    icd9_cols = [f"{BG_ICD9_OHE_PREFIX}_{_safe_icd9_code(c)}" for c in top_k_icd9_codes]
    lab_cols = [f"{col}_Z" for col in LAB_SELECTED_TESTS]
    return [
        "Patient",
        "Medical_Record",
        "Case_Number",
        "Surgical_Department",
        "Surgery_Date",
        "Age",
        "Male",
        "Female",
        "BMI",
        "Normalized_BMI",
        "Procedure_Code",
        "Procedure_Side",
        "Oxygen_Saturation",
        "BP_Systolic_Before_Surgery",
        "BP_Diastolic_Before_Surgery",
        "Smoking",
        "CT_Hospitalized",
        "CT_Ambulatory",
        "ST_Elective",
        "ST_Elective_Sesia",
        "ST_Emergency",
        "ST_Non_Surgical_Operation",
        "ST_Organ_Procurement",
        "ST_Sesia",
        "ST_Urgent",
        "AT_General",
        "AT_General_Local",
        "AT_General_Regional",
        "AT_Local",
        "AT_Local_Regional",
        "AT_Other",
        "AT_Regional",
        "AT_Sedation",
        "AT_Spinal",
        "AT_Spinal_Epidural",
        "AT_Epidural",
        "AT_Local_Sedation",
        "AT_Regional_Sedation",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Drug_Names",
        "Allergy_Drug_Names",
        "Disease_ICD9_Codes",
        "Total_Surgery_Time",
        ESTIMATED_SURGERY_TIME_COL,
        "Cleaned_Drug_Names",
        "Cleaned_Allergy_Drug_Names",
        NUM_MEDICATIONS_COL,
        NUM_ALLERGY_MEDICATIONS_COL,
        SURGERY_TIME_ANOMALY_COL,
        "Age_Group_Young",
        "Age_Group_Middle-aged",
        "Age_Group_Elderly",
        "BMI_Category_Underweight",
        "BMI_Category_Normal",
        "BMI_Category_Overweight",
        "BMI_Category_Obese",
        *icd9_cols,
        *lab_cols,
    ]


def finalize_english_ml_dataset(df, top_k_icd9=BG_TOP_K_ICD9):
    """Translate columns to English, add ICD9 OHE, drop extras, and reorder."""
    df = df.copy()

    if "duration_quality" in df.columns:
        df[SURGERY_TIME_ANOMALY_COL] = (
            df["duration_quality"] != "valid"
        ).astype(int)

    if CLEANED_DRUGS_COL in df.columns:
        df[NUM_MEDICATIONS_COL] = df[CLEANED_DRUGS_COL].apply(count_comma_values)
    if CLEANED_DRUG_ALLERGIES_COL in df.columns:
        df[NUM_ALLERGY_MEDICATIONS_COL] = df[CLEANED_DRUG_ALLERGIES_COL].apply(
            count_comma_values
        )
    elif DRUG_ALLERGY_COUNT_COL in df.columns:
        df[NUM_ALLERGY_MEDICATIONS_COL] = df[DRUG_ALLERGY_COUNT_COL]

    if DRUG_ALLERGY_COUNT_COL in df.columns:
        df = df.drop(columns=[DRUG_ALLERGY_COUNT_COL])

    df[ESTIMATED_SURGERY_TIME_COL] = np.nan

    df, top_k_codes = encode_top_k_icd9_ohe(df, k=top_k_icd9)

    drop_cols = [c for c in ML_COLUMNS_DROP if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    rename_map = _build_column_rename_map()
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    ordered = _build_final_column_order(top_k_codes)
    ordered_existing = [c for c in ordered if c in df.columns]
    extra_cols = [c for c in df.columns if c not in ordered_existing]
    df = df[ordered_existing + extra_cols]

    return df


def export_ml_dataset_by_department(df, export_dir=ML_EXPORT_DIR):
    """Split final ML dataset by surgical department and export CSV per department."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if SURGICAL_DEPARTMENT_COL not in df.columns:
        raise KeyError(f"Missing column: {SURGICAL_DEPARTMENT_COL}")

    exported = {}
    for dept, filename in DEPARTMENT_EXPORT_FILES.items():
        subset = df[df[SURGICAL_DEPARTMENT_COL] == dept]
        path = export_dir / filename
        subset.to_csv(path, index=False)
        exported[dept] = path
        print(f"{dept}: {len(subset):,} rows -> {path.resolve()}")

    missing = sorted(
        set(df[SURGICAL_DEPARTMENT_COL].dropna().unique())
        - set(DEPARTMENT_EXPORT_FILES.keys())
    )
    if missing:
        print(f"warning: unmapped departments (not exported): {missing}")

    return exported


def invalid_height(row):
    age = row["גיל"]
    h = row["גובה"]
    if pd.isna(age) or pd.isna(h):
        return False
    if 1 <= age <= 4:
        return (h < 48.0) or (h > 120.0)
    if 5 <= age <= 12:
        return (h < 80.0) or (h > 200.0)
    if 13 <= age <= 17:
        return (h < 100.0) or (h > 220.0)
    return (h < 100.0) or (h > 220.0)


def invalid_weight(row):
    w = row["משקל"]
    if pd.isna(w):
        return False
    return (w < 20.0) or (w > 300.0)


def invalid_bmi_value(bmi):
    if pd.isna(bmi):
        return False
    return (bmi < 10.0) or (bmi > 80.0)


def _fill_by_age_gender_neighborhood(df, value_col, max_k=10):
    """Fill one column using same-gender mean by age, widening ±1..max_k years."""
    valid_ages = df["גיל"].dropna()
    if valid_ages.empty:
        return df

    min_age = int(valid_ages.min())
    max_age = int(valid_ages.max())

    age_gender_mean = (
        df.groupby(["_Male", "גיל"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "mean_age"})
    )

    full_grid = pd.MultiIndex.from_product(
        [[0, 1], range(min_age, max_age + 1)],
        names=["_Male", "גיל"],
    ).to_frame(index=False)

    lookup = full_grid.merge(age_gender_mean, on=["_Male", "גיל"], how="left")
    ages = lookup["גיל"].to_numpy()
    males = lookup["_Male"].to_numpy()
    mean_vals = lookup["mean_age"].to_numpy()
    knn_vals = np.full_like(mean_vals, np.nan, dtype=float)

    for g in (0, 1):
        idxs = np.where(males == g)[0]
        ages_g = ages[idxs]
        means_g = mean_vals[idxs]

        for i_local, age0 in enumerate(ages_g):
            if not np.isnan(means_g[i_local]):
                knn_vals[idxs[i_local]] = means_g[i_local]
                continue
            for k in range(1, max_k + 1):
                in_window = (ages_g >= age0 - k) & (ages_g <= age0 + k)
                candidates = means_g[in_window]
                candidates = candidates[~np.isnan(candidates)]
                if candidates.size > 0:
                    knn_vals[idxs[i_local]] = candidates.mean()
                    break

    lookup["knn_mean"] = knn_vals
    df = df.merge(
        lookup[["_Male", "גיל", "mean_age", "knn_mean"]],
        on=["_Male", "גיל"],
        how="left",
    )
    df[value_col] = df[value_col].fillna(df["mean_age"]).fillna(df["knn_mean"])
    df = df.drop(columns=["mean_age", "knn_mean"])
    return df


def impute_merged_bmi_metrics(df, max_k=BMI_IMPUTE_MAX_AGE_K):
    """Invalidate outliers, impute height/weight by age+gender, recalc BMI."""
    df = df.copy()

    for col in ["גובה", "משקל", "BMI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["גיל"] = pd.to_numeric(df["גיל"], errors="coerce")
    df["_Male"] = df["מין"].map({"זכר": 1, "נקבה": 0})

    invalid_h = df.apply(invalid_height, axis=1)
    invalid_w = df.apply(invalid_weight, axis=1)
    invalid_bmi = df["BMI"].apply(invalid_bmi_value)

    # rows that need BMI recalc after height/weight fix
    needs_bmi_recalc = (
        df["BMI"].isna() | invalid_bmi
        | invalid_h | invalid_w
        | df["גובה"].isna() | df["משקל"].isna()
    )

    df.loc[invalid_h, "גובה"] = np.nan
    df.loc[invalid_w, "משקל"] = np.nan
    df.loc[invalid_bmi, "BMI"] = np.nan

    df = _fill_by_age_gender_neighborhood(df, "גובה", max_k=max_k)
    df = _fill_by_age_gender_neighborhood(df, "משקל", max_k=max_k)

    recalc_mask = needs_bmi_recalc & df["גובה"].notna() & df["משקל"].notna()
    df.loc[recalc_mask, "BMI"] = (
        df.loc[recalc_mask, "משקל"] / (df.loc[recalc_mask, "גובה"] / 100) ** 2
    )

    df = df.drop(columns=["_Male"])
    return df