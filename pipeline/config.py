from pathlib import Path
import re

import numpy as np
import pandas as pd

# --- paths ---
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
STAGED_DIR = CLEAN_DIR / "staged"
STAGED_DIR.mkdir(parents=True, exist_ok=True)

# --- local files (dev only — replace with SQL on server) ---
FILES = {
    "surgeries": "ניתוחים_260415.xlsx",
    "bmi": "סוג דם גובה משקל BMI.xlsx",
    "smoking": "עישון.xlsx",
    "bp": "לחץ דם.xlsx",
    "saturation": "סטורציה.xlsx",
    "surgery_meds": "תרופות בניתוח.xlsx",
    "labs": "בדיקות מעבדה.csv",
    "background_diseases": "מחלות רקע.xlsx",
    "other_allergy": "רגישות אחרת.xlsx",
    "drug_allergy": "רגישות לתרופות.xlsx",
    "chronic_meds": "תרופות קבועות.xlsx",
    "anesthesia": "הרדמה.xlsx",
}

LAB_COLUMNS = [
    "Patient", "מספר מקרה", "קטגוריית בדיקה",
    "שם בדיקה", "ערך", "תאריך בדיקה",
]

# --- SQL queries (production) ---
SQL = {
    "surgeries": "SELECT * FROM surgeries",
    "bmi": "SELECT * FROM bmi",
    "smoking": "SELECT * FROM smoking",
    "bp": "SELECT * FROM blood_pressure",
    "saturation": "SELECT * FROM saturation",
    "background_diseases": "SELECT * FROM background_diseases",
    "drug_allergy": "SELECT * FROM drug_allergy",
    "other_allergy": "SELECT * FROM other_allergy",
    "chronic_meds": "SELECT * FROM chronic_meds",
    "surgery_meds": "SELECT * FROM surgery_meds",
    "labs": "SELECT * FROM labs",
    # add rest when moving to server
}

# --- surgeries cleaning rules ---
SURGERIES_GROUP_KEYS = ["מספר מקרה", "Medical Record"]

SURGERIES_COLS_TO_DROP = [
    "פעולה ניתוחית מיילדותית", "F45", "F46", "הרדמה פירוט",
    "מחלקה מנתחת נוספת", "הערה לפרוצדורה", "הערה לאבחנה",
    "עוזר מנתח", "מנתח ראשי", "מרדים", "אחות רחוצה", "מנתח אחראי","אחות מסתובבת"
]

# --- total_surgery_time (OR occupancy, minutes) ---
ENTRY_DATE_COL = "תאריך כניסה לחדר ניתוח"
ENTRY_TIME_COL = "שעת כניסה לחדר ניתוח"
ANES_START_DATE = "תאריך תחילת הרדמה"
ANES_START_TIME = "שעת תחילת הרדמה"
SURG_START_DATE = "תאריך תחילת ניתוח"
SURG_START_TIME = "שעת תחילת ניתוח"
EXIT_DATE_COL = "תאריך יציאה מחדר ניתוח"
EXIT_TIME_COL = "שעת יציאה מחדר ניתוח"
ANES_END_DATE = "תאריך סיום הרדמה"
ANES_END_TIME = "שעת סיום הרדמה"
SURG_END_DATE = "תאריך סיום ניתוח"
SURG_END_TIME = "שעת סיום ניתוח"
DISC_DATE_COL = "תאריך שחרור מח ניתוח"
DISC_TIME_COL = "שעת שחרור מח ניתוח"
TARGET_COL = "total_surgery_time"
MAX_START_GAP_HOURS = 2
MAX_END_GAP_HOURS = 2
DISCHARGE_CLOSE_MINUTES = 30
SURGERY_DATE_COL = "תאריך ניתוח"

SURGERIES_DROP_IF_MISSING = [TARGET_COL, "גיל"]
SURGERIES_FILL_UNKNOWN = [
    "סוג ניתוח", "סוג הרדמה",
    "קוד אבחנה", "שם אבחנה", "קוד פרוצדורה", "שם פרוצדורה",
]
SURGERIES_COLS_TO_DROP_EXTRA = ["חדר"]
SURGERIES_FILL_NON = ["צד פרוצדורה"]
PROCEDURE_SIDE_COL = "צד פרוצדורה"
SURGERIES_INT_COLS = ["מספר מקרה", "גיל", TARGET_COL]

SURGERIES_RAW_TIMESTAMP_COLS = [
    ENTRY_DATE_COL, ENTRY_TIME_COL,
    ANES_START_DATE, ANES_START_TIME,
    SURG_START_DATE, SURG_START_TIME,
    EXIT_DATE_COL, EXIT_TIME_COL,
    ANES_END_DATE, ANES_END_TIME,
    SURG_END_DATE, SURG_END_TIME,
    DISC_DATE_COL, DISC_TIME_COL,
]

SURGERIES_COMPUTED_DATETIME_COLS = [
    "entry_to_or_datetime", "anesthesia_start_datetime", "surgery_start_datetime",
    "exit_from_or_datetime", "anesthesia_end_datetime", "surgery_end_datetime",
    "discharge_datetime", "selected_start_datetime", "selected_end_datetime",
]

# --- BMI cleaning rules ---
BMI_GROUP_KEYS = ["מספר מקרה", "תאריך ניתוח"]
BMI_COLS_TO_DROP = ["דם"]
MERGED_SURGERIES_NAME = "surgeries_with_bmi"
BMI_IMPUTE_MAX_AGE_K = 10  # widen age window up to ±10 years

# --- smoking cleaning rules ---
SMOKING_GROUP_KEYS = ["מספר מקרה", "מועד ניתוח"]
SMOKING_COL = "Smoking"
SMOKING_FILL_UNKNOWN = "UNKNOWN"
MERGED_AFTER_SMOKING_NAME = "surgeries_with_bmi_smoking"

# --- one-hot encoding ---
OHE_CATEGORICAL_COLS = ["מין", "סוג מקרה", "יום ניתוח", "סוג ניתוח", "סוג הרדמה"]
ANESTHESIA_TYPE_COL = "סוג הרדמה"
SURGERY_TYPE_COL = "סוג ניתוח"
SMOKING_OHE_PREFIX = "עישון"
SMOKER_LABEL = "מעשן"
NON_SMOKER_LABEL = "לא מעשן"

# --- blood pressure ---
BP_CASE_COL_RAW = "מקרה"
BP_GROUP_KEYS = ["מספר מקרה", "תאריך ניתוח"]
BP_SBP_COL = "היסטולי לפני ניתוח"
BP_DBP_COL = "דיאסטולי לפני ניתוח"

# --- saturation ---
SAT_CASE_COL_RAW = "מקרה"
SAT_GROUP_KEYS = ["מספר מקרה", "תאריך ניתוח"]
SAT_COL = "סטורציה לפני ניתוח"

# --- patient-level tables ---
PATIENT_KEY = "patient"
BG_OUTPUT_COL = "מחלות_רקע"
DRUG_ALLERGY_OUTPUT_COL = "רגישות_לתרופות"
OTHER_ALLERGY_OUTPUT_COL = "רגישות_אחרת"
CHRONIC_MEDS_OUTPUT_COL = "תרופות_קבועות"

# --- surgery meds ---
SURGERY_MEDS_KEY = "מספר מקרה"
SURGERY_MEDS_VALUE_COL = "תרופה"
SURGERY_MEDS_OUTPUT_COL = "תרופות_בניתוח"
SURGERY_MEDS_DROP_COLS = ["אופן מתן"]

# --- staged merge outputs ---
MERGED_AFTER_BP_NAME = "surgeries_with_bmi_smoking_bp"
MERGED_AFTER_SAT_NAME = "surgeries_with_bmi_smoking_bp_sat"
MERGED_AFTER_PATIENT_NAME = "surgeries_with_bmi_smoking_bp_sat_patient"
MERGED_AFTER_SURGERY_MEDS_NAME = "surgeries_with_bmi_smoking_bp_sat_patient_meds"

# --- feature engineering ---
COLS_TO_DROP = ["רגישות_אחרת", "תרופות_קבועות", "חדר"]
AGE_YOUNG_MAX = 18
AGE_ELDERLY_MIN = 65
BMI_UNDERWEIGHT_MAX = 18.5
BMI_NORMAL_MAX = 25.0
BMI_OVERWEIGHT_MAX = 30.0

# --- drugs index ---
DRUGS_DIR = Path("data/Drugs")
DRUGS_OUTPUT_DIR = DRUGS_DIR / "Output"
DRUGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DRUG_INDEX_PATH = DRUGS_DIR / "unique_drug_names.csv"
CLEANED_DRUGS_COL = "CLEANED_DRUGS"
CLEANED_DRUG_ALLERGIES_COL = "CLEANED_DRUG_ALLERGIES"
CLEANED_DRUGS_EXPORT_PATH = DRUGS_OUTPUT_DIR / "cleaned_drugs_comparison.csv"
BG_COUNT_COL = "מחלות_רקע_count"
DRUG_ALLERGY_COUNT_COL = "רגישות_לתרופות_count"
PROCEDURE_CODE_COUNT_COL = "קוד_פרוצדורה_count"

# --- labs ---
LAB_DATE_COL = "תאריך בדיקה"
LAB_TEST_COL = "שם בדיקה"
LAB_VALUE_COL = "ערך"
LABS_FILL_VALUE = 0
LAB_SELECTED_TESTS = [
    "ALT(GPT)-B", "AST(GOT)-B", "Albumin-B", "Albumin/Globulin-B",
    "Alkaline Phosphatase-B", "Amylase-B", "BASO abs", "BASO%", "BUN-B",
    "Bilirubin, total-B", "CK-B", "CRP", "Calcium-B", "Creatinine-B",
    "EOS abs", "EOS%", "GGT- Gamma Glutam.Trans.-B", "Globulin - blood",
    "Glucose-B", "HCT", "HGB", "Hemolysis Index -B", "Icteric Index-B",
    "LD Lactate dehydrogenase-B", "LYMPHO abs", "LYMPHO%", "Lipemic Index-B",
    "MCH", "MCHC", "MCV", "MONO abs", "MONO%", "MPV", "Magnesium-B",
    "NEUTRO abs", "NEUTRO%", "NRBC  abs",
    "NRBC (Nucleated red blood cells) / leukocytes % - blood",
    "OSM-cal-B", "PCT", "PDW", "PT,SEC", "PT-INR-B", "PTT-blood",
    "Phosphor -B", "Platelet, automated count - blood", "Potassium-B",
    "Protein, total-B", "RBC", "RDW", "Sodium-B", "Uric acid-B", "WBC",
]
MERGED_AFTER_LABS_NAME = "surgeries_merged_final"

# --- final dataset normalization ---
ANESTHESIA_OHE_PREFIX = "סוג הרדמה"
SURGERY_TYPE_OHE_PREFIX = "סוג ניתוח"
OHE_DROP_UNKNOWN_COLS = [
    f"{SURGERY_TYPE_OHE_PREFIX}_UNKNOWN",
    f"{ANESTHESIA_OHE_PREFIX}_UNKNOWN",
]
SAT_NORM_COL = "oxygen_saturation_norm"
SBP_Z_COL = "systolic_bp_Z"
DBP_Z_COL = "diastolic_bp_Z"
BMI_Z_COL = "BMI_Z"
SMOKER_COL = "smoker"

# --- English ML dataset export ---
BG_TOP_K_ICD9 = 65
BG_ICD9_OHE_PREFIX = "ICD9"
ESTIMATED_SURGERY_TIME_COL = "Estimated_Surgery_Time"
SURGERY_TIME_ANOMALY_COL = "Surgery_Time_Anomaly"
NUM_MEDICATIONS_COL = "Num_Medications"
NUM_ALLERGY_MEDICATIONS_COL = "Num_Allergy_Medications"
ML_DATASET_NAME = "surgeries_ml_ready"
SURGICAL_DEPARTMENT_COL = "Surgical_Department"
ML_EXPORT_DIR = CLEAN_DIR / "export"
ML_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

DEPARTMENT_EXPORT_FILES = {
    "אורטופדיה": "surgeries_ml_ready_orthopedics.csv",
    "כירורגיה": "surgeries_ml_ready_general_surgery.csv",
    "אף אוזן גרון": "surgeries_ml_ready_ent.csv",
}

ML_COLUMNS_DROP = [
    "שם פרוצדורה",
    "שם אבחנה",
    "start_time_source",
    "end_time_source",
    "duration_quality",
    BP_SBP_COL,
    BP_DBP_COL,
    SAT_COL,
    "קוד אבחנה",
    "גובה",
    "משקל",
    BG_COUNT_COL,
    PROCEDURE_CODE_COUNT_COL,
]
