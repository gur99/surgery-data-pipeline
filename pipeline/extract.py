from pathlib import Path

import pandas as pd

from pipeline.config import FILES, LAB_COLUMNS, RAW_DIR, SQL

def extract_surgeries(conn=None):
    """Pull surgeries table from DB or local file."""
    if conn is not None:
        return pd.read_sql(SQL["surgeries"], conn)
    return pd.read_excel(RAW_DIR / FILES["surgeries"])


def extract_bmi(conn=None):
    """Pull BMI table from DB or local file."""
    if conn is not None:
        return pd.read_sql(SQL["bmi"], conn)
    return pd.read_excel(RAW_DIR / FILES["bmi"])


def extract_smoking(conn=None):
    """Pull smoking table from DB or local file."""
    if conn is not None:
        return pd.read_sql(SQL["smoking"], conn)
    return pd.read_excel(RAW_DIR / FILES["smoking"])


def extract_bp(conn=None):
    if conn is not None:
        return pd.read_sql(SQL["bp"], conn)
    return pd.read_excel(RAW_DIR / FILES["bp"])


def extract_saturation(conn=None):
    if conn is not None:
        return pd.read_sql(SQL["saturation"], conn)
    return pd.read_excel(RAW_DIR / FILES["saturation"])


def extract_background_diseases(conn=None):
    if conn is not None:
        return pd.read_sql(SQL["background_diseases"], conn)
    return pd.read_excel(RAW_DIR / FILES["background_diseases"])


def extract_drug_allergy(conn=None):
    if conn is not None:
        return pd.read_sql(SQL["drug_allergy"], conn)
    return pd.read_excel(RAW_DIR / FILES["drug_allergy"])


def extract_other_allergy(conn=None):
    if conn is not None:
        return pd.read_sql(SQL["other_allergy"], conn)
    return pd.read_excel(RAW_DIR / FILES["other_allergy"])


def extract_chronic_meds(conn=None):
    if conn is not None:
        return pd.read_sql(SQL["chronic_meds"], conn)
    return pd.read_excel(RAW_DIR / FILES["chronic_meds"])


def extract_surgery_meds(conn=None):
    if conn is not None:
        return pd.read_sql(SQL["surgery_meds"], conn)
    return pd.read_excel(RAW_DIR / FILES["surgery_meds"])


def extract_labs(conn=None):
    """Pull labs table from DB or local CSV (no header)."""
    if conn is not None:
        return pd.read_sql(SQL["labs"], conn)
    return pd.read_csv(RAW_DIR / FILES["labs"], header=None, names=LAB_COLUMNS)
