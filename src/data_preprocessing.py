"""
data_preprocessing.py
----------------------
Loads, cleans, and prepares the Frankfurt Diabetes dataset for modelling.

Pipeline:
  1. Load CSV
  2. Remove rows with too many zero-valued physiological measurements
  3. Remove outliers (Pregnancies, SkinThickness, etc.)
  4. Impute remaining zeros with column means
  5. Scale numeric features (StandardScaler)
  6. Define categorical / numeric feature splits
  7. Train/test split (80/20, stratified)
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["Pregnancies"]
NUMERIC_FEATURES = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_FEATURE = "Outcome"

# Columns that should not legitimately be zero (physiological impossibility)
ZERO_CHECK_COLS = ["Insulin", "BMI", "SkinThickness", "BloodPressure", "Glucose"]

# Outlier thresholds (inclusive upper bound → rows *above* are removed)
OUTLIER_THRESHOLDS = {
    "Pregnancies": 14,   # >= 15 considered outlier
    "SkinThickness": 80, # physiologically implausible above ~80 mm
}

# Maximum number of zero-valued physiological fields allowed per row
MAX_ZERO_COUNT = 2


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
    return df


def remove_sparse_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where too many physiological columns are zero.
    Rows with more than MAX_ZERO_COUNT zeros across ZERO_CHECK_COLS are dropped.
    """
    zero_count = (df[ZERO_CHECK_COLS] == 0).sum(axis=1)
    filtered = df[zero_count <= MAX_ZERO_COUNT].copy()
    print(f"[remove_sparse_rows] {len(df) - len(filtered)} rows removed → {len(filtered)} remaining.")
    return filtered


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physiologically implausible outlier rows."""
    original = len(df)
    for col, threshold in OUTLIER_THRESHOLDS.items():
        if col in df.columns:
            df = df[df[col] <= threshold].copy()
    print(f"[remove_outliers] {original - len(df)} outlier rows removed → {len(df)} remaining.")
    return df


def impute_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace zero values in physiological columns with column means.
    Zeros in these columns represent missing measurements, not true zeros.
    """
    imputer = SimpleImputer(missing_values=0, strategy="mean")
    df[ZERO_CHECK_COLS] = imputer.fit_transform(df[ZERO_CHECK_COLS])
    print("[impute_zeros] Zero-imputation with column means complete.")
    return df


def scale_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise numeric features (zero mean, unit variance).
    Returns the transformed DataFrame and the fitted scaler for inference use.
    """
    scaler = StandardScaler()
    df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES]).astype("float32")
    print("[scale_numeric] Numeric features standardised.")
    return df, scaler


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split the processed DataFrame into train/test feature and target arrays.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    """
    X = df[FEATURES]
    y = df[TARGET_FEATURE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[split_data] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def run_preprocessing(filepath: str) -> tuple:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV (e.g. 'data/raw/frankfurt_diabetes.csv').

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, df_processed
    """
    df = load_data(filepath)
    df = remove_sparse_rows(df)
    df = remove_outliers(df)
    df = impute_zeros(df)
    df, scaler = scale_numeric(df)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test, scaler, df
