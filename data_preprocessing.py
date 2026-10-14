"""
Data Preprocessing Module for Telco Customer Churn Prediction.
Modular helper functions for loading, cleaning, encoding, splitting, and scaling data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, Optional, List


def load_data(filepath_or_buffer: Any) -> pd.DataFrame:
    """
    Load customer churn dataset from a CSV filepath or file-like buffer.

    Parameters:
        filepath_or_buffer: Path to CSV or uploaded file buffer.

    Returns:
        pd.DataFrame: Raw dataframe loaded from CSV.
    """
    df = pd.read_csv(filepath_or_buffer)
    return df


def clean_dataset(df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
    """
    Clean the Telco Customer Churn dataframe:
    - Convert 'TotalCharges' to numeric (coercing spaces and invalid entries to NaN)
    - Drop missing values resulting from empty charges
    - Remove duplicate records
    - Drop 'customerID' identifier column if requested

    Parameters:
        df (pd.DataFrame): Raw dataframe.
        drop_id (bool): Whether to drop the 'customerID' column.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df_clean = df.copy()

    # Convert TotalCharges to numeric
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # Drop missing values
    df_clean.dropna(inplace=True)

    # Drop duplicate records
    df_clean.drop_duplicates(inplace=True)

    # Drop identifier column if present
    if drop_id and "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    return df_clean


def extract_features_and_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "Churn"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract specified input features and map target variable to binary (1/0).

    Default features based on notebook:
    - 'tenure'
    - 'MonthlyCharges'
    - 'TotalCharges'
    - 'SeniorCitizen'

    Parameters:
        df (pd.DataFrame): Cleaned dataframe.
        feature_cols (List[str], optional): List of feature columns.
        target_col (str): Column name of target.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and binary target vector y.
    """
    if feature_cols is None:
        feature_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

    # Verify columns exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns in dataset: {missing_features}")

    X = df[feature_cols].copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y_raw = df[target_col]

    # Map target string to binary integer if needed
    if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
        y = y_raw.map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    else:
        y = y_raw.astype(int)

    return X, y


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Perform one-hot encoding on categorical features with drop_first=True.

    Parameters:
        X (pd.DataFrame): Feature dataframe.

    Returns:
        pd.DataFrame: Numerically encoded dataframe.
    """
    return pd.get_dummies(X, drop_first=True, dtype=int)


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing partitions.

    Parameters:
        X (pd.DataFrame): Input features.
        y (pd.Series): Target labels.
        test_size (float): Fraction of dataset for testing. Default 0.2.
        random_state (int): Seed for reproducibility.
        stratify (bool): Whether to stratify splits based on y.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """
    Fit or apply StandardScaler to features.

    Parameters:
        X_train (pd.DataFrame or np.ndarray): Training features.
        X_test (pd.DataFrame or np.ndarray, optional): Testing features.
        scaler (StandardScaler, optional): Pre-fitted scaler instance.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
            Scaled X_train, Scaled X_test (if provided), and the fitted scaler.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_test_scaled, scaler


def preprocess_single_customer(
    tenure: int,
    monthly_charges: float,
    total_charges: float,
    senior_citizen: int,
    scaler: StandardScaler
) -> np.ndarray:
    """
    Preprocess a single customer's attributes for live inference.

    Parameters:
        tenure (int): Tenure in months.
        monthly_charges (float): Monthly bill amount.
        total_charges (float): Total bill amount.
        senior_citizen (int): 1 if senior citizen, 0 otherwise.
        scaler (StandardScaler): Fitted StandardScaler.

    Returns:
        np.ndarray: Scaled 2D array of shape (1, 4) ready for model prediction.
    """
    df_input = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": senior_citizen
    }])

    return scaler.transform(df_input)


def generate_eda_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate exploratory data analysis summary metrics for display.

    Parameters:
        df (pd.DataFrame): Dataset dataframe.

    Returns:
        Dict[str, Any]: Statistical summary and data health metrics.
    """
    total_rows = len(df)
    churn_count = (df["Churn"].map({"Yes": 1, "No": 0}) == 1).sum() if "Churn" in df.columns else 0
    churn_rate = (churn_count / total_rows * 100) if total_rows > 0 else 0.0

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "column_names": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "churn_count": int(churn_count),
        "churn_rate": float(churn_rate),
        "numerical_describe": df.describe().to_dict() if not df.empty else {}
    }
