import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for E-Commerce Customer Churn dataset.
    - trim column names
    - drop ID cols
    - handle missing values in numeric and categorical columns
    - ensure target is 0/1
    """
    # tidy headers
    df.columns = df.columns.str.strip()

    # drop IDs (not needed for model)
    for col in ["CustomerID", "customer_id", "customerID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # target should already be 0/1 for this dataset, but ensure it
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1, 0: 0, 1: 1})

    # Handle missing values in numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Remove target from numeric cols if present
    numeric_cols = [col for col in numeric_cols if col != target_col]

    # For numeric columns, fill missing values with median (more robust than mean for churn data)
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Handle missing values in categorical columns - fill with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
            df[col] = df[col].fillna(mode_val)

    # Ensure binary columns are 0/1 where expected
    binary_cols = ["Complain"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df
