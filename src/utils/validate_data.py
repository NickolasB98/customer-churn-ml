from typing import Tuple, List
import pandas as pd


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Data validation for Telco Customer Churn dataset using pandas.

    Validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    """
    print("🔍 Starting data validation...")

    # Convert TotalCharges to numeric (may have empty strings)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    failed_checks = []
    passed_checks = 0
    total_checks = 0

    # === SCHEMA VALIDATION ===
    print("   📋 Validating schema and required columns...")

    required_cols = ["customerID", "gender", "Partner", "Dependents", "PhoneService",
                     "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges"]

    for col in required_cols:
        total_checks += 1
        if col not in df.columns:
            failed_checks.append(f"Missing column: {col}")
        else:
            passed_checks += 1

    # === NULL VALUES CHECK ===
    critical_cols = ["customerID", "tenure", "MonthlyCharges"]
    for col in critical_cols:
        total_checks += 1
        if col in df.columns and df[col].isnull().any():
            failed_checks.append(f"Null values in {col}")
        else:
            passed_checks += 1

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    # Gender values
    total_checks += 1
    if "gender" in df.columns:
        valid_genders = set(df["gender"].unique()) <= {"Male", "Female"}
        if valid_genders:
            passed_checks += 1
        else:
            failed_checks.append("Invalid gender values")

    # Yes/No fields
    for col in ["Partner", "Dependents", "PhoneService"]:
        total_checks += 1
        if col in df.columns:
            valid_vals = set(df[col].unique()) <= {"Yes", "No"}
            if valid_vals:
                passed_checks += 1
            else:
                failed_checks.append(f"Invalid values in {col}")

    # Contract types
    total_checks += 1
    if "Contract" in df.columns:
        valid_contracts = set(df["Contract"].unique()) <= {"Month-to-month", "One year", "Two year"}
        if valid_contracts:
            passed_checks += 1
        else:
            failed_checks.append("Invalid contract types")

    # Internet service
    total_checks += 1
    if "InternetService" in df.columns:
        valid_internet = set(df["InternetService"].unique()) <= {"DSL", "Fiber optic", "No"}
        if valid_internet:
            passed_checks += 1
        else:
            failed_checks.append("Invalid internet service types")

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges...")

    # Tenure: 0-120 months
    total_checks += 1
    if "tenure" in df.columns:
        if (df["tenure"] >= 0).all() and (df["tenure"] <= 120).all():
            passed_checks += 1
        else:
            failed_checks.append("Tenure values out of valid range")

    # Monthly charges: 0-200
    total_checks += 1
    if "MonthlyCharges" in df.columns:
        if (df["MonthlyCharges"] >= 0).all() and (df["MonthlyCharges"] <= 200).all():
            passed_checks += 1
        else:
            failed_checks.append("MonthlyCharges values out of valid range")

    # Total charges: >= 0 (skip NaN values - they're acceptable for new customers)
    total_checks += 1
    if "TotalCharges" in df.columns:
        valid_charges = (df["TotalCharges"] >= 0).all() or df["TotalCharges"].isna().any()
        # Check that non-NaN values are >= 0
        non_nan_valid = df["TotalCharges"].dropna().apply(lambda x: x >= 0).all() if df["TotalCharges"].notna().any() else True
        if non_nan_valid:
            passed_checks += 1
        else:
            failed_checks.append("TotalCharges has negative values")

    # === CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")

    total_checks += 1
    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        # Only check consistency for rows where TotalCharges is not NaN
        valid_rows = df[df["TotalCharges"].notna()]
        if len(valid_rows) > 0:
            consistency_check = (valid_rows["TotalCharges"] >= valid_rows["MonthlyCharges"]).sum() / len(valid_rows)
            if consistency_check >= 0.95:
                passed_checks += 1
            else:
                failed_checks.append("TotalCharges consistency issue (>5% of rows violate logic)")
        else:
            passed_checks += 1  # No valid rows to check

    # === RESULTS ===
    is_valid = len(failed_checks) == 0

    if is_valid:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {len(failed_checks)} checks failed")
        for check in failed_checks:
            print(f"   - {check}")

    return is_valid, failed_checks
