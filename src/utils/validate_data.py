from typing import Tuple, List
import pandas as pd


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Data validation for E-Commerce Customer Churn dataset using pandas.

    Validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    """
    print("🔍 Starting data validation...")

    failed_checks = []
    passed_checks = 0
    total_checks = 0

    # === SCHEMA VALIDATION ===
    print("   📋 Validating schema and required columns...")

    required_cols = ["CustomerID", "Churn", "Tenure", "PreferredLoginDevice",
                     "CityTier", "Gender", "PreferredPaymentMode", "PreferedOrderCat"]

    for col in required_cols:
        total_checks += 1
        if col not in df.columns:
            failed_checks.append(f"Missing column: {col}")
        else:
            passed_checks += 1

    # === NULL VALUES CHECK ===
    critical_cols = ["CustomerID", "Churn", "Gender"]
    for col in critical_cols:
        total_checks += 1
        if col in df.columns and df[col].isnull().any():
            failed_checks.append(f"Null values in critical column: {col}")
        else:
            passed_checks += 1

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    # Gender values
    total_checks += 1
    if "Gender" in df.columns:
        valid_genders = set(df["Gender"].unique()) <= {"Male", "Female"}
        if valid_genders:
            passed_checks += 1
        else:
            failed_checks.append("Invalid gender values")

    # Marital Status
    total_checks += 1
    if "MaritalStatus" in df.columns:
        valid_status = set(df["MaritalStatus"].dropna().unique()) <= {"Single", "Married", "Divorced"}
        if valid_status:
            passed_checks += 1
        else:
            failed_checks.append("Invalid marital status values")

    # City Tier (1-3)
    total_checks += 1
    if "CityTier" in df.columns:
        if set(df["CityTier"].unique()) <= {1, 2, 3}:
            passed_checks += 1
        else:
            failed_checks.append("Invalid CityTier values (should be 1-3)")

    # Satisfaction Score (1-5)
    total_checks += 1
    if "SatisfactionScore" in df.columns:
        if set(df["SatisfactionScore"].unique()) <= {1, 2, 3, 4, 5}:
            passed_checks += 1
        else:
            failed_checks.append("Invalid SatisfactionScore values (should be 1-5)")

    # Churn values (0 or 1)
    total_checks += 1
    if "Churn" in df.columns:
        if set(df["Churn"].dropna().unique()) <= {0, 1}:
            passed_checks += 1
        else:
            failed_checks.append("Invalid Churn values (should be 0 or 1)")

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges...")

    # Tenure: 0-61 months
    total_checks += 1
    if "Tenure" in df.columns:
        valid_tenure = df["Tenure"].dropna().apply(lambda x: 0 <= x <= 61).all()
        if valid_tenure or len(df["Tenure"].dropna()) == 0:
            passed_checks += 1
        else:
            failed_checks.append("Tenure values out of valid range (0-61)")

    # HourSpendOnApp: 0-5 hours
    total_checks += 1
    if "HourSpendOnApp" in df.columns:
        valid_hours = df["HourSpendOnApp"].dropna().apply(lambda x: 0 <= x <= 5).all()
        if valid_hours or len(df["HourSpendOnApp"].dropna()) == 0:
            passed_checks += 1
        else:
            failed_checks.append("HourSpendOnApp values out of valid range (0-5)")

    # WarehouseToHome: 5-127 km
    total_checks += 1
    if "WarehouseToHome" in df.columns:
        valid_distance = df["WarehouseToHome"].dropna().apply(lambda x: 5 <= x <= 127).all()
        if valid_distance or len(df["WarehouseToHome"].dropna()) == 0:
            passed_checks += 1
        else:
            failed_checks.append("WarehouseToHome values out of valid range (5-127)")

    # OrderCount: >= 0
    total_checks += 1
    if "OrderCount" in df.columns:
        valid_orders = df["OrderCount"].dropna().apply(lambda x: x >= 0).all()
        if valid_orders or len(df["OrderCount"].dropna()) == 0:
            passed_checks += 1
        else:
            failed_checks.append("OrderCount has negative values")

    # === CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")

    total_checks += 1
    # NumberOfAddress should be >= 1
    if "NumberOfAddress" in df.columns:
        if (df["NumberOfAddress"] >= 1).all():
            passed_checks += 1
        else:
            failed_checks.append("NumberOfAddress should be >= 1")

    # === RESULTS ===
    is_valid = len(failed_checks) == 0

    if is_valid:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {len(failed_checks)} checks failed")
        for check in failed_checks:
            print(f"   - {check}")

    return is_valid, failed_checks
