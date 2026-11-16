import pandas as pd
from pathlib import Path

def validate_dataset(path: str) -> dict:

    # Check if dataset exists
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Load dataset
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    # Check required columns
    required_columns = [
        "student_id",
        "age",
        "gender",
        "year_of_study",
        "sleep_hours",
        "study_hours_per_week",
        "gpa",
        "social_media_hours",
        "screen_time_hours",
        "physical_activity",
        "financial_stress",
        "family_support",
        "attendance_rate",
        "part_time_job",
        "diet_quality",
        "previous_mental_health_history",
        "mental_health_risk"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Check target values
    if df["mental_health_risk"].isna().any():
        raise ValueError("Missing target values detected")

    valid_classes = {"Low", "Medium", "High"}
    invalid_classes = set(df["mental_health_risk"]) - valid_classes

    if invalid_classes:
        raise ValueError(f"Invalid target classes: {invalid_classes}")

    # Check student identifiers
    if df["student_id"].isna().any():
        raise ValueError("Missing student identifiers detected")

    if not df["student_id"].is_unique:
        raise ValueError("Duplicated student identifiers detected")

    # Check duplicate feature profiles
    feature_duplicates = df.drop(
        columns=["student_id", "mental_health_risk"]
    ).duplicated().sum()

    if feature_duplicates > 0:
        raise ValueError(f"Duplicate feature profiles: {feature_duplicates}")

    # Return validation results
    results = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "target_distribution": df["mental_health_risk"].value_counts().to_dict()
    }

    return results
