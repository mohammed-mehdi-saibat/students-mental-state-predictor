import pandas as pd
from sklearn.model_selection import train_test_split

def data_clues_and_target(df: pd.DataFrame) -> dict:
    data_clues_and_target = {
        "X" : df.drop(columns=["student_id", "mental_health_risk"]),
        "y" : df["mental_health_risk"],
    }   

    return data_clues_and_target

def data_train_test_split(df: pd.DataFrame, random_state = 42,test_size=0.2) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        data_clues_and_target(df)["X"],
        data_clues_and_target(df)["y"],
        test_size=test_size,
        stratify=data_clues_and_target(df)["y"],
        random_state=random_state   
    )
    return X_train, X_test, y_train, y_test

def identify_feature_groups(df: pd.DataFrame) -> dict:
    feature_groups = {
        "numerical_group" : df.select_dtypes(include="number"),
        "ordinal_categorical_group" : df[["physical_activity", "diet_quality"]],
        "nominal_categorical_group" : df[["gender", "part_time_job", "previous_mental_health_history"]]
    }

    return feature_groups
  