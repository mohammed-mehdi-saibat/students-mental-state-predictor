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
  