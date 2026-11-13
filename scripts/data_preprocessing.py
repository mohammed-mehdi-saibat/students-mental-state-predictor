import pandas as pd
from sklearn.model_selection import train_test_split

def data_clues_and_target(df: pd.DataFrame):
    data_clues_and_target = {
        "X" : df.drop(columns=["student_id", "mental_health_risk"]),
        "y" : df["mental_health_risk"],
    }   

    return data_clues_and_target
