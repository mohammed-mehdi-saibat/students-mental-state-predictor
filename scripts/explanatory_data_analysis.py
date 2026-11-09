import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from pathlib import Path

root = Path.cwd().resolve().parents[0]
raw_data_path = root / "data" / "raw" / "student_mental_health.csv"

def fetch_data_and_turn_into_dataframe_and_convert_column_names_to_lowercase(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=str.lower)
    return df

def extract_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include="object")
    return categorical_columns

def extract_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    numerical_columns = df.select_dtypes(include="number")
    return numerical_columns












