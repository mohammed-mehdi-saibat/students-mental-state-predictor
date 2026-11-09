import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from pathlib import Path

root = Path.cwd().resolve().parents[0]
raw_data_path = root / "data" / "raw" / "student_mental_health.csv"

def fetch_dataset_and_turn_into_dataframe_and_convert_column_names_to_lowercase(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=str.lower)
    return df

def extract_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return categorical_columns

def extract_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    return numerical_columns

def dataset_structure_and_basic_inspection(df: pd.DataFrame) -> None:
    # Inspect dataset shape
    print("=" * 30)
    print(f"DATASET SHAPE: \n{df.shape[0]} rows, {df.shape[1]} columns")
    # Inspect column name
    print("=" * 30)
    print(f"COLUMN NAMES: \n{df.columns.tolist()}")
    # Inspect data types
    print("=" * 30)
    print(f"DATA TYPES: \n{df.dtypes}")
    # Display the first and last rows
    print("=" * 30)
    print("FIRST 5 ROWS: \n")
    display(df.head())
    print("LAST 5 ROWS: \n")
    display(df.tail())
    # Inspect dataset info
    print("=" * 30)
    df.info()
    print("=" * 30)







