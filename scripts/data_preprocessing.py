import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def data_clues_and_target(df: pd.DataFrame) -> dict:
    data_clues_and_target = {
        "X": df.drop(columns=["student_id", "mental_health_risk"]),
        "y": df["mental_health_risk"],
    }
    return data_clues_and_target


def data_train_test_split(df: pd.DataFrame, random_state=42, test_size=0.2) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        data_clues_and_target(df)["X"],
        data_clues_and_target(df)["y"],
        test_size=test_size,
        stratify=data_clues_and_target(df)["y"],
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def identify_feature_groups(df: pd.DataFrame) -> dict:
    feature_groups = {
        "numerical_group": df.select_dtypes(include="number").columns.tolist(),
        "ordinal_categorical_group": df[
            ["physical_activity", "diet_quality"]
        ].columns.tolist(),
        "nominal_categorical_group": df[["gender"]].columns.tolist(),
        "binary_categorical_group": df[
            ["part_time_job", "previous_mental_health_history"]
        ].columns.tolist(),
    }
    return feature_groups


def create_numerical_imputer() -> SimpleImputer:
    numerical_imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")
    return numerical_imputer


def create_categorical_imputer() -> SimpleImputer:
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    return categorical_imputer


def create_ordinal_encoder() -> OrdinalEncoder:
    ordinal_encoder = OrdinalEncoder(
        categories=[["Low", "Moderate", "High"], ["Poor", "Average", "Good"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    return ordinal_encoder


def create_nominal_encoder(drop_first: bool = False) -> OneHotEncoder:
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first" if drop_first else None,
        sparse_output=False,
    )
    return one_hot_encoder


def create_binary_encoder() -> OrdinalEncoder:
    binary_encoder = OrdinalEncoder(
        categories=[["No", "Yes"], ["No", "Yes"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    return binary_encoder


def handle_social_media_screen_time_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    social_media_screen_time_inconsistencies = (
        df["social_media_hours"] > df["screen_time_hours"]
    )
    df.loc[social_media_screen_time_inconsistencies, "social_media_hours"] = (
        df[["social_media_hours", "screen_time_hours"]]
    ).min(axis=1)
    return df


# Make the above function usable by sklearn in a pipeline using FunctionTransfrormer
def create_social_media_screen_time_transformer() -> FunctionTransformer:
    social_media_screen_time_transformer = FunctionTransformer(
        func=handle_social_media_screen_time_inconsistencies,
        validate=False,
        feature_names_out="one-to-one",
    )
    return social_media_screen_time_transformer


def create_numerical_scaler() -> StandardScaler:
    numerical_scaler = StandardScaler()
    return numerical_scaler


def create_numerical_pipeline(
    imputer: SimpleImputer, transformer: FunctionTransformer, scaler: StandardScaler
) -> Pipeline:
    numerical_pipeline = Pipeline(
        [("imputer", imputer), ("transformer", transformer), ("scaler", scaler)]
    )
    return numerical_pipeline


def create_ordinal_pipeline(
    imputer: SimpleImputer, encoder: OrdinalEncoder
) -> Pipeline:
    ordinal_pipeline = Pipeline([("imputer", imputer), ("encoder", encoder)])
    return ordinal_pipeline


def create_nominal_pipeline(imputer: SimpleImputer, encoder: OneHotEncoder) -> Pipeline:
    nominal_pipeline = Pipeline([("imputer", imputer), ("encoder", encoder)])
    return nominal_pipeline


def create_binary_pipeline(imputer: SimpleImputer, encoder: OrdinalEncoder) -> Pipeline:
    binary_pipeline = Pipeline([("imputer", imputer), ("encoder", encoder)])
    return binary_pipeline


def create_preprocessor(
    numerical_pipeline: Pipeline,
    ordinal_pipeline: Pipeline,
    nominal_pipeline: Pipeline,
    binary_pipeline: Pipeline,
    numerical_group: list,
    ordinal_categorical_group: list,
    nominal_categorical_group: list,
    binary_categorical_group: list,
) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        [
            ("num_pipeline", numerical_pipeline, numerical_group),
            ("ordinal_pipeline", ordinal_pipeline, ordinal_categorical_group),
            ("nominal_pipeline", nominal_pipeline, nominal_categorical_group),
            ("binary_pipeline", binary_pipeline, binary_categorical_group),
        ]
    )
    return preprocessor
