import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_validate, StratifiedKFold

def create_model_pipeline(preprocessor: ColumnTransformer, classifier: ClassifierMixin) -> Pipeline:
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return model_pipeline

def evaluate_model_with_cross_validation(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5) -> dict:
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scoring = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro"
    ]

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1
    )

    return cv_results