import os
from pathlib import Path

import pendulum
from airflow.sdk import dag, task


# Define project and dataset paths inside Docker
project_root = Path(os.environ.get("SC1_PROJECT_ROOT", "/opt/airflow/project"))
data_path = project_root / "data" / "raw" / "student_mental_health.csv"
processed_path = project_root / "data" / "processed"
models_path = project_root / "models"
metrics_path = project_root / "reports" / "metrics" / "airflow_evaluation.json"


@dag(
    dag_id="sc1_ml_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 11, 16, tz="Africa/Casablanca"),
    catchup=False,
    tags=["sc1", "machine-learning"],
)
def sc1_pipeline():

    @task
    def validate_data():
        from scripts.data_validation import validate_dataset

        # Check dataset before training
        results = validate_dataset(str(data_path))
        print("DATASET VALIDATION RESULTS")
        print(results)

    @task
    def prepare_data():
        import pandas as pd
        from scripts.data_preprocessing import data_train_test_split

        # Load dataset and create stratified split
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.lower()
        X_train, X_test, y_train, y_test = data_train_test_split(df)

        # Save datasets for the next tasks
        processed_path.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(processed_path / "X_train.csv", index=False)
        X_test.to_csv(processed_path / "X_test.csv", index=False)
        y_train.to_csv(processed_path / "y_train.csv", index=False)
        y_test.to_csv(processed_path / "y_test.csv", index=False)

        print("DATA PREPARATION COMPLETED")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")

    @task
    def train_model() -> str:
        import joblib
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from scripts.data_preprocessing import (
            identify_feature_groups,
            create_numerical_imputer,
            create_categorical_imputer,
            create_ordinal_encoder,
            create_nominal_encoder,
            create_binary_encoder,
            create_social_media_screen_time_transformer,
            create_numerical_scaler,
            create_numerical_pipeline,
            create_ordinal_pipeline,
            create_nominal_pipeline,
            create_binary_pipeline,
            create_preprocessor,
        )
        from scripts.model_training import create_model_pipeline

        # Load original training data
        X_train = pd.read_csv(processed_path / "X_train.csv")
        y_train = pd.read_csv(processed_path / "y_train.csv")["mental_health_risk"]
        groups = identify_feature_groups(X_train)

        # Reuse the same preprocessing as the notebook
        numerical_pipeline = create_numerical_pipeline(
            imputer=create_numerical_imputer(),
            transformer=create_social_media_screen_time_transformer(),
            scaler=create_numerical_scaler(),
        )
        ordinal_pipeline = create_ordinal_pipeline(
            imputer=create_categorical_imputer(),
            encoder=create_ordinal_encoder(),
        )
        nominal_pipeline = create_nominal_pipeline(
            imputer=create_categorical_imputer(),
            encoder=create_nominal_encoder(),
        )
        binary_pipeline = create_binary_pipeline(
            imputer=create_categorical_imputer(),
            encoder=create_binary_encoder(),
        )
        preprocessor = create_preprocessor(
            numerical_pipeline=numerical_pipeline,
            ordinal_pipeline=ordinal_pipeline,
            nominal_pipeline=nominal_pipeline,
            binary_pipeline=binary_pipeline,
            **groups,
        )

        # Train Logistic Regression with the selected C value
        classifier = LogisticRegression(C=10, max_iter=1000, random_state=42)
        model = create_model_pipeline(preprocessor, classifier)
        model.fit(X_train, y_train)

        # Store a candidate model until evaluation succeeds
        models_path.mkdir(parents=True, exist_ok=True)
        candidate_path = models_path / "sc1_airflow_candidate.joblib"
        joblib.dump(model, candidate_path)
        print(f"MODEL TRAINED: {candidate_path}")
        return str(candidate_path)

    @task
    def evaluate_model(candidate_path: str) -> str:
        import json
        import joblib
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report,
            confusion_matrix,
        )
        from scripts.model_training import evaluate_model_with_cross_validation

        # Load fitted model and the independent test split
        model = joblib.load(candidate_path)
        X_train = pd.read_csv(processed_path / "X_train.csv")
        y_train = pd.read_csv(processed_path / "y_train.csv")["mental_health_risk"]
        X_test = pd.read_csv(processed_path / "X_test.csv")
        y_test = pd.read_csv(processed_path / "y_test.csv")["mental_health_risk"]

        # Measure cross-validation performance on training data
        cv_results = evaluate_model_with_cross_validation(
            pipeline=model, X_train=X_train, y_train=y_train, n_splits=5
        )

        # Evaluate the independent test set once for this project
        predictions = model.predict(X_test)
        labels = ["Low", "Medium", "High"]
        results = {
            "model": "Logistic Regression",
            "cv_f1_macro": float(cv_results["test_f1_macro"].mean()),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "test_f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
            "classification_report": classification_report(
                y_test, predictions, labels=labels, output_dict=True, zero_division=0
            ),
        }

        # Save evaluation results for monitoring and Streamlit
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        print("MODEL EVALUATION COMPLETED")
        print(f"CV F1 Macro: {results['cv_f1_macro']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Test F1 Macro: {results['test_f1_macro']:.4f}")
        print(f"Metrics saved: {metrics_path}")
        return str(metrics_path)

    @task
    def save_model(candidate_path: str, evaluation_path: str):
        import shutil

        # Publish model only after evaluation succeeded
        final_path = models_path / "logistic_regression_airflow.joblib"
        shutil.copy2(candidate_path, final_path)
        print(f"MODEL SAVED: {final_path}")
        print(f"EVALUATION AVAILABLE: {evaluation_path}")

    # Execute all tasks in order during one DAG run
    validation = validate_data()
    preparation = prepare_data()
    candidate = train_model()
    evaluation = evaluate_model(candidate)
    saved_model = save_model(candidate, evaluation)

    validation >> preparation >> candidate >> evaluation >> saved_model


sc1_pipeline()
