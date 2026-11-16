
import os
from pathlib import Path

import pendulum
from airflow.sdk import dag, task


# Define dataset path inside Docker
project_root = Path(
    os.environ.get("SC1_PROJECT_ROOT", "/opt/airflow/project")
)

data_path = project_root / "data" / "raw" / "student_mental_health.csv"


@dag(
    dag_id="sc1_ml_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 11, 16, tz="Africa/Casablanca"),
    catchup=False,
    tags=["sc1", "machine-learning"]
)
def sc1_pipeline():

    @task
    def validate_data():

        # Import dataset validation function
        from scripts.data_validation import validate_dataset

        # Validate raw dataset
        results = validate_dataset(str(data_path))

        print("DATASET VALIDATION RESULTS")
        print(results)

        return results

    validate_data()


sc1_pipeline()
