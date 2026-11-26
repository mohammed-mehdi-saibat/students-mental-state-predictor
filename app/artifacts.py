import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TARGET = "mental_health_risk"
LABELS = ["Low", "Medium", "High"]
MODEL = ROOT / "models/logistic_regression_airflow.joblib"
METRICS = ROOT / "reports/metrics/airflow_evaluation.json"
DATA = ROOT / "data/raw/student_mental_health.csv"
PROCESSED = ROOT / "data/processed"


def published_paths():
    try:
        run = (PROCESSED / "published_run.txt").read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return MODEL, METRICS
    except (OSError, UnicodeError):
        run = ""
    if re.fullmatch(r"[0-9a-f]{64}", run):
        folder = PROCESSED / "runs" / run
        return folder / "model.joblib", folder / "evaluation.json"
    st.warning("Published run metadata could not be read. Showing the committed artifacts.")
    return MODEL, METRICS


@st.cache_data(show_spinner=False)
def read_data(path, modified):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.lower()
    frame = frame.drop(columns="student_id", errors="ignore").dropna(subset=[TARGET])
    if frame.empty or len(frame.columns) < 2 or not frame[TARGET].isin(LABELS).all():
        raise ValueError("Invalid dataset")
    return frame


@st.cache_data(show_spinner=False)
def read_json(path, modified):
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def read_model(path, modified):
    # Load only the repository's trusted, fixed-path artifact.
    model = joblib.load(path)
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Invalid model")
    return model


def available(path, loader, label, validate=None):
    try:
        result = loader(path, path.stat().st_mtime_ns)
        if validate is not None:
            validate(result)
        return result
    except Exception:
        st.warning(f"{label} is unavailable. Check the project setup and artifact files.")
        return None


def evaluation():
    def validate(report):
        labels = report["labels"]
        if set(labels) != set(LABELS):
            raise ValueError("Invalid label set")
        matrix = pd.DataFrame(report["confusion_matrix"], index=labels, columns=labels)
        if matrix.isna().any().any() or (matrix < 0).any().any():
            raise ValueError("Invalid confusion matrix")
        for metric in ("test_accuracy", "test_f1_macro", "cv_f1_macro"):
            if not 0 <= report[metric] <= 1:
                raise ValueError("Invalid score")
        for label in labels:
            row = report["classification_report"][label]
            for metric in ("precision", "recall", "f1-score"):
                if not 0 <= row[metric] <= 1:
                    raise ValueError("Invalid class score")
            if not row["support"] >= 0:
                raise ValueError("Invalid class support")

    return available(published_paths()[1], read_json, "Holdout evaluation", validate)
