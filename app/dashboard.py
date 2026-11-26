from math import isclose

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from artifacts import DATA, LABELS, ROOT, TARGET, available, evaluation, published_paths, read_data, read_json, read_model
from design import COLORS, GOLD, counts, heading, plot


def overview():
    heading(None, "Explore a synthetic student dataset and see how a classification model performs on held-out records.")
    data = available(DATA, read_data, "Dataset")
    report = evaluation()
    cards = st.columns(4)
    cards[0].metric("Dataset records", f"{len(data):,}" if data is not None else "—")
    cards[1].metric("Predictive features", len(data.columns) - 1 if data is not None else "—")
    cards[2].metric("Holdout accuracy", f"{report['test_accuracy']:.1%}" if report else "—")
    cards[3].metric("Holdout macro F1", f"{report['test_f1_macro']:.3f}" if report else "—")
    left, right = st.columns([1.65, 1])
    with left.container(border=True):
        st.subheader("Dataset class distribution")
        if data is not None:
            plot(counts(LABELS, data[TARGET].value_counts().reindex(LABELS, fill_value=0).tolist()))
    with right.container(border=True):
        st.subheader("Model at a glance")
        if report:
            st.write("**Logistic Regression** with a fitted preprocessing pipeline.")
            st.write(f"Evaluated on **{sum(map(sum, report['confusion_matrix']))} held-out records**.")
            st.write(f"Five-fold cross-validation macro F1 **{report['cv_f1_macro']:.3f}**.")
            st.write(f"On the holdout, the Medium class had an F1 of **{report['classification_report']['Medium']['f1-score']:.3f}**.")
        st.caption("For educational use with fictional profiles. Dataset labels have no clinical interpretation.")


def field_name(name):
    return "GPA" if name == "gpa" else name.replace("_", " ").capitalize()


def data_insights():
    heading("Data insights", "Compare feature distributions and missingness across dataset classes.")
    data = available(DATA, read_data, "Dataset")
    if data is None:
        return
    columns = [name for name in data if name != TARGET]
    left, right = st.columns(2)
    feature = left.selectbox("Feature", columns, index=columns.index("sleep_hours") if "sleep_hours" in columns else 0, format_func=field_name)
    classes = right.multiselect("Dataset classes", LABELS, default=LABELS)
    if not classes:
        st.info("Select at least one dataset class.")
        return
    subset = data[data[TARGET].isin(classes)]
    if subset.empty:
        st.info("There are no records for the selected classes.")
        return
    observed = subset.dropna(subset=[feature])
    metrics = st.columns(3)
    metrics[0].metric("Selected records", f"{len(subset):,}")
    metrics[1].metric("Observed values", f"{len(observed):,}")
    metrics[2].metric("Missing values", f"{subset[feature].isna().mean():.1%}")
    left, right = st.columns([1.6, 1])
    colors = dict(zip(LABELS, COLORS))
    with left.container(border=True):
        st.subheader(f"{field_name(feature)} by class")
        if pd.api.types.is_numeric_dtype(subset[feature]):
            fig = go.Figure()
            for label in LABELS:
                values = observed.loc[observed[TARGET] == label, feature]
                if values.empty:
                    continue
                q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
                spread = 1.5 * (q3 - q1)
                fig.add_box(
                    name=label, x0=label, q1=[q1], median=[median], q3=[q3],
                    lowerfence=[values[values >= q1 - spread].min()],
                    upperfence=[values[values <= q3 + spread].max()],
                    marker_color=colors[label], boxpoints=False, showlegend=False,
                )
            fig.update_xaxes(type="category", categoryorder="array", categoryarray=LABELS)
            fig.update_yaxes(title=field_name(feature))
        else:
            summary = observed.groupby([feature, TARGET], observed=True).size().reset_index(name="Records")
            fig = px.bar(summary, x=feature, y="Records", color=TARGET, barmode="group",
                         color_discrete_map=colors, category_orders={TARGET: LABELS})
        fig.update_xaxes(title=None)
        plot(fig)
    with right.container(border=True):
        st.subheader("Missing values by feature")
        missing = subset[columns].isna().sum().sort_values().loc[lambda values: values > 0]
        if missing.empty:
            st.caption("All selected records have complete feature values.")
        else:
            plot(go.Figure(go.Bar(x=missing, y=[field_name(name) for name in missing.index],
                                  orientation="h", marker_color=GOLD)).update_xaxes(title="Missing records"))
    st.caption("Charts display aggregates from the synthetic dataset. Feature differences describe associations within these data.")


def comparison_rows(report):
    rows = pd.DataFrame(report["models"])
    if rows.empty or rows["name"].isna().any() or not rows[["baseline", "tuned"]].apply(lambda c: c.between(0, 1)).all().all():
        raise ValueError("Invalid comparison")
    return rows


def model_evaluation():
    heading("Model evaluation", "Compare cross-validation scores and inspect errors from the selected model’s holdout evaluation.")
    comparison = available(ROOT / "reports/metrics/notebook_comparison.json", read_json, "Notebook comparison", comparison_rows)
    report = evaluation()
    left, right = st.columns([1.1, 1])
    with left.container(border=True):
        st.subheader("Baseline and tuned models")
        if comparison:
            rows = comparison_rows(comparison)
            fig = go.Figure()
            for column, label, color in [("baseline", "Baseline", "#756747"), ("tuned", "Tuned", GOLD)]:
                fig.add_bar(y=rows["name"], x=rows[column], name=label, orientation="h",
                            marker_color=color, text=rows[column].map("{:.4f}".format), textposition="outside", cliponaxis=False)
            fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.13, x=0))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[0, 1], title="Five-fold CV macro F1")
            plot(fig, height=275)
            st.caption("Source · Saved comparison table in the model training notebook.")
    with right.container(border=True):
        st.subheader("Holdout confusion matrix")
        if report:
            fig = go.Figure(go.Heatmap(z=report["confusion_matrix"], x=report["labels"], y=report["labels"],
                texttemplate="%{z}", colorscale=[[0, "#fff9d9"], [1, "#e1bb12"]], showscale=False,
                hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z} records<extra></extra>"))
            fig.update_xaxes(title="Predicted class")
            fig.update_yaxes(title="Actual class", autorange="reversed")
            plot(fig, height=275)
            st.caption("Source · Airflow holdout evaluation JSON.")
    if report:
        st.subheader("Per-class holdout metrics")
        rows = pd.DataFrame.from_dict({label: report["classification_report"][label] for label in report["labels"]}, orient="index").reset_index(names="Class")
        rows["support"] = rows["support"].astype(int)
        rows = rows.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support"})
        st.table(rows.set_index("Class").style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}))


def prediction_lab():
    heading("Prediction lab", "Try a fictional student profile and inspect the fitted model’s probabilities for the three dataset classes.")
    model = available(published_paths()[0], read_model, "Saved model")
    if model is None:
        return

    with st.form("fictional_profile"):
        academic, habits, context = st.columns(3)
        with academic:
            st.subheader("Academic profile")
            age = st.number_input("Age", 17, 29, 21)
            year = st.number_input("Year of study", 1, 5, 2)
            study = st.number_input("Study hours per week", 0.0, 168.0, 20.0, step=0.5)
            gpa = st.number_input("GPA (out of 4)", 0.0, 4.0, 2.6, step=0.1)
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 85.0, step=1.0)
        with habits:
            st.subheader("Daily habits")
            sleep = st.number_input("Sleep hours per day", 0.0, 24.0, 7.0, step=0.5)
            screen = st.number_input("Screen hours per day", 0.0, 24.0, 8.5, step=0.5)
            social = st.number_input("Social media hours per day", 0.0, 24.0, 6.0, step=0.5)
            activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"], index=1)
            diet = st.selectbox("Diet quality", ["Poor", "Average", "Good"], index=1)
        with context:
            st.subheader("Profile context")
            gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
            job = st.selectbox("Part-time job", ["No", "Yes"])
            history = st.selectbox("Previous mental health history", ["No", "Yes"])
            stress = st.number_input("Financial stress (1–10)", 1.0, 10.0, 6.0, step=0.5)
            support = st.number_input("Family support (1–10)", 1.0, 10.0, 6.5, step=0.5)
        st.caption("Use fictional values only. This demo does not save submitted profiles.")
        submitted = st.form_submit_button("Run prediction", type="primary")
    if not submitted:
        return
    if social > screen:
        st.error("Social media hours must be at or below total screen hours.")
        return
    profile = pd.DataFrame([{
        "age": age, "gender": gender, "year_of_study": year, "sleep_hours": sleep,
        "study_hours_per_week": study, "gpa": gpa, "social_media_hours": social,
        "screen_time_hours": screen, "physical_activity": activity, "financial_stress": stress,
        "family_support": support, "attendance_rate": attendance, "part_time_job": job,
        "diet_quality": diet, "previous_mental_health_history": history,
    }])
    try:
        label = model.predict(profile)[0]
        probabilities = None
        if callable(getattr(model, "predict_proba", None)):
            probabilities = dict(zip(model.classes_, model.predict_proba(profile)[0], strict=True))
            if not all(0 <= p <= 1 for p in probabilities.values()) or not isclose(sum(probabilities.values()), 1):
                raise ValueError("Invalid probabilities")
    except Exception:
        st.error("The saved model could not score this profile.")
        return
    with st.container(border=True):
        st.subheader(f"Predicted dataset class · {label}")
        if probabilities is not None:
            for col, (name, probability) in zip(st.columns(len(probabilities)), probabilities.items()):
                col.metric(f"P({name})", f"{probability:.1%}")
        else:
            st.caption("This model does not provide class probabilities.")
        st.caption("These are model estimates for synthetic dataset labels. They have no established clinical validity.")


def methodology():
    heading("Methodology", "Follow the path from synthetic source data to reproducible holdout evaluation and interactive inference.")
    first, second = st.columns(2)
    with first.container(border=True):
        st.subheader("Data preparation")
        st.markdown("""
- Validate the schema, target labels, identifiers and exact duplicate feature profiles.
- Remove the identifier and target from predictors. Create a stratified 80/20 split with seed 42.
- Impute missing values, cap social-media time at total screen time, standardize numeric features, and encode categorical inputs inside the fitted pipeline.
""")
    with second.container(border=True):
        st.subheader("Training and evaluation")
        st.markdown("""
- Compare Logistic Regression, Random Forest, and RBF SVC using five-fold stratified cross-validation.
- Tune hyperparameters by macro F1. Evaluate the selected Logistic Regression on the 400-record holdout.
- Load the fitted pipeline for inference and use its class ordering to interpret probability columns.
""")
    with st.container(border=True):
        st.subheader("Airflow orchestration")
        st.write("The Docker Compose stack runs the scheduled training workflow. The dashboard reads saved artifacts and can run independently.")
        st.code("Validate → Prepare split → Fit candidate → Evaluate → Publish", language=None)
    with st.container(border=True):
        st.subheader("Limitations and responsible use")
        st.write("The source data are synthetic and the label-generation mechanism has not been independently verified. Repeated runs on the same data reuse the holdout. Performance on real students, probability calibration and subgroup generalization have not been established.")
        st.write("Use fictional profiles for educational demonstrations. The model is unsuitable for diagnosis, clinical screening, admissions, discipline, or decisions about access to support.")
