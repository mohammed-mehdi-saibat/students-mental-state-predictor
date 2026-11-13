import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from IPython.display import display
from pathlib import Path


root = Path.cwd().resolve().parents[0]
raw_data_path = root / "data" / "raw" / "student_mental_health.csv"
plots_path = root / "reports" / "figures" / "eda"
plots_path.mkdir(parents=True, exist_ok=True)

def fetch_dataset_and_turn_into_dataframe_and_convert_column_names_to_lowercase(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=str.lower)
    return df

def extract_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include=["object", "category"])
    return categorical_columns

def extract_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    numerical_columns = df.select_dtypes(include="number")
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

def dataset_quality_inspection(df: pd.DataFrame) -> None:
    # Inspect total missing values
    print("=" * 30)
    missing_values_total = df.isnull().sum()
    print(f"MISSING VALUES TOTAL: \n{missing_values_total[missing_values_total > 0].sort_values(ascending=False)}")
    # Inspect missing values percentage
    print("=" * 30)
    missing_percentage = (df.isnull().sum() / df.shape[0]) * 100
    print(f"MISSING VALUES PERCENTAGE: \n{missing_percentage[missing_percentage > 0].sort_values(ascending=False)}")
    # Inspect duplicates
    print("=" * 30)
    duplicates_total = df.duplicated().sum()
    print(f"DUPLICATED VALUES: \n{duplicates_total}")
    # Make sure student_id is unique and has no missing values, because it should never be duplicated
    print("=" * 30)
    print(f"student_id values are completely unique? --{df["student_id"].is_unique}--")
    print(f"student_id missing values: {df["student_id"].isnull().sum()}")
    # Inspect inconsistent values
    print("=" * 30)
    print("UNIQUE VALUES PER COLUMN:")
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        print(f"Unique values in '{col}':")
        print(df[col].unique())
        print("-" * 40)
    # Check impossible numerical values
    print("=" * 30)
    print("CHECK IF VALUES ARE LOGIC")
    print("-" * 30)
    age_bellow_zero = df["age"] <= 0
    print(f"Age bellow 0 count: {age_bellow_zero.sum()}")
    age_above_one_hundred = df["age"] > 100
    print(f"Age above 100 count: {age_above_one_hundred.sum()}")
    print("-" * 30)
    sleep_hours_bellow_zero = df["sleep_hours"] < 0
    print(f"Sleep hours bellow 0: {sleep_hours_bellow_zero.sum()}")
    sleep_hours_above_twenty_four = df["sleep_hours"] > 24
    print(f"Sleep hours above 24: {sleep_hours_above_twenty_four.sum()}")
    print("-" * 30)
    gpa_bellow_zero = df["gpa"] < 0
    print(f"GPA bellow 0: {gpa_bellow_zero.sum()}")
    gpa_above_four = df["gpa"] > 4
    print(f"GPA above 4: {gpa_above_four.sum()}")
    print("-" * 30)
    attendance_rate_bellow_0 = df["attendance_rate"] < 0
    print(f"ATTENDANCE RATE BELLOW 0: {attendance_rate_bellow_0.sum()}")
    attendance_rate_above_one_hundred = df["attendance_rate"] > 100
    print(f"ATTENDANCE RATE ABOVE 100: {attendance_rate_above_one_hundred.sum()}")
    print("-" * 30)
    print("SCREEN TIME HOURS ARE LOWER THAN SOCIAL MEDIA HOURS:")
    print(f"{df[df["social_media_hours"] > df["screen_time_hours"]].shape[0]} rows")
    print("-" * 30)
    print("TOTAL ACTIVITIES HOURS THAT ARE GREATER THAN 168 (24 HOURS X 7 DAYS):")
    week_total_sleeping_hours = df["sleep_hours"] * 7
    week_total_screen_time_hours = df["screen_time_hours"] * 7
    study_hours_per_week = df["study_hours_per_week"]
    print(f"{df[(week_total_sleeping_hours + week_total_screen_time_hours + study_hours_per_week) > 168].shape[0]} rows")  
    print("THE ROWS ARE:")
    df["total_activities_hours"] = week_total_sleeping_hours + week_total_screen_time_hours + study_hours_per_week
    display(df[df["total_activities_hours"] > 168])
    print("-" * 30)
    # Inspect outliers
    print("-" * 30)
    print("OUTLIERS OF EACH COLUMN")
    def detect_outliers(df: pd.DataFrame) -> None:
        numerical_features = df.select_dtypes(include="number")

        for col in numerical_features.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower_fence = Q1 - 1.5 * IQR
            upper_fence = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_fence) | (df[col] > upper_fence)]

            print(f"COLUMN: {col}")
            print(f"LOWER FENCE: {lower_fence}")
            print(f"UPPER FENCE: {upper_fence}")
            print(f"NUMBER OF OUTLIERS: {len(outliers)}")
            if len(outliers) > 0:
                display(outliers[["student_id", col]])
            print("-" * 40)
    detect_outliers(df)


def descriptive_statistics(df: pd.DataFrame) -> None:
    # Inspect numerical statistics
    print("NUMERICAL STATISTICS")
    display(df.describe().T)
    # Inspect categorical statistics
    print("=" * 30)
    print("CATEGORICAL STATISTICS")
    display(df.describe(include=["str", "object", "category"]).T)
    # Inspect skewness
    print("Skewness Coefficient")
    all_skews = df.skew(numeric_only=True)
    print(all_skews)
    # • -0.5 to 0.5: Fairly symmetrical data.
    # • -1.0 to -0.5 or 0.5 to 1.0: Moderate skewness.
    # • Less than -1.0 or Greater than 1.0: High/severe skewness.

   

def target_variable_analysis(df: pd.DataFrame) -> None:
     # Inspect target distribution
    counts = df["mental_health_risk"].value_counts().reindex(["Low", "Medium", "High"])
    print("HOW MANY STUDENTS BELONG TO EACH CLASS")
    print(counts)
    # Inspect target percentage
    print("=" * 30)
    print("TARGET DISTRIBUTION PERCENTAGE")
    print((counts / df.shape[0]) * 100)
    # Visualize target distribution
    print("=" * 30)
    print("TARGET DISTRIBUTION VISUALIZATION")
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values, color=["#4CAF50", "#FFC107", "#F44336"])
    for i, value in enumerate(counts.values):
        plt.text(i, value + 20, f"{value}\n({value / len(df) * 100:.1f}%)", ha="center")
    plt.title("Distribution of Mental Health Risk")
    plt.xlabel("Risk Level")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(plots_path / "01_target_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

def univariate_analysis(df: pd.DataFrame) -> matplotlib.figure.Figure:
    # Plot numerical distribution
    numerical_features = df.select_dtypes(include="number")
    print("NUMERICAL FEATURES HISTOGRAM")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, numerical_features.columns):
        sns.histplot(data=df, x=col, bins=20, kde=True, ax=ax)
        ax.set_title(col)

    plt.tight_layout()
    plt.savefig(plots_path / "02_numerical_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("=" * 30)

    # Plot categorical distribution
    categorical_features = df.select_dtypes(include=["object", "str", "category"]).drop(columns="student_id")
    print("CATEGORICAL FEATURES BAR PLOT")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, categorical_features.columns):
        sns.countplot(data=df, x=col, ax=ax)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)

    # Hide empty subplots
    for ax in axes[len(categorical_features.columns):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(plots_path / "03_categorical_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("=" * 30)

    # Plot numerical boxplots
    print("NUMERICAL FEATURES BOXPLOTS")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, numerical_features.columns):
        sns.boxplot(data=df, y=col, ax=ax)
        ax.set_title(col)

    plt.tight_layout()
    plt.savefig(plots_path / "04_numerical_feature_boxplots.png", dpi=300, bbox_inches="tight")
    plt.show()

def bevariate_analysis(df: pd.DataFrame) -> matplotlib.figure.Figure:
    # Analyze numerical features vs target
    numerical_features = df.select_dtypes(include="number")
    print("NUMERICAL FEATURES VS TARGET")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, numerical_features.columns):
        sns.boxplot(data=df, x="mental_health_risk", y=col, order=["Low", "Medium", "High"], ax=ax)
        ax.set_title(col)

    plt.tight_layout()
    plt.savefig(plots_path / "05_numerical_features_vs_target.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("=" * 30)

    # Analyze categorical features vs target
    categorical_features = df.select_dtypes(include=["category", "str", "object"]).drop(columns=["student_id", "mental_health_risk"])
    print("CATEGORICAL FEATURES VS TARGET")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, categorical_features.columns):
        sns.countplot(data=df, x=col, hue="mental_health_risk", hue_order=["Low", "Medium", "High"], ax=ax)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)

    # Hide empty subplots
    for ax in axes[len(categorical_features.columns):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(plots_path / "06_categorical_features_vs_target.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("=" * 30)

    # Analyze feature relationships
    pairs = [
        ("sleep_hours", "gpa"),
        ("sleep_hours", "study_hours_per_week"),
        ("screen_time_hours", "social_media_hours"),
        ("gpa", "attendance_rate"),
        ("study_hours_per_week", "screen_time_hours"),
        ("sleep_hours", "screen_time_hours"),
    ]
    print("FEATURE RELATIONSHIPS")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    mental_health_risk_palette = {
        "High": "red",
        "Medium": "yellow",
        "Low": "green"
    }

    for ax, (x_col, y_col) in zip(axes, pairs):
        sns.scatterplot(data=df, x=x_col, y=y_col, hue="mental_health_risk", hue_order=["Low", "Medium", "High"], alpha=0.5, ax=ax, palette=mental_health_risk_palette)
        ax.set_title(f"{x_col} vs {y_col}")

    plt.tight_layout()
    plt.savefig(plots_path / "07_feature_relationships.png", dpi=300, bbox_inches="tight")
    plt.show()


def correlation_and_feature_relevance(df: pd.DataFrame) -> None:

    # Work with a copy and exclude non-predictive columns
    eda_df = df.drop(
        columns=["student_id", "total_activities_hours", "mental_health_risk_num"],
        errors="ignore"
    ).copy()

    # Calculate numerical feature correlations
    numerical_features = eda_df.select_dtypes(include="number")

    print("NUMERICAL FEATURE CORRELATION HEATMAP")
    plt.figure(figsize=(20, 6))

    corr_matrix = numerical_features.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5
    )

    plt.title("Numerical Features Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plots_path / "08_numerical_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Inspect multicollinearity
    print("=" * 30)
    print("HIGHLY CORRELATED PAIRS (|corr| > 0.8):")

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                print(
                    f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]}: "
                    f"{corr_matrix.iloc[i, j]:.2f}"
                )

    # Create temporary numerical target
    target_num = eda_df["mental_health_risk"].map({
        "Low": 0,
        "Medium": 1,
        "High": 2
    })

    # Numerical features vs target
    print("=" * 30)
    print("NUMERICAL FEATURES VS TARGET")

    numerical_corr = numerical_features.corrwith(target_num)

    print(numerical_corr.sort_values(key=abs, ascending=False))

    # Categorical features vs target
    print("=" * 30)
    print("CATEGORICAL FEATURES VS TARGET")

    categorical_features = eda_df.select_dtypes(
        include=["object", "category", "string"]
    ).drop(columns="mental_health_risk", errors="ignore")

    for col in categorical_features.columns:
        print(col)
        print(
            eda_df.assign(mental_health_risk_num=target_num)
            .groupby(col)["mental_health_risk_num"].mean()
        )
        print("=" * 30)
