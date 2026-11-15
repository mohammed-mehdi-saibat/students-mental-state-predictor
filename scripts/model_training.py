from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin

def create_model_pipeline(preprocessor: ColumnTransformer, classifier: ClassifierMixin) -> Pipeline:
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return model_pipeline