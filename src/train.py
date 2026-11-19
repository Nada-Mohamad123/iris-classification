import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src import config


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])


def train_model(X_train, y_train) -> Pipeline:
    model = build_model()
    model.fit(X_train, y_train)
    return model


def save_model(model) -> None:
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)