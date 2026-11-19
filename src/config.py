from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
RAW_PATH = ROOT / "data" / "raw" / "iris.csv"
MODEL_PATH = ROOT / "models" / "iris_model.joblib"
REPORTS_DIR = ROOT / "reports"

COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
TARGET = "species"
TEST_SIZE = 0.2
RANDOM_STATE = 42