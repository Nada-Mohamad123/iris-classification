import joblib
import pandas as pd
from src import config


def predict(sepal_length, sepal_width, petal_length, petal_width) -> str:
    model = joblib.load(config.MODEL_PATH)
    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=config.COLUMNS[:-1],
    )
    return model.predict(sample)[0]


if __name__ == "__main__":
    print(predict(5.1, 3.5, 1.4, 0.2))