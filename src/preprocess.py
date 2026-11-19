import pandas as pd
from sklearn.model_selection import train_test_split
from src import config


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )