import urllib.request
import pandas as pd
from src import config


def download_data() -> None:
    """Download the raw dataset if it doesn't exist yet."""
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not config.RAW_PATH.exists():
        urllib.request.urlretrieve(config.DATA_URL, config.RAW_PATH)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_PATH, names=config.COLUMNS)
    return df.dropna().reset_index(drop=True)