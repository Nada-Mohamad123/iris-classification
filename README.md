# Iris Flower Classification

Refactored machine learning project that classifies iris flowers into three species
(Iris-setosa, Iris-versicolor, Iris-virginica) using modular `.py` files instead of notebooks.

## Dataset
UCI Machine Learning Repository - Iris dataset (150 samples, 4 features).
The dataset is downloaded automatically on first run into `data/raw/iris.csv`.

## Workflow
Download data -> Load -> Split (80/20, stratified) -> Train (StandardScaler + Logistic Regression pipeline) -> Evaluate -> Save model -> Predict

## Project Structure
```
iris-classification/
├── data/raw/            # raw dataset
├── models/              # saved model (ignored by git)
├── reports/             # metrics and confusion matrix
├── src/
│   ├── config.py        # paths and settings
│   ├── data_loader.py   # download and load data
│   ├── preprocess.py    # train/test split
│   ├── train.py         # build, train, save model
│   ├── evaluate.py      # metrics and confusion matrix
│   └── predict.py       # predict on new samples
├── main.py              # runs the full pipeline
├── requirements.txt
└── README.md
```

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Usage
```bash
python main.py              # train and evaluate
python -m src.predict       # example prediction
```

## Results
Test accuracy: **0.933** (see `reports/metrics.txt` and `reports/confusion_matrix.png`).