from src.data_loader import download_data, load_data
from src.preprocess import split_data
from src.train import train_model, save_model
from src.evaluate import evaluate_model


def main():
    download_data()
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)


if __name__ == "__main__":
    main()