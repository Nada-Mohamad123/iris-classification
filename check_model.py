import joblib
from src import config
from src.data_loader import load_data
from src.preprocess import split_data

model = joblib.load(config.MODEL_PATH)
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

preds = model.predict(X_test)
wrong = X_test[preds != y_test].copy()
wrong["actual"] = y_test[preds != y_test]
wrong["predicted"] = preds[preds != y_test]

print(f"Test accuracy: {(preds == y_test).mean():.3f}")
print(f"Wrong predictions: {len(wrong)} of {len(y_test)}")
print(wrong)