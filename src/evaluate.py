import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay)
from src import config


def evaluate_model(model, X_test, y_test) -> float:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    print(f"Accuracy: {acc:.3f}\n{report}")
    (config.REPORTS_DIR / "metrics.txt").write_text(
        f"Accuracy: {acc:.3f}\n\n{report}"
    )

    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.savefig(config.REPORTS_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
    return acc