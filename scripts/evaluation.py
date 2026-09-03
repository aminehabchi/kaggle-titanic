"""Step 4 - evaluation."""

from sklearn.metrics import accuracy_score, classification_report


def evaluate(model, X_test, y_test):
    """Score the fitted model on the held-out test set."""
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    print(f"test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, pred, target_names=["died", "survived"]))

    return {"accuracy": accuracy}
