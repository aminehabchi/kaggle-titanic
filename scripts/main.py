"""Orchestrator: cleaning -> feature engineering -> training -> evaluation -> submission."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from cleaning import clean
from evaluation import evaluate
from family import combine, family_survival
from feature_engineering import build_features
from predict import make_submission
import rules
from training import train

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "model"))


def main():
    train_csv = os.getenv("TRAIN_CSV", DATA_DIR / "train.csv")
    df = pd.read_csv(train_csv)
    print(f"loaded {len(df)} rows from {train_csv}")

    # 1 - cleaning
    df = clean(df)

    # 2 - feature engineering
    X, y = build_features(df)

    # test.csv has no Survived column, so the test set is a split of train.csv
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3 - training, on the split only, so the evaluation below stays honest
    model = train(X_train, y_train)

    # 4 - evaluation
    evaluate(model, X_test, y_test)

    # 5 - the rules, scored on the same held-out rows for a fair comparison.
    # Their family feature may only look at labels from the training split.
    test_csv = os.getenv("TEST_CSV", DATA_DIR / "test.csv")
    kaggle_test = pd.read_csv(test_csv)
    both, labels, n_train = combine(df, kaggle_test)
    known = np.zeros(len(both), bool)
    known[X_train.index] = True
    fam = family_survival(both, labels, known)
    rule_pred = rules.predict(df.loc[X_test.index], fam[:n_train][X_test.index])
    print(f"rules accuracy on the same 179 rows: "
          f"{(rule_pred == y_test.values).mean():.4f}")

    # 6 - refit on all 891 rows and keep the model for reference
    train(X, y, MODEL_DIR / "model.pkl")

    # 7 - submission, written on every run, from the rules
    make_submission(df, kaggle_test)


if __name__ == "__main__":
    main()
