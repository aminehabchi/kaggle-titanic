"""Step 5 - predict on Kaggle's test.csv and write a submission file."""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import rules
from family import combine, family_survival

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "model"))
SUBMISSION_DIR = Path(os.getenv("SUBMISSION_DIR", ROOT / "submissions"))


def load(name, env):
    return pd.read_csv(os.getenv(env, DATA_DIR / name))


def make_submission(train=None, test=None):
    """Predict the test set with the rules and write submission.csv."""
    train = load("train.csv", "TRAIN_CSV") if train is None else train
    test = load("test.csv", "TEST_CSV") if test is None else test

    both, labels, n_train = combine(train, test)
    # every train label is known when predicting the real test set
    known = np.zeros(len(both), bool)
    known[:n_train] = True
    fam = family_survival(both, labels, known)

    pred = rules.predict(test, fam[n_train:])
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred})

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(out, index=False)
    known_rel = int((fam[n_train:] != 0.5).sum())
    print(f"submission saved to {out} ({len(submission)} rows, "
          f"{known_rel} with a known relative)")
    return submission


def main():
    """Standalone run: rebuild the submission without retraining."""
    make_submission()


if __name__ == "__main__":
    main()
