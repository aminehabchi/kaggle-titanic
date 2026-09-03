"""Step 3 - training."""

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def train(X, y, model_path=None):
    """Cross-validate, fit on all of X, and pickle the model if asked."""
    # min_samples_leaf=3 is the whole anti-overfitting knob: with the default
    # of 1 the trees grow to ~5 rows a leaf and memorise Fare and Age, which
    # cost more on the leaderboard than the extra fit was worth.
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, random_state=42
    )

    # 5 folds repeated 3x: a single 5-fold split on this little data swings
    # ~4 points fold to fold, which is wider than any change worth making.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    print(f"cross-validation on {len(X)} rows (5 folds x 3): "
          f"{scores.mean():.4f} +- {scores.std():.4f}")

    model.fit(X, y)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"model saved to {model_path}")

    return model
