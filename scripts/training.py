from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def train(df):
    df = df.set_index("PassengerId")

    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Define models and parameter grids
    models_params = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 500, 1000],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=2000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs"],
                "penalty": ["l2"]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5]
            }
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 500],
                "learning_rate": [0.01, 0.1, 0.5, 1]
            }
        },
        "SVC": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            }
        }
    }

    best_model = None
    best_acc = 0
    best_name = ""
    best_params = None

    # Loop through each model
    for name, mp in models_params.items():
        print(f"Starting GridSearch for {name}...")
        grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Model: {name}")
        print(f"Best Params: {grid.best_params_}")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        print("="*50)

        if acc > best_acc:
            best_acc = acc
            best_model = grid.best_estimator_
            best_name = name
            best_params = grid.best_params_

    print(f"Best Model: {best_name} with Accuracy: {best_acc:.4f}")
    print(f"Best Hyperparameters: {best_params}")
    return best_model
