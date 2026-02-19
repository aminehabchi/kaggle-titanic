import pandas as pd
from preprocessing import preprocess_data

TEST_DATA_X = "../data/test.csv"

def evaluate(model):
    X_test = pd.read_csv(TEST_DATA_X)
    passenger_ids = X_test["PassengerId"]
    X_test.set_index("PassengerId", inplace=True)

    X_test = preprocess_data(X_test)
    
    y_pred = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": y_pred
    })

    submission.to_csv("../submissions/submission.csv", index=False)
    print("Submission file saved as submission.csv!")
