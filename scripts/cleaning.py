"""Step 1 - cleaning."""

# Median Age of train.csv. Constants, not df["Age"].median(), so the 86 rows
# missing Age in test.csv get the same fill the model was trained on - the
# test set's own median is 27.0, which the model never saw.
AGE_FILL = 28.0
EMBARKED_FILL = "S"


def clean(df):
    """Drop Cabin, fill Age with the train median, fill Embarked with 'S'."""
    df = df.drop(columns=["Cabin"], errors="ignore")
    df["Age"] = df["Age"].fillna(AGE_FILL)
    df["Embarked"] = df["Embarked"].fillna(EMBARKED_FILL)
    return df
