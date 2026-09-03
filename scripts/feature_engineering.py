"""Step 2 - feature engineering."""

FEATURES = ["Sex", "Pclass", "Age", "Fare"]
TARGET = "Survived"

# Median Fare of train.csv. A constant, not df["Fare"].median(), so the one
# missing Fare in test.csv is filled with the value the model was trained on.
FARE_FILL = 14.4542

CHILD_AGE = 12

# Mlle/Ms are Miss, Mme is Mrs; everything else is too rare to keep on its own.
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
TITLE_CODES = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}


def extract_title(names):
    """Pull the title out of 'Last, Title. First' and label-encode it."""
    titles = names.str.extract(r" ([A-Za-z]+)\.", expand=False)
    titles = titles.replace(TITLE_ALIASES)
    return titles.map(TITLE_CODES).fillna(TITLE_CODES["Rare"]).astype(int)


def build_features(df):
    """Keep the 4 base features, label-encode Sex, add 5 derived columns.

    Every derived column is computed from one row only - no median or count
    over the whole file - so train and test are built identically.
    """
    X = df[FEATURES].copy()
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    X["Fare"] = X["Fare"].fillna(FARE_FILL)
    X["Title"] = extract_title(df["Name"])

    # siblings/spouses + parents/children + the passenger themselves
    family = df["SibSp"] + df["Parch"] + 1
    X["FamilySize"] = family
    X["IsAlone"] = (family == 1).astype(int)
    # a group booking repeats the whole ticket price on every row
    X["FarePerPax"] = X["Fare"] / family
    X["IsChild"] = (X["Age"] < CHILD_AGE).astype(int)

    # Sex*Age was dropped here: 0.87 correlated with Sex, and giving the trees
    # one more continuous column to memorise cost accuracy rather than adding.
    y = df[TARGET] if TARGET in df.columns else None
    return X, y
