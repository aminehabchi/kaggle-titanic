"""The predictor: "women and children first", with three corrections.

Sex alone scores 0.7655 on Kaggle. A random forest on these features scored
0.7608 - it overruled sex on 73 rows and was right on only 40% of them. These
three corrections are the ones the train data supports out-of-sample, measured
leave-one-out over all 891 rows.
"""

from family import DIED, LIVED

BIG_FAMILY = 5


def predict(df, fam_surv):
    """Sex baseline, then the three corrections. Returns an int array."""
    female = (df["Sex"] == "female").values
    third = (df["Pclass"] == 3).values
    master = df["Name"].str.contains(r" Master\.", regex=True).values
    big_family = (df["SibSp"] + df["Parch"] + 1).values >= BIG_FAMILY

    pred = female.astype(int)
    # A third-class woman whose family is known to have died died with them:
    # 87% of 55 such rows in train. Third class only, on purpose - first and
    # second class women survived at 97% and 92% whatever their family did,
    # and demoting those was wrong on 22 of 25 rows.
    pred[female & third & (fam_surv == DIED)] = 0
    # third-class women in a large family: 3 of 27 survived in train
    pred[female & third & big_family] = 0
    # boys travelling with a family that lived, lived too: 17 of 18 in train
    pred[master & (fam_surv == LIVED)] = 1
    return pred
