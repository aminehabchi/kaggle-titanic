"""Relatives' known outcomes - the strongest signal available for the test set.

A group booking repeats the exact same Fare on every member's row, so
(Surname, Fare) identifies a family; Ticket catches the rest. 63% of
multi-passenger tickets share a single outcome, and 41% of test surnames also
appear in train - so for 171 of the 418 test passengers a relative's fate is
already known from train.csv.
"""

import numpy as np
import pandas as pd

LIVED, DIED, UNKNOWN = 1.0, 0.0, 0.5
GROUP_KEYS = (["Surname", "Fare"], ["Ticket"])


def combine(train, test):
    """Stack train on test, add Surname, and return labels (-1 where unknown)."""
    both = pd.concat([train, test], ignore_index=True)
    both["Surname"] = both["Name"].str.split(",").str[0].str.strip()
    labels = np.concatenate([
        train["Survived"].values, np.full(len(test), -1)
    ])
    return both, labels, len(train)


def family_survival(both, labels, known):
    """LIVED/DIED if some relative's label is known, else UNKNOWN.

    A row never sees its own label - only its relatives' - so this is safe to
    build on the training rows for evaluation as well as on the test rows.
    """
    out = np.full(len(both), UNKNOWN)
    for keys in GROUP_KEYS:
        for idx in both.groupby(keys, dropna=False).indices.values():
            if len(idx) < 2:
                continue
            for i in idx:
                if out[i] != UNKNOWN:      # the Ticket pass only fills gaps
                    continue
                relatives = idx[idx != i]
                seen = relatives[known[relatives]]
                if len(seen):
                    out[i] = LIVED if (labels[seen] == 1).any() else DIED
    return out
