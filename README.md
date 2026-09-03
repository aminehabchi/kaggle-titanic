# Kaggle Titanic

Predicting which passengers survived the Titanic, for the
[Kaggle Titanic competition](https://www.kaggle.com/c/titanic).

**Kaggle username:** `ahabchi`  
**Best leaderboard score: 0.80861** (338 of 418 test passengers correct).

The interesting part of this project is not the model. It is that a random
forest with 0.84 cross-validated accuracy scored *worse* on Kaggle than
guessing "every woman survived, every man died" - and what we did about it.

## Result history

Kaggle scores the full 418-row test set, so every score is an exact row count
and carries no sampling noise. The `gender_submission.csv` baseline that ships
with the competition - women live, men die - scores **320/418 = 0.7655**.

| submission | rows correct | score | rows changed vs baseline | of those, right |
|---|---|---|---|---|
| random forest, no regularisation | ~306 | 0.73 | 73 | 40% |
| random forest, `min_samples_leaf=3` | 318 | 0.76076 | 54 | 48% |
| rules, all passenger classes | 328 | 0.78468 | 42 | 60% |
| **rules, third class only** | **338** | **0.80861** | **28** | **82%** |

Every improvement came from making *fewer* but better-supported changes to the
baseline, never from a stronger model.

## Setup

The competition CSVs are not committed. Download them into `data/`:

```bash
kaggle competitions download -c titanic -p data && unzip -o data/titanic.zip -d data
```

Then create the environment and a `.env`:

```bash
python3 -m venv titanic
./titanic/bin/pip install -r requirements.txt
cp .env.example .env      # then edit the absolute paths inside
```

`.env` holds the paths (`DATA_DIR`, `MODEL_DIR`, `SUBMISSION_DIR`, ...); every
script falls back to the repo-relative defaults if it is missing.

## Running it

```bash
./run.sh                                  # the whole pipeline
./titanic/bin/python scripts/predict.py   # rebuild the submission only
```

`run.sh` trains, evaluates, saves `model/model.pkl` and writes
`submissions/submission.csv`. Submit that file:

```bash
./titanic/bin/kaggle competitions submit -c titanic -f submissions/submission.csv -m "notes"
```

Kaggle API credentials go in `~/.config/kaggle/kaggle.json` (`chmod 600`), and
you must click "Join Competition" once in the browser before the API accepts a
submission.

## Layout

```
scripts/
  main.py                 orchestrator - calls the steps below in order
  cleaning.py             1. missing values
  feature_engineering.py  2. the feature matrix
  training.py             3. random forest + cross-validation
  evaluation.py           4. scoring on held-out rows
  family.py                  relatives' known outcomes
  rules.py                   the predictor that actually gets submitted
  predict.py              5. writes submissions/submission.csv
notebook/main.ipynb       exploratory analysis: missing values, outliers
data/                     competition CSVs (not committed)
model/model.pkl           the fitted forest (not committed)
submissions/submission.csv the file to upload (committed)
```

## The pipeline

**1. Cleaning** (`cleaning.py`) drops `Cabin` (77% missing), fills `Age` with
`28.0` and `Embarked` with `"S"`. The fill values are module constants rather
than `df["Age"].median()` on purpose: the test set's own Age median is 27.0, so
computing it per file would have filled 86 test rows with a value the model was
never trained on.

**2. Feature engineering** (`feature_engineering.py`) builds nine columns:

| column | how |
|---|---|
| `Sex` | label encoded, male=0 female=1 |
| `Pclass`, `Age`, `Fare` | as given, `Fare` filled with the train median 14.4542 |
| `Title` | parsed from `"Last, Title. First"`, mapped to Mr/Miss/Mrs/Master/Rare |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | `FamilySize == 1` |
| `FarePerPax` | `Fare / FamilySize` - a group booking repeats the whole ticket price on every row |
| `IsChild` | `Age < 12` |

Every derived column is computed from a single row. Nothing uses a median or a
count over the whole file, so train and test are always built identically.

**3. Training** (`training.py`) fits a `RandomForestClassifier` with 300 trees
and `min_samples_leaf=3`. That one setting matters more than anything else in
the file: at the default of 1 the trees grow to about five rows per leaf,
reaching 0.98 accuracy on data they have seen and memorising individual fares.

**4. Evaluation** (`evaluation.py`) scores a 20% hold-out split, and
`main.py` scores the rules on those same rows for a fair comparison.
Cross-validation uses `RepeatedStratifiedKFold(5, n_repeats=3)`, because a
single 5-fold split on 891 rows swings about 4 points between folds - wider
than any improvement worth chasing.

**5. Submission** (`predict.py`) writes `submissions/submission.csv` on every
run, using `rules.py`.

## Why the submission comes from rules, not the forest

The forest is still trained, cross-validated and pickled, but it does not
produce the submission. The numbers are the reason:

| | cross-validation | Kaggle |
|---|---|---|
| sex alone | 0.7867 | 0.7655 |
| random forest | 0.8395 | 0.7608 |

Cross-validation rated the forest 5 points above the trivial baseline. On the
real test set it came in half a point *below* it. Its overrides of the sex
baseline were right 40% then 48% of the time - coin flips wearing a confidence
interval. Out-of-fold those same overrides measured 71% accurate, so the gap is
not something cross-validation could see.

We checked the obvious explanation and it was wrong: `GroupKFold` on ticket and
on surname scored 0.8350 and 0.8339, versus 0.8383 for plain folds, so families
spanning folds were not leaking anything.

What worked instead was to stop trusting aggregate accuracy and measure each
individual decision - how often does *this* specific override beat the
baseline, leave-one-out across all 891 rows.

## The rules

`family.py` finds relatives by `(Surname, Fare)` and then by `Ticket`, and
reports whether a relative's fate is already **known from `train.csv`**. This
works because 63% of multi-passenger tickets share a single outcome and 41% of
test surnames also appear in train: **171 of the 418** test passengers have a
relative whose label we know. A row never sees its own label, so the same
function is used for evaluation and for prediction.

`rules.py` then starts from the sex baseline and applies three corrections:

| correction | support in train |
|---|---|
| third-class woman whose family is known to have died → dies | 87% of 55 rows |
| third-class woman in a family of 5+ → dies | 3 of 27 survived |
| `Master` (a boy) whose family lived → survives | 17 of 18 rows |

The class restriction on the first rule is what took the score from 0.78468 to
0.80861. Applying it to every class demoted 25 first- and second-class women,
and it was wrong on 22 of them - those women survived at 97% and 92%
regardless of what happened to their families. Restricting the rule to third
class removed 14 flips from the submission and gained 10 rows.

## What did not help

Measured and rejected, so they are not in the code:

- `Sex * Age` - 0.87 correlated with `Sex`, pure memorisation surface.
- `Embarked`, `SibSp` and `Parch` as raw columns, `NameLength`, `log(Fare)`,
  ticket digit counts, ticket prefix flags - all neutral or negative.
- Binning `Fare` or `Age` - cut the overfitting gap but cost accuracy.
- Filling `Age` by title instead of one median - worth +0.03 points.
- Promoting men whose family survived - right only 43% of the time. The
  family signal works for women and boys, not for adult men.
- Feeding family survival to the forest as a feature - +0.0004 in leak-free
  cross-validation. The signal only pays off applied directly.

## The lesson

On 891 rows, cross-validation could not tell a real improvement from noise, and
it was optimistic by 8 points for the model that overfit hardest. What
correlated with the leaderboard was the number of bets placed against a simple,
strong baseline and the measured hit rate of each one - 73 bets at 40% lost
points, 28 bets at 82% won them.
