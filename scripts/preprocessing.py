import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def fill_missing_value(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

def drop_unused_columns(df):
    df.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)
    return df

def encode_ordinal_category(df):
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    return df


def feature_engineering(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # include self

    df['FamilySize'] = df['FamilySize'].apply(lambda x: 1 if x > 4 else x)
    
    return df


def scale_features(df):
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    return df
    

def preprocess_data(df):
    
    df = fill_missing_value(df)
    df = drop_unused_columns(df)
    df = encode_ordinal_category(df)
    df = feature_engineering(df)
    df =scale_features(df)

    return df
