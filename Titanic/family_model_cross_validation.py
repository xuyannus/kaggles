import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from Titanic.titanic_common import age_encoding, sex_encoding


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv", index_col="PassengerId")
    return train_df


def missing_data_filling(df):
    # https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html
    df['Embarked'].fillna("S", inplace=True)
    df.drop("Cabin", inplace=True, axis=1)
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    return df


def get_family_id(x):
    return "{}_{}_{}".format(x['Surename'], x['Pclass'], x['Embarked'])


def extract_common_features(df):
    df['AgeEncoded'] = df['Age'].map(age_encoding)
    df['SexEncoded'] = df['Sex'].map(sex_encoding)
    df['Surename'] = df['Name'].apply(lambda x: x.split(",")[0].strip().lower())
    df['Family'] = df.apply(lambda x: get_family_id(x), axis=1)
    return df


def generate_family_survive_rate(x):
    x_filtered = x[(x['Sex'] == 'female') | (x['AgeEncoded'] == 1)]
    if x_filtered.shape[0] >= 0:
        return np.mean(x_filtered['Survived'])
    else:
        return None


def cal_family_survive_rate(df):
    family_survive_rate = df.groupby('Family').apply(lambda x: generate_family_survive_rate(x)).to_frame()
    family_survive_rate.reset_index(inplace=True)
    family_survive_rate.columns = ['Family', 'Family_Survive_Rate']
    return family_survive_rate


def model(x, family_survive_column="Family_Survive_Rate"):
    if x['Sex'] == 'female':
        if 0 <= x[family_survive_column] <= 0.5:
            return 0
        else:
            return 1
    else:
        if x['AgeEncoded'] == 1:
            if x[family_survive_column] > 0.5:
                return 1
            else:
                return 0
        else:
            return 0


def model_accuracy(X_train, X_test):
    family_survive_rate_train = cal_family_survive_rate(X_train)
    # default_survive_rate = np.mean(X_train['Survived'])
    default_survive_rate = -1

    temp = X_train.merge(family_survive_rate_train, left_on='Family', right_on='Family', how='left')
    temp['Family_Survive_Rate'].fillna(default_survive_rate, inplace=True)
    X_train['Family_Survive_Rate'] = temp['Family_Survive_Rate'].values

    temp2 = X_test.merge(family_survive_rate_train, left_on='Family', right_on='Family', how='left')
    temp2['Family_Survive_Rate'].fillna(default_survive_rate, inplace=True)
    X_test['Family_Survive_Rate'] = temp2['Family_Survive_Rate'].values

    train_y = X_train['Survived'].values
    prediction_train_y = X_train.apply(lambda x: model(x), axis=1).values

    test_y = X_test['Survived'].values
    prediction_test_y = X_test.apply(lambda x: model(x), axis=1).values

    return accuracy_score(train_y, prediction_train_y), accuracy_score(test_y, prediction_test_y)


def cross_validate(k=5):
    train_df = load_data()
    train_df = missing_data_filling(train_df)
    train_df = extract_common_features(train_df)

    kf3 = KFold(n_splits=k, shuffle=True)
    train_accuracy_list = []
    test_accuracy_list = []

    for tune_train_index, tune_test_index in kf3.split(train_df):
        X_train = train_df.iloc[tune_train_index].copy()
        X_test = train_df.iloc[tune_test_index].copy()
        train_accuracy, test_accuracy = model_accuracy(X_train, X_test)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

    print({
        "train_accuracy": train_accuracy_list,
        "mean_train_accuracy": np.mean(train_accuracy_list),
        "test_accuracy": test_accuracy_list,
        "mean_test_accuracy": np.mean(test_accuracy_list),
    })


# example:
# training accuracy: 0.8978670595836551
# testing accuracy: 0.8417362375243236
# leaderboard score: 0.79
if __name__ == "__main__":
    cross_validate()
