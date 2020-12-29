import os
from random import randrange
from subprocess import call

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz

from titanic.titanic_common import age_encoding, sex_encoding, cabin_encoding, family_size_encoding, fare_encoding
from common import explore_none


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv", index_col="PassengerId")
    return train_df


def missing_data_filling(df):
    # https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html
    df['Embarked'].fillna("S", inplace=True)
    df['Cabin'].fillna('M', inplace=True)
    df['Cabin'] = df['Cabin'].astype(str).str[0].str.upper()
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    df['Fare'].fillna(7.33, inplace=True)  # average fare for Pclass 3
    # explore_none(df)
    return df


def get_family_id(x):
    return "{}_{}_{}".format(x['Surename'], x['Pclass'], x['Embarked'])


def extract_common_features(df):
    df['AgeEncoded'] = df['Age'].map(age_encoding)
    df['CabinEncoded'] = df['Cabin'].map(cabin_encoding)
    df['Surename'] = df['Name'].apply(lambda x: x.split(",")[0].strip().lower())
    df['Family'] = df.apply(lambda x: get_family_id(x), axis=1)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilySizeEncode'] = df['FamilySize'].map(family_size_encoding)

    # fare -> individual person
    temp1 = df.groupby('Ticket')['Surename'].count().to_frame()
    temp1.reset_index(inplace=True)
    temp1.columns = ['Ticket', 'TicketSize']

    temp2 = df.merge(temp1, left_on='Ticket', right_on='Ticket', how='left')
    df['TicketSize'] = temp2['TicketSize'].values
    df['FareAdjusted'] = df['Fare'] / df['TicketSize']
    df['FareAdjustedEncoded'] = df['FareAdjusted'].map(fare_encoding)

    return df


def one_hot_encoding(df):
    pclass_encoder = OneHotEncoder(handle_unknown='ignore')
    pclass_encoder.fit(df[['Pclass']])
    pclass_temp_df = pd.DataFrame(pclass_encoder.transform(df[['Pclass']]).toarray(),
                                  columns=pclass_encoder.get_feature_names(['Pclass']), index=df.index)

    embarked_encoder = OneHotEncoder(handle_unknown='ignore')
    embarked_encoder.fit(df[['Embarked']])
    embarked_temp_df = pd.DataFrame(embarked_encoder.transform(df[['Embarked']]).toarray(),
                                    columns=embarked_encoder.get_feature_names(['Embarked']), index=df.index)

    cabin_encoder = OneHotEncoder(handle_unknown='ignore')
    cabin_encoder.fit(df[['CabinEncoded']])
    cabin_temp_df = pd.DataFrame(cabin_encoder.transform(df[['CabinEncoded']]).toarray(),
                                 columns=cabin_encoder.get_feature_names(['CabinEncoded']), index=df.index)

    family_size_encoder = OneHotEncoder(handle_unknown='ignore')
    family_size_encoder.fit(df[['FamilySizeEncode']])
    family_size_temp_df = pd.DataFrame(family_size_encoder.transform(df[['FamilySizeEncode']]).toarray(),
                                       columns=family_size_encoder.get_feature_names(['FamilySizeEncode']),
                                       index=df.index)

    return pd.concat([df, pclass_temp_df, embarked_temp_df, cabin_temp_df, family_size_temp_df], axis=1)


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


def encode_survive_rate(x):
    if x == -1:
        return -1
    elif x <= 0.5:
        return 0
    else:
        return 1


def combine_sex_survive_rate(x):
    if x['Sex'] == 'female':
        if 0 <= x['Family_Survive_Rate'] <= 0.5:
            return 'female_special'
        else:
            return 'female_normal'
    else:
        if x['AgeEncoded'] == 1:
            if x['Family_Survive_Rate'] > 0.5:
                return 'male_special'
            else:
                return 'male_normal'
        else:
            return 'male_normal'


def model_accuracy(clf_model, X_train, X_test):
    family_survive_rate_train = cal_family_survive_rate(X_train)
    default_survive_rate = -1

    temp = X_train.merge(family_survive_rate_train, left_on='Family', right_on='Family', how='left')
    temp['Family_Survive_Rate'].fillna(default_survive_rate, inplace=True)
    X_train['Family_Survive_Rate'] = [encode_survive_rate(x) for x in temp['Family_Survive_Rate'].values]
    X_train['Sex_Survive'] = X_train.apply(lambda x: combine_sex_survive_rate(x), axis=1)

    temp2 = X_test.merge(family_survive_rate_train, left_on='Family', right_on='Family', how='left')
    temp2['Family_Survive_Rate'].fillna(default_survive_rate, inplace=True)
    X_test['Family_Survive_Rate'] = [encode_survive_rate(x) for x in temp2['Family_Survive_Rate'].values]
    X_test['Sex_Survive'] = X_test.apply(lambda x: combine_sex_survive_rate(x), axis=1)

    temp_combine = pd.concat([X_train, X_test])
    surviveEncoder = OneHotEncoder(handle_unknown='ignore')
    surviveEncoder.fit(temp_combine[['Sex_Survive']])
    temp_df = pd.DataFrame(surviveEncoder.transform(X_train[['Sex_Survive']]).toarray(),
                           columns=surviveEncoder.get_feature_names(['Sex_Survive']),
                           index=X_train.index)
    X_train = pd.concat([X_train, temp_df], axis=1)
    temp2_df = pd.DataFrame(surviveEncoder.transform(X_test[['Sex_Survive']]).toarray(),
                            columns=surviveEncoder.get_feature_names(['Sex_Survive']),
                            index=X_test.index)
    X_test = pd.concat([X_test, temp2_df], axis=1)

    selected_columns = ['FareAdjustedEncoded', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
                        'SibSp', 'Parch', 'AgeEncoded', 'Sex_Survive_female_normal', 'Sex_Survive_female_special',
                        'Sex_Survive_male_normal', 'Sex_Survive_male_special', 'CabinEncoded_ABCT', 'CabinEncoded_DE',
                        'CabinEncoded_FG', 'CabinEncoded_M', 'FamilySizeEncode_1',
                        'FamilySizeEncode_2', 'FamilySizeEncode_3', 'FamilySizeEncode_4']

    clf_model.fit(X_train[selected_columns], X_train['Survived'])

    train_y = X_train['Survived'].values
    prediction_train_y = clf_model.predict(X_train[selected_columns])

    test_y = X_test['Survived'].values
    prediction_test_y = clf_model.predict(X_test[selected_columns])

    return accuracy_score(train_y, prediction_train_y), accuracy_score(test_y, prediction_test_y)


def cross_validate(params, k=3):
    train_df = load_data()
    train_df = missing_data_filling(train_df)
    train_df = extract_common_features(train_df)
    train_df = one_hot_encoding(train_df)

    kf3 = KFold(n_splits=k, shuffle=True)
    train_accuracy_list = []
    test_accuracy_list = []

    for tune_train_index, tune_test_index in kf3.split(train_df):
        X_train = train_df.iloc[tune_train_index].copy()
        X_test = train_df.iloc[tune_test_index].copy()

        clf = LogisticRegression(random_state=0, C=params['C'])

        train_accuracy, test_accuracy = model_accuracy(clf, X_train, X_test)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

    return {
        "train_accuracy": train_accuracy_list,
        "mean_train_accuracy": np.mean(train_accuracy_list),
        "test_accuracy": test_accuracy_list,
        "mean_test_accuracy": np.mean(test_accuracy_list),
    }


def hyper_params_tuning():
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
    }

    optimal_params = None
    optimal_accuracy = 0.0
    for params in ParameterGrid(param_grid):
        result = cross_validate(params)
        print("---------------")
        print(params)
        print(result)
        if optimal_accuracy < result['mean_test_accuracy']:
            optimal_accuracy = result['mean_test_accuracy']
            optimal_params = params

    print({
        "optimal_params": optimal_params,
        "optimal_accuracy": optimal_accuracy
    })


if __name__ == "__main__":
    hyper_params_tuning()
