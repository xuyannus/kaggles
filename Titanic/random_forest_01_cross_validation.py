import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import OneHotEncoder

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


def model_accuracy(rf_model, X_train, X_test):
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

    selected_columns = ['AgeEncoded', 'Sex_Survive_female_normal', 'Sex_Survive_female_special', 'Sex_Survive_male_normal', 'Sex_Survive_male_special']
    rf_model.fit(X_train[selected_columns], X_train['Survived'])

    train_y = X_train['Survived'].values
    prediction_train_y = rf_model.predict(X_train[selected_columns])

    test_y = X_test['Survived'].values
    prediction_test_y = rf_model.predict(X_test[selected_columns])

    return accuracy_score(train_y, prediction_train_y), accuracy_score(test_y, prediction_test_y)


def cross_validate(params, k=5):
    train_df = load_data()
    train_df = missing_data_filling(train_df)
    train_df = extract_common_features(train_df)

    kf3 = KFold(n_splits=k, shuffle=True)
    train_accuracy_list = []
    test_accuracy_list = []

    for tune_train_index, tune_test_index in kf3.split(train_df):
        X_train = train_df.iloc[tune_train_index].copy()
        X_test = train_df.iloc[tune_test_index].copy()
        rf_model = RandomForestClassifier(random_state=31, max_depth=params['max_depth'],
                                          n_estimators=params['n_estimators'])

        train_accuracy, test_accuracy = model_accuracy(rf_model, X_train, X_test)
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
        'max_depth': [2, 3, 4, 5],
        'n_estimators': [5, 10, 20, 50, 100]
    }

    optimal_params = None
    optimal_accuracy = 0.0
    for params in ParameterGrid(param_grid):
        result = cross_validate(params)
        print(result)
        if optimal_accuracy < result['mean_test_accuracy']:
            optimal_accuracy = result['mean_test_accuracy']
            optimal_params = params

    print({
        "optimal_params": optimal_params,
        "optimal_accuracy": optimal_accuracy
    })


# {'max_depth': 5, 'n_estimators': 50}
# example:
# training accuracy: 0.8978690294214979
# testing accuracy: 0.8485594124662608
if __name__ == "__main__":
    hyper_params_tuning()
