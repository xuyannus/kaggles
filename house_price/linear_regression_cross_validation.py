import os
from random import randrange
from subprocess import call

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import export_graphviz
from common import explore_none

FEATURES = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
            '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
            'Fireplaces', 'BsmtFinSF1', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'LotArea']


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv", index_col="Id")
    return train_df


def selected_features(df):
    return df[FEATURES]


def missing_data_filling(df):
    # explore_none(df)
    return df


def extract_common_features(df):
    return df


def one_hot_encoding(df):
    return df


def normalize_features(df):
    scaled_features = StandardScaler().fit_transform(df.values)
    return pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


def cal_accuracy(y_actual, y_predicted):
    return mean_squared_error(y_actual, y_predicted, squared=False)


def cross_validate(features_df, y, params, k=3):
    kf3 = KFold(n_splits=k, shuffle=True)
    train_accuracy_list = []
    test_accuracy_list = []

    for tune_train_index, tune_test_index in kf3.split(features_df):
        X_train = features_df.iloc[tune_train_index].copy()
        X_test = features_df.iloc[tune_test_index].copy()
        y_train = np.log(y[tune_train_index])
        y_test = np.log(y[tune_test_index])

        lm = Ridge(random_state=0, alpha=params['alpha'])
        lm.fit(X_train, y_train)

        print("===========")
        for a, b in zip(lm.coef_, FEATURES):
            print("{}->{}".format(b, a))

        y_train_hat = lm.predict(X_train)
        y_test_hat = lm.predict(X_test)

        train_accuracy_list.append(cal_accuracy(y_train, y_train_hat))
        test_accuracy_list.append(cal_accuracy(y_test, y_test_hat))

    return {
        "train_accuracy": train_accuracy_list,
        "mean_train_accuracy": np.mean(train_accuracy_list),
        "test_accuracy": test_accuracy_list,
        "mean_test_accuracy": np.mean(test_accuracy_list),
    }


def hyper_params_tuning():
    train_df = load_data()

    features_df = selected_features(train_df)
    features_df = missing_data_filling(features_df)
    features_df = extract_common_features(features_df)
    features_df = one_hot_encoding(features_df)
    features_df = normalize_features(features_df)

    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    }

    optimal_params = None
    optimal_accuracy = 0.0
    for params in ParameterGrid(param_grid):
        result = cross_validate(features_df, train_df['SalePrice'].values, params)
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
