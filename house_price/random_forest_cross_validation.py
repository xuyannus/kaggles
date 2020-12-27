import os
from random import randrange
from subprocess import call

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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


def fill_missing_garage_area(x, default_garage_area):
    if pd.isna(x['GarageArea']):
        return default_garage_area[x['YearBuilt']]
    else:
        return x['GarageArea']


def fill_missing_garage_cars(x, default_garage_cars):
    if pd.isna(x['GarageCars']):
        return default_garage_cars[x['YearBuilt']]
    else:
        return x['GarageCars']


def fill_missing_total_bsmt(x, df):
    if pd.isna(x['TotalBsmtSF']):
        benchmark = x['1stFlrSF']
        return df[(df['1stFlrSF'] >= (benchmark - 100)) & (df['1stFlrSF'] <= (benchmark + 100))]['TotalBsmtSF'].median()
    else:
        return x['TotalBsmtSF']


def fill_missing_bsmt_fin_1(x, df):
    if pd.isna(x['BsmtFinSF1']):
        benchmark = x['1stFlrSF']
        return df[(df['1stFlrSF'] >= (benchmark - 100)) & (df['1stFlrSF'] <= (benchmark + 100))]['BsmtFinSF1'].median()
    else:
        return x['BsmtFinSF1']


def missing_data_filling(df):
    default_garage_area = df.groupby('YearBuilt')['GarageArea'].median().to_dict()
    df['GarageArea'] = df.apply(lambda x: fill_missing_garage_area(x, default_garage_area), axis=1)

    default_garage_cars = df.groupby('YearBuilt')['GarageCars'].median().to_dict()
    df['GarageCars'] = df.apply(lambda x: fill_missing_garage_cars(x, default_garage_cars), axis=1)

    df['TotalBsmtSF'] = df.apply(lambda x: fill_missing_total_bsmt(x, df), axis=1)
    df['BsmtFinSF1'] = df.apply(lambda x: fill_missing_bsmt_fin_1(x, df), axis=1)

    explore_none(df)
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

        rf_model = RandomForestRegressor(random_state=31,
                                         max_depth=params['max_depth'],
                                         n_estimators=params['n_estimators'],
                                         max_features=params['max_features'])

        rf_model.fit(X_train, y_train)

        # print("===========")
        # for a, b in zip(rf_model.feature_importances_, FEATURES):
        #     print("{}->{}".format(b, a))

        y_train_hat = rf_model.predict(X_train)
        y_test_hat = rf_model.predict(X_test)

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
    # features_df = normalize_features(features_df)

    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'max_features': ["sqrt", None],
        'n_estimators': [20, 50, 100, 500]
    }

    optimal_params = None
    optimal_accuracy = 100.0
    for params in ParameterGrid(param_grid):
        result = cross_validate(features_df, train_df['SalePrice'].values, params)
        print("---------------")
        print(params)
        print(result)
        if optimal_accuracy > result['mean_test_accuracy']:
            optimal_accuracy = result['mean_test_accuracy']
            optimal_params = params

    print({
        "optimal_params": optimal_params,
        "optimal_accuracy": optimal_accuracy
    })


# {'optimal_params': {'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 500}, 'optimal_accuracy': 0.14826421440179333}
if __name__ == "__main__":
    hyper_params_tuning()
