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

ALL_FEATURES = ['GrLivArea',
                'MSSubClass',
                'TotalBsmtSF',
                'LotArea',
                'BsmtFinSF1',
                '1stFlrSF',
                'GarageArea',
                'OpenPorchSF',
                'Neighborhood',
                'YearBuilt',
                'YearRemodAdd',
                '2ndFlrSF',
                'OverallQual',
                'OverallCond',
                'WoodDeckSF',
                'MoSold',
                'YrSold',
                'GarageQual',
                'GarageCars',
                'BedroomAbvGr',
                'ExterQual',
                'Exterior1st',
                'TotRmsAbvGrd',
                'FullBath',
                'HalfBath',
                'BsmtFullBath',
                'BsmtHalfBath',
                'EnclosedPorch']

CAT_FEATURES = ['Neighborhood',
                'MSSubClass',
                'GarageQual',
                'ExterQual',
                'Exterior1st',
                'MoSold',
                'YrSold',
                ]


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv", index_col="Id")
    return train_df


def selected_features(df):
    return df[ALL_FEATURES]


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

    df['GarageQual'].fillna('NoGarage', inplace=True)
    # df['GarageFinish'].fillna('NoGarage', inplace=True)
    # df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)
    df['TotalBsmtSF'] = df.apply(lambda x: fill_missing_total_bsmt(x, df), axis=1)
    df['BsmtFinSF1'] = df.apply(lambda x: fill_missing_bsmt_fin_1(x, df), axis=1)

    explore_none(df)
    return df


def house_renovated(x):
    if x['YearBuilt'] < x['YearRemodAdd']:
        return 1
    else:
        return 0


def house_new(x):
    if x['YrSold'] <= x['YearBuilt'] + 1:
        return 1
    else:
        return 0


def bin_negihbour(x):
    if x in ['StoneBr', 'NridgHt', 'NoRidge']:
        return 'HighPriceSuburb'
    elif x in ['MeadowV', 'IDOTRR', 'BrDale']:
        return 'LowPriceSuburb'
    else:
        return 'MediumPriceSuburb'


def extract_common_features(df):
    df.loc[:, 'TotBathrooms'] = df['FullBath'] + df['HalfBath'] * 0.5 + df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5
    df.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1, inplace=True)

    df.loc[:, 'HouseAge'] = (df['YrSold'] - df['YearRemodAdd']).clip(0, 1000)
    df.loc[:, 'HouseRemod'] = df.apply(lambda x: house_renovated(x), axis=1)
    df.loc[:, 'HouseNew'] = df.apply(lambda x: house_new(x), axis=1)
    df.loc[:, 'Neighborhood'] = df['Neighborhood'].apply(lambda x: bin_negihbour(x))

    return df


def drop_outliers(df):
    # those two houses are huge and high quality but low price
    df.drop([524, 1299], inplace=True)
    return df


def one_hot_encoding(df):
    for cat_fea in CAT_FEATURES:
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(df[[cat_fea]])
        temp_df = pd.DataFrame(encoder.transform(df[[cat_fea]]).toarray(), columns=encoder.get_feature_names([cat_fea]), index=df.index)
        df = pd.concat([df, temp_df], axis=1)
        df.drop(cat_fea, axis=1, inplace=True)
    return df


def cal_accuracy(y_actual, y_predicted):
    return mean_squared_error(y_actual, y_predicted, squared=False)


def cross_validate(features_df, y, params, k=3):
    kf3 = KFold(n_splits=k, shuffle=True)
    train_accuracy_list = []
    test_accuracy_list = []

    some_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',
                     'TotRmsAbvGrd', 'YearRemodAdd', 'BsmtFinSF1', 'WoodDeckSF', '2ndFlrSF',
                     'OpenPorchSF', 'LotArea']
    features_df = features_df[some_features]

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
    features_df = drop_outliers(features_df)
    features_df = one_hot_encoding(features_df)

    param_grid = {
        'max_depth': [5, 6, 7, 8, 9, 10, 11, 12],
        'max_features': ['sqrt', None],
        'n_estimators': [20, 50, 100, 500, 1000]
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


# {'optimal_params': {'max_depth': 9, 'max_features': None, 'n_estimators': 500}, 'optimal_accuracy': 0.1462004605199809}
if __name__ == "__main__":
    hyper_params_tuning()
