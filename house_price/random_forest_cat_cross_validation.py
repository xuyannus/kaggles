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
                'Neighborhood',
                'YearBuilt',
                'YearRemodAdd',
                'OverallQual',
                'OverallCond',
                'WoodDeckSF',
                'MoSold',
                'YrSold',
                'GarageQual',
                'GarageCars',
                'BedroomAbvGr',
                'ExterQual',
                'BsmtQual',
                'Exterior1st',
                'KitchenQual',
                'FireplaceQu',
                'TotRmsAbvGrd',
                'FullBath',
                'HalfBath',
                'BsmtFullBath',
                'BsmtHalfBath']

CAT_FEATURES = ['Neighborhood',
                'MSSubClass',
                'GarageQual',
                'ExterQual',
                'BsmtQual',
                'KitchenQual',
                'FireplaceQu',
                'Exterior1st',
                'MoSold',
                'YrSold']


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
    # default_garage_area = df.groupby('YearBuilt')['GarageArea'].median().to_dict()
    # df['GarageArea'] = df.apply(lambda x: fill_missing_garage_area(x, default_garage_area), axis=1)

    default_garage_cars = df.groupby('YearBuilt')['GarageCars'].median().to_dict()
    df['GarageCars'] = df.apply(lambda x: fill_missing_garage_cars(x, default_garage_cars), axis=1)

    df['GarageQual'].fillna('NoGarage', inplace=True)
    df['BsmtQual'].fillna('NoBasement', inplace=True)
    df['FireplaceQu'].fillna('NoFireplace', inplace=True)
    # df['GarageFinish'].fillna('NoGarage', inplace=True)
    # df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)
    df['TotalBsmtSF'] = df.apply(lambda x: fill_missing_total_bsmt(x, df), axis=1)
    # df['BsmtFinSF1'] = df.apply(lambda x: fill_missing_bsmt_fin_1(x, df), axis=1)

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


def bin_exterior1st(x):
    if x not in ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood']:
        return 'Others'
    else:
        return x


def bin_negihbour(x):
    if x in ['StoneBr', 'NridgHt', 'NoRidge']:
        return 'HighPriceSuburb'
    elif x in ['MeadowV', 'IDOTRR', 'BrDale']:
        return 'LowPriceSuburb'
    else:
        return 'MediumPriceSuburb'


def bin_total_sq_feet(x):
    if x <= 1000:
        return 1
    elif x<= 2000:
        return 2
    elif x <= 2500:
        return 3
    elif x <= 3000:
        return 4
    elif x <= 4000:
        return 5
    else:
        return 6


def extract_common_features(df):
    df.loc[:, 'TotBathrooms'] = df['FullBath'] + df['HalfBath'] * 0.5 + df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5
    df.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1, inplace=True)

    df.loc[:, 'HouseAge'] = (df['YrSold'] - df['YearRemodAdd']).clip(0, 1000)
    df.loc[:, 'HouseRemod'] = df.apply(lambda x: house_renovated(x), axis=1)
    df.loc[:, 'HouseNew'] = df.apply(lambda x: house_new(x), axis=1)
    df.drop(['YearBuilt', 'YearRemodAdd'], axis=1, inplace=True)

    df.loc[:, 'Neighborhood'] = df['Neighborhood'].apply(lambda x: bin_negihbour(x))
    df.loc[:, 'MoSold'] = df['MoSold'].apply(lambda x: int((x - 1) / 3))

    df.loc[:, 'TotalSqFeet'] = df['GrLivArea'] + df['TotalBsmtSF']
    df.loc[:, 'TotalSqFeet'] = df['TotalSqFeet'].apply(lambda x: bin_total_sq_feet(x))
    df.drop(['TotalBsmtSF'], axis=1, inplace=True)

    df.loc[:, 'Exterior1st'] = df['Exterior1st'].apply(lambda x: bin_exterior1st(x))

    return df


def drop_outliers(df, y):
    # those two houses are huge and high quality but low price
    df.drop([524, 1299], inplace=True)
    y = np.delete(y, [524, 1299])
    return df, y


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


def print_top_k_features(model, features):
    print("---------")
    for x, y in sorted(zip(model.feature_importances_, features), reverse=True)[:20]:
        print({"feature": y, "score": x})
    print("---------")


def cross_validate(features_df, y, params, k=5):
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
                                         min_samples_leaf=params['min_samples_leaf'],
                                         max_features=params['max_features'])
        rf_model.fit(X_train, y_train)
        # print_top_k_features(rf_model, X_train.columns)

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
    y = train_df['SalePrice'].values

    features_df = selected_features(train_df)
    features_df = missing_data_filling(features_df)
    features_df = extract_common_features(features_df)

    features_df, y = drop_outliers(features_df, y)
    features_df = one_hot_encoding(features_df)

    param_grid = {
        'max_depth': [5, 10, 50],
        'max_features': [None],
        'min_samples_leaf': [4, 8, 16, 32],
        'n_estimators': [20, 50, 100, 500]
    }

    optimal_params = None
    optimal_accuracy = 100.0
    for params in ParameterGrid(param_grid):
        result = cross_validate(features_df, y, params)
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


# {'max_depth': 10, 'max_features': None, 'min_samples_leaf': 4, 'n_estimators': 100}, 'optimal_accuracy': 0.14305061923002066}
if __name__ == "__main__":
    hyper_params_tuning()
