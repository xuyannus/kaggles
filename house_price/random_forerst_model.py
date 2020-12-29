import pandas as pd
import numpy as np
import matplotlib
import os
import xgboost as xgb
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, ParameterGrid, KFold

from common import explore_none


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv")
    test_df = pd.read_csv(os.path.dirname(__file__) + "/data/test.csv")
    return train_df, test_df


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
    elif x <= 2000:
        return 2
    elif x <= 2500:
        return 3
    elif x <= 3000:
        return 4
    elif x <= 4000:
        return 5
    else:
        return 6


# it did not work for Lasso
def extract_common_features(df):
    df.loc[:, 'TotBathrooms'] = df['FullBath'] + df['HalfBath'] * 0.5 + df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5
    df.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1, inplace=True)

    df.loc[:, 'HouseAge'] = (df['YrSold'] - df['YearRemodAdd']).clip(0, 1000)
    df.loc[:, 'HouseRemod'] = df.apply(lambda x: house_renovated(x), axis=1)
    df.loc[:, 'HouseNew'] = df.apply(lambda x: house_new(x), axis=1)
    df.drop(['YearBuilt', 'YearRemodAdd'], axis=1, inplace=True)

    # df.loc[:, 'Neighborhood'] = df['Neighborhood'].apply(lambda x: bin_negihbour(x))
    # df.loc[:, 'MoSold'] = df['MoSold'].apply(lambda x: int((x - 1) / 3))
    df.loc[:, 'MoSold'] = df['MoSold'].astype('category')
    df.loc[:, 'YrSold'] = df['YrSold'].astype('category')

    # df.loc[:, 'TotalSqFeet'] = df['GrLivArea'] + df['TotalBsmtSF']
    # df.loc[:, 'TotalSqFeet'] = df['TotalSqFeet'].apply(lambda x: bin_total_sq_feet(x))
    # df.drop(['TotalBsmtSF'], axis=1, inplace=True)

    # df.loc[:, 'Exterior1st'] = df['Exterior1st'].apply(lambda x: bin_exterior1st(x))
    return df


def config_categorical_features(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df.loc[:, col] = df[col].astype('category')
        elif col in ['MSSubClass', 'MoSold', 'YrSold']:
            df.loc[:, col] = df[col].astype('category')


def log_transform_features(df):
    numeric_feats = df.dtypes[(df.dtypes != "category") & (df.dtypes != "object")].index
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.5]
    skewed_feat_names = skewed_feats.index
    df.loc[:, skewed_feat_names] = np.log1p(df[skewed_feat_names])


def one_hot_encoding(df):
    # features with `object` or `category` dtype will be converted
    return pd.get_dummies(df)


def missing_value_fill(df):
    df.fillna(df.median(), inplace=True)


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
                                         min_samples_leaf=params['min_samples_leaf'],
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


def tuning(X_train, y):
    param_grid = {
        'max_depth': [5, 10, 50],
        'max_features': [None],
        'min_samples_leaf': [4, 8, 16, 32],
        'n_estimators': [20, 50, 100, 500]
    }

    optimal_params = None
    optimal_accuracy = 100.0
    for params in ParameterGrid(param_grid):
        result = cross_validate(X_train, y, params)
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
    return optimal_params


def build_rf(params=None):
    train_df, test_df = load_data()
    combined_df = pd.concat((train_df.loc[:, 'MSSubClass':'SaleCondition'],
                             test_df.loc[:, 'MSSubClass':'SaleCondition']))

    # feature engineering
    config_categorical_features(combined_df)
    # combined_df = extract_common_features(combined_df)
    log_transform_features(combined_df)
    combined_df = one_hot_encoding(combined_df)
    missing_value_fill(combined_df)

    X_train = combined_df[:train_df.shape[0]]
    X_test = combined_df[train_df.shape[0]:]
    y = np.log1p(train_df["SalePrice"])

    # model tuning
    if params is None:
        params = tuning(X_train, y)

    # model training
    model = RandomForestRegressor(random_state=31,
                                  max_depth=params['max_depth'],
                                  n_estimators=params['n_estimators'],
                                  min_samples_leaf=params['min_samples_leaf'],
                                  max_features=params['max_features'])
    model.fit(X_train, y)

    print("cross_validation_rmse:",
          np.mean(np.sqrt(-cross_val_score(model, X_train, y, cv=3, scoring="neg_mean_squared_error"))))

    # model prediction
    lasso_preds = np.expm1(model.predict(X_test))
    solution = pd.DataFrame({"id": test_df.Id, "SalePrice": lasso_preds})
    solution.to_csv("./house_price/submission_rf_v1.csv", index=False)


if __name__ == "__main__":
    # build_rf()
    build_rf(params={'max_depth': 50, 'max_features': None, 'min_samples_leaf': 4, 'n_estimators': 50})
