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
    model = xgb.XGBRegressor(**params)
    cross_accuracy = np.sqrt(-cross_val_score(model, features_df, y, cv=k, scoring="neg_mean_squared_error"))
    return {
        "test_accuracy": cross_accuracy,
        "mean_test_accuracy": np.mean(cross_accuracy),
    }


def tuning(X_train, y):
    # param_grid = {
    #     'learning_rate': [0.1],
    #     'n_estimators': [50, 100, 200, 500, 1000],
    #     'min_child_weight': [1],
    #     'max_depth': [5],
    #     'subsample': [0.8],
    #     'gamma': [0],
    # }

    # param_grid = {
    #     'learning_rate': [0.1, 0.03],
    #     'n_estimators': [200, 500, 1000],
    #     'min_child_weight': [1, 4, 8],
    #     'max_depth': [5, 7, 9, 11],
    #     'subsample': [0.8],
    #     'gamma': [0]
    # }

    param_grid = {
        'learning_rate': [0.03],
        'n_estimators': [500, 1000, 1500],
        'min_child_weight': [4],
        'max_depth': [5, 7],
        'subsample': [1.0, 0.8],
        'gamma': [0, 0.1, 0.2, 0.3]
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


def build_xgb(params=None):
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
    model = xgb.XGBRegressor(**params)  # the params were tuned using xgb.cv
    model.fit(X_train, y)

    print("cross_validation_rmse:",
          np.mean(np.sqrt(-cross_val_score(model, X_train, y, cv=3, scoring="neg_mean_squared_error"))))

    # model prediction
    lasso_preds = np.expm1(model.predict(X_test))
    solution = pd.DataFrame({"id": test_df.Id, "SalePrice": lasso_preds})
    solution.to_csv("./house_price/submission_xgb_v1.csv", index=False)


if __name__ == "__main__":
    build_xgb(params={'max_depth': 2, 'n_estimators': 360})
