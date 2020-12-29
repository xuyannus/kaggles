import pandas as pd
import numpy as np
import matplotlib
import lightgbm as lgb
import os
import xgboost as xgb
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, ParameterGrid, KFold


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv")
    test_df = pd.read_csv(os.path.dirname(__file__) + "/data/test.csv")
    return train_df, test_df


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


def missing_value_fill(df):
    df.fillna(df.median(), inplace=True)


def cal_accuracy(y_actual, y_predicted):
    return mean_squared_error(y_actual, y_predicted, squared=False)


def cross_validate(features_df, y, params, k=3):
    model = lgb.LGBMRegressor(**params)
    cross_accuracy = np.sqrt(-cross_val_score(model, features_df, y, cv=k, scoring="neg_mean_squared_error"))
    return {
        "test_accuracy": cross_accuracy,
        "mean_test_accuracy": np.mean(cross_accuracy),
    }


def tuning(X_train, y):
    param_grid = {
        'num_leaves': [10, 50, 100, 500],
        'min_child_samples': [4, 8, 16, 32],
        'lambda_l2': [0, 0.001, 0.01, 0.1],
        'learning_rate': [0.1, 0.05, 0.02, 0.01],
        'n_estimators': [5, 10, 20, 50, 100, 200, 500]
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
    # log_transform_features(combined_df)
    missing_value_fill(combined_df)

    X_train = combined_df[:train_df.shape[0]]
    X_test = combined_df[train_df.shape[0]:]
    y = np.log1p(train_df["SalePrice"])

    # model tuning
    if params is None:
        params = tuning(X_train, y)

    # model training
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y)
    print("cross_validation_rmse:",
          np.mean(np.sqrt(-cross_val_score(model, X_train, y, cv=3, scoring="neg_mean_squared_error"))))

    # model prediction
    lgbm_preds = np.expm1(model.predict(X_test))
    solution = pd.DataFrame({"id": test_df.Id, "SalePrice": lgbm_preds})
    solution.to_csv("./house_price/submission_lgbm_v1.csv", index=False)


if __name__ == "__main__":
    # build_xgb()
    build_xgb({'lambda_l2': 0, 'learning_rate': 0.05, 'min_child_samples': 4, 'n_estimators': 500, 'num_leaves': 10})
