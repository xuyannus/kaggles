import pandas as pd
import numpy as np
import matplotlib
import os
import xgboost as xgb
from scipy.stats import skew
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, Lasso
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.preprocessing import StandardScaler

from common import explore_none
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv", index_col="Id")
    test_df = pd.read_csv(os.path.dirname(__file__) + "/data/test.csv", index_col="Id")
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


# it did not work for Lasso
def extract_common_features(df):
    df.loc[:, 'TotBathrooms'] = df['FullBath'] + df['HalfBath'] * 0.5 + df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5
    # df.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1, inplace=True)

    df.loc[:, 'HouseRemod'] = df.apply(lambda x: house_renovated(x), axis=1)
    df.loc[:, 'HouseNew'] = df.apply(lambda x: house_new(x), axis=1)
    # df.drop(['YearBuilt', 'YearRemodAdd'], axis=1, inplace=True)

    # df.loc[:, 'Neighborhood'] = df['Neighborhood'].apply(lambda x: bin_negihbour(x))
    # df.loc[:, 'MoSold'] = df['MoSold'].apply(lambda x: int((x - 1) / 3))
    df.loc[:, 'HouseRemod'] = df['HouseRemod'].astype('category')
    df.loc[:, 'HouseNew'] = df['HouseNew'].astype('category')
    df.loc[:, 'MoSold'] = df['MoSold'].astype('category')
    df.loc[:, 'YrSold'] = df['YrSold'].astype('category')

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

    # check boxcox_normmax against log() or 0.0
    # for col in skewed_feat_names:
    #     print({
    #         "col": col,
    #         "boxcox_lambda": boxcox_normmax(df[col].dropna() + 1)
    #     })
    # for col in skewed_feat_names:
    #     df.loc[:, col] = boxcox1p(df[col], boxcox_normmax(df[col].dropna() + 1))
    df.loc[:, skewed_feat_names] = np.log1p(df[skewed_feat_names])


def one_hot_encoding(df):
    # features with `object` or `category` dtype will be converted
    return pd.get_dummies(df)


def missing_value_fill_individual(df):
    df["MSZoning"].fillna(df["MSZoning"].mode()[0], inplace=True)
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    df["Alley"] = df["Alley"].astype('object')
    df["Alley"].fillna("NA", inplace=True)
    df["Alley"] = df["Alley"].astype('category')

    df["Utilities"].fillna(df["Utilities"].mode()[0], inplace=True)
    df["Exterior1st"].fillna(df["Exterior1st"].mode()[0], inplace=True)
    df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0], inplace=True)

    df["MasVnrType"] = df["MasVnrType"].astype('object')
    df["MasVnrType"].fillna("NA", inplace=True)
    df["MasVnrType"] = df["MasVnrType"].astype('category')

    df["MasVnrArea"].fillna(0, inplace=True)

    df["BsmtQual"] = df["BsmtQual"].astype('object')
    df["BsmtQual"].fillna("NA", inplace=True)
    df["BsmtQual"] = df["BsmtQual"].astype('category')

    df["BsmtCond"] = df["BsmtCond"].astype('object')
    df["BsmtCond"].fillna("NA", inplace=True)
    df["BsmtCond"] = df["BsmtCond"].astype('category')

    df["BsmtExposure"] = df["BsmtExposure"].astype('object')
    df["BsmtExposure"].fillna("NA", inplace=True)
    df["BsmtExposure"] = df["BsmtExposure"].astype('category')

    df["BsmtFinType1"] = df["BsmtFinType1"].astype('object')
    df["BsmtFinType1"].fillna("NA", inplace=True)
    df["BsmtFinType1"] = df["BsmtFinType1"].astype('category')

    df["BsmtFinSF1"].fillna(df["BsmtFinSF1"].median(), inplace=True)

    df["BsmtFinType2"] = df["BsmtFinType2"].astype('object')
    df["BsmtFinType2"].fillna("NA", inplace=True)
    df["BsmtFinType2"] = df["BsmtFinType2"].astype('category')

    df["BsmtFinSF2"].fillna(df["BsmtFinSF2"].median(), inplace=True)
    df["BsmtUnfSF"].fillna(0, inplace=True)
    df["TotalBsmtSF"].fillna(df["TotalBsmtSF"].median(), inplace=True)

    df["Electrical"].fillna(df["Electrical"].mode()[0], inplace=True)
    df["BsmtFullBath"].fillna(0, inplace=True)
    df["BsmtHalfBath"].fillna(0, inplace=True)

    df["KitchenQual"].fillna(df["KitchenQual"].mode()[0], inplace=True)
    df["Functional"].fillna(df["Functional"].mode()[0], inplace=True)

    df["FireplaceQu"] = df["FireplaceQu"].astype('object')
    df["FireplaceQu"].fillna("NA", inplace=True)
    df["FireplaceQu"] = df["FireplaceQu"].astype('category')

    df["GarageType"] = df["GarageType"].astype('object')
    df["GarageType"].fillna("NA", inplace=True)
    df["GarageType"] = df["GarageType"].astype('category')

    df["GarageYrBlt"].fillna(df["GarageYrBlt"].mode()[0], inplace=True)

    df["GarageFinish"] = df["GarageFinish"].astype('object')
    df["GarageFinish"].fillna("NA", inplace=True)
    df["GarageFinish"] = df["GarageFinish"].astype('category')

    df["GarageCars"].fillna(df["GarageCars"].mode()[0], inplace=True)
    df["GarageArea"].fillna(df["GarageArea"].median(), inplace=True)

    df["GarageQual"] = df["GarageQual"].astype('object')
    df["GarageQual"].fillna("NA", inplace=True)
    df["GarageQual"] = df["GarageQual"].astype('category')

    df["GarageCond"] = df["GarageCond"].astype('object')
    df["GarageCond"].fillna("NA", inplace=True)
    df["GarageCond"] = df["GarageCond"].astype('category')

    df["PoolQC"] = df["PoolQC"].astype('object')
    df["PoolQC"].fillna("NA", inplace=True)
    df["PoolQC"] = df["PoolQC"].astype('category')

    df["Fence"] = df["Fence"].astype('object')
    df["Fence"].fillna("NA", inplace=True)
    df["Fence"] = df["Fence"].astype('category')

    df["MiscFeature"] = df["MiscFeature"].astype('object')
    df["MiscFeature"].fillna("NA", inplace=True)
    df["MiscFeature"] = df["MiscFeature"].astype('category')

    df["SaleType"].fillna(df["SaleType"].mode()[0], inplace=True)


def missing_value_fill_median(df):
    df.fillna(df.median(), inplace=True)


def normalize_numerical_features(df):
    for col in df.columns:
        if df[col].dtype.name not in ["object", "category"]:
            df.loc[:, col] = StandardScaler().fit_transform(df[col].values.reshape(-1, 1))
    return df


def cross_validate(features_df, y, params, k=3):
    model = KernelRidge(**params)
    cross_accuracy = np.sqrt(-cross_val_score(model, features_df, y, cv=k, scoring="neg_mean_squared_error"))
    return {
        "test_accuracy": cross_accuracy,
        "mean_test_accuracy": np.mean(cross_accuracy),
    }


def tuning(X_train, y):
    param_grid = {
        'alpha': np.logspace(0, 4, base=0.1, num=20),
        'kernel': ["polynomial"],
        'degree': [1, 2, 3]
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


def drop_outlier(df):
    print("before drop_outlier:", df.shape)
    df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index, axis=0, inplace=True)
    print("after drop_outlier:", df.shape)


def build_kernel_ridge(params=None):
    train_df, test_df = load_data()
    # drop_outlier(train_df)

    combined_df = pd.concat((train_df.loc[:, 'MSSubClass':'SaleCondition'],
                             test_df.loc[:, 'MSSubClass':'SaleCondition']))

    # feature engineering
    config_categorical_features(combined_df)
    missing_value_fill_individual(combined_df)
    # missing_value_fill_median(combined_df)
    # combined_df = extract_common_features(combined_df)
    log_transform_features(combined_df)
    combined_df = normalize_numerical_features(combined_df)
    combined_df = one_hot_encoding(combined_df)

    X_train = combined_df[:train_df.shape[0]]
    X_test = combined_df[train_df.shape[0]:]
    y = np.log1p(train_df["SalePrice"])

    # model training
    if params is None:
        params = tuning(X_train, y)

    model = KernelRidge(**params)
    model.fit(X_train, y)
    print("cross_validation_rmse:",
          np.mean(np.sqrt(-cross_val_score(model, X_train, y, cv=3, scoring="neg_mean_squared_error"))))

    # model prediction
    lasso_preds = np.expm1(model.predict(X_test))
    solution = pd.DataFrame({"id": test_df.index, "SalePrice": lasso_preds})
    solution.to_csv("./house_price/submission_kernel_ridge_v1.csv", index=False)


if __name__ == "__main__":
    build_kernel_ridge(params={'alpha': 0.054555947811685206, 'degree': 2, 'kernel': 'polynomial'})
