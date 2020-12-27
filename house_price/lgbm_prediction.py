import os
from random import randrange

import pandas as pd
import numpy as np
import lightgbm as lgb

from house_price.house_common import plot_first_tree
from house_price.lgbm_cat_cross_validation import selected_features, missing_data_filling, extract_common_features, \
    drop_outliers, one_hot_encoding, print_top_k_features, cal_accuracy


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv", index_col="Id")
    test_df = pd.read_csv(os.path.dirname(__file__) + "/data/test.csv", index_col="Id")
    return train_df, test_df


def prediction(params):
    train_df, test_df = load_data()
    train_df = drop_outliers(train_df)

    combined_df = pd.concat([train_df, test_df])
    y_train = np.log(train_df['SalePrice'].values)

    features_df = selected_features(combined_df)
    features_df = missing_data_filling(features_df)
    features_df = extract_common_features(features_df)
    features_df = one_hot_encoding(features_df)

    train_feature_df = features_df.head(train_df.shape[0])
    test_feature_df = features_df.tail(test_df.shape[0])

    model = lgb.LGBMRegressor(random_state=1,
                              num_leaves=params['num_leaves'],
                              min_child_samples=params['min_child_samples'],
                              lambda_l2=params['lambda_l2'],
                              learning_rate=params['learning_rate'],
                              n_estimators=params['n_estimators'])

    ll = len(train_feature_df)
    model.fit(train_feature_df.iloc[:ll-1], y_train[:ll-1], eval_set=(train_feature_df.iloc[:ll-1], y_train[:ll-1]))
    # model.fit(train_feature_df, y_train, eval_set=(train_feature_df, y_train))
    print_top_k_features(model, train_feature_df.columns)

    # plot_first_tree(model, feature_names=train_feature_df.columns,
    #                 file_name_id="{}_{}_{}".format(params['learning_rate'], params['n_estimators'], randrange(100)))

    print({
        "training": cal_accuracy(y_train, model.predict(train_feature_df))
    })

    test_df['Prediction'] = np.exp(model.predict(test_feature_df))
    test_df['Id'] = test_df.index
    submit = test_df[['Id', 'Prediction']]
    submit.columns = ['Id', 'SalePrice']
    submit[['Id', 'SalePrice']].to_csv("./house_price/submission_lgbm_01.csv", index=False)


if __name__ == "__main__":
    prediction(params={'lambda_l2': 0.1, 'learning_rate': 0.05, 'min_child_samples': 8, 'n_estimators': 500, 'num_leaves': 10})
