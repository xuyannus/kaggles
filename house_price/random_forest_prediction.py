import os
from random import randrange

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from house_price.house_common import plot_first_tree
from house_price.random_forest_cross_validation import selected_features, missing_data_filling, extract_common_features, \
    one_hot_encoding, normalize_features, FEATURES, cal_accuracy


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv", index_col="Id")
    test_df = pd.read_csv(os.path.dirname(__file__) + "/data/test.csv", index_col="Id")
    return train_df, test_df


def prediction(params):
    train_df, test_df = load_data()
    combined_df = pd.concat([train_df, test_df])

    features_df = selected_features(combined_df)
    features_df = missing_data_filling(features_df)
    features_df = extract_common_features(features_df)
    features_df = one_hot_encoding(features_df)
    features_df = normalize_features(features_df)

    train_feature_df = features_df.head(train_df.shape[0])
    test_feature_df = features_df.tail(test_df.shape[0])
    y_train = np.log(train_df['SalePrice'].values)

    rf_model = RandomForestRegressor(random_state=31,
                                     max_depth=params['max_depth'],
                                     n_estimators=params['n_estimators'],
                                     max_features=params['max_features'])
    rf_model.fit(train_feature_df, y_train)

    for x, y in zip(FEATURES, rf_model.feature_importances_):
        print("{}->{}".format(x, y))

    plot_first_tree(rf_model, feature_names=FEATURES,
                    file_name_id="{}_{}_{}".format(params['max_depth'], params['n_estimators'], randrange(100)))

    print({
        "training": cal_accuracy(y_train, rf_model.predict(train_feature_df))
    })

    test_df['Prediction'] = np.exp(rf_model.predict(test_feature_df))
    test_df['Id'] = test_df.index
    submit = test_df[['Id', 'Prediction']]
    submit.columns = ['Id', 'SalePrice']
    submit[['Id', 'SalePrice']].to_csv("./house_price/submission_rf_01.csv", index=False)


if __name__ == "__main__":
    prediction(params={'max_depth': 9, 'max_features': 'sqrt', 'n_estimators': 500})
