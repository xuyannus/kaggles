import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from titanic.family_model_cross_validation import missing_data_filling, extract_common_features, cal_family_survive_rate
from titanic.random_forest_01_cross_validation import encode_survive_rate, combine_sex_survive_rate


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv", index_col="PassengerId")
    test_df = pd.read_csv(os.path.dirname(__file__) + "/data/test.csv", index_col="PassengerId")
    return train_df, test_df


def prediction(params):
    train_df, test_df = load_data()
    combined_df = pd.concat([train_df, test_df])
    combined_df = missing_data_filling(combined_df)
    combined_df = extract_common_features(combined_df)
    train_df = combined_df.head(train_df.shape[0])
    test_df = combined_df.tail(test_df.shape[0])

    rf_model = RandomForestClassifier(random_state=31, max_depth=params['max_depth'], n_estimators=params['n_estimators'])

    family_survive_rate_train = cal_family_survive_rate(train_df)
    default_survive_rate = -1

    temp = train_df.merge(family_survive_rate_train, left_on='Family', right_on='Family', how='left')
    temp['Family_Survive_Rate'].fillna(default_survive_rate, inplace=True)
    train_df['Family_Survive_Rate'] = [encode_survive_rate(x) for x in temp['Family_Survive_Rate'].values]
    train_df['Sex_Survive'] = train_df.apply(lambda x: combine_sex_survive_rate(x), axis=1)

    temp2 = test_df.merge(family_survive_rate_train, left_on='Family', right_on='Family', how='left')
    temp2['Family_Survive_Rate'].fillna(default_survive_rate, inplace=True)
    test_df['Family_Survive_Rate'] = [encode_survive_rate(x) for x in temp2['Family_Survive_Rate'].values]
    test_df['Sex_Survive'] = test_df.apply(lambda x: combine_sex_survive_rate(x), axis=1)

    temp_combine = pd.concat([train_df, test_df])
    surviveEncoder = OneHotEncoder(handle_unknown='ignore')
    surviveEncoder.fit(temp_combine[['Sex_Survive']])
    temp_df = pd.DataFrame(surviveEncoder.transform(train_df[['Sex_Survive']]).toarray(),
                           columns=surviveEncoder.get_feature_names(['Sex_Survive']),
                           index=train_df.index)
    train_df = pd.concat([train_df, temp_df], axis=1)
    temp2_df = pd.DataFrame(surviveEncoder.transform(test_df[['Sex_Survive']]).toarray(),
                            columns=surviveEncoder.get_feature_names(['Sex_Survive']),
                            index=test_df.index)
    test_df = pd.concat([test_df, temp2_df], axis=1)

    selected_columns = ['AgeEncoded', 'Sex_Survive_female_normal', 'Sex_Survive_female_special',
                        'Sex_Survive_male_normal', 'Sex_Survive_male_special']
    rf_model.fit(train_df[selected_columns], train_df['Survived'])

    train_df['Prediction'] = rf_model.predict(train_df[selected_columns]).astype(int)
    print({
        "training": accuracy_score(train_df['Survived'].values, train_df['Prediction'].values)
    })

    test_df['Prediction'] = rf_model.predict(test_df[selected_columns]).astype(int)
    test_df['PassengerId'] = test_df.index
    submit = test_df[['PassengerId', 'Prediction']]
    submit.columns = ['PassengerId', 'Survived']
    submit[['PassengerId', 'Survived']].to_csv("./titanic/submission_rf_01.csv", index=False)


if __name__ == "__main__":
    prediction(params={'max_depth': 5, 'n_estimators': 50})
