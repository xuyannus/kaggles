import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from Titanic.random_forest_02_cross_validation import missing_data_filling, extract_common_features, one_hot_encoding, \
    cal_family_survive_rate, encode_survive_rate, combine_sex_survive_rate


def load_data():
    train_df = pd.read_csv(os.path.dirname(__file__) + "/data/train.csv", index_col="PassengerId")
    test_df = pd.read_csv(os.path.dirname(__file__) + "/data/test.csv", index_col="PassengerId")
    return train_df, test_df


def prediction():
    train_df, test_df = load_data()
    combined_df = pd.concat([train_df, test_df])

    combined_df = missing_data_filling(combined_df)
    combined_df = extract_common_features(combined_df)
    combined_df = one_hot_encoding(combined_df)

    train_df = combined_df.head(train_df.shape[0])
    test_df = combined_df.tail(test_df.shape[0])

    family_survive_rate_train = cal_family_survive_rate(train_df)
    default_survive_rate = -1

    temp = train_df.merge(family_survive_rate_train, left_on='Family', right_on='Family', how='left')
    temp['Family_Survive_Rate'].fillna(default_survive_rate, inplace=True)
    train_df.loc[:, 'Family_Survive_Rate'] = [encode_survive_rate(x) for x in temp['Family_Survive_Rate'].values]
    train_df.loc[:, 'Sex_Survive'] = train_df.apply(lambda x: combine_sex_survive_rate(x), axis=1)

    temp2 = test_df.merge(family_survive_rate_train, left_on='Family', right_on='Family', how='left')
    temp2['Family_Survive_Rate'].fillna(default_survive_rate, inplace=True)
    test_df.loc[:, 'Family_Survive_Rate'] = [encode_survive_rate(x) for x in temp2['Family_Survive_Rate'].values]
    test_df.loc[:, 'Sex_Survive'] = test_df.apply(lambda x: combine_sex_survive_rate(x), axis=1)

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

    selected_columns = ['FareAdjusted', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
                        'SibSp', 'Parch', 'AgeEncoded', 'Sex_Survive_female_normal', 'Sex_Survive_female_special',
                        'Sex_Survive_male_normal', 'Sex_Survive_male_special', 'CabinEncoded_ABCT', 'CabinEncoded_DE',
                        'CabinEncoded_FG', 'CabinEncoded_M', 'FamilySizeEncode_1',
                        'FamilySizeEncode_2', 'FamilySizeEncode_3', 'FamilySizeEncode_4']

    clf = LinearDiscriminantAnalysis()
    clf.fit(train_df[selected_columns], train_df['Survived'])

    train_df['Prediction'] = clf.predict(train_df[selected_columns]).astype(int)
    print({
        "training": accuracy_score(train_df['Survived'].values, train_df['Prediction'].values)
    })

    test_df['Prediction'] = clf.predict(test_df[selected_columns]).astype(int)

    print({
        "train_df_truth": train_df.groupby(['Sex']).agg({'Survived': ['count', 'mean']}),
        "train_df_predict": train_df.groupby(['Sex']).agg({'Prediction': ['count', 'mean']}),
        "test_df_predict": test_df.groupby(['Sex']).agg({'Prediction': ['count', 'mean']})
    })

    test_df['PassengerId'] = test_df.index
    submit = test_df[['PassengerId', 'Prediction']]
    submit.columns = ['PassengerId', 'Survived']
    submit[['PassengerId', 'Survived']].to_csv("lda_submission_01.csv", index=False)


if __name__ == "__main__":
    prediction()
