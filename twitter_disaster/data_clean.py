import pandas as pd
import re

train = pd.read_csv("/Users/yanxu/Documents/kaggles/twitter_disaster/data/train.csv")
test = pd.read_csv("/Users/yanxu/Documents/kaggles/twitter_disaster/data/test.csv")

SPACE_PATTERN = re.compile(r'%20')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
TAG_PATTERN = re.compile(r'<.*?>')
TOKEN_PATTERN = re.compile(r'[^A-Za-z0-9 ]+')


def concat_string_and_clean(x):
    message = ""
    if type(x["keyword"]) == str and len(x["keyword"]) > 0:
        message += x["keyword"] + " "
    if type(x["location"]) == str and len(x["location"]) > 0:
        message += x["location"] + " "
    if type(x["text"]) == str and len(x["text"]) > 0:
        message += x["text"]

    message = message.lower()
    message = SPACE_PATTERN.sub(' ', message)
    message = URL_PATTERN.sub('', message)
    message = TAG_PATTERN.sub('', message)
    message = TOKEN_PATTERN.sub('', message)
    return message


train.loc[:, "text"] = train.apply(lambda x: concat_string_and_clean(x), axis=1)
test.loc[:, "text"] = test.apply(lambda x: concat_string_and_clean(x), axis=1)

train.drop(columns=['keyword', 'location'], inplace=True)
test.drop(columns=['keyword', 'location'], inplace=True)

train.to_csv("/Users/yanxu/Documents/kaggles/twitter_disaster/data/train_v2.csv", index=None)
test.to_csv("/Users/yanxu/Documents/kaggles/twitter_disaster/data/test_v2.csv", index=None)
