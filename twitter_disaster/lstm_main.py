import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

import spacy
import torch
from torchtext import data
import torch.nn as nn
import re

from twitter_disaster.df_set import DataFrameDataset
from twitter_disaster.lstm_model import LSTM_net

train = pd.read_csv("/Users/yanxu/Documents/kaggles/twitter_disaster/data/train.csv")
test = pd.read_csv("/Users/yanxu/Documents/kaggles/twitter_disaster/data/test.csv")

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
    message = URL_PATTERN.sub('', message)
    message = TAG_PATTERN.sub('', message)
    message = TOKEN_PATTERN.sub('', message)
    return message


train.loc[:, "combined_text"] = train.apply(lambda x: concat_string_and_clean(x), axis=1)
test.loc[:, "combined_text"] = test.apply(lambda x: concat_string_and_clean(x), axis=1)

train.drop(columns=['id', 'keyword', 'location', 'text'], inplace=True)
test.drop(columns=['keyword', 'location', 'text'], inplace=True)

train_df, valid_df = train_test_split(train, test_size=0.3)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

spacy_en = spacy.load("en_core_web_sm")


def tokenize_fun(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


TEXT = data.Field(sequential=True, use_vocab=True, tokenize=tokenize_fun, include_lengths=True)
LABEL = data.Field(sequential=False, use_vocab=False)
fields = [('combined_text', TEXT), ('label', LABEL)]

train_ds, val_ds, test_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=valid_df, test_df=test)

TEXT.build_vocab(train_ds, max_size=10000, vectors='glove.6B.200d', min_freq=2)
LABEL.build_vocab(train_ds)

BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_ds, val_ds),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device)

num_epochs = 30
learning_rate = 0.001
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # padding

model = LSTM_net(INPUT_DIM,
                 EMBEDDING_DIM,
                 HIDDEN_DIM,
                 N_LAYERS,
                 BIDIRECTIONAL,
                 DROPOUT,
                 PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

model.to(device)  # CNN to GPU
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# training function
def train(model, iterator):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        text, text_lengths = batch.combined_text

        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label.type_as(predictions))
        acc = binary_accuracy(predictions, batch.label.type_as(predictions))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator):
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.combined_text
            predictions = model(text, text_lengths).squeeze(1)
            acc = binary_accuracy(predictions, batch.label.type_as(predictions))
            epoch_acc += acc.item()
    return epoch_acc / len(iterator)


def predict(model, iterator):
    res = np.array([])
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.combined_text
            predictions = model(text, text_lengths).squeeze(1)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            res = np.append(res, rounded_preds.cpu().numpy())
    return res


t = time.time()
loss = []
acc = []
val_acc = []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_iterator)
    valid_acc = evaluate(model, valid_iterator)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Acc: {valid_acc * 100:.2f}%')

    loss.append(train_loss)
    acc.append(train_acc)
    val_acc.append(valid_acc)

print(f'time:{time.time() - t:.3f}')

test_iterator = data.BucketIterator(test_ds, batch_size=BATCH_SIZE, train=False, device=device)
valid_acc = predict(model, test_iterator)

test['target'] = [int(x) for x in valid_acc]
test[['id', 'target']].to_csv("/Users/yanxu/Documents/kaggles/twitter_disaster/data/test_submission.csv", index=False)
