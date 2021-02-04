import pandas as pd
import torch
from torch.utils.data import DataLoader
from twitter_disaster.bag_of_word_net import BagOfWordNet, BagWordDataSet, BagWordTestDataSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATA_PATH = "/Users/yanxu/Documents/kaggles/twitter_disaster/data/train_v2.csv"
TEST_DATA_PATH = "/Users/yanxu/Documents/kaggles/twitter_disaster/data/test_v2.csv"
SUB_DATA_PATH = "/Users/yanxu/Documents/kaggles/twitter_disaster/data/sample_submission.csv"
PRED_DATA_PATH = "/Users/yanxu/Documents/kaggles/twitter_disaster/data/sample_submission_2.csv"

dataset = BagWordDataSet(TRAIN_DATA_PATH)
train_size = int(0.8 * len(dataset))
evaluator_size = len(dataset) - train_size
train_dataset, evaluator_dataset = torch.utils.data.random_split(dataset, [train_size, evaluator_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=64)
evaluator_loader = DataLoader(evaluator_dataset, batch_size=64)

model = BagOfWordNet(epochs=1, vocab_size=len(dataset.token2idx), hidden1=128, hidden2=64, verbose=True)
model.fit(train_loader, evaluator_loader)

test_dataset = BagWordTestDataSet(TEST_DATA_PATH, dataset.vectorizer)
test_loader = DataLoader(test_dataset, batch_size=64)
pred = model.predict(test_loader)

sub = pd.read_csv(SUB_DATA_PATH)
sub['target'] = pred
sub.to_csv(PRED_DATA_PATH, index=False)

