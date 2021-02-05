import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from twitter_disaster.glove_gru_net import GloveDataSet, GloveNet, GloveTestDataSet, EMBEDDING_DIMENSION

TRAIN_DATA_PATH = os.path.dirname(__file__) + "data/train_v2.csv"
TEST_DATA_PATH = os.path.dirname(__file__) + "data/test_v2.csv"
SUB_DATA_PATH = os.path.dirname(__file__) + "data/sample_submission.csv"
PRED_DATA_PATH = os.path.dirname(__file__) + "data/sample_submission_glove_gru_v3.csv"

dataset = GloveDataSet(TRAIN_DATA_PATH, max_seq_len=30)
train_size = int(0.8 * len(dataset))
evaluator_size = len(dataset) - train_size
train_dataset, evaluator_dataset = torch.utils.data.random_split(dataset, [train_size, evaluator_size], generator=torch.Generator().manual_seed(42))


def collate(batch):
    inputs = torch.FloatTensor([item[0] for item in batch])
    target = torch.FloatTensor([item[1] for item in batch])
    return inputs, target


def collate_test(batch):
    inputs = torch.FloatTensor([item for item in batch])
    return inputs


train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate, shuffle=True)
evaluator_loader = DataLoader(evaluator_dataset, batch_size=64, collate_fn=collate, shuffle=True)

model = GloveNet(epochs=10, embedding_dimension=EMBEDDING_DIMENSION, batch_size=64, verbose=True)
model.fit(train_loader, evaluator_loader)

test_dataset = GloveTestDataSet(TEST_DATA_PATH, max_seq_len=30, glove_embedding=dataset.glove_embedding)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_test)
pred = model.predict(test_loader)

sub = pd.read_csv(SUB_DATA_PATH)
sub['target'] = pred
sub.to_csv(PRED_DATA_PATH, index=False)


