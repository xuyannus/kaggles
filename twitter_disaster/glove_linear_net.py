import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np


EMBEDDING_DIMENSION = 200
COMMON_STOP_WORDS = set(['the', 'a', 'in', 'to', 'of', 'i', 'and', 'is', 'you', 'for', 'on', 'it', 'my', 'that',
                         'with', 'are', 'at', 'by', 'this', 'have', 'from', 'be', 'was', 'do', 'will', 'as', 'up',
                         'me', 'am', 'so', 'we', 'your', 'has', 'when', 'an', 's', 'they', 'about', 'been', 'there',
                         'who', 'would', 'into', 'his', 'them', 'did', 'w', 'their', 'm', 'its', 'does', 'where', 'th',
                         'b', 'd', 'x', 'p', 'o', 'r', 'c', 'n', 'e', 'g', 'v', 'k', 'l', 'f', 'j', 'z', 'us', 'our',
                         'all', 'can', 'may'])


class GloveClassifier(nn.Module):
    def __init__(self, embedding_dimension=1, hidden1=128, hidden2=64, device='cpu',):
        super(GloveClassifier, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.device = device

        # in this model, we will import embedding vector directly
        self.fc1 = nn.Linear(embedding_dimension, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GloveNet:
    def __init__(self, lr=0.001, epochs=1, embedding_dimension=1, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.model = GloveClassifier(embedding_dimension=embedding_dimension)
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.verbose = verbose
        if self.verbose:
            print(self.model)

    def fit(self, data_loader, evaluator_loader=None):
        if evaluator_loader:
            epoch_accuracies = []

        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(data_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output.squeeze(), target.float())
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                if self.verbose:
                    if (batch_idx + 1) % 10 == 0:
                        print({"Epoch": epoch,
                               "Data Scanned": (batch_idx + 1) * len(data),
                               "Total": len(data_loader.dataset),
                               "Local Loss": loss.data.item()
                               })

            if evaluator_loader:
                evaluation_accuracy = self.validate_accuracy(evaluator_loader)
                print({"epoch": epoch, "evaluation_accuracy": evaluation_accuracy})
                epoch_accuracies.append({"epoch": epoch, "evaluation_accuracy": evaluation_accuracy})

        if evaluator_loader:
            return epoch_accuracies

    def validate_accuracy(self, data_loader):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = Variable(data), Variable(target)
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                output = self.model(data) > 0
                output = output.squeeze().type_as(target)
                correct += output.eq(target).cpu().sum()

        if self.verbose:
            print({
                "Total": len(data_loader.dataset),
                "Correct": correct,
                "Accuracy": 1.00 * correct / len(data_loader.dataset)
            })
        return 1.00 * correct / len(data_loader.dataset)

    def predict(self, data_loader):
        self.model.eval()
        test_pred = torch.LongTensor()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                data = Variable(data)
                if torch.cuda.is_available():
                    data = data.cuda()
                output = self.model(data) > 0
                output = output.squeeze()
                test_pred = torch.cat((test_pred, output), dim=0)
            return test_pred.numpy()


def load_glove_embedding(dimension, verbose=False):
    path_to_glove_file = os.path.dirname(__file__) + "/glove/glove.twitter.27B.{}d.txt".format(dimension)
    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    if '<PAD>' not in embeddings_index:
        embeddings_index['<PAD>'] = np.zeros(dimension)
    if verbose:
        print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index


def text_embed(words, dimension, glove_embedding):
    words_list = [w for w in words.split() if w not in COMMON_STOP_WORDS]
    unknown_indices = []
    mean = np.zeros(dimension)
    for i in range(len(words_list)):
        if words_list[i] in glove_embedding:
            mean += glove_embedding[words_list[i]]
        else:
            unknown_indices.append(i)
    if len(words_list) - len(unknown_indices) == 0:
        return None
    else:
        return mean / (len(words_list) - len(unknown_indices))


class GloveDataSet(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.glove_embedding = load_glove_embedding(dimension=EMBEDDING_DIMENSION, verbose=True)
        df.loc[:, "text"] = df["text"].apply(lambda x: text_embed(x, EMBEDDING_DIMENSION, self.glove_embedding))
        df.dropna(inplace=True)
        self.sequences = df.text.tolist()
        self.labels = df.target.tolist()

    def __getitem__(self, i):
        return self.sequences[i], self.labels[i]

    def __len__(self):
        return len(self.sequences)


class GloveTestDataSet(Dataset):
    def __init__(self, path, glove_embedding):
        self.glove_embedding = glove_embedding
        df = pd.read_csv(path)

        df.loc[:, "text"] = df["text"].apply(lambda x: text_embed(x, EMBEDDING_DIMENSION, self.glove_embedding))
        df.dropna(inplace=True)
        self.sequences = df.text.tolist()

    def __getitem__(self, i):
        return self.sequences[i]

    def __len__(self):
        return len(self.sequences)