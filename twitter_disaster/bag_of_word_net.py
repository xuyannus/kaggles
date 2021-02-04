import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_size, hidden1, hidden2):
        super(BagOfWordsClassifier, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.squeeze(1).float()))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BagOfWordNet:
    def __init__(self, lr=0.001, epochs=30, mini_batch=256, vocab_size=1000, hidden1=128, hidden2=64, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.mini_batch = mini_batch
        self.vocab_size = vocab_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.model = BagOfWordsClassifier(vocab_size, hidden1, hidden2)
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.verbose = verbose

    def fit(self, data_loader, evaluator_loader=None):
        if evaluator_loader:
            epoch_accuracies = []

        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = Variable(data), Variable(target)
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


class BagWordDataSet(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=2)
        self.sequences = self.vectorizer.fit_transform(df.text.tolist())
        self.labels = df.target.tolist()
        self.token2idx = self.vectorizer.vocabulary_
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def __getitem__(self, i):
        return self.sequences[i, :].toarray(), self.labels[i]

    def __len__(self):
        return self.sequences.shape[0]


class BagWordTestDataSet(Dataset):
    def __init__(self, path, vectorizer):
        df = pd.read_csv(path)
        self.vectorizer = vectorizer
        self.sequences = self.vectorizer.transform(df.text.tolist())
        self.token2idx = self.vectorizer.vocabulary_
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def __getitem__(self, i):
        return self.sequences[i, :].toarray()

    def __len__(self):
        return self.sequences.shape[0]