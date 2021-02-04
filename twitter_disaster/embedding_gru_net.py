import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from torch.autograd import Variable
from torch.utils.data import Dataset


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, batch_size, embedding_dimension=100, hidden_size=128, n_layers=2, l1_size=128, l2_size=64, device='cpu',):
        super(GRUClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        # in this model, we will train the embedding layer
        self.embed = nn.Embedding(vocab_size, embedding_dimension)
        self.rnn = nn.GRU(embedding_dimension, hidden_size, num_layers=n_layers, batch_first=True,)
        self.lc1 = nn.Linear(hidden_size, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.lc3 = nn.Linear(l2_size, 1)

    def init_hidden(self):
        return torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device)

    def forward(self, inputs):
        # Avoid breaking if the last batch has a different size
        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        encoded = self.embed(inputs)
        output, hidden_status = self.rnn(encoded, self.init_hidden())
        output = output[:, -1, :].squeeze()

        # only use the last dimension
        output = F.relu(self.lc1(output))
        output = F.relu(self.fc2(output))
        return self.lc3(output)


class GRUNet:
    def __init__(self, lr=0.1, epochs=30, vocab_size=1000, batch_size=100, hidden_size=128, n_layers=2, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.model = GRUClassifier(vocab_size=vocab_size, batch_size=batch_size, hidden_size=hidden_size, n_layers=n_layers)
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


class GRUDataSet(Dataset):
    def __init__(self, path, max_seq_len):
        self.max_seq_len = max_seq_len
        df = pd.read_csv(path)

        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=2)
        self.vectorizer.fit(df.text.tolist())

        self.token2idx = self.vectorizer.vocabulary_
        self.token2idx['<PAD>'] = max(self.token2idx.values()) + 1

        self.tokenizer = self.vectorizer.build_analyzer()
        self.text_encoding_fun = lambda x: [self.token2idx[token] for token in self.tokenizer(x) if token in self.token2idx]
        self.padding_fun = lambda x: x + (max_seq_len - len(x)) * [self.token2idx['<PAD>']]

        sequences = [self.text_encoding_fun(x)[:max_seq_len] for x in df.text.tolist()]
        self.sequences = [self.padding_fun(sequence) for sequence in sequences]
        self.labels = df.target.tolist()

    def __getitem__(self, i):
        return self.sequences[i], self.labels[i]

    def __len__(self):
        return len(self.sequences)


class GRUTestDataSet(Dataset):
    def __init__(self, path, max_seq_len, vectorizer, tokenizer):
        self.max_seq_len = max_seq_len
        df = pd.read_csv(path)
        self.vectorizer = vectorizer

        self.token2idx = self.vectorizer.vocabulary_
        self.text_encoding_fun = lambda x: [self.token2idx[token] for token in tokenizer(x) if token in self.token2idx]
        self.padding_fun = lambda x: x + (max_seq_len - len(x)) * [self.token2idx['<PAD>']]

        sequences = [self.text_encoding_fun(x)[:max_seq_len] for x in df.text.tolist()]
        self.sequences = [self.padding_fun(sequence) for sequence in sequences]

    def __getitem__(self, i):
        return self.sequences[i]

    def __len__(self):
        return len(self.sequences)