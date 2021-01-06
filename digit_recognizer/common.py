import math
import os
import sys

import pandas as pd
import numpy as np

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

IMG_PIXELS = 784
IMG_HEIGHT = 28
IMG_WIDTH = 28


class MNIST_data(Dataset):
    def __init__(self, file_path, n_pixels, height, width, transforms):
        df = pd.read_csv(file_path)

        if len(df.columns) == n_pixels:
            # test data
            self.X = df.values.reshape((-1, height, width)).astype(np.uint8)[:, :, :, None]
            self.y = None
        else:
            # training data
            self.X = df.iloc[:, 1:].values.reshape((-1, height, width)).astype(np.uint8)[:, :, :, None]
            self.y = torch.from_numpy(df.iloc[:, 0].values)
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transforms(self.X[idx]), self.y[idx]
        else:
            return self.transforms(self.X[idx])


class Net(nn.Module):
    def __init__(self, dropout=0.5, batch_norm=True):
        super(Net, self).__init__()

        if batch_norm:
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(64 * 7 * 7, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, 10),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, 10),
                nn.Softmax(10)
            )

        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class NeuralNetworkWrapper:
    def __init__(self, dropout=0.5, lr=0.003, epochs=20, mini_batch=64, step_size=7, gamma=0.1, batch_norm=True, verbose=True):
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.epochs = epochs
        self.mini_batch = mini_batch
        self.model = Net(dropout=dropout, batch_norm=batch_norm)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
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
                loss = self.criterion(output, target)
                loss.backward()

                if self.verbose:
                    if (batch_idx + 1) % 100 == 0:
                        print({"Epoch": epoch,
                               "Data Scanned": (batch_idx + 1) * len(data),
                               "Total": len(data_loader.dataset),
                               "Local Loss": loss.data.item()
                               })
                self.optimizer.step()
            self.exp_lr_scheduler.step()

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
                output = self.model(data)
                pred = output.cpu().data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
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
                output = self.model(data)
                pred = output.cpu().data.max(1, keepdim=True)[1]
                test_pred = torch.cat((test_pred, pred), dim=0)
            return test_pred.numpy()