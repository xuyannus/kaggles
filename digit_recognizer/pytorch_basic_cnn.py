import math
import os

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomAffine

IMG_PIXELS = 784
IMG_HEIGHT = 28
IMG_WIDTH = 28

MODEL_PATH = os.path.dirname(__file__) + "/cnn_digit_v1.pth"


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


def load_train_img(batch_size=64):
    train_img_dir = os.path.dirname(__file__) + "/data/train.csv"
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          RandomAffine(degrees=20, translate=(0.1, 0.2), scale=(.9, 1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5,), std=(0.5,))])
    train_dataset = MNIST_data(train_img_dir, IMG_PIXELS, IMG_HEIGHT, IMG_WIDTH, train_transform)
    return torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


def load_test_img(batch_size=64):
    test_img_dir = os.path.dirname(__file__) + "/data/test.csv"
    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5,), std=(0.5,))])
    test_dataset = MNIST_data(test_img_dir, IMG_PIXELS, IMG_HEIGHT, IMG_WIDTH, test_transform)
    return torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def show_tranformed_figures(data_loader, training=True):
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.figure(figsize=(12, 9))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(data_loader)
    if training:
        images, _ = dataiter.next()
    else:
        images = dataiter.next()
    imshow(torchvision.utils.make_grid(images[:16]))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
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


def train(epoch, model, exp_lr_scheduler, train_loader, optimizer, criterion):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.data.item()))
    exp_lr_scheduler.step()


def evaluate(model, data_loader):
    model.eval()
    loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = Variable(data), Variable(target)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(data_loader.dataset)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))


def predict(model, data_loader):
    model.eval()
    test_pred = torch.LongTensor()
    for i, data in enumerate(data_loader):
        data = Variable(data, volatile=True)
        if torch.cuda.is_available():
            data = data.cuda()
        output = model(data)

        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
    return test_pred.numpy()


def build_cnn(load_model=False):
    train_loader = load_train_img()
    test_loader = load_test_img()

    # show_tranformed_figures(train_loader)
    # show_tranformed_figures(test_loader, training=False)

    if load_model:
        model = Net()
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model = Net()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            print("there is no GPU available")

        n_epochs = 1
        for epoch in range(n_epochs):
            train(epoch, model, exp_lr_scheduler, train_loader, optimizer, criterion)
            evaluate(model, train_loader)

        torch.save(model.state_dict(), MODEL_PATH)

    out_df = pd.read_csv(os.path.dirname(__file__) + "/data/sample_submission.csv")
    out_df.loc[:, 'Label'] = predict(model, test_loader)
    out_df.to_csv(os.path.dirname(__file__) + '/digit_recognizer/submission_v2.csv', index=False)


if __name__ == "__main__":
    build_cnn()