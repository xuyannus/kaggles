import os
import pandas as pd
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomAffine
from digit_recognizer.common import NeuralNetworkWrapper, MNIST_data, IMG_PIXELS, IMG_HEIGHT, IMG_WIDTH


def random_split_train_test(csv_path, train_rate=0.75):
    temp_df = pd.read_csv(csv_path)
    mask = np.random.rand(len(temp_df)) < train_rate
    temp_df[mask].to_csv(os.path.dirname(__file__) + "/data/train_p1.csv", index=False)
    temp_df[~mask].to_csv(os.path.dirname(__file__) + "/data/train_p2.csv", index=False)


def load_train_and_evaluation_img(batch_size=64):
    random_split_train_test(os.path.dirname(__file__) + "/data/train.csv", train_rate=0.75)
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          RandomAffine(degrees=20, translate=(0.1, 0.2), scale=(.9, 1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5,), std=(0.5,))])
    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5,), std=(0.5,))])

    train_dataset = MNIST_data(os.path.dirname(__file__) + "/data/train_p1.csv", IMG_PIXELS, IMG_HEIGHT, IMG_WIDTH,
                               train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MNIST_data(os.path.dirname(__file__) + "/data/train_p2.csv", IMG_PIXELS, IMG_HEIGHT, IMG_WIDTH,
                              test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def epoch_tuning():
    params = {'dropout': 0, 'epochs': 500, 'gamma': 0.5, 'lr': 0.003, 'mini_batch': 128}
    model_wrapper = NeuralNetworkWrapper(**params)
    train_loader, test_loader = load_train_and_evaluation_img()
    epoch_accuracies = model_wrapper.fit(train_loader, evaluator_loader=test_loader)
    epoch_accuracies_df = pd.DataFrame(epoch_accuracies)
    epoch_accuracies_df.to_csv(os.path.dirname(__file__) + '/data/epoch_accuracies.csv', index=False)


if __name__ == "__main__":
    epoch_tuning()
