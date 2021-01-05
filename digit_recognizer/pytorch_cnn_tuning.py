import os
import pandas as pd
import numpy as np
import logging

import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.model_selection import ParameterGrid, ParameterSampler

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

    train_dataset = MNIST_data(os.path.dirname(__file__) + "/data/train_p1.csv", IMG_PIXELS, IMG_HEIGHT, IMG_WIDTH, train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MNIST_data(os.path.dirname(__file__) + "/data/train_p2.csv", IMG_PIXELS, IMG_HEIGHT, IMG_WIDTH, test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def model_evaluation(params):
    model_wrapper = NeuralNetworkWrapper(**params)
    train_loader, test_loader = load_train_and_evaluation_img()
    model_wrapper.fit(train_loader)
    return model_wrapper.validate_accuracy(test_loader)


def cnn_tuning():
    # param_grid = {
    #     'lr': [0.001, 0.003, 0.005, 0.01],
    #     'dropout': [0, 0.3, 0.5],
    #     'mini_batch': [64, 128],
    #     'gamma': [0.1, 0.5, 1.0],
    #     'epochs': [20, 50]
    # }
    param_grid = {
        'lr': [0.003],
        'dropout': [0],
        'mini_batch': [128],
        'gamma': [0.5],
        'epochs': [50]
    }

    k = 3
    optimal_param = None
    optimal_accuracy = 0

    for params in ParameterGrid(param_grid):
    # for params in ParameterSampler(param_grid, n_iter=20):
        accuracy_list = []
        for index in range(k):
            print({
                "params": params,
                "index": index
            })
            accuracy_list.append(model_evaluation(params))

        logging.info({
            "params": params,
            "accuracy": np.mean(accuracy_list),
            "accuracy_list": np.np.array2string(accuracy_list, precision=6, separator=','),
        })

        if np.mean(accuracy_list) > optimal_accuracy:
            optimal_accuracy = np.mean(accuracy_list)
            optimal_param = params

    logging.info({
        "OptimalParams": optimal_param,
        "OptimalAccuracy": optimal_accuracy
    })


# optimal: {'dropout': 0, 'epochs': 50, 'gamma': 0.5, 'lr': 0.003, 'mini_batch': 128}
# Validation Accuracy: 'accuracy_list': [tensor(0.9949), tensor(0.9958), tensor(0.9942)], 'accuracy': 0.9949905
if __name__ == "__main__":
    logging.basicConfig(filename=os.path.dirname(__file__) + '/data/digit.log', level=logging.INFO)
    cnn_tuning()
