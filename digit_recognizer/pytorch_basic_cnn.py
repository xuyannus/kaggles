import os
import random
import pandas as pd
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomAffine
from digit_recognizer.common import NeuralNetworkWrapper, MNIST_data, IMG_PIXELS, IMG_HEIGHT, IMG_WIDTH


def seed_init_fn(worker_id):
    seed = 31 + + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def load_train_img(batch_size=64):
    train_img_dir = os.path.dirname(__file__) + "/data/train.csv"
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          RandomAffine(degrees=20, translate=(0.1, 0.2), scale=(.9, 1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5,), std=(0.5,))])
    train_dataset = MNIST_data(train_img_dir, IMG_PIXELS, IMG_HEIGHT, IMG_WIDTH, train_transform)
    return torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_init_fn)


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


def build_cnn(params):
    train_loader = load_train_img(batch_size=params['mini_batch'])
    test_loader = load_test_img(batch_size=params['mini_batch'])

    # show_tranformed_figures(train_loader)
    # show_tranformed_figures(test_loader, training=False)

    model_wrapper = NeuralNetworkWrapper(**params)
    model_wrapper.fit(train_loader)
    print("training_accuracy:", model_wrapper.validate_accuracy(train_loader))

    out_df = pd.read_csv(os.path.dirname(__file__) + "/data/sample_submission.csv")
    out_df.loc[:, 'Label'] = model_wrapper.predict(test_loader)
    out_df.to_csv(os.path.dirname(__file__) + '/submission_v100.csv', index=False)


if __name__ == "__main__":
    build_cnn(params={'dropout': 0, 'epochs': 100, 'gamma': 0.5, 'lr': 0.003, 'mini_batch': 128, 'batch_norm': True})