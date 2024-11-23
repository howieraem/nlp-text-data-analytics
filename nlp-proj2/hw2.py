"""Author: J Lin"""
import os
# Setting the following environmental variable is required
# if forcing pytorch to use deterministic algorithms
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
SEED = 1

import random
random.seed(SEED)

# Imports
import nltk
import numpy as np
np.random.seed(SEED)

import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
torch.use_deterministic_algorithms(True)
torch.manual_seed(SEED)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models
import argparse

# Global definitions - data
DATA_FN = 'data/crowdflower_data.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 32
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    device = torch.device('cuda:0') if USE_CUDA else torch.device('cpu')
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    pre_dev_loss = float('inf')
    epoch = 1
    while True:
        model.train()
        for batch_x, batch_y in train_generator:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        cur_dev_loss = 0.
        for batch_x, batch_y in dev_generator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()
            with torch.no_grad():
                out = model(batch_x)
            cur_dev_loss += loss_fn(out, batch_y).item()
        print("Epoch %d Dev Loss:" % epoch, round(cur_dev_loss, 5))
        if cur_dev_loss > pre_dev_loss:
            break
        pre_dev_loss = cur_dev_loss
        epoch += 1
    return model.cpu()


class WarmupCosineAnnealLR(optim.lr_scheduler._LRScheduler):
    """
    Implements the cosine annealing learning rate scheduler with warmup strategy.

    References:
        [1] I. Loshchilov en F. Hutter, “SGDR: Stochastic gradient descent with warm restarts”, arXiv preprint arXiv:1608. 03983, 2016.
        [2] P. Goyal et al., “Accurate, large minibatch SGD: Training imagenet in 1 hour”, arXiv preprint arXiv:1706. 02677, 2017.

    extension-grading
    """
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_epoch: int,
                 max_epoch: int,
                 min_lr: float):
        """
        Constructor of the cosine annealing scheduler with warmup.

        Args:
            optimizer: PyTorch optimizer object, SGD, Adam, etc.
            warmup_epoch: how many epochs for warmup
            max_epoch: the epoch where lr is reduced to the minimum
            min_lr: minimum learning rate
        """
        self.warmup_epoch = max(0, warmup_epoch)
        self.max_epoch = max(max_epoch, warmup_epoch + 1)
        self.min_lr = min_lr
        super().__init__(optimizer, -1)

    def calc_decayed_lr(self, base_lr):
        # See equation (5) on page 4 of the paper [1]
        return (self.min_lr +
                0.5 * (base_lr - self.min_lr) *
                (1 + np.cos(self.last_epoch / self.max_epoch * np.pi)))

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            # After warm-up, adjust learning rate in a cosine manner
            return [self.calc_decayed_lr(base_lr) for base_lr in self.base_lrs]
        # At the warm-up stage, gradually increase the learning rate, see paper [2]
        factor = (self.last_epoch + 1) / self.warmup_epoch
        return [base_lr * factor for base_lr in self.base_lrs]


def train_model_extension(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    This version is for model extension2, with a learning rate scheduler, and 
    defines the maximum number of epochs to tolerate if there is no improvement 
    in loss.

    extension-grading

    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    max_tol_epochs = 3
    epochs_no_improve = 0
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-8)
    scheduler = WarmupCosineAnnealLR(optimizer, 3, 10, 1e-8)

    device = torch.device('cuda:0') if USE_CUDA else torch.device('cpu')
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    pre_dev_loss = float('inf')
    epoch = 1
    while True:
        model.train()
        for batch_x, batch_y in train_generator:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        cur_dev_loss = 0.
        for batch_x, batch_y in dev_generator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()
            with torch.no_grad():
                out = model(batch_x)
            cur_dev_loss += loss_fn(out, batch_y).item()
        print("Epoch %d Dev Loss:" % epoch, round(cur_dev_loss, 5))
        epochs_no_improve += (cur_dev_loss > pre_dev_loss)
        if epochs_no_improve >= max_tol_epochs:
            break
        pre_dev_loss = cur_dev_loss
        epoch += 1
    return model.cpu()


def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    train_func = train_model
    if args.model == 'dense':
        model = models.DenseNetwork(embeddings, EMBEDDING_DIM, NUM_CLASSES)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    elif args.model == 'RNN':
        model = models.RecurrentNetwork(embeddings, EMBEDDING_DIM, NUM_CLASSES)
        optimizer = optim.AdamW(model.parameters(), lr=0.021)
    elif args.model == 'extension1':
        # extension-grading
        # A 2-layer 1D-convolutional model, with 1D batch normalizations.
        model = models.ExperimentalNetwork(embeddings, EMBEDDING_DIM, NUM_CLASSES)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    elif args.model == 'extension2':
        # extension-grading
        # Model is the same as 'dense', but with:
        # 1. a different learning rate
        # 2. a learning rate scheduler
        # 3. the maximum number of epochs without improvement to tolerate
        model = models.DenseNetwork(embeddings, EMBEDDING_DIM, NUM_CLASSES)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        train_func = train_model_extension
    else:
        raise ValueError("Unsupported argument for model type.")

    model = train_func(model, loss_fn, optimizer, train_generator, dev_generator)
    test_model(model, loss_fn, test_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
