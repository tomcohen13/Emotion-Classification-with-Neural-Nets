"""
COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Tom Cohen | tc2955
Emotion Classification with Neural Networks - Main File

"""

# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
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
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

EXTENSION = False 

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator, model_type):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    ########## YOUR CODE HERE ##########
    # TODO: Given a model, data, and loss function, you should do the following:
    # TODO: 1) Loop through the whole train dataset performing batch optimization with the optimizer of your choice,
    # TODO: updating the model parameters with each batch (we suggest you use torch.optim.Adam to start);
    # TODO: 2) Each time you reach the end of the train dataset (one "epoch"), calculate the loss on the whole dev set;
    # TODO and 3) stop training and return the model once the development loss stops improving (called early stopping).
    # TODO: Make sure to print the dev set loss each epoch to stdout.

    gold = []
    predicted = []
    epochs = 10 if model_type == "RNN" else 25

    # extension-grading
    if model_type == 'extension2':
        lmbda = lambda epoch: 0.999 ** epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    
    lrs = [] # just for fun I keep track of the decreasing values of the learning rate.
    
    for _ in range(1, epochs + 1):
        
        model.train()
        
        lrs.append(optimizer.param_groups[0]["lr"])

        print("epoch: {}/{} ==================".format(_,epochs))
        for xb, yb in train_generator:

            optimizer.zero_grad()

            # make a prediction by passing the batch through the model
            pred = model(xb) 

            #calcuate loss given current parameters
            loss = loss_fn(pred, yb)
            
            # backprop
            loss.backward() 
            
            #update gradients
            optimizer.step()
        
        model.eval()
        dev_loss = torch.zeros(1) 

        with torch.no_grad():
            for xb, yb in dev_generator:

                pred = model(xb)
                
                # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
                gold.extend(yb.cpu().detach().numpy())
                predicted.extend(pred.argmax(1).cpu().detach().numpy())

                dev_loss += loss_fn(pred.double(), yb.long()).data
        
        print("Dev loss: ")
        print(dev_loss)
        print("F-score: ")
        f1 = f1_score(gold, predicted, average='macro')
        print(f1)

        if model_type == 'extension2': 
            scheduler.step()
    
    #extension-grading: uncomment me to get a graph of the diminishing learning rate :)
    if model_type == 'extension2':
        plt.plot(range(epochs), lrs)
        plt.show()

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
    # train, dev, test = utils.get_data(DATA_FN)
    # print(train)
    # Prepare the data and the pretrained embedding matrix
    EXTENSION = False 

    if args.model == 'extension1':
        EXTENSION = True

    if FRESH_START or EXTENSION:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM, extension = EXTENSION)
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
            
    print(embeddings.shape[0])
    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()
    
    ########## YOUR CODE HERE ##########
    # TODO: for each of the two models, you should 1) create it,
    # TODO 2) run train_model() to train it, and
    # TODO: 3) run test_model() on the result

    if args.model == 'dense':
        model = models.DenseNetwork(embeddings = embeddings)
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)    

    elif args.model == 'RNN':
        print("Running RNN")
        model = models.RecurrentNetwork(embeddings, hidden = 32, layers = 2)
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)

    elif args.model == 'extension1':
        print("Running Dense Model with Extension 1: Tweet Tokenizer")
        model = models.DenseNetwork(embeddings = embeddings)
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)  

    else:
        print("Running Dense Model with Extension 2: Learning Rate Scheduler")
        model = models.DenseNetwork(embeddings = embeddings)
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)  

    train_model(model, loss_fn, optimizer, train_generator, dev_generator, model_type = args.model)
    test_model(model, loss_fn, test_generator)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
