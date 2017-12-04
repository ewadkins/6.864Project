import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt

import utils
import train
import encode
import cnn
import evaluate

#################################################
# Data loader


def init():
    print 'Loading training samples..'
    training_samples = utils.load_samples('../data/askubuntu/train_random.txt')
    print len(training_samples)

    print 'Loading dev samples..'
    dev_samples = utils.load_samples('../data/askubuntu/dev.txt')
    print len(dev_samples)

    print 'Loading test samples..'
    test_samples = utils.load_samples('../data/askubuntu/test.txt')
    print len(test_samples)

    print 'Loading corpus..'
    question_map = utils.load_corpus('../data/askubuntu/text_tokenized.txt')
    print len(question_map)

    print 'Loading embeddings..'
    embedding_map = utils.load_embeddings(
        '../data/pruned_askubuntu_android_vector.txt')
    print len(embedding_map)
    print

    utils.store_embedding_map(embedding_map)

    return (training_samples,
            dev_samples, test_samples, question_map, embedding_map)


#################################################
# Plot configuration
fig = plt.figure()

losses = []


def display_callback(loss):
    losses.append(loss)
    fig.clear()
    plt.plot(list(range(len(losses))), losses)
    plt.pause(0.0001)

#################################################
# LSTM configuration

lstm_input_size = 200
lstm_hidden_size = 300
lstm_num_layers = 2

lstm_learning_rate = 1e-1

lstm = nn.LSTM(
    lstm_input_size,
    lstm_hidden_size,
    lstm_num_layers,
    batch_first=True)

# Dont remove; values needed to encode
encode.store_config(lstm_hidden_size, lstm_num_layers)

print lstm
print

#################################################
# CNN configuration

cnn = cnn.CNN()

print cnn
print

#################################################
# Data loading

training_samples, dev_samples, test_samples, question_map, embedding_map =\
    init()

#################################################
# MAIN                                          #
#################################################

def midpoint_eval(i):
    if (i + 1) % 100 == 0:
        evaluate.evaluate_model(cnn, encode.encode_cnn, dev_samples, question_map)


# NOTE: Trains CNN
epoch = 0
while True:
    epoch += 1
    print 'Epoch', epoch
    train.train_batch(cnn, encode.encode_cnn, training_samples,
                      cnn_learning_rate, question_map, display_callback, midpoint_eval)

#def midpoint_eval(i):
#    if (i + 1) % 50 == 0:
#        evaluate.evaluate_model(lstm, encode.encode_lstm, dev_samples, question_map)
#
## NOTE: Trains LSTM
#epoch = 0
#while True:
#    epoch += 1
#    print 'Epoch', epoch
#    train.train_batch(lstm, encode.encode_lstm, training_samples,
#                      lstm_learning_rate, question_map, display_callback, midpoint_eval)
