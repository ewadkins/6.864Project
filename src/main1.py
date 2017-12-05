import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt

import utils
import real_train
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
    if len(losses) % 20 == 0:
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

cnn_learning_rate = 1e-5

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


model = cnn
encode_fn = encode.encode_cnn
learning_rate = cnn_learning_rate

# Trains models
def midpoint_eval(i):
    if (i + 1) % 1000 == 0:
        evaluate.evaluate_model(model, encode_fn, dev_samples, question_map)    
real_train.train_batch(model, encode_fn, training_samples,
                  learning_rate, question_map, display_callback, midpoint_eval)

print
print 'EVALUATION'
print
print 'Askubuntu dev'
evaluate.evaluate_model(model, encode_fn, dev_samples, question_map)
print 'Askubuntu test'
evaluate.evaluate_model(model, encode_fn, test_samples, question_map)
