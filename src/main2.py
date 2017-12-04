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
    print 'Loading askubuntu training samples..'
    askubuntu_training_samples = utils.load_samples('../data/askubuntu/train_random.txt')
    print len(askubuntu_training_samples)

    print 'Loading askubuntu dev samples..'
    askubuntu_dev_samples = utils.load_samples('../data/askubuntu/dev.txt')
    print len(askubuntu_dev_samples)

    print 'Loading askubuntu test samples..'
    askubuntu_test_samples = utils.load_samples('../data/askubuntu/test.txt')
    print len(askubuntu_test_samples)

    print 'Loading askubuntu corpus..'
    askubuntu_question_map = utils.load_corpus('../data/askubuntu/text_tokenized.txt')
    print len(askubuntu_question_map)

    print 'Loading android dev samples..'
    android_dev_samples = utils.load_samples_stupid_format(
        '../data/android/dev.pos.txt', '../data/android/dev.neg.txt')
    print len(android_dev_samples)

    print 'Loading android test samples..'
    android_test_samples = utils.load_samples_stupid_format(
        '../data/android/test.pos.txt', '../data/android/test.neg.txt')
    print len(android_test_samples)

    print 'Loading android corpus..'
    android_question_map = utils.load_corpus('../data/android/corpus.tsv')
    print len(android_question_map)

    print 'Loading embeddings..'
    embedding_map = utils.load_embeddings('../data/pruned_askubuntu_android_vector.txt')
    print len(embedding_map)
    print

    utils.store_embedding_map(embedding_map)

    return (askubuntu_training_samples, askubuntu_dev_samples, askubuntu_test_samples,
            askubuntu_question_map, android_dev_samples, android_test_samples,
            android_question_map, embedding_map)


#################################################
# Plot configuration
fig = plt.figure()

losses = []


def display_callback(loss):
    losses.append(loss)
    if len(losses) % 100 == 0:
        fig.clear()
        plt.plot(list(range(len(losses))), losses)
        plt.pause(0.0001)

#################################################
# LSTM configuration

lstm_input_size = 200
lstm_hidden_size = 300
lstm_num_layers = 1

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

embedding_size = 200
filter_size = 5
hidden_size = 300
sequence_state_size = 250

cnn_learning_rate = 1e-1

cnn = cnn.LanguageCNN(embedding_size, filter_size, hidden_size, sequence_state_size)

print cnn
print

#################################################
# Data loading

(askubuntu_training_samples, askubuntu_dev_samples, askubuntu_test_samples,
 askubuntu_question_map, android_dev_samples, android_test_samples,
 android_question_map, embedding_map) = init()

#################################################
# MAIN                                          #
#################################################


def midpoint_eval(i):
    if (i + 1) % 200 == 0:
        evaluate.evaluate_model(cnn, encode.encode_cnn, askubuntu_dev_samples, askubuntu_question_map)

# NOTE: Trains CNN
train.train_batch(cnn, encode.encode_cnn, askubuntu_training_samples[:100],
                  cnn_learning_rate, askubuntu_question_map, display_callback, midpoint_eval)


#print
#print 'EVALUATION'
#print
#print 'Askubuntu dev'
#evaluate.evaluate_model(cnn, encode.encode_cnn, askubuntu_dev_samples, askubuntu_question_map)
#print 'Askubuntu test'
#evaluate.evaluate_model(cnn, encode.encode_cnn, askubuntu_test_samples, askubuntu_question_map)
#print 'Android dev'
#evaluate.evaluate_model(cnn, encode.encode_cnn, android_dev_samples, android_question_map)
#print 'Android dev'
#evaluate.evaluate_model(cnn, encode.encode_cnn, android_test_samples, android_question_map)
