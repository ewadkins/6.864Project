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
import rcnn
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
    # IMPORTANT: Make sure to store the correct question map before using utils
    utils.store_question_map(askubuntu_question_map)

    return (askubuntu_training_samples, askubuntu_dev_samples, askubuntu_test_samples,
            askubuntu_question_map, android_dev_samples, android_test_samples,
            android_question_map, embedding_map)


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
# RCNN configuration


rcnn_input_size = 200  # size of word embedding
# sizes of the convolutional layers; determines # of conv layers
rcnn_hidden_sizes = [300, 200, 150]
rcnn_output_size = 100  # size of state vector

# NOTE: assert len(rcnn_kernel_sizes) == len(rcnn_hidden_sizes)
rcnn_kernel_sizes = [5, 4, 3]
# NOTE: assert len(rcnn_pooling_sizes) == len(rcnn_hidden_sizes)
rcnn_pooling_sizes = [2, 2, 2]

rcnn_learning_rate = 1e-1

rcnn = rcnn.RCNN(rcnn_input_size, rcnn_hidden_sizes, rcnn_output_size,
                 rcnn_kernel_sizes, rcnn_pooling_sizes,
                 padding=max(rcnn_kernel_sizes))

print rcnn
print

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
# Data loading

(askubuntu_training_samples, askubuntu_dev_samples, askubuntu_test_samples,
 askubuntu_question_map, android_dev_samples, android_test_samples,
 android_question_map, embedding_map) = init()

#################################################
# MAIN                                          #
#################################################

# title, body = android_question_map[android_dev_samples[0].id]
# print title
# embeddings = utils.get_embeddings(title)
# print np.shape(embeddings)
# encoded = encode.encode_rcnn(rcnn, embeddings)
# print encoded

# NOTE: Trains RCNN without batching
# train.train(
#    rcnn,
#    encode.encode_rcnn,
#    askubuntu_training_samples,
#    rcnn_learning_rate,
#    display_callback)


# title, body = question_map[askubuntu_training_samples[0].id]
# print title
# embeddings = utils.get_embeddings(title)
# encoded = encode.encode_lstm(lstm, embeddings)
# print encoded

# NOTE: Trains LSTM without batching
# train.train(lstm, encode.encode_lstm, askubuntu_training_samples, lstm_learning_rate,
#           display_callback)


# batch_ids = [askubuntu_training_samples[0].id] + list(sample.candidate_map.keys())
# embeddings_batch = map(lambda id:
#                       utils.get_embeddings(question_map[askubuntu_training_samples[0].id][0]),
#                       batch_ids)
# print np.shape(embeddings_batch)
# encoded_batch = encode.encode_lstm_batch(lstm, embeddings_batch)
# print np.shape(encoded_batch)

# NOTE: Trains LSTM with batching
# train.train_batch(lstm, encode.encode_lstm_batch, askubuntu_training_samples,
#                  lstm_learning_rate, display_callback)


# EVALUATION EXAMPLE
#evaluate.evaluate_model(lstm, encode.encode_lstm, askubuntu_dev_samples)
