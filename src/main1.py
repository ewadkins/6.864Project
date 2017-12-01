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

#################################################
# Data loader

def init():
    print 'Loading training samples..'
    training_samples = utils.load_samples('../data/train_random.txt')
    print len(training_samples)

    print 'Loading dev samples..'
    dev_samples = utils.load_samples('../data/dev.txt')
    print len(dev_samples)

    print 'Loading test samples..'
    test_samples = utils.load_samples('../data/test.txt')
    print len(test_samples)

    print 'Loading corpus..'
    question_map = utils.load_corpus('../data/text_tokenized.txt')
    print len(question_map)

    print 'Loading embeddings..'
    embedding_map = utils.load_embeddings('../data/vectors_pruned.200.txt')
    print len(embedding_map)
    print
    
    utils.store_embedding_map(embedding_map)
    utils.store_question_map(question_map)
    
    return training_samples, dev_samples, test_samples, question_map, embedding_map

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

rcnn_input_size = 200 # size of word embedding
rcnn_hidden_sizes = [300, 200, 150] # sizes of the convolutional layers; determines # of conv layers
rcnn_output_size = 100 # size of state vector

rcnn_kernel_sizes = [5, 4, 3] # NOTE: assert len(rcnn_kernel_sizes) == len(rcnn_hidden_sizes)
rcnn_pooling_sizes = [2, 2, 2] # NOTE: assert len(rcnn_pooling_sizes) == len(rcnn_hidden_sizes)

rcnn_learning_rate = 1e-1

rcnn = rcnn.RCNN(rcnn_input_size, rcnn_hidden_sizes, rcnn_output_size,
                 rcnn_kernel_sizes, rcnn_pooling_sizes,
                 padding=max(rcnn_kernel_sizes));

print rcnn
print 

#################################################
# LSTM configuration

lstm_input_size = 200
lstm_hidden_size = 300
lstm_num_layers = 1

lstm_learning_rate = 1e-1

lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)

# Dont remove; values needed to encode
encode.store_config(lstm_hidden_size, lstm_num_layers)

print lstm
print 

#################################################
# Data loading

training_samples, dev_samples, test_samples, question_map, embedding_map = init();

#################################################
# MAIN                                          #
#################################################

#title, body = question_map[training_samples[0].id]
#print title
#embeddings = utils.get_embeddings(title, embedding_map)
#encoded = encode.encode_rcnn(rcnn, embeddings)
#print encoded

# NOTE: Trains RCNN without batching
#train.train(rcnn, encode.encode_rcnn, training_samples, rcnn_learning_rate, display_callback)



#title, body = question_map[training_samples[0].id]
#print title
#embeddings = utils.get_embeddings(title, embedding_map)
#encoded = encode.encode_lstm(lstm, embeddings)
#print encoded

# NOTE: Trains LSTM without batching
#train.train(lstm, encode.encode_lstm, training_samples, lstm_learning_rate, display_callback)



#batch_ids = [training_samples[0].id] + list(sample.candidate_map.keys())
#embeddings_batch = map(lambda id: 
#                       utils.get_embeddings(question_map[training_samples[0].id][0], embedding_map),
#                       batch_ids)
#print np.shape(embeddings_batch)
#encoded_batch = encode.encode_lstm_batch(lstm, embeddings_batch)
#print np.shape(encoded_batch)

# NOTE: Trains LSTM with batching
train.train_batch(lstm, encode.encode_lstm_batch, training_samples, lstm_learning_rate, display_callback)

