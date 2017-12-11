import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt

import data_loader1
import utils
import train
import encode
import evaluate


#################################################
# Plot configuration
fig = plt.figure()

losses = []


def display_callback(loss):
    losses.append(loss)
    if len(losses) % 1 == 0:
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
    lstm_num_layers)

print lstm
print

#################################################
# CNN configuration

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(200, 667, 3, 1, 2)

    def forward(self, x):
        x = F.tanh(self.conv(x))
        x = F.avg_pool1d(x, x.size()[-1])
        return x.squeeze(2)
    
cnn_learning_rate = 1e-5

cnn = CNN()

print cnn
print

#################################################
# Data loading

training_samples, dev_samples, test_samples, question_map, embedding_map = data_loader1.init()

#################################################
# MAIN                                          #
#################################################

train_indefinitely = False

##########
##########
##########
# Uncomment for part 1.2.2.1: CNN
model = cnn
encode_fn = encode.encode_cnn
optimizer = optim.Adam
learning_rate = cnn_learning_rate
batch_size = 10
num_batches = 100
##########
##########
##########



##########
##########
##########
# Uncomment for part 1.2.2.2: LSTM
#model = lstm
#encode_fn = encode.encode_lstm
#optimizer = optim.SGD
#learning_rate = lstm_learning_rate
#batch_size = 10
#num_batches = 100
##########
##########
##########



##########
# Trains models
def midpoint_eval(batch):
    if (batch + 1) % 10 == 0:
        evaluate.evaluate_model(model, encode_fn, dev_samples, question_map)
        
train.train(model, encode_fn, optimizer, training_samples,
            batch_size, num_batches, learning_rate,
            question_map, display_callback, midpoint_eval)
if train_indefinitely:
    while True:
        train.train(model, encode_fn, optimizer, training_samples,
                    batch_size, num_batches, learning_rate,
                    question_map, display_callback, midpoint_eval)
print
print 'EVALUATION'
print
print 'Askubuntu dev'
evaluate.evaluate_model(model, encode_fn, dev_samples, question_map)
print 'Askubuntu test'
evaluate.evaluate_model(model, encode_fn, test_samples, question_map)
