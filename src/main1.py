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
# LSTM configuration

lstm_input_size = 200
lstm_hidden_size = 240
lstm_num_layers = 1

lstm_learning_rate = 1e-1

lstm = nn.LSTM(
    lstm_input_size,
    lstm_hidden_size,
    lstm_num_layers,
    bidirectional=True)

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


cnn_learning_rate = 1e-1

cnn = CNN()

print cnn
print

#################################################
# Plot configuration
fig = plt.figure()

losses = []


def display_callback(loss):
    losses.append(loss)
    if len(losses) % 5 == 0:
        fig.clear()
        plt.plot(list(range(len(losses))), losses)
        plt.pause(0.0001)

#################################################
# Data loading


training_samples, dev_samples, test_samples, question_map, embedding_map =\
    data_loader1.init()

#################################################
# MAIN                                          #
#################################################


##########
##########
##########
# Uncomment for part 1.2.2.1: CNN
#model = cnn
#encode_fn = encode.encode_cnn
#optimizer = optim.Adam
#learning_rate = cnn_learning_rate
#batch_size = 20
#num_batches = 6000
#save_name = 'models/part_1_cnn.pt'
##########
##########
##########


##########
##########
##########
# Uncomment for part 1.2.2.2: LSTM
model = lstm
encode_fn = encode.encode_lstm
optimizer = optim.Adam
learning_rate = lstm_learning_rate
batch_size = 20
num_batches = 60000
save_name = 'models/part_1_lstm.pt'
##########
##########
##########


#model = torch.load('part_1_cnn_good.pt')
#print '\nMODEL LOADED\n'


##########
# Trains models
def midpoint_eval(batch):
    if (batch + 1) % 25 == 0:
        print 'Askubuntu dev'
        evaluate.evaluate_model(model, encode_fn, dev_samples, question_map)
        print 'Askubuntu test'
        evaluate.evaluate_model(model, encode_fn, test_samples, question_map)
        torch.save(model, save_name + str((batch + 1) * batch_size))
        print '\nMODEL SAVED\n'
        
train.train(model, encode_fn, optimizer, training_samples,
            batch_size, num_batches, learning_rate,
            question_map, display_callback, midpoint_eval)

torch.save(model, save_name)
print '\nMODEL SAVED\n'

print
print 'EVALUATION'
print
print 'Askubuntu dev'
evaluate.evaluate_model(model, encode_fn, dev_samples, question_map)
print 'Askubuntu test'
evaluate.evaluate_model(model, encode_fn, test_samples, question_map)
