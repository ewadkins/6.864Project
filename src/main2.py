import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt

import data_loader2
import utils
import train
import encode
import evaluate
import domain_transfer


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
lstm_hidden_size = 240
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
# CNN Domain Transfer Net Configuration
# (LSTM domain transfer net can be built the same way)

# cnn or lstm
feature_extractor = cnn

cnn_domain_transfer_net = domain_transfer.DomainTransferNet(feature_extractor)

#################################################
# Data loading

(askubuntu_training_samples, askubuntu_dev_samples, askubuntu_test_samples,
 askubuntu_question_map, android_dev_samples, android_test_samples,
 android_question_map, embedding_map) = data_loader2.init()

#################################################
# MAIN                                          #
#################################################


##########
##########
##########
# Uncomment for part 2.3.1.a.1: Evaluate bag of words on Askubuntu dataset
#print 'Bag of words evaluation askubuntu:'
#question_map = askubuntu_question_map
#samples = askubuntu_dev_samples
#vocabulary_map = utils.get_vocabulary_map(question_map)
#evaluate.evaluate_bag_of_words(samples, question_map, vocabulary_map)
##########
##########
##########



##########
##########
##########
# Uncomment for part 2.3.1.a.2: Evaluate bag of words on Android dataset
#print 'Bag of words evaluation android:'
#question_map = android_question_map
#samples = android_dev_samples
#vocabulary_map = utils.get_vocabulary_map(question_map)
#evaluate.evaluate_bag_of_words(samples, question_map, vocabulary_map)
##########
##########
##########



##########
##########
##########
# Uncomment for part 2.3.3.1: Evaluate with domain transfer
#def midpoint_eval(i):
#    if (i + 1) % 200 == 0:
#        evaluate.evaluate_model(cnn_domain_transfer_net, encode.encode_cnn_dt_label,
#                                askubuntu_dev_samples, askubuntu_question_map) 
#    if (i + 1) % 3000 == 0:
#        evaluate.evaluate_model(cnn_domain_transfer_net, encode.encode_cnn_dt_label,
#                                android_dev_samples, android_question_map)
#train.train_batch_domain_transfer(cnn_domain_transfer_net,
#                                       encode.encode_cnn_dt_label, encode.encode_cnn_dt_domain,
#                                       askubuntu_training_samples,
#                                       cnn_learning_rate, 
#                                       askubuntu_question_map, android_question_map,
#                                       display_callback, midpoint_eval)
##########
##########
##########



##########
##########
##########
# Uncomment for part 2.3.1.b: Train on askubuntu and evaluate android, no transfer learning
model = cnn
encode_fn = encode.encode_cnn
optimizer = optim.Adam
learning_rate = cnn_learning_rate
batch_size = 10
num_batches = 100
def midpoint_eval(batch):
    if (batch + 1) % 25 == 0:
        print 'Evaluation of askubuntu dev'
        evaluate.evaluate_model(model, encode_fn, askubuntu_dev_samples, askubuntu_question_map) 
    if (batch + 1) % 100 == 0:
        print 'Evaluation of android dev'
        evaluate.evaluate_model(model, encode_fn, android_dev_samples, android_question_map)
        
train.train(model, encode_fn, optimizer, askubuntu_training_samples,
            batch_size, num_batches, learning_rate,
            askubuntu_question_map, display_callback, midpoint_eval)
##########
##########
##########



##########
##########
##########
# Uncomment for part 2.3.3.1: Evaluate with domain transfer
#model = cnn_domain_transfer_net
#encode_fn = encode.encode_cnn
#encode_domain_fn = encode.encode_cnn_domain
#optimizer1 = optim.Adam
#optimizer2 = optim.Adam
#learning_rate1 = cnn_learning_rate
#learning_rate2 = -1e-7
#gamma = 1e-5
#batch_size = 10
#num_batches = 100
#def midpoint_eval(batch):
#    if (batch + 1) % 25 == 0:
#        print 'Evaluation of askubuntu dev'
#        evaluate.evaluate_model(model, encode_fn, askubuntu_dev_samples, askubuntu_question_map) 
#    if (batch + 1) % 100 == 0:
#        print 'Evaluation of android dev'
#        evaluate.evaluate_model(model, encode_fn, android_dev_samples, android_question_map)
#train.train_domain_transfer(model,
#                            encode_fn, encode_domain_fn,
#                            optimizer1, optimizer2,
#                            askubuntu_training_samples, batch_size, num_batches,
#                            learning_rate1, learning_rate2,
#                            gamma,
#                            askubuntu_question_map, android_question_map,
#                            display_callback, midpoint_eval)
##########
##########
##########


##########
print
print 'EVALUATION'
print
print 'Evaluation of askubuntu dev'
evaluate.evaluate_model(model, encode_fn, askubuntu_dev_samples, askubuntu_question_map)
print 'Evaluation of askubuntu test'
evaluate.evaluate_model(model, encode_fn, askubuntu_test_samples, askubuntu_question_map)
print 'Evaluation of android dev'
evaluate.evaluate_model(model, encode_fn, android_dev_samples, android_question_map)
print 'Evaluation of askubuntu test'
evaluate.evaluate_model(model, encode_fn, android_test_samples, android_question_map)
