import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

import data_loader2
import utils
import train
import encode
import evaluate
import domain_transfer


#################################################
# Plot configuration


fig = plt.figure()
losses1 = []
losses2 = []
def display_callback(loss1, loss2):
    losses1.append(loss1)
    losses2.append(loss2)
    if len(losses1) % 1 == 0:
        fig.clear()
        plt.subplot(211)
        plt.plot(list(range(len(losses1))), losses1)
        plt.subplot(212)
        plt.plot(list(range(len(losses2))), losses2)
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


cnn_learning_rate = 1e-1

cnn = CNN()
#cnn = torch.load('auc_tuned_cnn.pt')
print cnn
print

#################################################
# CNN/LSTM Domain Transfer Net Configuration

cnn_domain_transfer_net = domain_transfer.CNNDomainTransferNet(cnn)
lstm_domain_transfer_net = domain_transfer.LSTMDomainTransferNet(lstm)

#################################################
# Data loading

(askubuntu_training_samples, askubuntu_dev_samples, askubuntu_test_samples,
 askubuntu_question_map, android_dev_samples, android_test_samples,
 android_question_map, embedding_map) = data_loader2.init()

#################################################
# MAIN                                          #
#################################################


# TF-IDF evaluation
#question_map = askubuntu_question_map
#samples1 = askubuntu_dev_samples
#samples2 = askubuntu_test_samples
#
#tfidf = TfidfVectorizer(min_df=1)
#corpus_texts = map(lambda (t, b): t + ' ' + b, question_map.values())
#tfidf.fit(corpus_texts)
#
#print 'TFIDF-weighted bag of word booleans evaluation askubuntu dev:'
#evaluate.evaluate_directly(samples1, encode.encode_tfidf_bag_of_word_booleans,
#                           question_map, tfidf)
#print 'TFIDF-weighted bag of word booleans evaluation askubuntu test:'
#evaluate.evaluate_directly(samples2, encode.encode_tfidf_bag_of_word_booleans,
#                           question_map, tfidf)



##########
##########
##########
# Uncomment for part 2.3.1.a.1.1: Evaluate bag of word booleans on Askubuntu dataset
#question_map = askubuntu_question_map
#samples1 = askubuntu_dev_samples
#samples2 = askubuntu_test_samples
#vocabulary_map = utils.get_vocabulary_map(question_map)
#print 'Bag of word booleans evaluation askubuntu dev:'
#evaluate.evaluate_directly(samples1, encode.encode_bag_of_word_booleans, question_map, vocabulary_map)
#print 'Bag of word booleans evaluation askubuntu test:'
#evaluate.evaluate_directly(samples2, encode.encode_bag_of_word_booleans, question_map, vocabulary_map)
##########
##########
##########


##########
##########
##########
# Uncomment for part 2.3.1.a.1.2: Evaluate bag of word counts on Askubuntu dataset
#question_map = askubuntu_question_map
#samples1 = askubuntu_dev_samples
#samples2 = askubuntu_test_samples
#vocabulary_map = utils.get_vocabulary_map(question_map)
#print 'Bag of word counts evaluation askubuntu dev:'
#evaluate.evaluate_directly(samples1, encode.encode_bag_of_word_counts, question_map, vocabulary_map)
#print 'Bag of word counts evaluation askubuntu test:'
#evaluate.evaluate_directly(samples2, encode.encode_bag_of_word_counts, question_map, vocabulary_map)
##########
##########
##########


##########
##########
##########
# Uncomment for part 2.3.1.a.1.3: Evaluate mean embeddings on Askubuntu dataset
#question_map = askubuntu_question_map
#samples1 = askubuntu_dev_samples
#samples2 = askubuntu_test_samples
#vocabulary_map = utils.get_vocabulary_map(question_map)
#print 'Mean embeddings evaluation askubuntu dev:'
#evaluate.evaluate_directly(samples1, encode.encode_mean_embeddings, question_map, embedding_map)
#print 'Mean embeddings evaluation askubuntu test:'
#evaluate.evaluate_directly(samples2, encode.encode_mean_embeddings, question_map, embedding_map)
##########
##########
##########


##########
##########
##########
# Uncomment for part 2.3.1.a.1.4: Evaluate TF-IDF BOW on Askubuntu dataset
#question_map = askubuntu_question_map
#samples1 = askubuntu_dev_samples
#samples2 = askubuntu_test_samples
#tfidf = TfidfVectorizer(min_df=1)
#corpus_texts = map(lambda (t, b): t + ' ' + b, question_map.values())
#tfidf.fit(corpus_texts)
#print 'TFIDF-weighted bag of word booleans evaluation askubuntu dev:'
#evaluate.evaluate_directly(samples1, encode.encode_tfidf_bag_of_word_booleans,
#                           question_map, tfidf)
#print 'TFIDF-weighted bag of word booleans evaluation askubuntu test:'
#evaluate.evaluate_directly(samples2, encode.encode_tfidf_bag_of_word_booleans,
#                           question_map, tfidf)
##########
##########
##########


##########
##########
##########
# Uncomment for part 2.3.1.a.2.1: Evaluate bag of word booleans on Android dataset
#question_map = android_question_map
#samples1 = android_dev_samples
#samples2 = android_test_samples
#vocabulary_map = utils.get_vocabulary_map(question_map)
#print 'Bag of word booleans evaluation android dev:'
#evaluate.evaluate_directly(samples1, encode.encode_bag_of_word_booleans, question_map, vocabulary_map)
#print 'Bag of word booleans evaluation android test:'
#evaluate.evaluate_directly(samples2, encode.encode_bag_of_word_booleans, question_map, vocabulary_map)
##########
##########
##########


##########
##########
##########
# Uncomment for part 2.3.1.a.2.2: Evaluate bag of word counts on Android dataset
#question_map = android_question_map
#samples1 = android_dev_samples
#samples2 = android_test_samples
#vocabulary_map = utils.get_vocabulary_map(question_map)
#print 'Bag of word counts evaluation android dev:'
#evaluate.evaluate_directly(samples1, encode.encode_bag_of_word_counts, question_map, vocabulary_map)
#print 'Bag of word counts evaluation android test:'
#evaluate.evaluate_directly(samples2, encode.encode_bag_of_word_counts, question_map, vocabulary_map)
##########
##########
##########


##########
##########
##########
# Uncomment for part 2.3.1.a.2.3: Evaluate mean embeddings on Android dataset
#question_map = android_question_map
#samples1 = android_dev_samples
#samples2 = android_test_samples
#vocabulary_map = utils.get_vocabulary_map(question_map)
#print 'Mean embeddings evaluation android dev:'
#evaluate.evaluate_directly(samples1, encode.encode_mean_embeddings, question_map, embedding_map)
#print 'Mean embeddings evaluation android test:'
#evaluate.evaluate_directly(samples2, encode.encode_mean_embeddings, question_map, embedding_map)
##########
##########
##########


##########
##########
##########
# Uncomment for part 2.3.1.a.2.4: Evaluate TF-IDF BOW on Android dataset
#question_map = android_question_map
#samples1 = android_dev_samples
#samples2 = android_test_samples
#tfidf = TfidfVectorizer(min_df=1)
#corpus_texts = map(lambda (t, b): t + ' ' + b, question_map.values())
#tfidf.fit(corpus_texts)
#print 'TFIDF-weighted bag of word booleans evaluation android dev:'
#evaluate.evaluate_directly(samples1, encode.encode_tfidf_bag_of_word_booleans,
#                           question_map, tfidf)
#print 'TFIDF-weighted bag of word booleans evaluation android test:'
#evaluate.evaluate_directly(samples2, encode.encode_tfidf_bag_of_word_booleans,
#                           question_map, tfidf)
##########
##########
##########


##########
##########
##########
# Uncomment for part 2.3.1.b.1: Train on askubuntu, no transfer learning
#model = cnn
#encode_fn = encode.encode_cnn
#optimizer = optim.Adam
#learning_rate = cnn_learning_rate
#batch_size = 20
#num_batches = 2000000
#save_name = 'transfer_models/preprocessed_vecs_transfer_cnn.pt'
##
##model = torch.load('part_1_lstm_good.pt')
##print '\nMODEL LOADED\n'
##
#def midpoint_eval(batch):
#    if (batch + 1) % 40 == 0:
#        print 'Evaluation of askubuntu dev'
#        evaluate.evaluate_model(model, encode_fn, askubuntu_dev_samples,
#            askubuntu_question_map)
#    if (batch + 1) % 40 == 0:
#        print 'Evaluation of android dev'
#        evaluate.evaluate_model(model, encode_fn, android_dev_samples[:100],
#            android_question_map)
#        torch.save(model, save_name + str((batch + 1) * batch_size))
#        print '\nMODEL SAVED\n'
#
#train.train(model, encode_fn, optimizer, askubuntu_training_samples,
#        batch_size, num_batches, learning_rate,
#        askubuntu_question_map, display_callback, midpoint_eval)
##########
##########
##########


##########
##########
##########

# Uncomment for part 2.3.3.1: Evaluate with domain transfer

#model = lstm_domain_transfer_net
##model = torch.load('transfer_models/preprocessed_vecs_transfer_cnn.pt8000')
#encode_fn = encode.encode_lstm
#encode_domain_fn = encode.encode_lstm_domain
#optimizer1 = optim.Adam
#optimizer2 = optim.Adam
#learning_rate1 = lstm_learning_rate
#learning_rate2 = -1e-1 #TODO: play with this
#gamma = 1e-6
#batch_size = 20
#num_batches = 2000000
#save_name = 'transfer_models/preprocessed_vecs_domain_transfer_cnn.pt'
#
#model = torch.load('transfer_models/preprocessed_vecs_domain_transfer_cnn.pt2500')
#print '\nMODEL LOADED\n'

#def midpoint_eval(batch):
#    if (batch+1) % 25 == 0:
#        print 'Evaluation of askubuntu dev'
#        evaluate.evaluate_model(
#            model,
#            encode_fn,
#            askubuntu_dev_samples,
#            askubuntu_question_map)
#    if (batch+1) % 25 == 0:
#        print 'Evaluation of android dev'
#        evaluate.evaluate_model(
#            model,
#            encode_fn,
#            android_dev_samples[:150],
#            android_question_map)
#        torch.save(model, save_name + str((batch + 1) * batch_size))
#        print '\nMODEL SAVED\n'


#train.train_domain_transfer(
#    model,
#    encode_fn,
#    encode_domain_fn,
#    optimizer1,
#    optimizer2,
#    askubuntu_training_samples,
#    batch_size,
#    num_batches,
#    learning_rate1,
#    learning_rate2,
#    gamma,
#    askubuntu_question_map,
#    android_question_map,
#    display_callback,
#    midpoint_eval)

##########
##########
##########

##########
#print
#print 'EVALUATION'
#print
#print 'Evaluation of askubuntu dev'
#evaluate.evaluate_model(
#    model,
#    encode_fn,
#    askubuntu_dev_samples,
#    askubuntu_question_map)
#print 'Evaluation of askubuntu test'
#evaluate.evaluate_model(
#    model,
#    encode_fn,
#    askubuntu_test_samples,
#    askubuntu_question_map)
#print 'Evaluation of android dev'
#evaluate.evaluate_model(
#    model,
#    encode_fn,
#    android_dev_samples,
#    android_question_map)
#print 'Evaluation of android test'
#evaluate.evaluate_model(
#    model,
#    encode_fn,
#    android_test_samples,
#    android_question_map)
