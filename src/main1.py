import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt

import utils    

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
    
    return training_samples, dev_samples, test_samples, question_map, embedding_map

#################################################
# Model

class RCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, kernel_sizes, pooling_sizes, **kwargs):
        super(RCNN, self).__init__()
        assert len(hidden_sizes) == len(kernel_sizes)
        assert len(hidden_sizes) == len(pooling_sizes)
        assert len(hidden_sizes) > 0
        
        self.hidden_size = hidden_sizes[-1]
        self.output_size = output_size
        
        self.convs = nn.ModuleList();
        self.pools = nn.ModuleList();
        
        # Convolutional and pooling layers
        for i in range(len(kernel_sizes)):
            self.convs.append(
                nn.Conv1d(
                    input_size if i == 0 else hidden_sizes[i-1], # input_channels
                    hidden_sizes[i], # output_channels
                    kernel_sizes[i], # kernel_size
                    **kwargs))
            self.pools.append(nn.AvgPool1d(pooling_sizes[i]))
            
        # Gated recurrent unit layer (uses size of last convolutional layer)
        self.gru = nn.GRU(hidden_sizes[-1], hidden_sizes[-1])
        
        # Output layer
        self.out = nn.Linear(hidden_sizes[-1], output_size)

        
    # NOTE TO TRISTAN: Use tanh or sigmoid??? Paper uses sigmoids for convolutional layers, but \
    #   output should probably be a tanh
    
    def forward(self, input, hidden):
        # Feed only input vector through convs+pools, instead of (input + hidden)
        # This is because the GRU incorporates the hidden layer using a gate instead
        
        x = input
        x = x.transpose(0, 1)
        x = x.unsqueeze(0)
                        
        # Convolutional and pooling layers
        for i in range(len(self.convs)):
            x = F.sigmoid(self.convs[i](x))
            x = self.pools[i](x)
                    
        x = x.transpose(1, 2).transpose(0, 1)
                
        # Gated recurrent unit layer
        output, hidden = self.gru(x, hidden)
                
        output = output.view(output.size(0), self.hidden_size)
                
        # Output layer
        output = F.relu(self.out(output))
        
        reduced = output.mean(0)
                
        return reduced, output, hidden

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

rcnn = RCNN(rcnn_input_size, rcnn_hidden_sizes, rcnn_output_size,
            rcnn_kernel_sizes, rcnn_pooling_sizes,
            padding=max(rcnn_kernel_sizes));

print rcnn
print 

#################################################
#################################################
# LSTM configuration

lstm_input_size = 200
lstm_hidden_size = 300
lstm_num_layers = 1

lstm_learning_rate = 1e-1

lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)

print lstm
print 

#################################################
# Data loading

training_samples, dev_samples, test_samples, question_map, embedding_map = init();

#################################################
# Encoding

# Returns the vector representation of a question, given the question's word embeddings
def encode_rcnn(rcnn, embeddings):
    input = Variable(torch.FloatTensor(embeddings))
    hidden = None
    encoded, output, hidden = rcnn(input, hidden)
    return encoded

def encode_lstm(lstm, embeddings):
    input = Variable(torch.FloatTensor(embeddings)).unsqueeze(0)
    hidden = Variable(torch.randn(lstm_num_layers, 1, lstm_hidden_size))
    cell = Variable(torch.randn(lstm_num_layers, 1, lstm_hidden_size))
    output, (hidden, cell) = lstm(input, (hidden, cell))
    output = output.view(output.size(1), lstm_hidden_size)
    encoded = output.mean(0)
    return encoded

def encode_lstm_batch(lstm, embeddings_batch):
    input = Variable(torch.FloatTensor(embeddings_batch))
    batch_size = input.size(0)
    hidden = Variable(torch.randn(lstm_num_layers, batch_size, lstm_hidden_size))
    cell = Variable(torch.randn(lstm_num_layers, batch_size, lstm_hidden_size))
    output, (hidden, cell) = lstm(input, (hidden, cell))
    encoded_batch = output.mean(1)
    return encoded_batch

#################################################
# Training

def train(rnn, encode_fn, training_samples, learning_rate, display_callback=None):
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CosineEmbeddingLoss()
    
    # Given a title and body, return embeddings to use
    # Currently, only use titles
    def get_embeddings(title, body):
        return utils.get_embeddings(title, embedding_map)
    
    #rcnn.train();
    for i in range(len(training_samples)):
        sample = training_samples[i]
        embeddings = get_embeddings(*question_map[sample.id])
        
        #print
        print i + 1, '/', len(training_samples)
        #print title
        
        candidate_ids = list(sample.candidate_map.keys())
        random.shuffle(candidate_ids)
        for candidate_id in candidate_ids:
            similar_indicator = sample.candidate_map[candidate_id]
            candidate_title, candidate_body = question_map[candidate_id]
            candidate_embeddings = get_embeddings(candidate_title, candidate_body)
            
            encoded = encode_fn(rnn, embeddings)
            candidate_encoded = encode_fn(rnn, candidate_embeddings)
            
            # Update
            optimizer.zero_grad();
            loss = criterion(encoded.unsqueeze(0), candidate_encoded.unsqueeze(0),
                             Variable(torch.IntTensor([similar_indicator])));
            
            loss.backward();
            #print loss.data[0]
            if display_callback is not None: display_callback(loss.data[0])
            optimizer.step();
            
def train_batch(rnn, encode_batch_fn, training_samples, learning_rate, display_callback=None):
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CosineEmbeddingLoss()
    
    # Given a title and body, return embeddings to use
    # Currently, only use titles
    def get_embeddings(title, body):
        return utils.get_embeddings(title, embedding_map)
    
    #rcnn.train();
    for i in range(len(training_samples)):
        sample = training_samples[i]
        title, body = question_map[sample.id]
        embeddings = get_embeddings(title, body)
        
        #print
        print i + 1, '/', len(training_samples)
        #print title
        
        batch_ids = [training_samples[0].id] + list(sample.candidate_map.keys())
        embeddings_batch = map(lambda id: 
                               get_embeddings(*question_map[training_samples[0].id]), batch_ids)
        
        encoded_batch = encode_batch_fn(rnn, embeddings_batch)
        print encoded_batch.size()
        
        encoded, encoded_candidates = encoded_batch[0], encoded_batch[1:]
        
        # Update
        loss = 0
        optimizer.zero_grad();
        for i in range(len(encoded_candidates)):
            candidate_id = batch_ids[i + 1]
            candidate_encoded = encoded_candidates[i]
            similar_indicator = sample.candidate_map[candidate_id]
            
            loss += criterion(encoded.unsqueeze(0), candidate_encoded.unsqueeze(0),
                             Variable(torch.IntTensor([similar_indicator])));
                
        loss.backward();
        #print loss.data[0]
        if display_callback is not None: display_callback(loss.data[0])
        optimizer.step();
        
#################################################

#title, body = question_map[training_samples[0].id]
#print title
#embeddings = utils.get_embeddings(title, embedding_map)
#encoded = encode_rcnn(rcnn, embeddings)
#print encoded

# NOTE: Trains with RCNN without batching
#train(rcnn, encode_rcnn, training_samples, rcnn_learning_rate, display_callback)


#title, body = question_map[training_samples[0].id]
#print title
#embeddings = utils.get_embeddings(title, embedding_map)
#encoded = encode_lstm(lstm, embeddings)
#print encoded

# NOTE: Trains with LSTM without batching
#train(lstm, encode_lstm, training_samples, lstm_learning_rate, display_callback)


#batch_ids = [training_samples[0].id] + list(sample.candidate_map.keys())
#embeddings_batch = map(lambda id: 
#                       utils.get_embeddings(question_map[training_samples[0].id][0], embedding_map),
#                       batch_ids)
#print np.shape(embeddings_batch)
#encoded_batch = encode_lstm_batch(lstm, embeddings_batch)
#print np.shape(encoded_batch)

# NOTE: Trains with LSTM with batching
train_batch(lstm, encode_lstm_batch, training_samples, lstm_learning_rate, display_callback)
