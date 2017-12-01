import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

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
            x = self.convs[i](x)
            x = F.sigmoid(x)
            x = self.pools[i](x)
                    
        x = x.transpose(1, 2).transpose(0, 1)
                
        # Gated recurrent unit layer
        output, hidden = self.gru(x, hidden)
                
        output = output.view(output.size(0), self.hidden_size)
                
        # Output layer
        output = F.tanh(self.out(output))
        
        reduced = output.mean(0)
                
        return reduced, output, hidden

#################################################
# RCNN configuration

input_size = 200 # size of word embedding
hidden_sizes = [300, 200, 150] # sizes of the convolutional layers; determines # of conv layers
output_size = 100 # size of state vector

kernel_sizes = [5, 4, 3] # NOTE: assert len(kernel_sizes) == len(hidden_sizes)
pooling_sizes = [2, 2, 2] # NOTE: assert len(pooling_sizes) == len(hidden_sizes)

learning_rate = 1e-1;

rcnn = RCNN(input_size, hidden_sizes, output_size, kernel_sizes, pooling_sizes,
            padding=max(kernel_sizes));

print rcnn
print 

#################################################
# Data loading

training_samples, dev_samples, test_samples, question_map, embedding_map = init();

#################################################
# Encoding

# Returns the vector representation of a question, given the question's word embeddings
def encode(rcnn, embeddings):
    input = Variable(torch.FloatTensor(embeddings))
    hidden = None
    encoded, output, hidden = rcnn(input, hidden)
    return encoded

#################################################
# Training

def train(rcnn, learning_rate=learning_rate):
    optimizer = optim.SGD(rcnn.parameters(), lr=learning_rate)
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
        
        candidate_ids = list(sample.candidate_map.keys())
        random.shuffle(candidate_ids)
        for candidate_id in candidate_ids:
            similar_indicator = sample.candidate_map[candidate_id]
            candidate_title, candidate_body = question_map[candidate_id]
            candidate_embeddings = get_embeddings(candidate_title, candidate_body)
            
            encoded = encode(rcnn, embeddings)
            candidate_encoded = encode(rcnn, candidate_embeddings)
            
            # Update
            optimizer.zero_grad();
            loss = criterion(encoded, candidate_encoded,
                             Variable(torch.IntTensor(similar_indicator)));
            loss.backward();
            optimizer.step();
            print loss.data[0]
        
    



#################################################

#print training_samples[0]
#title, body = question_map[training_samples[0].id]
#title2, body2 = question_map[training_samples[0].similar[0]]

#print
#print title
#embeddings = utils.get_embeddings(title, embedding_map)
#encoded = encode(rcnn, embeddings)
#print
#print encoded.data.numpy()

#print
#print title2
#embeddings = utils.get_embeddings(title2, embedding_map)
#encoded = encode(rcnn, embeddings)
#print
#print encoded.data.numpy()

train(rcnn, training_samples)
