import torch
from torch.autograd import Variable

import utils

#################################################
# Encoding

# Returns the vector representation of a question, given the question's
# word embeddings


def encode_bag_of_words(string, vocabulary_map):
    encoded = [0.0] * len(vocabulary_map)
    for word in string.split():
        if word in vocabulary_map:
            encoded[vocabulary_map[word]] += 1
    return Variable(torch.FloatTensor(encoded))


def encode_cnn(cnn, embeddings):
    input = torch.transpose(Variable(
        torch.FloatTensor(embeddings)), 0, 1).unsqueeze(0)
    return cnn(input).squeeze()


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
    hidden = Variable(
        torch.randn(
            lstm_num_layers,
            batch_size,
            lstm_hidden_size))
    cell = Variable(torch.randn(lstm_num_layers, batch_size, lstm_hidden_size))
    output, (hidden, cell) = lstm(input, (hidden, cell))
    encoded_batch = output.mean(1)
    return encoded_batch


def store_config(_lstm_hidden_size, _lstm_num_layers):
    global lstm_hidden_size
    global lstm_num_layers
    lstm_hidden_size = _lstm_hidden_size
    lstm_num_layers = _lstm_num_layers
