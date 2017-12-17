import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import utils


# Returns the vector representation of a question, given the question's
# word embeddings

use_body_lstm = False


def encode_lstm(net, q_id, question_map):
    state_size = net.hidden_size
    def encode(net, embeddings):
        if len(embeddings) == 0:
            return None
        state = (Variable(torch.zeros(2, 1, state_size)),
                 Variable(torch.zeros(2, 1, state_size)))
        out, state = net(Variable(
            torch.FloatTensor(embeddings)).view(len(embeddings), 1, -1), state)
        out = torch.transpose(out.squeeze(1), 0, 1).unsqueeze(0)
        return F.avg_pool1d(out, out.size()[-1]).squeeze(2)
    def get_embeddings(title, body):
        return utils.get_embeddings(title), utils.get_embeddings(body)
    title_embeddings, body_embeddings = get_embeddings(*question_map[q_id])
    body_embeddings = body_embeddings[:80]
    title_state = encode(net, title_embeddings)
    if use_body_lstm:
        body_state = encode(net, body_embeddings)
        if body_state is not None:
            return F.avg_pool1d(torch.cat((
                title_state,
                body_state), 1).unsqueeze(0), 2).squeeze()
    return title_state.squeeze()


def encode_lstm_domain(net, q_id, question_map):
    state_size = net.hidden_size
    def encode(net, embeddings):
        if len(embeddings) == 0:
            return None
        state = (Variable(torch.zeros(2, 1, state_size)),
                 Variable(torch.zeros(2, 1, state_size)))
        out = net(Variable(
            torch.FloatTensor(embeddings)).view(len(embeddings), 1, -1), state,
                         return_domain=True)
        return out.squeeze()
    def get_embeddings(title, body):
        return utils.get_embeddings(title), utils.get_embeddings(body)
    title_embeddings, body_embeddings = get_embeddings(*question_map[q_id])
    body_embeddings = body_embeddings[:80]
    title_state = encode(net, title_embeddings)
    if use_body_lstm:
        body_state = encode(net, body_embeddings)
        if body_state is not None:
            return F.avg_pool1d(torch.cat((
                title_state,
                body_state), 1).unsqueeze(0), 2).squeeze()
    return title_state.squeeze()


def encode_cnn(net, q_id, question_map):
    def encode(net, embeddings):
        if len(embeddings) == 0:
            return None
        input = torch.transpose(Variable(
            torch.FloatTensor(embeddings)), 0, 1).unsqueeze(0)
        return net(input).squeeze()
    def get_embeddings(title, body):
        return utils.get_embeddings(title), utils.get_embeddings(body)
    title_embeddings, body_embeddings = get_embeddings(*question_map[q_id])
    title = encode(net, title_embeddings)
    body = encode(net, body_embeddings)
    if body is not None:
        return F.avg_pool1d(
            torch.cat(
                (title.unsqueeze(1),
                 body.unsqueeze(1)),
                1).unsqueeze(0),
            2).squeeze()
    return title


def encode_cnn_domain(net, q_id, question_map):
    def encode(net, embeddings):
        if len(embeddings) == 0:
            return None
        input = torch.transpose(Variable(
            torch.FloatTensor(embeddings)), 0, 1).unsqueeze(0)
        return net(input, return_domain=True).squeeze()
    def get_embeddings(title, body):
        return utils.get_embeddings(title), utils.get_embeddings(body)
    title_embeddings, body_embeddings = get_embeddings(*question_map[q_id])
    title = encode(net, title_embeddings)
    body = encode(net, body_embeddings)
    if body is not None:
        return F.avg_pool1d(
            torch.cat(
                (title.unsqueeze(1),
                 body.unsqueeze(1)),
                1).unsqueeze(0),
            2).squeeze()
    return title


def encode_tfidf_bag_of_word_booleans(string, tfidf):
    encoded = tfidf.transform([string])
    return Variable(torch.FloatTensor(encoded.toarray()[0]))


def encode_bag_of_word_booleans(string, vocabulary_map):
    encoded = [0] * len(vocabulary_map)
    for word in string.split():
        if word in vocabulary_map:
            encoded[vocabulary_map[word]] = 1
    return Variable(torch.FloatTensor(encoded))


def encode_bag_of_word_counts(string, vocabulary_map):
    encoded = [0] * len(vocabulary_map)
    for word in string.split():
        if word in vocabulary_map:
            encoded[vocabulary_map[word]] += 1
    return Variable(torch.FloatTensor(encoded))


def encode_mean_embeddings(string, embedding_map):
    embeddings = utils.get_embeddings(string, embedding_map)
    encoded = np.mean(embeddings, axis=0)
    return Variable(torch.FloatTensor(encoded))
