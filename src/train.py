import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random

import utils

#################################################
# Training


def train(
        net,
        encode_fn,
        training_samples,
        learning_rate,
        display_callback=None):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CosineEmbeddingLoss()

    # Given a title and body, return embeddings to use
    # Currently, only use titles
    def get_embeddings(title, body):
        return utils.get_embeddings(title)

    # nn.train();
    for i in range(len(training_samples)):
        sample = training_samples[i]
        embeddings = get_embeddings(*utils.get_question(sample.id))

        # print
        print i + 1, '/', len(training_samples)
        # print title

        candidate_ids = list(sample.candidate_map.keys())
        random.shuffle(candidate_ids)
        for candidate_id in candidate_ids:
            similar_indicator = sample.candidate_map[candidate_id]
            candidate_title, candidate_body = utils.get_question(candidate_id)
            candidate_embeddings = get_embeddings(
                candidate_title, candidate_body)

            encoded = encode_fn(net, embeddings)
            candidate_encoded = encode_fn(net, candidate_embeddings)

            # Update
            optimizer.zero_grad()
            loss = criterion(
                encoded.unsqueeze(0),
                candidate_encoded.unsqueeze(0),
                Variable(
                    torch.IntTensor(
                        [similar_indicator])))

            loss.backward()
            # print loss.data[0]
            if display_callback is not None:
                display_callback(loss.data[0])
            optimizer.step()


def train_batch_cnn(
        net,
        encode_fn,
        training_samples,
        learning_rate,
        display_callback=None):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CosineEmbeddingLoss()

    # Given a title and body, return embeddings to use
    # Currently, only use titles
    def get_embeddings(title, body):
        return utils.get_embeddings(title)

    # nn.train();
    for i in range(len(training_samples)):
        sample = training_samples[i]
        embeddings = get_embeddings(*utils.get_question(sample.id))

        # print
        print i + 1, '/', len(training_samples)
        # print title

        candidate_ids = list(sample.candidate_map.keys())
        random.shuffle(candidate_ids)
        encoded = []
        candidate_encoded = []
        similar_indicators = []
        for candidate_id in candidate_ids:
            candidate_title, candidate_body = utils.get_question(candidate_id)
            candidate_embeddings = get_embeddings(
                candidate_title, candidate_body)
            if len(candidate_embeddings) != 0:  # NOTE: Probably do something
                                                # else.
                encoded.append(encode_fn(net, embeddings).unsqueeze(0))
                candidate_encoded.append(
                    encode_fn(net, candidate_embeddings).unsqueeze(0))
                similar_indicators.append(sample.candidate_map[candidate_id])

        # Update
        encoded = torch.cat(encoded)
        candidate_encoded = torch.cat(candidate_encoded)
        similar_indicators = Variable(torch.IntTensor(similar_indicators))
        optimizer.zero_grad()
        loss = criterion(
            encoded,
            candidate_encoded,
            similar_indicators)

        loss.backward()
        # print loss.data[0]
        if display_callback is not None:
            display_callback(loss.data[0])
        optimizer.step()


def train_batch(
        rnn,
        encode_batch_fn,
        training_samples,
        learning_rate,
        display_callback=None):
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CosineEmbeddingLoss()

    # Given a title and body, return embeddings to use
    # Currently, only use titles
    def get_embeddings(title, body):
        return utils.get_embeddings(title)

    # rnn.train();
    for i in range(len(training_samples)):
        sample = training_samples[i]
        title, body = utils.get_question(sample.id)
        embeddings = get_embeddings(title, body)

        # print
        print i + 1, '/', len(training_samples)
        # print title

        batch_ids = [training_samples[0].id] + \
            list(sample.candidate_map.keys())
        embeddings_batch = map(
            lambda id: get_embeddings(
                *
                utils.get_question(
                    training_samples[0].id)),
            batch_ids)

        encoded_batch = encode_batch_fn(rnn, embeddings_batch)

        encoded, encoded_candidates = encoded_batch[0], encoded_batch[1:]

        # Update
        loss = 0
        optimizer.zero_grad()
        for i in range(len(encoded_candidates)):
            candidate_id = batch_ids[i + 1]
            candidate_encoded = encoded_candidates[i]
            similar_indicator = sample.candidate_map[candidate_id]

            loss += criterion(encoded.unsqueeze(0),
                              candidate_encoded.unsqueeze(0),
                              Variable(torch.IntTensor([similar_indicator])))

        loss.backward()
        # print loss.data[0]
        if display_callback is not None:
            display_callback(loss.data[0])
        optimizer.step()
