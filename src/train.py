import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np

import utils


def train(net,
          encode,
          optimizer,
          training_samples,
          batch_size,
          num_batches,
          learning_rate,
          question_map,
          display_callback=None,
          callback=None):

    optimizer = optimizer(net.parameters(), lr=learning_rate)
    criterion = nn.MultiMarginLoss(margin=0.3)
    similarity = nn.CosineSimilarity()
    def next_batch(batch_size, training_samples, batch_num):
        start_index = (batch_num * batch_size) % (len(training_samples) - batch_size)
        return training_samples[start_index:start_index+batch_size]
    for batch_num in range(num_batches):
        print (batch_num + 1) * batch_size, '/', num_batches * batch_size
        batch = next_batch(batch_size, training_samples, batch_num)
        sample_similarities = []
        for sample in batch:
            q = encode(net, sample.id, question_map)
            similar_p = encode(
                net, np.random.choice(
                    sample.similar), question_map)
            dissimilar_ps = map(
                lambda question_id: encode(net, question_id, question_map),
                np.random.choice(sample.dissimilar, 20))
            similarities = map(
                lambda question: similarity(q.unsqueeze(0),
                                            question.unsqueeze(0)),
                [similar_p] + dissimilar_ps)
            similarities = torch.cat(similarities).unsqueeze(0)
            sample_similarities.append(similarities)
        sample_similarities = torch.cat(sample_similarities)
        sample_targets = Variable(torch.LongTensor([0] * batch_size))
        optimizer.zero_grad()
        loss = criterion(sample_similarities, sample_targets)
        loss.backward()
        optimizer.step()
        if display_callback:
            display_callback(loss.data[0])
        if callback:
            callback(batch_num)


def train_domain_transfer(net,
                          encode_label,
                          encode_domain,
                          optimizer1,
                          optimizer2,
                          training_samples,
                          batch_size,
                          num_batches,
                          learning_rate1,
                          learning_rate2,
                          gamma,
                          question_map1,
                          question_map2,
                          display_callback=None,
                          callback=None):

    optimizer1 = optimizer1(net.parameters(), lr=learning_rate1)
    optimizer2 = optimizer2(net.parameters(), lr=learning_rate2)
    criterion1 = nn.MultiMarginLoss(margin=0.3)
    criterion2 = nn.CrossEntropyLoss()
    similarity = nn.CosineSimilarity()
    for batch_num in range(num_batches):
        print (batch_num + 1) * batch_size, '/', num_batches * batch_size
        batch = np.random.choice(training_samples, batch_size)
        sample_similarities = []
        for sample in batch:
            q = encode_label(net, sample.id, question_map1)
            similar_p = encode_label(
                net, np.random.choice(
                    sample.similar), question_map1)
            dissimilar_ps = map(
                lambda question_id: encode_label(
                    net, question_id, question_map1), np.random.choice(
                    sample.dissimilar, 20))
            similarities = map(
                lambda question: similarity(q.unsqueeze(0),
                                            question.unsqueeze(0)),
                [similar_p] + dissimilar_ps)
            similarities = torch.cat(similarities).unsqueeze(0)
            sample_similarities.append(similarities)
        sample_similarities = torch.cat(sample_similarities)
        sample_targets = Variable(torch.LongTensor([0] * batch_size))

        encoded = torch.cat(
            map(
                lambda question_id: encode_domain(
                    net, question_id, question_map1), np.random.choice(
                    question_map1.keys(), 20 * batch_size)) + map(
                lambda question_id: encode_domain(
                    net, question_id, question_map2), np.random.choice(
                    question_map2.keys(), 20 * batch_size)))
        encoded = encoded.view(2 * 20 * batch_size, 2)
        targets = Variable(torch.LongTensor(
            ([0] * 20 * batch_size) + ([1] * 20 * batch_size)))

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss1 = criterion1(sample_similarities, sample_targets)
        loss2 = criterion2(encoded, targets)
        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()

        if display_callback:
            display_callback(loss1.data[0], loss2.data[0])
        if callback:
            callback(batch_num)
