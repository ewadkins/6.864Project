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
    criterion = nn.MultiMarginLoss()
    similarity = nn.CosineSimilarity()
    for batch_num in range(num_batches):
        print (batch_num + 1) * batch_size, '/', num_batches * batch_size
        batch = np.random.choice(training_samples, batch_size)
        sample_similarities = []
        for sample in batch:
            q = encode(net, sample.id, question_map)
            similar_p = encode(net, np.random.choice(sample.similar), question_map)
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
        sample_targets = Variable(torch.LongTensor([0]*batch_size))
        optimizer.zero_grad()
        loss = criterion(sample_similarities, sample_targets)
        loss.backward()
        optimizer.step()
        if display_callback:
            display_callback(loss.data[0])
        if callback:
            callback(batch_num)