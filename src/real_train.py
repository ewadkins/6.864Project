import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np

import utils


def train_batch(net,
                encode_fn,
                training_samples,
                learning_rate,
                question_map,
                display_callback=None,
                callback=None):
    # Given a title and body, return embeddings to use
    # Currently, only use titles
    criterion = nn.MultiMarginLoss()
    cosine_similarity = nn.CosineSimilarity()

    def get_embeddings(title, body):
        return utils.get_embeddings(title)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for i in range(len(training_samples)):
        print i + 1, '/', len(training_samples)
        sample = training_samples[i]
        main_encoded = encode_fn(net, get_embeddings(*question_map[sample.id]))
        similar_question_id = np.random.choice(sample.similar)
        dissimilar_question_ids = sample.dissimilar[:20]
        question_ids = [similar_question_id] + dissimilar_question_ids
        similarities = []
        targets = Variable(torch.LongTensor([0]))
        for question_id in question_ids:
            encoded = encode_fn(net, get_embeddings(*question_map[question_id]))
            similarities.append(cosine_similarity(main_encoded.unsqueeze(0), encoded.unsqueeze(0)))
        similarities = torch.cat(similarities).unsqueeze(0)
        optimizer.zero_grad()
        loss = criterion(similarities, targets)
        loss.backward()
        if display_callback is not None:
            display_callback(loss.data[0])
        optimizer.step()
        if callback is not None:
            callback(i)
        
            
        
