import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

import utils

#################################################
# Training


#def train(
#        net,
#        encode_fn,
#        training_samples,
#        learning_rate,
#        display_callback=None):
#    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
#    criterion = nn.CosineEmbeddingLoss()
#
#    # Given a title and body, return embeddings to use
#    # Currently, only use titles
#    def get_embeddings(title, body):
#        return utils.get_embeddings(title)
#
#    # nn.train();
#    for i in range(len(training_samples)):
#        sample = training_samples[i]
#        embeddings = get_embeddings(*utils.get_question(sample.id))
#
#        # print
#        print i + 1, '/', len(training_samples)
#        # print title
#
#        candidate_ids = list(sample.candidate_map.keys())
#        random.shuffle(candidate_ids)
#        for candidate_id in candidate_ids:
#            similar_indicator = sample.candidate_map[candidate_id]
#            candidate_title, candidate_body = utils.get_question(candidate_id)
#            candidate_embeddings = get_embeddings(
#                candidate_title, candidate_body)
#
#            encoded = encode_fn(net, embeddings)
#            candidate_encoded = encode_fn(net, candidate_embeddings)
#
#            # Update
#            optimizer.zero_grad()
#            loss = criterion(
#                encoded.unsqueeze(0),
#                candidate_encoded.unsqueeze(0),
#                Variable(
#                    torch.IntTensor(
#                        [similar_indicator])))
#
#            loss.backward()
#            # print loss.data[0]
#            if display_callback is not None:
#                display_callback(loss.data[0])
#            optimizer.step()

'''
def train_batch(
        net,
        encode_fn,
        training_samples,
        learning_rate,
        question_map,
        display_callback=None,
        callback=None):
    
    def cosine_similarity(x1, x2):
        return 1 - F.cosine_embedding_loss(x1, x2, Variable(torch.IntTensor(1)), 0, True)

    # Given a title and body, return embeddings to use
    # Currently, only use titles
    def get_embeddings(title, body):
        return utils.get_embeddings(title)


    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for i in range(len(training_samples)):
        sample = training_samples[i]
        sample_id = sample.id
        sample_embeddings = get_embeddings(*question_map[sample_id])
        sample_encoded = encode_fn(net, sample_embeddings).unsqueeze(0)
        
        print i + 1, '/', len(training_samples)

        for similar_id in sample.similar:
            similar_embeddings = get_embeddings(*question_map[similar_id])
            similar_encoded = encode_fn(net, sample_embeddings).unsqueeze(0)

            optimizer.zero_grad()
            
            losses = []
            targets = []
            for candidate_id in sample.dissimilar:
                sample_encoded = encode_fn(net, sample_embeddings).unsqueeze(0)
                similar_encoded = encode_fn(net, sample_embeddings).unsqueeze(0)
                
                candidate_embeddings = get_embeddings(*question_map[candidate_id])
                candidate_encoded = encode_fn(net, candidate_embeddings).unsqueeze(0)

                val = cosine_similarity(sample_encoded, candidate_encoded) - \
                        cosine_similarity(sample_encoded, similar_encoded)
                    
                #val = F.margin_ranking_loss(cosine_similarity(sample_encoded, candidate_encoded),
                #                            cosine_similarity(sample_encoded, similar_encoded),
                #                                              Variable(torch.FloatTensor([-1])))
                losses.append(val)
                #targets.append()
            
            #loss = F.multi_margin_loss(torch.cat(losses))
            loss = max(losses, key=lambda x: x.data[0])
            #print similar_id
            #print loss
            loss *= -1

            # Update
            loss.backward()
            optimizer.step()
            if display_callback is not None:
                display_callback(loss.data[0])
            if callback is not None:
                callback(i)
'''













        
            
'''
# def train_batch_adverserial(
#        net,
#        encode_fn,
#        training_samples,
#        learning_rate,
#        askubuntu_question_map,
#        android_question_map,
#        display_callback=None,
#        callback=None):
#    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
#    criterion = nn.CosineEmbeddingLoss()
#
#    # Given a title and body, return embeddings to use
#    # Currently, only use titles
#    def get_embeddings(title, body):
#        return utils.get_embeddings(title) + utils.get_embeddings(body)
#
#    # nn.train();
#    for i in range(len(training_samples)):
#        sample = training_samples[i]
#        embeddings = get_embeddings(*askubuntu_question_map[sample.id])
#
#        # print
#        print i + 1, '/', len(training_samples)
#        # print title
#
#        encoded_var = encode_fn(net, embeddings).unsqueeze(0)
#
#        candidate_ids = list(sample.candidate_map.keys())
#        random.shuffle(candidate_ids)
#        encoded = []
#        candidate_encoded = []
#        similar_indicators = []
#        for candidate_id in candidate_ids:
#            candidate_title, candidate_body =\
#               askubuntu_question_map[candidate_id]
#            candidate_embeddings = get_embeddings(
#                candidate_title, candidate_body)
#            if len(candidate_embeddings) != 0:  # NOTE: Probably do something
#                                                # else.
#                encoded.append(encoded_var)
#                candidate_encoded.append(
#                    encode_fn(net, candidate_embeddings).unsqueeze(0))
#                similar_indicators.append(sample.candidate_map[candidate_id])
#
#        # Update
#        encoded = torch.cat(encoded)
#        candidate_encoded = torch.cat(candidate_encoded)
#        similar_indicators = Variable(torch.IntTensor(similar_indicators))
#        optimizer.zero_grad()
#        loss = criterion(
#            encoded,
#            candidate_encoded,
#            similar_indicators)
#
#        loss.backward()
#        # print loss.data[0]
#        if display_callback is not None:
#            display_callback(loss.data[0])
#        if callback is not None:
#            callback(i)
#        optimizer.step()
'''
