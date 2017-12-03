import torch
import torch.nn as nn
from torch.autograd import Variable

import utils

#################################################
# Evaluation


def evaluate_model(rnn, encode_fn, samples, k=10):
    samples = filter(lambda s: len(s.similar) > 0, samples)
    criterion = nn.CosineEmbeddingLoss()

    # Given a title and body, return embeddings to use
    # Currently, only use titles
    def get_embeddings(title, body):
        return utils.get_embeddings(title)

    results_matrix = []
    for i in range(len(samples)):
        sample = samples[i]
        embeddings = get_embeddings(*utils.get_question(sample.id))
        if len(embeddings) == 0:
            continue

        print i + 1, '/', len(samples)

        results = []
        for candidate_id in sample.candidate_map:
            similar_indicator = sample.candidate_map[candidate_id]
            candidate_title, candidate_body = utils.get_question(candidate_id)
            candidate_embeddings = get_embeddings(
                candidate_title, candidate_body)
            if len(candidate_embeddings) == 0:
                continue

            encoded = encode_fn(rnn, embeddings)
            candidate_encoded = encode_fn(rnn, candidate_embeddings)

            # Compare similarity
            difference = criterion(
                encoded.unsqueeze(0),
                candidate_encoded.unsqueeze(0),
                Variable(
                    torch.IntTensor(
                        [1]))).data[0]
            results.append((difference, candidate_id))

        results.sort()
        results = map(lambda x: x[1], results)
        # print sample.similar
        # print results
        results_matrix.append(results)
        print 'AP:', average_precision(sample, results)
        print 'RR:', reciprocal_rank(sample, results)
        print 'P@' + str(k) + ':', precision_at_k(sample, results, k)
        print

    MAP = mean_average_precision(samples, results_matrix)
    MRR = mean_reciprocal_rank(samples, results_matrix)
    MPK = mean_precision_at_k(samples, results_matrix, k)

    print
    print 'MAP:', MAP
    print 'MRR:', MRR
    print 'MP@' + str(k) + ':', MPK
    print

    return MAP, MRR, MPK


def reciprocal_rank(sample, results):
    relevant = set(sample.similar)
    for i in range(len(results)):
        if results[i] in relevant:
            return 1.0 / (i + 1)
    return 0


def precision_at_k(sample, results, k):
    relevant = set(sample.similar)
    count = 0
    for i in range(min(k, len(results))):
        if results[i] in relevant:
            count += 1
    return float(count) / k


def average_precision(sample, results):
    relevant = set(sample.similar)
    total_precision = 0.0
    for i in filter(
        lambda i: results[i] in relevant,
        list(
            range(
            len(results)))):
        total_precision += precision_at_k(sample, results, i + 1)
    return total_precision / len(relevant)


def mean_fn(samples, results_matrix, fn, *varargs):
    x = map(lambda s_r: fn(s_r[0], s_r[1], *varargs),
            zip(samples, results_matrix))
    return sum(x) / len(x)

# samples: a length-n list Sample objects
# results: a length-n list of lists, where the inner lists contain the ids
# of candidation questions ranked in order of similarity


def mean_reciprocal_rank(samples, results_matrix):
    return mean_fn(samples, results_matrix, reciprocal_rank)


def mean_precision_at_k(samples, results_matrix, k):
    return mean_fn(samples, results_matrix, precision_at_k, k)


def mean_average_precision(samples, results_matrix):
    return mean_fn(samples, results_matrix, average_precision)


# training_samples = utils.load_samples('../data/train_random.txt')

# samples = [training_samples[0], training_samples[1]]
# results = [[training_samples[0].dissimilar[0]] + training_samples[0].similar
#                + \
#                training_samples[0].dissimilar[1:],
#           training_samples[1].similar + training_samples[1].dissimilar]

# print mean_average_precision(samples, results)