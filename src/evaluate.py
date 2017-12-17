import torch
import torch.nn as nn
from torch.autograd import Variable
import sys

import utils
import encode
import meter


def evaluate_model(net, encode, samples, question_map):
    samples = filter(lambda s: len(s.similar) > 0, samples)
    criterion = nn.CosineSimilarity()

    print
    print 'Evaluating',
    results_matrix = []
    scores_matrix = []
    for i in range(len(samples)):
        sample = samples[i]
        encoded = encode(net, sample.id, question_map)

        sys.stdout.write('.')
        sys.stdout.flush()

        results = []
        for candidate_id in sample.candidate_map:
            similar_indicator = sample.candidate_map[candidate_id]
            candidate_title, candidate_body = question_map[candidate_id]

            candidate_encoded = encode(net, candidate_id, question_map)

            # Compare similarity
            similarity = criterion(
                encoded.unsqueeze(0),
                candidate_encoded.unsqueeze(0)).data[0]
            results.append((1.0 - similarity, candidate_id))

        results.sort()
        scores = map(lambda x: 1.0 - x[0], results)
        results = map(lambda x: x[1], results)
        results_matrix.append(results)
        scores_matrix.append(scores)

    MAP = mean_average_precision(samples, results_matrix)
    MRR = mean_reciprocal_rank(samples, results_matrix)
    MPK1 = mean_precision_at_k(samples, results_matrix, 1)
    MPK5 = mean_precision_at_k(samples, results_matrix, 5)
    MAUC = mean_area_under_curve(samples, results_matrix)
    AUC05 = area_under_curve_fpr(samples, results_matrix, scores_matrix, 0.05)

    print
    print 'MAP:', MAP
    print 'MRR:', MRR
    print 'MP@1:', MPK1
    print 'MP@5:', MPK5
    print 'MAUC:', MAUC
    print 'AUC(0.05):', AUC05
    print

    return MAP, MRR, MPK1, MPK5, MAUC, AUC05


def evaluate_directly(samples, encode_fn, question_map, util_map):
    samples = filter(lambda s: len(s.similar) > 0, samples)
    criterion = nn.CosineEmbeddingLoss()

    # Given a title and body, return embeddings to use
    # Currently, only use titles
    def transform(title, body):
        return title + ' ' + body

    print
    print 'Evaluating',
    results_matrix = []
    scores_matrix = []
    for i in range(len(samples)):
        sample = samples[i]
        sample_text = transform(*question_map[sample.id])
        # print i + 1, '/', len(samples)
        sys.stdout.write('.')
        sys.stdout.flush()

        results = []
        for candidate_id in sample.candidate_map:
            similar_indicator = sample.candidate_map[candidate_id]
            candidate_title, candidate_body = question_map[candidate_id]
            candidate_text = transform(candidate_title, candidate_body)

            encoded = encode_fn(sample_text, util_map)
            candidate_encoded = encode_fn(candidate_text, util_map)

            # Compare difference
            difference = criterion(
                encoded.unsqueeze(0),
                candidate_encoded.unsqueeze(0),
                Variable(
                    torch.IntTensor(
                        [1]))).data[0]
            results.append((difference, candidate_id))

        results.sort()
        scores = map(lambda x: 1.0 - x[0], results)
        results = map(lambda x: x[1], results)
        results_matrix.append(results)
        scores_matrix.append(scores)

    MAP = mean_average_precision(samples, results_matrix)
    MRR = mean_reciprocal_rank(samples, results_matrix)
    MPK1 = mean_precision_at_k(samples, results_matrix, 1)
    MPK5 = mean_precision_at_k(samples, results_matrix, 5)
    MAUC = mean_area_under_curve(samples, results_matrix)
    AUC05 = area_under_curve_fpr(samples, results_matrix, scores_matrix, 0.05)

    print
    print 'MAP:', MAP
    print 'MRR:', MRR
    print 'MP@1:', MPK1
    print 'MP@5:', MPK5
    print 'MAUC:', MAUC
    print 'AUC(0.05):', AUC05
    print

    return MAP, MRR, MPK1, MPK5, MAUC, AUC05


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


def area_under_curve(sample, results):
    index_map = {}
    for i in reversed(range(len(results))):
        index_map[results[i]] = i
    count = 0
    for pos in sample.similar:
        for neg in sample.dissimilar:
            if pos in index_map and neg in index_map and index_map[pos] <\
                    index_map[neg]:
                count += 1
    return count and float(count) / (len(sample.similar)
                                     * len(sample.dissimilar))


def area_under_curve_fpr(samples, results_matrix, scores_matrix, max_fpr):
    auc_meter = meter.AUCMeter()
    for i in range(len(samples)):
        sample = samples[i]
        results = results_matrix[i]
        scores = scores_matrix[i]
        relevant = set(sample.similar)
        targets = [
            1 if results[i] in relevant else 0 for i in range(
                len(results))]
        auc_meter.add(torch.FloatTensor(scores), torch.LongTensor(targets))
    return auc_meter.value(max_fpr)


def area_under_curve_fpr2(sample, results, scores, max_fpr):
    relevant = set(sample.similar)
    auc_meter = meter.AUCMeter()
    targets = [1 if results[i] in relevant else 0 for i in range(len(results))]
    auc_meter.add(torch.FloatTensor(scores), torch.LongTensor(targets))
    print auc_meter.value(max_fpr)
    return auc_meter.value(max_fpr)


def mean_fn(samples, results_matrix, fn, *varargs):
    x = map(lambda s_r: fn(s_r[0], s_r[1], *varargs),
            zip(samples, results_matrix))
    return sum(x) / len(x)


def mean_fn2(samples, results_matrix, scores_matrix, fn, *varargs):
    x = map(lambda s_r: fn(s_r[0], s_r[1], s_r[2], *varargs),
            zip(samples, results_matrix, scores_matrix))
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


def mean_area_under_curve(samples, results_matrix):
    return mean_fn(samples, results_matrix, area_under_curve)


def mean_area_under_curve_fpr(samples, results_matrix, scores_matrix, max_fpr):
    return mean_fn2(
        samples,
        results_matrix,
        scores_matrix,
        area_under_curve_fpr2,
        max_fpr)


