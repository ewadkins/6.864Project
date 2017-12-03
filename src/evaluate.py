
#################################################
# Evaluation

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
    for i in filter(lambda i: results[i] in relevant, list(range(len(results)))):
        total_precision += precision_at_k(sample, results, i+1)
    return total_precision / len(relevant)

def mean_fn(samples, results_matrix, fn, *varargs):
    x = map(lambda (s, r): fn(s, r, *varargs), zip(samples, results_matrix))
    return sum(x) / len(x)

# samples: a length-n list Sample objects
# results: a length-n list of lists, where the inner lists contain the ids of candidation questions ranked in order of similarity

def mean_reciprocal_rank(samples, results_matrix):
    return mean_fn(samples, results_matrix, reciprocal_rank)

def mean_precision_at_k(samples, results_matrix, k):
    return mean_fn(samples, results_matrix, precision_at_k, k)

def mean_average_precision(samples, results_matrix):
    return mean_fn(samples, results_matrix, average_precision)


#training_samples = utils.load_samples('../data/train_random.txt')

#samples = [training_samples[0], training_samples[1]]
#results = [[training_samples[0].dissimilar[0]] + training_samples[0].similar + \
#                training_samples[0].dissimilar[1:],
#           training_samples[1].similar + training_samples[1].dissimilar]

#print mean_average_precision(samples, results)