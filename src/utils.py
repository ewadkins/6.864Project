import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class Sample:

    def __init__(self, id, similar, dissimilar, scores=None):
        self.id = id
        self.similar = similar
        self.dissimilar = dissimilar
        self.scores = scores and map(float, scores)
        self.candidate_map = {}
        for similar_id in self.similar:
            self.candidate_map[similar_id] = 1
        for dissimilar_id in self.dissimilar:
            self.candidate_map[dissimilar_id] = -1

    def __repr__(self):
        return '{' + 'id=' + str(self.id) + \
            ', similar=' + str(self.similar) + \
            ', dissimilar=' + str(self.dissimilar) + \
            (', scores=' + str(self.scores) if self.scores else '') + \
            '}'

# Returns Samples from the given filepath


def load_samples(filepath):
    with open(filepath, 'r') as f:
        samples = [line.strip() for line in f.readlines()]
        return map(
            lambda x: Sample(
                *map(
                    lambda i_y: i_y[1].split() if i_y[0] != 0 else i_y[1],
                    enumerate(
                        x.split('\t')))),
            samples)


def load_samples_stupid_format(pos_filepath, neg_filepath):
    pos_map = {}
    neg_map = {}
    with open(pos_filepath, 'r') as f:
        content = [line.strip().lower() for line in f.readlines()]
        for pair in map(lambda x: tuple(x.split()), content):
            pos_map[pair[0]] = pos_map.get(pair[0], []) + [pair[1]]
    with open(neg_filepath, 'r') as f:
        content = [line.strip().lower() for line in f.readlines()]
        for pair in map(lambda x: tuple(x.split()), content):
            neg_map[pair[0]] = neg_map.get(pair[0], []) + [pair[1]]

    assert list(sorted(pos_map.keys())) == list(sorted(neg_map.keys()))
    return map(
        lambda x: Sample(
            x, pos_map[x], neg_map[x]), list(
            sorted(
                pos_map.keys())))


# Returns a dictionary mapping question id's to their (title, body)


def load_corpus(filepath):
    with open(filepath, 'r') as f:
        corpus = [line.strip().lower() for line in f.readlines()]
        corpus = map(lambda x: x.split('\t'), corpus)
    return {x[0]: tuple(x[1:] + ([''] * max(0, 3 - len(x))))
                for x in corpus}

# Returns a dictionary mapping words to their 200-dimension pre-trained
# embeddings


def load_embeddings(filepath, corpus_texts, stop_words):
    cv = CountVectorizer(stop_words='english')  # token_pattern=r"(?u)\b\w+\b|!|[.,!?;:()\[\]{}]")  #stop_words='english') #min_df=2, stop_words=stop_words) # token_pattern=r"(?u)\b\w+\b|!|[.,!?;:()\[\]{}]")
    cv.fit(corpus_texts)
    vocabulary = set(cv.get_feature_names())
    with open(filepath, 'r') as f:
        embeddings = [line.strip() for line in f.readlines()]
        embeddings = map(
            lambda x: map(
                lambda i_y1: float(
                    i_y1[1]) if i_y1[0] != 0 else i_y1[1], enumerate(
                    x.split())), embeddings)
        return {x[0]: tuple(x[1:]) for x in embeddings if x[0] in vocabulary}


def load_stop_words(filepath):
    with open(filepath, 'r') as f:
        stop_words = set([line.strip() for line in f.readlines()])
        return stop_words


def store_embedding_map(_embedding_map):
    global embedding_map
    embedding_map = _embedding_map


def store_question_map(_question_map):
    global question_map
    question_map = _question_map

# Maps a string of words to an array of word embeddings, shape(num_words,
# embedding_length)


def get_embeddings(string, _embedding_map=None):
    global embedding_map
    if _embedding_map is None:
        _embedding_map = embedding_map
    return np.array(map(lambda x: _embedding_map[x] if x in _embedding_map else [
        0.0 for _ in range(len(embedding_map['apple']))], string.split()))  # TODO: do not hard code embedding dim here


def get_vocabulary_map(question_map):
    vocabulary = list(set(
        ' '.join([' '.join(
            question_map[id]).lower() for id in question_map]).split()))
    vocabulary_map = {}
    for i in range(len(vocabulary)):
        vocabulary_map[vocabulary[i]] = i
    return vocabulary_map
