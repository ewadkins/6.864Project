import numpy as np

class Sample:
    
    def __init__(self, id, similar, dissimilar, scores=None):
        self.id = id
        self.similar = similar
        self.dissimilar = dissimilar
        self.scores = scores and map(float, scores)
        self.candidate_map = {}
        for similar_id in similar:
            self.candidate_map[similar_id] = 1
        for dissimilar_id in dissimilar:
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
        return map(lambda x: Sample(*map(lambda (i, y):
                                         y.split() if i != 0 else y, enumerate(x.split('\t')))), samples)

# Returns a dictionary mapping question id's to their (title, body)
def load_corpus(filepath):
    with open(filepath, 'r') as f:
        corpus = [line.strip() for line in f.readlines()]
        corpus = map(lambda x: x.split('\t'), corpus)
        return {x[0]: tuple(x[1:] + ([''] * max(0, 3 - len(x)))) for x in corpus}
    
# Returns a dictionary mapping words to their 200-dimension pre-trained embeddings
def load_embeddings(filepath):
    with open(filepath, 'r') as f:
        embeddings = [line.strip() for line in f.readlines()]
        embeddings = map(lambda x: map(lambda (i, y):
                                       float(y) if i != 0 else y, enumerate(x.split())), embeddings)
        return {x[0]: tuple(x[1:]) for x in embeddings}

# Maps a string of words to an array of word embeddings, shape(num_words, embedding_length)
def get_embeddings(string, embedding_map):
    return np.array(map(lambda x: embedding_map[x],
                        filter(lambda x: x in embedding_map, string.split())))