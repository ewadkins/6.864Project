
class Sample:
    
    def __init__(self, id, similar, candidates, scores=None):
        self.id = id
        self.similar = similar
        self.candidates = candidates
        self.scores = scores and map(float, scores)
    
    def __repr__(self):
        return '{' + 'id=' + str(self.id) + \
                ', similar=' + str(self.similar) + \
                ', candidates=' + str(self.candidates) + \
                (', scores=' + str(self.scores) if self.scores else '') + \
                '}'

# Returns Samples from the given filepath
def load_data(filepath):
    with open(filepath, 'r') as f:
        samples = [line.strip() for line in f.readlines()]
        return map(lambda x: Sample(*map(lambda (i, y):
                                         y.split() if i != 0 else y, enumerate(x.split('\t')))), samples)
