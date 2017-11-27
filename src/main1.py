import utils

print 'Loading training samples..'
training_samples = utils.load_samples('../data/train_random.txt')
print len(training_samples)

print 'Loading dev samples..'
dev_samples = utils.load_samples('../data/dev.txt')
print len(dev_samples)

print 'Loading test samples..'
test_samples = utils.load_samples('../data/test.txt')
print len(test_samples)

print 'Loading corpus..'
question_map = utils.load_corpus('../data/text_tokenized.txt')
print len(question_map)

print 'Loading embeddings..'
embedding_map = utils.load_embeddings('../data/vectors_pruned.200.txt')
print len(embedding_map)

print

print training_samples[0]

title, body = question_map[training_samples[0].id]
print
print title
print
print body