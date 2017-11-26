import utils

print 'Loading training samples..'
training_samples = utils.load_data('../data/train_random.txt')
print len(training_samples)

print 'Loading dev samples..'
dev_samples = utils.load_data('../data/dev.txt')
print len(dev_samples)

print 'Loading test samples..'
test_samples = utils.load_data('../data/test.txt')
print len(test_samples)

print

print training_samples[0]