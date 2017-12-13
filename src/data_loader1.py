import utils


def init():
    print 'Loading training samples..'
    training_samples = utils.load_samples('../data/askubuntu/train_random.txt')
    print len(training_samples)

    print 'Loading dev samples..'
    dev_samples = utils.load_samples('../data/askubuntu/dev.txt')
    print len(dev_samples)

    print 'Loading test samples..'
    test_samples = utils.load_samples('../data/askubuntu/test.txt')
    print len(test_samples)

    print 'Loading corpus..'
    question_map = utils.load_corpus('../data/askubuntu/text_tokenized.txt')
    print len(question_map)
    
    print 'Loading stop words..'
    stop_words = utils.load_stop_words('../data/english_stop_words.txt')
    print len(stop_words)
        
    corpus_texts = map(lambda (t, b): t + ' ' + b, question_map.values())

    print 'Loading embeddings..'
    embedding_map = utils.load_embeddings(
        '../data/pruned_askubuntu_android_vector.txt', corpus_texts, stop_words)
    print len(embedding_map)
    print

    utils.store_embedding_map(embedding_map)

    return (training_samples,
            dev_samples, test_samples, question_map, embedding_map)
