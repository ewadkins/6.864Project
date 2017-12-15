import utils


def init():
    print 'Loading askubuntu training samples..'
    askubuntu_training_samples = utils.load_samples(
        '../data/askubuntu/train_random.txt')
    print len(askubuntu_training_samples)

    print 'Loading askubuntu dev samples..'
    askubuntu_dev_samples = utils.load_samples('../data/askubuntu/dev.txt')
    print len(askubuntu_dev_samples)

    print 'Loading askubuntu test samples..'
    askubuntu_test_samples = utils.load_samples('../data/askubuntu/test.txt')
    print len(askubuntu_test_samples)

    print 'Loading askubuntu corpus..'
    askubuntu_question_map = utils.load_corpus(
        '../data/askubuntu/text_tokenized.txt')
    print len(askubuntu_question_map)

    print 'Loading android dev samples..'
    android_dev_samples = utils.load_samples_stupid_format(
        '../data/android/dev.pos.txt', '../data/android/dev.neg.txt')
    print len(android_dev_samples)

    print 'Loading android test samples..'
    android_test_samples = utils.load_samples_stupid_format(
        '../data/android/test.pos.txt', '../data/android/test.neg.txt')
    print len(android_test_samples)

    print 'Loading android corpus..'
    android_question_map = utils.load_corpus('../data/android/corpus.tsv')
    print len(android_question_map)
    
    print 'Loading stop words..'
    stop_words = utils.load_stop_words('../data/english_stop_words.txt')
    print len(stop_words)

    corpus_texts = map(lambda (t, b): t + ' ' + b,
                       askubuntu_question_map.values() + android_question_map.values())
    
    print 'Loading embeddings..'
    embedding_map = utils.load_embeddings(
        '../data/pruned_android_vector.txt', corpus_texts, stop_words)  # pruned_askubuntu_android_vector.txt
    print len(embedding_map)
    print

    utils.store_embedding_map(embedding_map)

    return (
        askubuntu_training_samples,
        askubuntu_dev_samples,
        askubuntu_test_samples,
        askubuntu_question_map,
        android_dev_samples,
        android_test_samples,
        android_question_map,
        embedding_map)
