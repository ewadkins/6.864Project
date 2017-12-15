filepath = 'vectors_stackexchange.txt'
relevant = ['android/corpus.tsv']  # ['askubuntu/text_tokenized.txt', 'android/corpus.tsv']

target = 'pruned_android_vector.txt'

relevant_set = set()
for fp in relevant:
    with open(fp, 'r') as f:
        for line in f:
            tmp = map(lambda x: x.split(), filter(bool, line.split('\t')[1:]))
            relevant_set |= set([word.lower() for sublist in tmp for word in sublist])


with open(filepath, 'r') as f:
    with open(target, 'a') as tf:
        for line in f:
            word = line.split()[0].lower()
            if word in relevant_set:
                tf.write(line.strip() + '\n')
