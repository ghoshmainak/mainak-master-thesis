import codecs
import re
import operator
import configuration as config

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def is_number(token):
    return bool(num_regex.match(token))


def create_vocab(maxlen=0, vocab_size=0, lang='en'):
    source = config.data_source[lang].format(config.filter_word_on)

    total_words, unique_words = 0, 0
    word_freqs = {}

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.split()
        if maxlen > 0 and len(words) > maxlen:
            continue

        for w in words:
            if not is_number(w):
                try:
                    word_freqs[w] += 1
                except KeyError:
                    unique_words += 1
                    word_freqs[w] = 1
                total_words += 1

    print('%i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('keep the top %i words' % vocab_size)
    return vocab

"""     # Write (vocab, frequence) to a txt file
    if lang == 'de':
        vocab_file = codecs.open(config.vocab_file['de'].format(config.filter_word_on), mode='w', encoding='utf8')
    else:
        vocab_file = codecs.open(config.vocab_file['en'].format(config.filter_word_on), mode='w', encoding='utf8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word + '\t' + str(0) + '\n')
            continue
        vocab_file.write(word + '\t' + str(word_freqs[word]) + '\n')
    vocab_file.close() """
    


def read_dataset(vocab, maxlen, lang='en'):
    source = config.data_source[lang].format(config.filter_word_on)
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.strip().split()
        if maxlen > 0 and len(words) > maxlen:
            words = words[:maxlen]
        if not len(words):
            continue

        indices = []
        for word in words:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)

    print('<num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return data_x, maxlen_x


def get_data(vocab_size=0, maxlen=0, lang='en'):
    print('Creating vocab ...')
    vocab = create_vocab(maxlen, vocab_size, lang)
    print('Reading dataset ...')
    print('train set')
    train_x, train_maxlen = read_dataset(vocab, maxlen, lang)
    return vocab, train_x, train_maxlen
