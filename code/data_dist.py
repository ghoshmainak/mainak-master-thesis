import argparse
import logging
import numpy as np
from time import time
import util as ut
import codecs
from keras.preprocessing import sequence
import reader as dataset
from model import create_model
import keras.backend as K
from optimizers import get_optimizer
import configuration as config
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200, help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000, help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14, help="The number of aspects specified by users (default=14)")
parser.add_argument("--emb-name", dest="emb_name", type=str, metavar='<str>', help="The path to the word embeddings file")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20, help="Number of negative instances (default=20)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1, help="The weight of orthogonol regularizaiton (default=0.1)")
parser.add_argument("--lang", "--lang", dest="lang", type=str, metavar='<str>', default='en',
                        help="dataset language")
parser.add_argument("-emb_tech", "--emb_technique", dest="emb_technique", type=str, metavar='<str>', default='w2v',
                        help="embedding technique (w2v or fasttext)")
parser.add_argument("-o", "--output", dest="out_file", type=str, metavar='<str>',
                        help="output file name")

args = parser.parse_args()
vocab, train_x, overall_maxlen = dataset.get_data(vocab_size=args.vocab_size,
                                                    maxlen=args.maxlen, lang=args.lang)
train_x = sequence.pad_sequences(train_x, maxlen=overall_maxlen)
print('Number of training examples: ', len(train_x))
print('Length of vocab: ', len(vocab))

optimizer = get_optimizer(args.algorithm)
model = create_model(args, overall_maxlen, vocab)

model_param = config.model_param_file[args.lang].format(args.emb_technique, config.word_emb_training_type, args.epochs, config.filter_word_on)
model.load_weights(model_param)
model.compile(optimizer=optimizer, loss=ut.max_margin_loss, metrics=[ut.max_margin_loss])

vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w

def sentence_batch_generator(data, batch_size):
    n_batch = len(data) // batch_size
    batch_count = 0

    while True:
        if batch_count == n_batch:
            batch_count = 0

        batch = data[batch_count * batch_size: (batch_count + 1) * batch_size]
        batch_count += 1
        yield batch

sen_gen = sentence_batch_generator(train_x, args.batch_size)
train_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()], 
        [model.get_layer('p_t').output])
number_batches = len(train_x)//args.batch_size
dist = {}
for b in range(number_batches):
    batch_x = next(sen_gen)
    aspect_probs = train_fn([batch_x, 0])
    aspect_ids = np.argsort(aspect_probs[0], axis=1)[:,-1]
    for id in aspect_ids:
        if id in dist:
            dist[id] += 1
        else:
            dist[id] = 0
w = csv.writer(open(os.path.join(os.path.dirname(model_param),args.out_file), "w"))
for key, val in dist.items():
    w.writerow([key, val])