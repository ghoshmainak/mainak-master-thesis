from keras.preprocessing import sequence
import reader as dataset
import argparse
import numpy as np
from optimizers import get_optimizer
from model import create_model
import keras.backend as K
import sys
import logging
if 'absl.logging' in sys.modules:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')

import util
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import configuration as config

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def sentence_batch_generator(data, batch_size):
    n_batch = len(data) // batch_size
    batch_count = 0
    np.random.shuffle(data)

    while True:
        if batch_count == n_batch:
            np.random.shuffle(data)
            batch_count = 0

        batch = data[batch_count * batch_size: (batch_count + 1) * batch_size]
        batch_count += 1
        yield batch

def negative_batch_generator(data, batch_size, neg_size):
    data_len = data.shape[0]
    dim = data.shape[1]

    while True:
        indices = np.random.choice(data_len, batch_size * neg_size)
        samples = data[indices].reshape(batch_size, neg_size, dim)
        yield samples

def train_model_each_cluster(args,cluster_size,embtype):
    logger.info("Cluster Size: {}".format(cluster_size))
    args.aspect_size = cluster_size
    if args.seed > 0:
        np.random.seed(args.seed)

    aspect_file_name = config.aspect_file_name[args.lang].format(args.emb_technique, config.word_emb_training_type, args.epochs, config.filter_word_on, embtype, cluster_size)
    model_path = config.model_param_file[args.lang].format(args.emb_technique, config.word_emb_training_type, args.epochs, config.filter_word_on)
    util.createPath(aspect_file_name)

    vocab, train_x, overall_maxlen = dataset.get_data(vocab_size=args.vocab_size,
                                                      maxlen=args.maxlen, lang=args.lang)
    train_x = sequence.pad_sequences(train_x, maxlen=overall_maxlen)
    print('Number of training examples: ', len(train_x))
    print('Length of vocab: ', len(vocab))

    optimizer = get_optimizer(args.algorithm)
    logger.info('Building {} based model for {}'.format(args.emb_technique, args.lang))
    model = create_model(args, overall_maxlen, vocab)
    # freeze the word embedding layer
    model.get_layer('word_emb').trainable = False
    model.compile(optimizer=optimizer, loss=util.max_margin_loss, metrics=[util.max_margin_loss])

    logger.info("-" * 80)

    vocab_inv = {}
    for w, ind in vocab.items():
        vocab_inv[ind] = w

    sen_gen = sentence_batch_generator(train_x, args.batch_size)
    neg_gen = negative_batch_generator(train_x, args.batch_size, args.neg_size)
    batches_per_epoch = len(train_x) // args.batch_size

    min_loss = float('inf')
    for ii in range(args.epochs):
        t0 = time()
        loss, max_margin_loss = 0., 0.

        for b in tqdm(range(batches_per_epoch)):
            sen_input = next(sen_gen)
            neg_input = next(neg_gen)

            batch_loss, batch_max_margin_loss = model.train_on_batch([sen_input, neg_input],
                                                                     np.ones((args.batch_size, 1)))
            loss += batch_loss / batches_per_epoch
            max_margin_loss += batch_max_margin_loss / batches_per_epoch

        tr_time = time() - t0

        if loss < min_loss:
            min_loss = loss
            word_emb = K.get_value(model.get_layer('word_emb').embeddings)
            aspect_emb = K.get_value(model.get_layer('aspect_emb').W)
            word_emb = word_emb / np.linalg.norm(word_emb, axis=-1, keepdims=True)
            aspect_emb = aspect_emb / np.linalg.norm(aspect_emb, axis=-1, keepdims=True)
            aspect_file = open(aspect_file_name, 'wt', encoding='utf-8')
            model.save(model_path)
            for ind in range(len(aspect_emb)):
                desc = aspect_emb[ind]
                sims = word_emb.dot(desc.T)
                ordered_words = np.argsort(sims)[::-1]
                desc_list = [vocab_inv[w] + "|" + str(sims[w]) for w in ordered_words[:50]]
                # print('Aspect %d:' % ind)
                # print(desc_list)
                aspect_file.write('Aspect %d:\n' % ind)
                aspect_file.write(' '.join(desc_list) + '\n\n')

        per_cluster_train_loss = loss

        logger.info('Epoch %d, train: %is' % (ii, tr_time))
        logger.info(
            'Total loss: %.4f, max_margin_loss: %.4f, ortho_reg: %.4f' % (
                loss, max_margin_loss, loss - max_margin_loss))

    return per_cluster_train_loss


def train_model(args, n_clusters_range):
    all_clust_err = []
    if args.emb_technique == "fasttext":
        embtype = config.fasttext_method+"_"+str(args.emb_dim)
    else:
        embtype = args.emb_dim
    for cluster_size in n_clusters_range:
        per_cluster_train_loss = train_model_each_cluster(args, cluster_size, embtype)
        all_clust_err.append(per_cluster_train_loss)
    return all_clust_err


def plot_graph(cluster_range, all_clust_err, args):
    if args.emb_technique == "fasttext":
        embtype = config.fasttext_method+"_"+str(args.emb_dim)
    else:
        embtype = args.emb_dim
    file_name = config.image_path['file_name'].format(embtype, args.clust_range)
    plt.plot(cluster_range, all_clust_err)
    plt.title("Error trend for embedding size {}, embedding algorithm {}".format(args.emb_dim,args.emb_technique))
    plt.xlabel("Number of clusters")
    plt.ylabel("Clustering Error")
    path = config.image_path[args.lang].format(args.emb_technique, config.word_emb_training_type, args.epochs, config.filter_word_on, file_name)
    util.createPath(path)
    plt.savefig(path)
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000,
                        help="Vocab size. '0' means no limit (default=9000)")
    parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=256,
                        help="Maximum allowed number of words during training. '0' means no limit (default=0)")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32,
                        help="Batch size (default=32)")
    parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15,
                        help="Number of epochs (default=15)")
    parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20,
                        help="Number of negative instances (default=20)")
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234,
                        help="Random seed (default=1234)")
    parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam',
                        help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
    parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200,
                        help="Embeddings dimension (default=100)")
    parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14,
                        help="The number of aspects specified by users (default=14)")
    parser.add_argument("--emb-name", type=str,
                        help="The name to the word embeddings file", default="w2v_64k_unigram_100d.model")
    parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1,
                        help="The weight of orthogonal regularization (default=0.1)")
    parser.add_argument("-emb_tech", "--emb_technique", dest="emb_technique", type=str, metavar='<str>', default='w2v',
                        help="embedding technique (w2v or fasttext)")
    parser.add_argument("--c_range", "--c_range", dest="clust_range", type=str, metavar='<str>', default='10-20',
                        help="cluster range")
    parser.add_argument("--cluster_step", "--cluster_step", dest="cluster_step", type=int, metavar='<int>', default='3',
                        help="cluster step")
    parser.add_argument("--lang", "--lang", dest="lang", type=str, metavar='<str>', default='en',
                        help="dataset language")
    args = parser.parse_args()

    scluster_range = args.clust_range.split('-')
    cluster_range = range(int(scluster_range[0]), int(scluster_range[1]), args.cluster_step)

    all_clust_err = train_model(args, cluster_range)
    plot_graph(cluster_range, all_clust_err, args)

if __name__ == "__main__":
    main()