import logging
import os
import re
import numpy as np
import gensim
from sklearn.cluster import KMeans
from gensim.models.fasttext import FastText as FT_gensim
import pickle
from scipy.spatial import procrustes
from sklearn.cross_decomposition import CCA
import rccaMod

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class W2VEmbReader:
    def __init__(self, data_path, emb_name):
        self.data_path = data_path
        emb_path = os.path.join(data_path, emb_name)
        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []
        model = gensim.models.KeyedVectors.load(emb_path)
        self.emb_dim = model.vector_size
        for word in model.wv.vocab:
            self.embeddings[word] = list(model[word])
            emb_matrix.append(list(model[word]))

        # if emb_dim != None:
        #     assert self.emb_dim == len(self.embeddings['nice'])

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)
        self.aspect_size = None
        logger.info('#vectors: %i, #dimensions: %i' %
                    (self.vector_size, self.emb_dim))

    def get_emb_given_word(self, word):
        try:
            return self.embeddings[word]
        except KeyError:
            return None

    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.items():
            try:
                emb_matrix[index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass

        logger.info(
            '%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100 * counter / len(vocab)))
        # L2 normalization
        norm_emb_matrix = emb_matrix / \
            np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        return norm_emb_matrix

    def get_aspect_matrix(self, n_clusters=0):
        self.aspect_size = n_clusters
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        km_aspects = km.cluster_centers_
        aspects = km_aspects
        # L2 normalization
        norm_aspect_matrix = aspects / \
            np.linalg.norm(aspects, axis=-1, keepdims=True)
        return norm_aspect_matrix

    def get_emb_dim(self):
        return self.emb_dim


class FastTextEmbReader(W2VEmbReader):
    def __init__(self, data_path, emb_name, pre_train=False):
        self.data_path = data_path
        emb_path = os.path.join(data_path, emb_name)
        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []
        if pre_train:
            model = gensim.models.KeyedVectors.load(emb_path)
        else:
            model = FT_gensim.load(emb_path)
        self.emb_dim = model.vector_size
        for word in model.wv.vocab:
            self.embeddings[word] = list(model[word])
            emb_matrix.append(list(model[word]))

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)
        self.aspect_size = None
        logger.info('#vectors: %i, #dimensions: %i' %
                    (self.vector_size, self.emb_dim))


class GloveEmbedding(W2VEmbReader):
    def __init__(self, data_path, emb_name):
        self.data_path = data_path
        emb_path = os.path.join(data_path, emb_name)
        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []

        model = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=False)
        self.emb_dim = model.vector_size
        for word in model.wv.vocab:
            self.embeddings[word] = list(model[word])
            emb_matrix.append(list(model[word]))

        # if emb_dim != None:
        #     assert self.emb_dim == len(self.embeddings['nice'])

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)
        self.aspect_size = None
        logger.info('#vectors: %i, #dimensions: %i' %
                    (self.vector_size, self.emb_dim))


class MUSEEmbedding(W2VEmbReader):
    def __init__(self, data_path, emb_name):
        self.data_path = data_path
        emb_path = os.path.join(data_path, emb_name)
        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []

        model = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=False)
        self.emb_dim = model.vector_size
        for word in model.wv.vocab:
            self.embeddings[word] = list(model[word])
            emb_matrix.append(list(model[word]))

        # if emb_dim != None:
        #     assert self.emb_dim == len(self.embeddings['nice'])

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)
        self.aspect_size = None
        logger.info('#vectors: %i, #dimensions: %i' %
                    (self.vector_size, self.emb_dim))


class FineTuneEmbed_ortho_procrustes(W2VEmbReader):
    def __init__(self, pre_train_data_path, pre_train_emb_name, full_train_data_path, full_train_emb_name):
        #self.data_path = data_path
        hypbrid_embed = os.path.join(pre_train_data_path, 'hybrid_embed_ortho_procrsutes')
        emb_matrix = []
        if os.path.exists(hypbrid_embed) and os.path.isfile(hypbrid_embed):
            logger.info('Loading hybrid embeddings for fine tuning from: ' + hypbrid_embed)
            fin = open(hypbrid_embed, 'rb')
            info = pickle.load(fin)
            self.embeddings = info['embeddings']
            self.emb_dim = info['emb_dim']
            emb_matrix = info['emb_matrix']
        else:
            pre_train_emb_path = os.path.join(pre_train_data_path, pre_train_emb_name)
            logger.info('Loading pre-train embeddings from: ' + pre_train_emb_path)
            full_train_emb_path = os.path.join(full_train_data_path, full_train_emb_name)
            self.embeddings = {}
            pre_train_model = gensim.models.KeyedVectors.load(pre_train_emb_path)
            logger.info('Loading trained embeddings from: ' + full_train_emb_path)

            full_train_model = gensim.models.KeyedVectors.load(full_train_emb_path)
            self.emb_dim = full_train_model.vector_size
            count = 0
            total = 0
            common_pre_train_embedding = []
            common_full_train_embedding = []
            common_word_order = []
            for word in full_train_model.wv.vocab:
                total += 1
                if word in pre_train_model.wv.vocab:
                    count += 1
                    common_word_order.append(word)
                    #self.embeddings[word] = list(pre_train_model[word])
                    #emb_matrix.append(list(pre_train_model[word]))
                    common_pre_train_embedding.append(list(pre_train_model[word]))
                    common_full_train_embedding.append(list(full_train_model[word]))
                #else:
                #    self.embeddings[word] = list(full_train_model[word])
                #    emb_matrix.append(list(full_train_model[word]))
            logger.info('hit: {}'.format(count/total))
            common_pre_train_embedding = np.asarray(common_pre_train_embedding)
            common_full_train_embedding = np.asarray(common_full_train_embedding)
            common_full_train_embedding, common_pre_train_embedding, disparity = procrustes(common_full_train_embedding,common_pre_train_embedding)
            logger.info('disparity: {}'.format(round(disparity)))
            emb_matrix = common_pre_train_embedding
            for i, word in enumerate(common_word_order):
                self.embeddings[word] = common_pre_train_embedding[i]
            fout = open(hypbrid_embed, 'wb')
            info = {}
            info['emb_dim'] = self.emb_dim
            info['embeddings'] = self.embeddings
            info['emb_matrix'] = emb_matrix
            pickle.dump(info, fout)
            fout.close()
            # if emb_dim != None:
            #     assert self.emb_dim == len(self.embeddings['nice'])

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)
        self.aspect_size = None
        logger.info('#vectors: %i, #dimensions: %i' %
                    (self.vector_size, self.emb_dim))


class FineTuneEmbed_cca(W2VEmbReader):
    def __init__(self, pre_train_data_path, pre_train_emb_name, full_train_data_path, full_train_emb_name):
        #self.data_path = data_path
        hypbrid_embed = os.path.join(pre_train_data_path, 'hybrid_embed_init_cca')
        emb_matrix = []
        if os.path.exists(hypbrid_embed) and os.path.isfile(hypbrid_embed):
            logger.info('Loading hybrid embeddings for fine tuning from: ' + hypbrid_embed)
            fin = open(hypbrid_embed, 'rb')
            info = pickle.load(fin)
            self.embeddings = info['embeddings']
            self.emb_dim = info['emb_dim']
            emb_matrix = info['emb_matrix']
        else:
            pre_train_emb_path = os.path.join(pre_train_data_path, pre_train_emb_name)
            logger.info('Loading pre-train embeddings from: ' + pre_train_emb_path)
            full_train_emb_path = os.path.join(full_train_data_path, full_train_emb_name)
            self.embeddings = {}
            pre_train_model = gensim.models.KeyedVectors.load(pre_train_emb_path)
            logger.info('Loading trained embeddings from: ' + full_train_emb_path)

            full_train_model = gensim.models.KeyedVectors.load(full_train_emb_path)
            self.emb_dim = full_train_model.vector_size
            count = 0
            total = 0
            common_pre_train_embedding = []
            common_full_train_embedding = []
            common_word_order = []
            for word in full_train_model.wv.vocab:
                total += 1
                if word in pre_train_model.wv.vocab:
                    count += 1
                    common_word_order.append(word)
                    #self.embeddings[word] = list(pre_train_model[word])
                    #emb_matrix.append(list(pre_train_model[word]))
                    common_pre_train_embedding.append(list(pre_train_model[word]))
                    common_full_train_embedding.append(list(full_train_model[word]))
                #else:
                #    self.embeddings[word] = list(full_train_model[word])
                #    emb_matrix.append(list(full_train_model[word]))
            logger.info('hit: {}'.format(count/total))
            common_pre_train_embedding = np.asarray(common_pre_train_embedding)
            common_full_train_embedding = np.asarray(common_full_train_embedding)
            # Canonical Correlation Analysis
            cca = CCA(n_components=self.emb_dim, max_iter=1000)
            logger.info("CCA fit started")
            common_pre_train_embedding, common_full_train_embedding = cca.fit_transform(common_pre_train_embedding,common_full_train_embedding)
            logger.info('CCA transformation complete')
            emb_matrix = np.add(0.5*common_pre_train_embedding, 0.5*common_full_train_embedding)
            for i, word in enumerate(common_word_order):
                self.embeddings[word] = 0.5*common_pre_train_embedding[i] + 0.5*common_full_train_embedding[i]
            fout = open(hypbrid_embed, 'wb')
            info = {}
            info['emb_dim'] = self.emb_dim
            info['embeddings'] = self.embeddings
            info['emb_matrix'] = emb_matrix
            pickle.dump(info, fout)
            fout.close()
            # if emb_dim != None:
            #     assert self.emb_dim == len(self.embeddings['nice'])

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)
        self.aspect_size = None
        logger.info('#vectors: %i, #dimensions: %i' %
                    (self.vector_size, self.emb_dim))


class FineTuneEmbed_kcca(W2VEmbReader):
    def __init__(self, pre_train_data_path, pre_train_emb_name, full_train_data_path, full_train_emb_name):
        #self.data_path = data_path
        hypbrid_embed = os.path.join(pre_train_data_path, 'hybrid_embed_init_kcca')
        emb_matrix = []
        if os.path.exists(hypbrid_embed) and os.path.isfile(hypbrid_embed):
            logger.info('Loading hybrid embeddings for fine tuning from: ' + hypbrid_embed)
            fin = open(hypbrid_embed, 'rb')
            info = pickle.load(fin)
            self.embeddings = info['embeddings']
            self.emb_dim = info['emb_dim']
            emb_matrix = info['emb_matrix']
        else:
            pre_train_emb_path = os.path.join(pre_train_data_path, pre_train_emb_name)
            logger.info('Loading pre-train embeddings from: ' + pre_train_emb_path)
            full_train_emb_path = os.path.join(full_train_data_path, full_train_emb_name)
            self.embeddings = {}
            pre_train_model = gensim.models.KeyedVectors.load(pre_train_emb_path)
            logger.info('Loading trained embeddings from: ' + full_train_emb_path)

            full_train_model = gensim.models.KeyedVectors.load(full_train_emb_path)
            self.emb_dim = full_train_model.vector_size
            count = 0
            total = 0
            common_pre_train_embedding = []
            common_full_train_embedding = []
            common_word_order = []
            for word in full_train_model.wv.vocab:
                total += 1
                if word in pre_train_model.wv.vocab:
                    count += 1
                    common_word_order.append(word)
                    #self.embeddings[word] = list(pre_train_model[word])
                    #emb_matrix.append(list(pre_train_model[word]))
                    common_pre_train_embedding.append(list(pre_train_model[word]))
                    common_full_train_embedding.append(list(full_train_model[word]))
                #else:
                #    self.embeddings[word] = list(full_train_model[word])
                #    emb_matrix.append(list(full_train_model[word]))
            logger.info('hit: {}'.format(count/total))
            common_pre_train_embedding = np.asarray(common_pre_train_embedding)
            common_full_train_embedding = np.asarray(common_full_train_embedding)
            # Kernel Canonical Correlation Analysis
            cca = rccaMod.CCA(reg=0.01, numCC=self.emb_dim, kernelcca=True, ktype="gaussian")
            logger.info("KCCA fit started")
            cancomps = cca.train([common_pre_train_embedding, common_full_train_embedding]).comps
            common_pre_train_embedding = cancomps[0]
            common_full_train_embedding = cancomps[1]
            logger.info('KCCA transformation complete')
            emb_matrix = np.add(0.25*common_pre_train_embedding, 0.75*common_full_train_embedding)
            for i, word in enumerate(common_word_order):
                self.embeddings[word] = 0.25*common_pre_train_embedding[i] + 0.75*common_full_train_embedding[i]
            fout = open(hypbrid_embed, 'wb')
            info = {}
            info['emb_dim'] = self.emb_dim
            info['embeddings'] = self.embeddings
            info['emb_matrix'] = emb_matrix
            pickle.dump(info, fout)
            fout.close()
            # if emb_dim != None:
            #     assert self.emb_dim == len(self.embeddings['nice'])

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)
        self.aspect_size = None
        logger.info('#vectors: %i, #dimensions: %i' %
                    (self.vector_size, self.emb_dim))


def length_normalize(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, np.newaxis]
    return matrix


def mean_center(matrix):
    avg = np.mean(matrix, axis=0)
    matrix -= avg
    return matrix


class FineTuneEmbed_procrsutes_expand(W2VEmbReader):
    def __init__(self, pre_train_data_path, pre_train_emb_name, full_train_data_path, full_train_emb_name):
        #self.data_path = data_path
        hypbrid_embed = os.path.join(pre_train_data_path, 'hybrid_embed_ortho_procrsutes_vecmap')
        emb_matrix = []
        if os.path.exists(hypbrid_embed) and os.path.isfile(hypbrid_embed):
            logger.info('Loading hybrid embeddings for fine tuning from: ' + hypbrid_embed)
            fin = open(hypbrid_embed, 'rb')
            info = pickle.load(fin)
            self.embeddings = info['embeddings']
            self.emb_dim = info['emb_dim']
            emb_matrix = info['emb_matrix']
        else:
            pre_train_emb_path = os.path.join(pre_train_data_path, pre_train_emb_name)
            logger.info('Loading pre-train embeddings from: ' + pre_train_emb_path)
            full_train_emb_path = os.path.join(full_train_data_path, full_train_emb_name)
            self.embeddings = {}
            pre_train_model = gensim.models.KeyedVectors.load(pre_train_emb_path)
            logger.info('Loading trained embeddings from: ' + full_train_emb_path)

            full_train_model = gensim.models.KeyedVectors.load(full_train_emb_path)
            logger.info('get embedding matrix for pre trained')
            pre_trained_word = {}
            pre_trained_vector = []
            index = 0
            for word in pre_train_model.wv.vocab:
                pre_trained_word[word] = index
                index += 1
                pre_trained_vector.append(list(pre_train_model[word]))
            pre_trained_vector = np.asarray(pre_trained_vector)
            pre_trained_vector = length_normalize(pre_trained_vector)
            pre_trained_vector = mean_center(pre_trained_vector)
            logger.info('get embedding matrix for fully trained')
            full_trained_word = []
            full_trained_vector = []
            for word in full_train_model.wv.vocab:
                full_trained_word.append(word)
                full_trained_vector.append(list(full_train_model[word]))
            full_trained_vector = np.asarray(full_trained_vector)
            full_trained_vector = length_normalize(full_trained_vector)
            full_trained_vector = mean_center(full_trained_vector)
            self.emb_dim = full_train_model.vector_size
            count = 0
            total = 0
            common_pre_train_embedding = []
            common_full_train_embedding = []
            common_word_order = []
            for i, word in enumerate(full_trained_word):
                total += 1
                if word in pre_trained_word:
                    count += 1
                    common_word_order.append(word)
                    #self.embeddings[word] = list(pre_train_model[word])
                    #emb_matrix.append(list(pre_train_model[word]))
                    j = pre_trained_word[word]
                    common_pre_train_embedding.append(pre_trained_vector[j])
                    common_full_train_embedding.append(full_trained_vector[i])
                #else:
                #    self.embeddings[word] = list(full_train_model[word])
                #    emb_matrix.append(list(full_train_model[word]))
            logger.info('hit: {}'.format(count/total))
            common_pre_train_embedding = np.asarray(common_pre_train_embedding)
            common_full_train_embedding = np.asarray(common_full_train_embedding)
            u, s, vt = np.linalg.svd(common_full_train_embedding.T.dot(common_pre_train_embedding))
            w = vt.T.dot(u.T)
            common_pre_train_embedding.dot(w, out=common_pre_train_embedding)
            emb_matrix = common_pre_train_embedding
            exclusive_full_trained_word = []
            exclusive_full_trained_vector = []
            for i, word in enumerate(full_trained_word):
                if word not in common_word_order:
                    exclusive_full_trained_word.append(word)
                    exclusive_full_trained_vector.append(full_trained_vector[i])
            exclusive_full_trained_vector = np.asarray(exclusive_full_trained_vector)
            emb_matrix = np.concatenate((emb_matrix, exclusive_full_trained_vector), axis=0)
            for i, word in enumerate(common_word_order):
                self.embeddings[word] = common_pre_train_embedding[i]
            for i, word in enumerate(exclusive_full_trained_word):
                self.embeddings[word] = exclusive_full_trained_vector[i]
            fout = open(hypbrid_embed, 'wb')
            info = {}
            info['emb_dim'] = self.emb_dim
            info['embeddings'] = self.embeddings
            info['emb_matrix'] = emb_matrix
            pickle.dump(info, fout)
            fout.close()
            # if emb_dim != None:
            #     assert self.emb_dim == len(self.embeddings['nice'])

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)
        self.aspect_size = None
        logger.info('#vectors: %i, #dimensions: %i' %
                    (self.vector_size, self.emb_dim))
