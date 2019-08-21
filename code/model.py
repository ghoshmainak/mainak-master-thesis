import logging
import os
import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from keras.constraints import MaxNorm
from word_embed_reader import W2VEmbReader as EmbReader, FastTextEmbReader, GloveEmbedding, MUSEEmbedding, FineTuneEmbed_ortho_procrustes, FineTuneEmbed_cca, FineTuneEmbed_kcca
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
import configuration as config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_model(args, maxlen, vocab):
    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = K.l2_normalize(weight_matrix, axis=-1)
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(w_n.shape[0].value)))
        return args.ortho_reg * reg

    vocab_size = len(vocab)

    if args.emb_name:
        if args.emb_technique == "w2v":
            logger.info("Load {} glove embedding for {}".format(args.lang, config.word_emb_training_type))
            if args.lang == 'en':
                emb_reader = EmbReader(config.emb_dir_en["w2v"].format(config.word_emb_training_type), 
                args.emb_name)
            elif args.lang == 'de':
                emb_reader = EmbReader(config.emb_dir_de["w2v"].format(config.word_emb_training_type), 
                args.emb_name)
            #emb_reader = FineTuneEmbed_kcca('../preprocessed_data/w2v/fine_tuned','w2v_emb_1000k','../preprocessed_data/w2v/full_trained', 'w2v_embedding_300')
        elif args.emb_technique == 'fasttext':
            if args.lang == 'en':
                emb_reader = FastTextEmbReader(config.emb_dir_en["fasttext"].format(config.word_emb_training_type),
                args.emb_name, config.fine_tuned_enabled)
            elif args.lang == 'de':
                emb_reader = FastTextEmbReader(config.emb_dir_de["fasttext"].format(config.word_emb_training_type),
                args.emb_name, config.fine_tuned_enabled)               
            #emb_reader = FineTuneEmbed_ortho_procrustes('../preprocessed_data/fasttext/fine_tuned','fasttext_pre_trained','../preprocessed_data/fasttext/full_trained', 'w2v_embedding_skipgram_300')
        elif args.emb_technique == "glove":
            if args.lang == 'de':
                logger.info('Load german glove embedding')
                emb_reader = GloveEmbedding(config.emb_dir_de["glove"], args.emb_name)
            else:
                logger.info("Load en glove embedding for {}".format(config.word_emb_training_type))
                emb_reader = GloveEmbedding(config.emb_dir_en["glove"].format(config.word_emb_training_type), args.emb_name)
        elif args.emb_technique == "MUSE_supervised":
            emb_reader = MUSEEmbedding(config.emb_dir_biling['supervised'], args.emb_name)
        elif args.emb_technique == "MUSE_unsupervised":
            emb_reader = MUSEEmbedding(config.emb_dir_biling['unsupervised'], args.emb_name)
        aspect_matrix = emb_reader.get_aspect_matrix(args.aspect_size)
        args.aspect_size = emb_reader.aspect_size
        args.emb_dim = emb_reader.emb_dim

    ##### Inputs #####
    sentence_input = Input(shape=(maxlen,), dtype='int32', name='sentence_input')
    neg_input = Input(shape=(args.neg_size, maxlen), dtype='int32', name='neg_input')

    ##### Construct word embedding layer #####
    word_emb = Embedding(vocab_size, args.emb_dim,
                         mask_zero=True, name='word_emb',
                         embeddings_constraint=MaxNorm(10))

    ##### Compute sentence representation #####
    e_w = word_emb(sentence_input)
    y_s = Average()(e_w)
    att_weights = Attention(name='att_weights',
                            W_constraint=MaxNorm(10),
                            b_constraint=MaxNorm(10))([e_w, y_s])
    z_s = WeightedSum()([e_w, att_weights])

    ##### Compute representations of negative instances #####
    e_neg = word_emb(neg_input)
    z_n = Average()(e_neg)

    ##### Reconstruction #####
    p_t = Dense(args.aspect_size)(z_s)
    p_t = Activation('softmax', name='p_t')(p_t)
    r_s = WeightedAspectEmb(args.aspect_size, args.emb_dim, name='aspect_emb',
                            W_constraint=MaxNorm(10),
                            W_regularizer=ortho_reg)(p_t)

    ##### Loss #####
    loss = MaxMargin(name='max_margin')([z_s, z_n, r_s])
    model = Model(inputs=[sentence_input, neg_input], outputs=[loss])

    ### Word embedding and aspect embedding initialization ######
    if args.emb_name:
        logger.info('Initializing word embedding matrix')
        embs = model.get_layer('word_emb').embeddings
        K.set_value(embs, emb_reader.get_emb_matrix_given_vocab(vocab, K.get_value(embs)))
        logger.info('Initializing aspect embedding matrix as centroid of kmean clusters')
        K.set_value(model.get_layer('aspect_emb').W, aspect_matrix)

    return model
