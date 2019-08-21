filter_word_on = '_filtered'
fasttext_method = 'skipgram'
num_most_common_words = 18000
word_emb_training_type = 'full_trained'
fine_tuned_enabled = False
data_source = {
    "en": '../preprocessed_data/whole_comments{}.txt',
    "de": '../preprocessed_data/german/whole_comments{}.txt',
    "biling": '../preprocessed_data/bilingual/whole_comments{}.txt'
}
vocab_file = {
    "en": '../preprocessed_data/vocab{}',
    "de": '../preprocessed_data/german/vocab{}',
    "biling": '../preprocessed_data/bilingual/vocab{}'
}
emb_dir_en = {
    "w2v": "../preprocessed_data/w2v/{}",
    "fasttext": "../preprocessed_data/fasttext/{}",
    "glove": "../preprocessed_data/glove/{}",
}
emb_dir_de = {
    "glove": "../preprocessed_data/german/glove/fine_tune",
    "w2v": "../preprocessed_data/german/w2v/{}",
    "fasttext": "../preprocessed_data/german/fasttext/{}"
}
emb_dir_biling = {
    "en_2_de_on": True,
    "supervided": "../MUSE_embedding/supervised/en_2_de",
    "unsupervided": "../MUSE_embedding/unsupervised/en_2_de"
}
aspect_file_name = {
    "en": "../post_train_output/{}/{}/epoch_{}{}/aspect_{}_{}.log",
    "de": "../post_train_output/german/{}/{}/epoch_{}{}/aspect_{}_{}.log",
    "biling": "../post_train_output/bilingual/{}{}/epoch_{}/aspect_{}_{}.log"
}
model_param_file = {
    "en": "../post_train_output/{}/{}/epoch_{}{}/model_param",
    "de": "../post_train_output/german/{}/{}/epoch_{}{}/model_param",
    "biling": "../post_train_output/bilingual/{}/epoch_{}{}/model_param"
}
image_path = {
    "en": "../images/{}/{}/epoch_{}{}/{}",
    "de": "../images/german/{}/{}/epoch_{}{}/{}",
    "biling": "../images/bilingual/{}/epoch_{}{}/{}",
    "file_name": "tr_err_vs_k_{}_{}.png"
}
glove_pretrained_emb = {
    'en': '../preprocessed_data/glove.6B/glove.6B.300d.txt',
    'de': '../preprocessed_data/german/glove/pretrained/vectors.txt'
}
glove_fine_tuned_emb = {
    'en': '../preprocessed_data/glove/fine_tuned/fine_tuned_glove_300',
    'de': '../preprocessed_data/german/glove/fine_tune/fine_tuned_glove_300'
}
glove_fine_tuned_vocab = {
    'en': '../preprocessed_data/glove/fine_tuned/vocab.pkl',
    'de': '../preprocessed_data/german/glove/fine_tune/vocab.pkl'
}
glove_fine_tuned_cooccurance = {
    'en': '../preprocessed_data/glove/fine_tuned/cooccurrence.pkl',
    'de': '../preprocessed_data/german/glove/fine_tune/cooccurrence.pkl'
}
garbage_words = ['uhh', 'uhm', '_blank', 'ww', 'www', 'com', 'http', 'umm', 'yep', 'hey',
'hmm', 'e', 'u', 'etc', 'te', 'ki', 'cu', 'en', 'ek', 'mn', 'se',
'pp', 'el', 'ho', 'mi', 'ei', 'su', 'sa', 'na', 'ot', 'ke', 'ft', 'hai', 'sc', 'ne',
'ko', 'mo', 'jo', 'si', 'og', 'basf', 'ei', 'ugh', 'ehp', 'efsa', 'rbst', 'rbgh'
, 'mm', 'html', 'href', 's', 'wa', 'n', 'oh', 'duh', 'ar', 'va']