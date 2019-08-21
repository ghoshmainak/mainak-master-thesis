import json
from pandas.io.json import json_normalize
import os
import keras.backend as K
import datasetUtil as file_reader
from functools import wraps
import pickle
import shutil
import codecs
import pandas as pd
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def listify(fn):
    """
    Use this decorator on a generator function to make it return a list
    instead.
    """

    @wraps(fn)
    def listified(*args, **kwargs):
        return list(fn(*args, **kwargs))

    return listified


def doesPathExist(path):
    if not os.path.exists(path):
        return False
    return True


def createPath(OUTPUT_PATH):
    directory = os.path.dirname(OUTPUT_PATH)
    if directory != "" and not doesPathExist(directory):
        os.makedirs(directory)


def writeFile4mListValues(outputPath, listValues, mode):
    outF = codecs.open(outputPath, mode, encoding='utf-8')
    for line in listValues:
        outF.write(line+"\n")
    outF.close()


def extractComment(inputFile, outputFile):
    READ_PATH = inputFile
    if not doesPathExist(READ_PATH):
        print("Input file does not exist")
        return
    OUTPUT_PATH = outputFile
    with open(READ_PATH) as f:
        data = json.load(f)
    pd_data = json_normalize(data, record_path="comments", meta=[
                             'article_source', 'resource_type', 'relevant'])
    print("total aticles", len(pd_data))
    rel_pd_data = pd_data[pd_data.relevant == 1]
    print("relevant aticles", len(rel_pd_data))
    comments_data = rel_pd_data['comment_text']
    createPath(OUTPUT_PATH)
    writeFile4mListValues(OUTPUT_PATH, comments_data.values, "a")


def readCSV2df(inputFile):
    if not doesPathExist(inputFile):
        return False
    df = pd.read_csv(inputFile)
    return df


def extractComment4mCSV(inputFile, outputFile):
    df = readCSV2df(inputFile)
    if isinstance(df, pd.DataFrame):
        comments_data = df['Text']
        createPath(outputFile)
        writeFile4mListValues(outputFile, comments_data.values, "w")
    else:
        print("Input file does not exist")


def extractCommentGivenFolder(folder_name, outputFile):
    no_comment = []
    for file in file_reader.find_files('*', folder_name):
        if file_reader.hasDatasetComments(file):
            extractComment(file, outputFile)
        else:
            no_comment.append(file)
    if not len(no_comment):
        print("all comments have been extracted.")
    else:
        print("These files have no comment and all others have been extracted: \n")
        print(no_comment)


@DeprecationWarning
def concatFiles(inFilenames, outfile):
    createPath(outfile)
    with open(outfile, 'w') as outfile:
        for fname in inFilenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    outfile.close()


def conateFileShutil(inFilenames, outfile):
    createPath(outfile)
    fout = codecs.open(outfile, 'w', 'utf-8')
    infile_objects = [codecs.open(infile, 'r', 'utf-8') for infile in inFilenames]
    for fo in infile_objects:
        shutil.copyfileobj(fo, fout)
        fo.close()
    fout.close()


def concat__embeddings(src_files, dest_file, read_1st_line=True, output_format='gensim_text'):
    get_word_nd_emb_size = lambda x: (x.split()[0], x.split()[1])
    fout = codecs.open(dest_file, 'w', 'utf-8')
    if output_format == 'gensim_text':
        if isinstance(src_files, list):
            file_objects = [codecs.open(infile, 'r', 'utf-8') for infile in src_files]
            srcfile_word_nd_emb_size = [get_word_nd_emb_size(fo.readline().strip()) for fo in file_objects]
            num_word, emb_size = 0, 0
            for word_nd_emb_size in srcfile_word_nd_emb_size:
                num_word = num_word + int(word_nd_emb_size[0])
                emb_size = word_nd_emb_size[1]
            fout.write('{} {}'.format(num_word, emb_size)+'\n')
            for fo in file_objects:
                shutil.copyfileobj(fo, fout)
                fo.close()
    fout.close()


def check_words_is_in_line_wrapper(words):
    def check_words_is_in_line(line):
        words_present = False
        for word in words:
            if word in line:
                words_present = True
                break
        return words_present
    return check_words_is_in_line


def filter_data_based_on_words(infile, words):
    output_path = infile.replace('.txt', '')+'_filtered.txt'
    fin = codecs.open(infile, 'r', 'utf-8')
    lines = [line.strip() for line in fin]
    fin.close()
    logger.info('#original_comments: {}'.format(len(lines)))
    filter_fn = check_words_is_in_line_wrapper(words)
    filtered_lines = list(filter(filter_fn, lines))
    logger.info('#filterd_comments: {}'.format(len(filtered_lines)))
    writeFile4mListValues(output_path, filtered_lines, 'w')


def max_margin_loss(_, y_pred):
    return K.mean(y_pred)


class DataCorrection():
    replacement_dict = {
        '!=': ' not equal ',
        '=': ' equal ',
        '_': ' ',
        b'\xc2\xae'.decode(): ' registered_sign ',  # ®
        '\x92': ' ',
        '\x91': ' ',
        '\x96': ' ',
        b'\xe2\x84\xa2'.decode(): ' trademark_sign ',  # ™
        b'\xe2\x80\x90'.decode(): '-',
        '}': ')',
        '{': '(',
        b'\xc2\xb2'.decode(): ' squared ',
        b'\xc2\xa7'.decode(): ' section ',  # §
        b'\xc2\xb0'.decode(): ' degrees ',
        b'\xe2\x80\xa6'.decode(): ' . ',   # …
        '\$': ' dollar ',
        b'\xe2\x82\xac'.decode(): ' euro ',
        '\|': ' , ',
        b'\xc2\xab'.decode(): ' \" ',
        b'\xc2\xbb'.decode(): ' \" ',
        '\+': ' plus ',
        b'\xc2\xa2'.decode(): ' , ',  # ¢
        b'\xe2\x80\x8b'.decode(): ' ',
        '\|': ',',
        b'\xe2\x80\x93'.decode(): '-',  # the long dash  –
        b'\xe2\x80\x94'.decode(): '-',  # another long dash
        '\[': '(',
        '\]': ')',
        '&': ' and ',
        b'\xe2\x80\x9c'.decode(): '\"',
        b'\xe2\x80\x9d'.decode(): '\"',
        b'\xc2\xbd'.decode(): ' half ',
        b'\xc2\xbc'.decode(): ' quarter ',
        b'\xe2\x80\x99'.decode(): '\'',
        b'\xe2\x80\x98'.decode(): '\'',
        b'\xc2\xb4'.decode(): '\'',
        b'\xc2\xb5g'.decode(): ' microgram ',
        '.': ' . ',
        ',': ' , ',
        'because': ' because '
    }
    relevant_terms = {
        'btw': ' by the way ',
        'tl;dr': ' summary ',
        'tbsp': ' table spoon ',
        'imho': ' in my opinion ',
        'imo': ' in my opinion ',
        'oganic': ' organic ',
        'orgainc': ' organic ',
        'tsp': ' tea spoon ',
        'faqs': ' frequently asked questions ',
        'fyi': ' for your information ',
        'pestdicides': ' pesticides ',
        'pestdicide': ' pesticide ',
        'pesiticides': ' pesticides ',
        'ogranic': ' organic ',
        'pestecides': ' pesticides ',
        'nonorganic': ' non organic ',
        'pestcides': ' pesticides ',
        '<3': ' love ',
        ' alot ': ' a lot ',
        'thier': ' their ',
        'breastmilk': ' breast milk ',
        'agribusinesses': ' agricultural businesses ',
        '<a href equal \"': ' ',
        'café': 'cafe',
        'theyre': 'they are',
        'buyorganic': 'buy organic',
        'kinda': 'kind of',
        'wanna': 'want to',
        'u': 'you',
        'bt': 'but',
        'ogm': 'gmo',
        'por': 'portugal',
        'esp': 'spain',
        'dont': 'do not',
        'havent': 'have not',
        'havested': 'harvested',
        'notvhave': 'not have',
        'ihave': 'i have',
        'ilive': 'i live',
        'organic_food': 'organic food',
        'wannabe': 'want to be',
        'jan': 'january',
        'scienceblogs': 'science blogs',
        'oxfordjournals': 'oxford journals',
        'health_and_science': 'health and science',
        'fb': 'facebook',
        'gmf': 'gmo',
        'nogmo': 'no gmo',
        'somewher': 'somewhere',
        'faq': ' frequently asked question ',
        'fud': 'food',
        'th': 'the',
        'wee': 'we',
        'ur': 'your',
        'efford': 'effort',
        'wasnt': 'was not',
        'cuz': 'because',
        'bcuz': 'because',
        'becuz': 'because',
        'soyabean': 'soybean',
        '_have_': 'have',
        'becausr': 'because',
        'becaue': 'because',
        'becaus': 'because',
        'animalagriculture': 'animal agriculture',
        'dunno': 'do not know',
        'fr': 'for',
        'feb': 'february',
        'mar': 'march',
        'apr': 'april',
        'aug': 'august',
        'sep': 'spetember',
        'oct': 'october',
        'nov': 'november',
        'dec': 'december',
        'consumesince': 'consume since'
    }
    contraction = {"I'd": 'I would',
                   "I'll": 'I will',
                   "I'm": 'I am',
                   "I've": 'I have',
                   "aren't": 'are not',
                   "can't": 'cannot',
                   "could've": 'could have',
                   "couldn't": 'could not',
                   "didn't": 'did not',
                   "doesn't": 'does not',
                   "don't": 'do not',
                   'gonna': 'going to',
                   'gotta': 'got to',
                   "hadn't": 'had not',
                   "hasn't": 'has not',
                   "haven't": 'have not',
                   "he'd": 'he would',
                   "he'll": 'he will',
                   "he's": 'he is',
                   "how's": 'how is',
                   "i'm": 'I am',
                   "i've": 'I have',
                   "isn't": 'is not',
                   "it'd": 'it would',
                   "it'll": 'it will',
                   "it's": 'it is',
                   "may've": 'may have',
                   "mayn't": 'may not',
                   "might've": 'might have',
                   "mightn't": 'might not',
                   "must've": 'must have',
                   "mustn't": 'must not',
                   "needn't": 'need not',
                   "o'clock": 'of the clock',
                   "oughtn't": 'ought not',
                   "she'd": 'she would',
                   "she'll": 'she will',
                   "she's": 'she is',
                   "should've": 'should have',
                   "shouldn't": 'should not',
                   "that's": 'that is',
                   "there're": 'there are',
                   "there's": 'there is',
                   "these're": 'these are',
                   "they'd": 'they would',
                   "they'll": 'they will',
                   "they're": 'they are',
                   "they've": 'they have',
                   "this's": 'this is',
                   "those're": 'those are',
                   "wasn't": 'was not',
                   "we'd": 'we would',
                   "we'll": 'we will',
                   "we're": 'we are',
                   "we've": 'we have',
                   "weren't": 'were not',
                   "what'll": 'what will',
                   "what're": 'what are',
                   "what's": 'what is',
                   "what've": 'what have',
                   "when's": 'when is',
                   "where're": 'where are',
                   "where's": 'where is',
                   "which's": 'which is',
                   "who'd": 'who would',
                   "who'll": 'who will',
                   "who're": 'who are',
                   "who's": 'who is',
                   "why's": 'why is',
                   "won't": 'will not',
                   "would've": 'would have',
                   "wouldn't": 'would not',
                   "you'd": 'you would',
                   "you'll": 'you will',
                   "you're": 'you are',
                   "you've": 'you have'}
