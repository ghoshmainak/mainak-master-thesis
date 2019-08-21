from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import util
import string
import spacy
import concurrent.futures
import logging
from util import DataCorrection
from string import digits
import nltk
import configuration as config
import re
nltk.download('stopwords')
nltk.download('wordnet')


class Preprocessor(object):
    def __init__(self, inFileName, outFile, lang='EN'):
        self.lang = lang
        self.inFile = codecs.open(inFileName, 'r', 'utf-8')
        util.createPath(outFile)
        self.outFile = codecs.open(outFile, 'w', 'utf-8')
        if lang == "EN":
            self.lmtzr = WordNetLemmatizer()
            self.stop = stopwords.words('english')
            self.replacement_dict = DataCorrection.replacement_dict
            self.relevant_terms = DataCorrection.relevant_terms
            self.contract = DataCorrection.contraction
        elif lang == "DE":
            self.spacy_model_de = spacy.load('de')
            german_stop = stopwords.words('german')
            self.stop = [self.umlauts(word) for word in german_stop]

    def umlauts(self, text):
        """
        Replace umlauts for a given text

        :param word: text as string
        :return: manipulated text as str
        """

        tempVar = text  # local variable

        # Using str.replace()

        tempVar = tempVar.replace('ä', 'ae')
        tempVar = tempVar.replace('ö', 'oe')
        tempVar = tempVar.replace('ü', 'ue')
        tempVar = tempVar.replace('Ä', 'Ae')
        tempVar = tempVar.replace('Ö', 'Oe')
        tempVar = tempVar.replace('Ü', 'Ue')
        tempVar = tempVar.replace('ß', 'ss')

        return tempVar

    def currency(self, text):
        """
        Removes the currency symbols from the text
        :param text: text as string
        :retrun: manipulated text as string
        """

        tempVar = text  # local variable

        tempVar = tempVar.replace('$', '')
        tempVar = tempVar.replace('€', '')
        tempVar = tempVar.replace('¥', '')
        tempVar = tempVar.replace('₹', '')
        tempVar = tempVar.replace('£', '')

        return tempVar

    def lemmatizer(self, text):
        """
        Lemmetize words using spacy
        :param: text as string
        :return: lemmetized text as string
        """
        sent = []
        doc = self.spacy_model_de(text.strip())
        for word in doc:
            sent.append(word.lemma_)
        return sent

    def reduce_same_letter(self, text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)

    def remove_non_ascii_words(self, word):
        non_ascii_indi = ['\\x', '\\u', '\\U']
        for c in non_ascii_indi:
            if c in word:
                return ""
        word_ = ascii(word)
        for c in non_ascii_indi:
            if c in word_:
                return ""
        return word

    def remove_link(self, text):
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        return text

    def parseSentence_EN(self, line):
        line = line.lower()
        line = self.remove_link(line)
        for key in self.replacement_dict:
            line = line.replace(key, self.replacement_dict[key])
        line = self.reduce_same_letter(line)
        tmp = line.split()
        for key in self.relevant_terms:
            tmp = [self.relevant_terms[key] if key.lower() == word else word for word in tmp]
        for key in self.contract:
            tmp = [self.contract[key] if key.lower() == word else word for word in tmp]
        line = " ".join(tmp)
        lmtzr = WordNetLemmatizer()
        text_token = CountVectorizer().build_tokenizer()(line)
        #print(text_token)
        # remove words having any digit
        text_token = list(filter(lambda x: not any(ch.isdigit() for ch in x), text_token))
        text_rmstop = [i for i in text_token if i not in self.stop]
        text_stem = [self.lmtzr.lemmatize(w) for w in text_rmstop]
        text = [self.remove_non_ascii_words(w) for w in text_stem]
        # remove numeric word
        text = [w for w in text if w not in config.garbage_words]
        return text

    def parseSentence_DE(self, line):
        # line=self.umlauts(line)
        remove_pun = str.maketrans('', '', string.punctuation)
        line_no_pun = line.translate(remove_pun)
        remove_digits = str.maketrans('', '', digits)
        line_no_num = line_no_pun.translate(remove_digits)
        text_rmstop = [word for word in line_no_num.split(
        ) if self.umlauts(word.lower()) not in self.stop]
        text_no_curr = [self.currency(word) for word in text_rmstop]
        return self.lemmatizer(" ".join(text_no_curr))

    def preprocess(self):
        for line in self.inFile:
            if len(line.strip().strip('\n')) > 14:
                line = line.strip()
                if self.lang == 'EN':
                    tokens = self.parseSentence_EN(line)
                elif self.lang == "DE":
                    tokens = self.parseSentence_DE(line)
                if len(tokens) > 0:
                    tmp = ' '.join(tokens)
                    if tmp.strip().strip('\n'):
                        self.outFile.write(tmp.strip().strip('\n') + '\n')

    def preprocess_multiprocess(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            if self.lang == "DE":
                values = executor.map(self.parseSentence_DE, self.inFile)
            elif self.lang == "EN":
                values = executor.map(self.parseSentence_EN, self.inFile)

        for i, s in enumerate(values):
            if i and i % 25000 == 0:
                logging.info('processed {} sentences'.format(i))
                self.outFile.flush()
            if s:
                self.outFile.write(' '.join(s)+'\n')
        logging.info('preprocessing of {} sentences finished!'.format(i))


def preprocessTrain(inFileName, outFile, lang='EN'):
    p = Preprocessor(inFileName, outFile, lang)
    p.preprocess()


def preprocessTrain_mp(inFileName, outFile, lang='EN'):
    p = Preprocessor(inFileName, outFile, lang)
    p.preprocess_multiprocess()
