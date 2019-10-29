import codecs
import argparse
from numpy import log
import numpy as np
from palmettopy.palmetto import Palmetto
import requests


def commentset(filename):
    comments = []
    fin = codecs.open(filename, 'r', 'utf-8')
    for line in fin:
        line = line.strip()
        if line:
            comments.append(line)
    fin.close()
    return comments


class CoherenceScore:

    def __init__(self, wordlist, commentset):
        self.wordlist = wordlist
        self.commentset = commentset
        self.doc_indx, self.word_not_available_in_doc = self.getDocIndx()

    @classmethod
    def fromFile(cls, wordlist, filename):
        scommentset = commentset(filename)
        return cls(wordlist, scommentset)

    def getWordListGivenCluster(self, cluster_no):
        return self.wordlist[cluster_no]

    def getDocIndxGivenWord(self, word):
        indx_set = []
        for indx, comment in enumerate(self.commentset):
            comment_words = comment.split()
            if word in comment_words:
                indx_set.append(indx)
        return set(indx_set)

    def getUniqueWords(self):
        word_set = []
        for t in self.wordlist:
            word_set += t
        unique_set = set(word_set)
        return unique_set

    def getDocIndx(self):
        doc_indx = {}
        word_not_avail = []
        word_set = self.getUniqueWords()
        for word in word_set:
            doc_indx_set = self.getDocIndxGivenWord(word)
            if len(doc_indx_set):
                doc_indx[word] = doc_indx_set
            else:
                word_not_avail.append(word)
        return doc_indx, word_not_avail

    def getCoherenceScoreGivenCluster(self, cluster_no):
        wordlist_cluster = self.getWordListGivenCluster(cluster_no)
        coherence_score = 0.0
        for indx, wordn in enumerate(wordlist_cluster):
            if indx == 0:
                continue
            for l in range(indx):
                wordl = wordlist_cluster[l]
                if wordl in self.doc_indx and wordn in self.doc_indx:
                    wordn_occ_set = self.doc_indx[wordn]
                    wordl_occ_set = self.doc_indx[wordl]
                    d1 = len(wordl_occ_set)
                    d2 = len(wordl_occ_set.intersection(wordn_occ_set))
                    coherence_score += log((d2+1)/d1)
        return float("{0:.2f}".format(coherence_score))

    def getCoherenceScoreAllClusters(self):
        all_coherence = []
        no_of_clusters = len(self.wordlist)
        for indx in range(no_of_clusters):
            all_coherence.append(self.getCoherenceScoreGivenCluster(indx))
        return no_of_clusters, all_coherence

    def getModelCoherenceScore(self, stat_method='mean'):
        overall_coherence_score = 0.0
        no_of_clusters, all_clust_coherence = self.getCoherenceScoreAllClusters()
        if stat_method == 'mean':
            for value in all_clust_coherence:
                overall_coherence_score += value
            overall_coherence_score = overall_coherence_score/no_of_clusters
        elif stat_method == 'median':
            overall_coherence_score = np.median(all_clust_coherence)
        return float("{0:.2f}".format(overall_coherence_score))


def getWordList(filename):
    wordListArray = []
    fin = codecs.open(filename, 'r', 'utf-8')
    cluster_size = 0
    for line in fin:
        t_line = line.strip()
        if t_line and "Aspect" in t_line:
            cluster_size += 1
        elif t_line:
            tmp = t_line.split()
            wordlist = [t.split("|")[0] for t in tmp]
            if '<unk>' in wordlist:
                wordlist.remove('<unk>')
            if '<pad>' in wordlist:
                wordlist.remove('<pad>')
            if '<num>' in wordlist:
                wordlist.remove('<num>')
            wordListArray.append(wordlist)
    fin.close()
    if cluster_size != len(wordListArray):
        del wordListArray[-1]
    # print(len(wordListArray))
    return wordListArray


def getCoherenceScore(file_name, corpus):
    wordListArray = getWordList(file_name)
    cs = CoherenceScore.fromFile(wordListArray, corpus)
    #cs = NPMI_score.fromFile(wordListArray, corpus)
    cluster_size, cluster_wise_cs = cs.getCoherenceScoreAllClusters()
    return cluster_size, file_name, cluster_wise_cs, cs.getModelCoherenceScore(stat_method='mean'), cs.word_not_available_in_doc


class NPMI_score(CoherenceScore):
    def getCoherenceScoreGivenCluster(self, cluster_no):
        wordlist_cluster = self.getWordListGivenCluster(cluster_no)
        coherence_score = 0.0
        for indx, wordn in enumerate(wordlist_cluster):
            if indx == 0 or wordn not in self.doc_indx:
                continue
            wordn_occ_set = self.doc_indx[wordn]
            dn = len(wordn_occ_set)
            for l in range(indx):
                wordl = wordlist_cluster[l]
                if wordl in self.doc_indx:
                    wordl_occ_set = self.doc_indx[wordl]
                    dl = len(wordl_occ_set)
                    dintersect = len(wordl_occ_set.intersection(wordn_occ_set))
                    if dintersect:
                        coherence_score += (log((dintersect*len(self.commentset))/(dn*dl))/-log(dintersect/len(self.commentset)))
        return float("{0:.2f}".format(coherence_score/(len(wordlist_cluster)*len(wordlist_cluster)-1)))


def getPalmettoCoherenceScoreGivenFile(filename, method):
    wordListArray = getWordList(filename)
    coherence_score_list = []
    cluster_size = 0
    for words in wordListArray:
        payload = {}
        payload["words"] = " ".join(words[0:10])
        r = requests.get("http://palmetto.aksw.org/palmetto-webapp/service/"+method, params=payload)
        if r.ok:
            coherence_score_list.append(float(r.text))
        cluster_size += 1
    return cluster_size, np.mean(coherence_score_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="file_name", type=str, metavar='<str>',
                        help="file name for word list")
    parser.add_argument("-c", "--corpus", dest="corpus", type=str, metavar='<str>',
                        help="corpus file")
    args = parser.parse_args()
    wordListArray = getWordList(args.file_name)
    cs = CoherenceScore.fromFile(wordListArray, args.corpus)
    # print(cs.getWordListGivenCluster(0))
    # print(cs.getUniqueWords())
    # print(cs.getDocIndxGivenWord("kiss"))
    # rint(cs.getDocIndx())
    # print(cs.getCoherenceScoreGivenCluster(0))
    #print("words not avialble in docs: ",cs.word_not_available_in_doc)
    print(cs.getCoherenceScoreAllClusters())
    print(cs.getModelCoherenceScore())


if __name__ == "__main__":
    main()
