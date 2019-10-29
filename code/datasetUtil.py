from os import listdir
from os.path import isfile, join
import json
import os,fnmatch
from pandas.io.json import json_normalize
import util

def find_files(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def hasDatasetComments(filePath):
    with open(filePath) as f:
        data = json.load(f)
    pd_data = json_normalize(data, record_path="comments")
    return not pd_data.empty

def whichDatasetHasNoComment(dirName):
    noCommentFileName=[]
    for file in listdir(dirName):
        path=join(dirName,file)
        if isfile(path) and not hasDatasetComments(path):
            noCommentFileName.append(file)
    return noCommentFileName


def writeDataStattoFile(file_path, outF, tc, rc):
    lang = 'English'
    if 'german' in file_path:
        lang = 'German'
    path_slice = file_path.split('/')
    file_name = path_slice[-1]
    source_type = path_slice[-2]
    data_type = path_slice[-3]
    outF.write('{}, {}, {}, {}, {}, {}\n'.format(lang, data_type, source_type, file_name, tc, rc))


def getDataStat_v1(inputFile, outF):
    READ_PATH = inputFile
    if not util.doesPathExist(READ_PATH):
        print("Input file does not exist")
        return
    with open(READ_PATH) as f:
        data = json.load(f)
    pd_data = json_normalize(data, record_path="comments", meta=[
                             'article_source', 'resource_type', 'relevant'])
    rel_pd_data = pd_data[pd_data.relevant == 1]
    writeDataStattoFile(READ_PATH, outF, len(pd_data), len(rel_pd_data))


def getDataStatGivenFolder(folder_name, outputFile):
    no_comment = []
    outF = open(outputFile, 'a')
    outF.write('Data Language,Type,Source Type,File Name,Total Comment,Relevant Comment\n')
    for file in find_files('*', folder_name):
        if hasDatasetComments(file):
            getDataStat_v1(file, outF)
        else:
            no_comment.append(file)
            writeDataStattoFile(file, outF, 0, 0)
    outF.close()