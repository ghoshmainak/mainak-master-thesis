from os import listdir
from os.path import isfile, join
import json
import os,fnmatch
from pandas.io.json import json_normalize

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
