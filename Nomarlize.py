import re
from constants import PATH
import pandas as pd
# from underthesea import sent_tokenize

# def getStopWord(file_name):
#     file = pd.read_csv(PATH + file_name)
#     sw = pd.Series(word for word in file['text']).apply(str)
    
#     return sw

# stopWord = getStopWord('stopword.csv')

def normalizeSentence(sentence):
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\b\d+[\.,]', '.', sentence)
    sentence = re.sub(r'\b(?:[1-5]\.\d+|[1-5])\b|\b\d+(\.\d+)?(l|ml|cm|m|mm|km|tr|k)\b', '', sentence)
    sentence = re.sub(r'\b\d+\b', '', sentence)

    # temp_sentence = []
    # for word in sentence.split():
        

    return sentence.lower()

def statusToNumber(sentence):
    if sentence == 'neg':
        return 1
    elif sentence == 'neu':
        return 2
    elif sentence == 'pos':
        return 3


def arrayToStatus(array):
    max_val = array[0]
    status = 1
    for i in range(1, 3):
        if array[i] > max_val:
            max = array[i]
            status = i + 1

    if status == 1:
        return 'neg'
    elif status == 2:
        return 'neu'
    elif status == 3:
        return 'pos'

