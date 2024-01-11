import re
from underthesea import sent_tokenize

def normalizeSentence(sentence):
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\b\d+[\.,]', '.', sentence)
    sentence = re.sub(r'\b(?:[1-5]\.\d+|[1-5])\b|\b\d+(\.\d+)?(l|ml|cm|m|mm|km|tr|k)\b', '', sentence)
    sentence = re.sub(r'\b\d+\b', '', sentence)

    return sentence.lower()

def statusToNumber(sentence):
    if sentence == 'neg':
        return 1
    elif sentence == 'neu':
        return 2
    elif sentence == 'pos':
        return 3

