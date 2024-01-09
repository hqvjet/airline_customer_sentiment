import re
from underthesea import sent_tokenize

def normalizeSentence(sentence):
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\b\d+[\.,]', '.', sentence)
    sentence = re.sub(r'\b(?:[1-5]\.\d+|[1-5])\b|\b\d+(\.\d+)?(l|ml|cm|m|mm|km|tr|k)\b', '', sentence)
    sentence = re.sub(r'\b\d+\b', '', sentence)

    return sentence.lower()

