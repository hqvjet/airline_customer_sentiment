import re
from phobert_we.constants import PATH
import pandas as pd

stopword = pd.read_csv(PATH + 'stopword.csv')
stopword = [word for word in stopword['stopword']]

def normalizeSentence(sentence):
    RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    sentence = RE_EMOJI.sub(r'', sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub(r'\b\d+[\.,/]\s*', '', sentence)

    for word in stopword:
        sentence = sentence.replace(' ' + word + ' ', ' ')
    
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.lower().strip()

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
