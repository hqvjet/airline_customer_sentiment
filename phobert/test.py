import pandas as pd
from constants import *
from Nomarlize import normalizeSentence

file = pd.read_csv(PATH + 'data.csv')['Content']

sum_len = sum([len(j) for i in file for j in normalizeSentence(i).split()])

print(sum_len / len(file))
