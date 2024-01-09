from glove import Glove
from constants import PATH

glove = Glove.load(PATH + 'gloveModel.model')

print(glove.most_similar('tệ', 100))