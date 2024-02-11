from glove import Glove
from constants import PATH

glove = Glove.load(PATH + 'gloveModel200.model')

print(glove.most_similar('máy bay', 100))
print(glove.most_similar('tuyệt vời', 100))
print(glove.most_similar('kinh khủng', 100))