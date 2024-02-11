from glove import Glove
from constants import PATH, MODEL, GLOVE_MODEL

glove = Glove.load(PATH + MODEL + GLOVE_MODEL)

print(glove.most_similar('máy_bay', 100))
print(glove.most_similar('tuyệt_vời', 100))
print(glove.most_similar('kinh_khủng', 100))