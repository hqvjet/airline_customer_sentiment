from loadModel import getModel
from appService import getRatingFromModel

print('LOADING MODEL..........................')
model = getModel()
print('PREDICTING.............................')
a = getRatingFromModel('Trên cả tuyệt vời', 'Khách sạn này thực sự quá là tuyệt vời, trên cả mong đợi')
print(a)
print('Donee.........................')
