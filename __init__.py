import tensorflow as tf
from sentimentModel import startLearning

def checkForGPU():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

if __name__ == '__main__':
    checkForGPU()
    startLearning()
