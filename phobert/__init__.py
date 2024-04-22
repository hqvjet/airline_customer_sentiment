import tensorflow as tf
from sentimentModel import startLearning

def checkForGPU():
    if tf.test.is_gpu_available():
        print("USING GPU....................................")
    else:
        print("USING CPU....................................")

if __name__ == '__main__':
    checkForGPU()
    startLearning()
