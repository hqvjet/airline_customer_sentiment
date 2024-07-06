import tensorflow as tf

def checkForGPU():
    if tf.test.is_gpu_available():
        print("USING GPU....................................")
    else:
        print("USING CPU....................................")

if __name__ == '__main__':
    checkForGPU()
