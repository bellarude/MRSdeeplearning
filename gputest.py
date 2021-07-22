# source for improvements: https://www.tensorflow.org/guide/gpu
# like how to assign run onto specific gpu or how to do parallelism and use both together


from tensorflow.python.client import device_lib
from timeit import default_timer as timer

print(device_lib.list_local_devices())

import tensorflow as tf
tf.test.gpu_device_name()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



# Create some tensors
def func():
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)


if __name__ == "__main__":
    start = timer()
    func()
    print("without GPU:", timer() - start)

    tf.debugging.set_log_device_placement(True)
    start = timer()
    func()
    print("wit GPU:", timer() - start)
