import keras.backend as K
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def cov_tf(x_val,y_val):
    x = tf.constant(x_val)
    y = tf.constant(y_val)
    cov = tf.matmul(x, y)
    return cov.eval(session=tf.Session())

def cov_keras(x_val,y_val):
    x = K.constant(x_val)
    y = K.constant(y_val)
    cov = K.dot(x, y)
    return cov.eval(session=tf.Session())

if __name__ == '__main__':
    x = np.array([[4]])
    y = np.array([[5]])
    print(cov_keras(x,y))
    delta = np.abs(cov_tf(x,y) - cov_keras(x,y)).max()
    print('Maximum absolute difference:', delta)
