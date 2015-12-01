import tensorflow as tf
import numpy as np


def build_graph():
    dim = 784
    x = tf.placeholder("float", shape=[None, dim])
    y_ = tf.placeholder("float", shape=[None, 10])

    W = tf.Variable(tf.zeros([dim, 10]))
    b = tf.Variable(tf.zeros([10]))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main():
    build_graph()

if __name__ == '__main__':
    main()