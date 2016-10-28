from Handin2.util import get_data
from Handin2.helper_function import plot_perf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

"""
To implement this we have used the guide for building a softmax regression model first as in handin 1
at https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html
"""

def main():
    # getting the data as mnist --> splits the training-data into a training part and a testing part.
    mnist = input_data.read_data_sets('Data/auTrain', one_hot=True)

    # Getting the data. images.shape = (10380, 784) labels.shape = (10380,)
    images, labels = get_data('Data/auTrain.npz')
    images_test, labels_test = get_data('Data/auTest.npz')

    # creating session
    sess = tf.InteractiveSession()

    # placeholder for input layer
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # placeholder for outputlayer
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # variables: weights W and biases b
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    # running the session
    sess.run(tf.initialize_all_variables())

    # implementation of regression model
    y = tf.matmul(x,W) + b

    # a loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

    # combines regression model and the loss function so we can train data in steps
    # applies the gradient descent updates to the parameters
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # training
    for i in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # evaluate model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    """
    Now to some neural networks:
    First convoutional layer
    """

    # computes 32 features for each 5xt patch
    # weight_variable params :
    # (dim of patch size, dim of patch size, number of input channels, number of output channels)
    W_conv1 = weight_variable([5, 5, 1, 32])

    # bias vector for each output channel, in this case 1
    b_conv1 = bias_variable([32])

    # reshape x to 4d tensor. 2nd and 3rd param = img widt and height, 4th param = num of color channels
    x_image = tf.reshape(x, [-1,28,28,1])

    # Convolve x_imgage with the weight tensor and adds the bias with ReLU function
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    """
    Second convolutional layer
    """

    # Just stacks another layer on top of the previous one
    # 64 features for each 5x5 patch
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    """
    Densely connected layer
    """

    # adds layer of 1024 fully connected neurons
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # reshapes tensor from pooling layer to a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Multiply by weight matrix, add a bias and apply a ReLU
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Handles scaling neuron outputs in addition to masking them
    # so dropout just works without any additional scaling
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    """
    Readout Layer
    """

    # Adds another layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    """
    Testing and evaluation
    """

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# initialising weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# initialising biases
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# makes outputs the same size as input
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Checks for the max value of 4 pixels in a 2x2 grid
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

if __name__ == "__main__":
    main()
