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
"""
# initialising weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# initialising biases
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
