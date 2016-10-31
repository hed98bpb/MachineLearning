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

    # placeholder for input layer
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # placeholder for outputlayer
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    """
    Densely connected layer
    """
    hidden_size = 300

    W_fc1 = weight_variable([28*28, hidden_size])
    b_fc1 = bias_variable([hidden_size])

    # Multiply by weight matrix, add a bias and apply a ReLU
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # Handles scaling neuron outputs in addition to masking them
    # so dropout just works without any additional scaling
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    """
    Readout Layer
    """
    # Adds another layer
    W_fc2 = weight_variable([hidden_size, 10])
    b_fc2 = bias_variable([10])

    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    """
    Testing and evaluation
    """
    learning_rate = 0.001
    batch_size = 25  # usually between 10 and 30
    nb_of_epoches = 30
    reg_rate = 0.0001

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    loss = tf.reduce_mean(cross_entropy)
    # L_2 weight decay
    reg = reg_rate * tf.reduce_sum(W_fc2 ** 2)
    # Minimization target is the sum of cross-entropy loss and regularization
    target = loss + reg

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(target)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Getting the data. images.shape = (10380, 784) labels.shape = (10380,)
    images, labels = get_data('Data/auTrain.npz')
    images_test, labels_test = get_data('Data/auTest.npz')

    nb_of_images = images.shape[0]


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(nb_of_epoches): #antal epoker
            perm = np.random.permutation(nb_of_images) #shufling the input
            images = images[perm]
            labels = labels[perm]
            for j in range(0, nb_of_images, batch_size):
                start = j
                end = min(start + batch_size, nb_of_images)

                img = images[start : end]
                lab = labels[start : end]

                if j == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: img, y_: lab, keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                train_step.run(feed_dict={x: img, y_: lab, keep_prob: 0.5})

        print("test accuracy %g" % accuracy.eval(feed_dict={
                x: images_test, y_: labels_test, keep_prob: 1.0}))

# initialising weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# initialising biases
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

if __name__ == "__main__":
    main()
