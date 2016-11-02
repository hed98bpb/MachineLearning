from Handin2.util import get_data_nn
import tensorflow as tf
import numpy as np
import math as math

"""
To implement this we have used the guide for building a softmax regression model first as in handin 1
at https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html
"""

def main():

    best_hidden_size = 0
    best_learning_rate = 0
    best_batch_size = 0
    best_nb_of_epoches = 0
    best_reg_rate = 0
    best_acc = 0

    for hidden_size in np.arange(400, 1201, 400):
        for learning_rate in np.arange(0.0005, 0.0016, 0.0005):
            for batch_size in np.arange(20, 41, 10):
                for nb_of_epoches in np.arange(10, 21, 10):
                    for reg_rate in np.arange(0.00001, 0.001, 0.0001):

                        # placeholder for input layer
                        x = tf.placeholder(tf.float32, shape=[None, 784])

                        # placeholder for outputlayer
                        y_ = tf.placeholder(tf.float32, shape=[None, 10])

                        """
                        Densely connected layer

                        hidden_size = 800
                        """

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

                        learning_rate = 0.001
                        batch_size = 25  # usually between 10 and 30
                        nb_of_epoches = 10
                        reg_rate = 0.0001
                        """

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
                        images, labels = get_data_nn('Data/auTrain.npz')
                        images_test, labels_test = get_data_nn('Data/auTest.npz')

                        nb_of_images = images.shape[0]
                        cv = 3
                        validation_size = math.floor(nb_of_images/10)

                        mean_of_val_acc = []
                        mean_of_test_acc = []


                        for k in range(cv):  # for k-fold cross validation where we can independently choose how large each test set is and how many trials you average over
                            perm1 = np.random.permutation(nb_of_images)  # shufling the input
                            images = images[perm1]
                            labels = labels[perm1]

                            images_training = images[validation_size: nb_of_images]
                            labels_training = labels[validation_size: nb_of_images]

                            images_validation = images[0: validation_size]
                            labels_validation = labels[0: validation_size]

                            with tf.Session() as sess:
                                sess.run(tf.initialize_all_variables())

                                for i in range(nb_of_epoches): #antal epoker
                                    size_of_training_data = images_training.shape[0]
                                    perm = np.random.permutation(size_of_training_data) #shufling the input
                                    images_training = images_training[perm]
                                    labels_training = labels_training[perm]

                                    for j in range(0, size_of_training_data, batch_size):
                                        start = j
                                        end = min(start + batch_size, size_of_training_data)

                                        img = images_training[start : end]
                                        lab = labels_training[start : end]

                                        if j == 0:
                                            train_accuracy = accuracy.eval(feed_dict={x: img, y_: lab, keep_prob: 1.0})
                                            # print("step %d, training accuracy %g" % (i, train_accuracy))
                                        train_step.run(feed_dict={x: img, y_: lab, keep_prob: 0.5})

                                in_sample_acc = sess.run(accuracy, feed_dict={x: images_validation, y_: labels_validation, keep_prob: 1.0})
                                # print('validation accuracy:', in_sample_acc)
                                mean_of_val_acc.append(in_sample_acc)

                                out_of_sample_acc = sess.run(accuracy, feed_dict={x: images_test, y_: labels_test, keep_prob: 1.0})
                                # print('out of sample accuracy:', out_of_sample_acc)
                                mean_of_test_acc.append(out_of_sample_acc)

                        print("Batch", best_batch_size)
                        print("Hidden size", best_hidden_size)
                        print("Learning rate", best_learning_rate)
                        print("Nb of epoches", best_nb_of_epoches)
                        print("Reg rate", best_reg_rate)

                        print('\nMEANS of %d runs: ' %(cv))
                        print('validation accuracy: ', np.mean(mean_of_val_acc))
                        print('out of sample accuracy: ',np.mean(mean_of_test_acc))

                        if best_acc < np.mean(mean_of_val_acc):
                            best_acc = np.mean(mean_of_val_acc)
                            best_batch_size = batch_size
                            best_hidden_size = hidden_size
                            best_learning_rate = learning_rate
                            best_nb_of_epoches = nb_of_epoches
                            best_reg_rate = reg_rate

    print("All time best batch", best_batch_size)
    print("All time best hidden size", best_hidden_size)
    print("All time best learning rate", best_learning_rate)
    print("All time best nb of epoches", best_nb_of_epoches)
    print("All time best reg rate", best_reg_rate)

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
