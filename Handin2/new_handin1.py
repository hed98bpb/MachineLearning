import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt




#First, we download the AU digits data set if it doesn't already exist.
from Handin2.helper_function import plot_perf

if not os.path.exists('Data/auTrain.npz'):
    os.system('wget https://users-cs.au.dk/rav/ml/auDigits/auTrain.npz')
if not os.path.exists('Data/auTest.npz'):
    os.system('wget https://users-cs.au.dk/rav/ml/auDigits/auTest.npz')

#Next, we load the training data.
auTrain = np.load('Data/auTrain.npz')
auTrain_images = auTrain['digits']
auTrain_labels = auTrain['labels']
auTest = np.load('Data/auTest.npz')
auTest_images = auTest['digits']
auTest_labels = auTest['labels']

d = auTrain_images.shape[1]

#For logistic regression we will try to separate the 2's from the 7's in the data set.
auTrain_label_is_27 = (auTrain_labels == 2) | (auTrain_labels == 7)
auTrain_labels27 = (auTrain_labels[auTrain_label_is_27] == 7).astype(np.int32)
auTrain_images27 = auTrain_images[auTrain_label_is_27]

auTest_label_is_27 = (auTest_labels == 2) | (auTest_labels == 7)
auTest_labels27 = (auTest_labels[auTest_label_is_27] == 7).astype(np.int32)
auTest_images27 = auTest_images[auTest_label_is_27]

#The following helper function returns a random permutation of the 2-7-data set.
def permute_data_27():
    # Return a random permutation of the training data
    perm = np.random.permutation(len(auTrain_labels27))
    return auTrain_images27[perm], auTrain_labels27[perm]

#Here we set the number of epochs (iterations through the data set when training) and the batch size for mini-batch gradient descent.
epochs = 10
batch_size = 16

#tensorflow:
with tf.Graph().as_default():
    # 1. The input to the computation is a data matrix (N x d) and label array (length N)
    data = tf.placeholder(tf.float32, shape=[None, d])
    labels = tf.placeholder(tf.int32, shape=[None])
    # The learning rate used in gradient descent
    learning_rate = tf.placeholder(tf.float32)

    # 2. Here we represent the bias term as a separate variable
    weights = tf.Variable(tf.zeros([d, 1]))
    bias = tf.Variable(tf.zeros([]))

    # 3. Our simple linear model is just sigmoid(X * w + b)
    logits = tf.matmul(data, weights) + bias
    # Reshape from column vector (N x 1) to array (length N)
    logits = tf.reshape(logits, [-1])

    # 4. The prediction is correct whenever the label is equal to "logit > 0".
    correct_prediction = tf.equal(
        labels,
        tf.cast(logits > 0, labels.dtype))
    # The accuracy of our classifier is the mean correct prediction.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 5. The built-in sigmoid_cross_entropy_with_logits applies sigmoid
    # and computes the cross entropy.
    # The labels we give to this function must be a float array, not an int array.
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits, tf.cast(labels, tf.float32))

    # The function we want to minimize in gradient descent is just the
    # mean cross entropy.
    loss = tf.reduce_mean(cross_entropy)

    # 6. The gradient descent optimizer we will use is the ADAM Optimizer.
    # There are many other implementations of gradient descent, but ADAM works well.
    opt = tf.train.AdamOptimizer(learning_rate)
    # train_step is an operation we can run in a session to perform one step of training.
    train_step = opt.minimize(loss)

    #fig, f = plot_perf(['E_in'], xlim=(0, epochs - 1), ylim=(0, 1), figsize=(6, 4))

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        for epoch in range(epochs):
            data_value, labels_value = permute_data_27()

            # Run train_step in mini-batches
            for i in range(0, len(auTrain_labels27), batch_size):
                j = min(len(auTrain_labels27), i + batch_size)
                session.run(
                    train_step,
                    feed_dict={data: data_value[i:j],
                               labels: labels_value[i:j],
                               learning_rate: 1e-4})

            # Evaluate on all of training data
            train_accuracy = session.run(
                accuracy,
                feed_dict={data: auTrain_images27,
                           labels: auTrain_labels27})
            in_sample_error = 1 - train_accuracy
            #f(in_sample_error)

        test_accuracy = session.run(
            accuracy,
            feed_dict={data: auTest_images27,
                       labels: auTest_labels27})
        test_error = 1 - test_accuracy
        print("%5.2f%% errors on test set" % (100 * test_error))

        # Extract and plot weight vector
        weight_vector = session.run(weights).reshape(28, 28)
        plt.imshow(weight_vector, cmap='bone')

