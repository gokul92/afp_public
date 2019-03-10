import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from functools import reduce
from operator import mul
import sys
import os
import copy
import get_summ_stats


def network(x, n_outputs, dropout = True, keep_prob = 0.5):

    n_hidden1 = 50
    n_hidden2 = 50
    n_hidden3 = 50
    n_hidden4 = 50
    n_hidden5 = 50
    n_hidden6 = 50
    n_hidden7 = 50
    n_hidden8 = 50

    # construct the DNN - with batch normalization
    hidden1 = tf.layers.dense(x[0], n_hidden1, name="hidden1", activation=tf.nn.relu)


    if dropout == True:

        h1_do = tf.nn.dropout(hidden1, keep_prob, name = "hidden1_dropout")
        hidden2 = tf.layers.dense(h1_do, n_hidden2, name="hidden2", activation=tf.nn.relu)

        h2_do = tf.nn.dropout(hidden2, keep_prob, name = "hidden2_dropout")
        hidden3 = tf.layers.dense(h2_do, n_hidden3, name="hidden3", activation=tf.nn.relu)

        h3_do = tf.nn.dropout(hidden3, keep_prob, name = "hidden3_dropout")
        hidden4 = tf.layers.dense(h3_do, n_hidden4, name="hidden4", activation=tf.nn.relu)

        h4_do = tf.nn.dropout(hidden4, keep_prob, name = "hidden4_dropout")
        hidden5 = tf.layers.dense(h4_do, n_hidden5, name="hidden5", activation=tf.nn.relu)

        h5_do = tf.nn.dropout(hidden5, keep_prob, name = "hidden5_dropout")
        hidden6 = tf.layers.dense(h5_do, n_hidden6, name="hidden6", activation=tf.nn.relu)

        h6_do = tf.nn.dropout(hidden6, keep_prob, name = "hidden6_dropout")
        hidden7 = tf.layers.dense(h6_do, n_hidden6, name="hidden7", activation=tf.nn.relu)

        h7_do = tf.nn.dropout(hidden7, keep_prob, name = "hidden7_dropout")
        hidden8 = tf.layers.dense(h7_do, n_hidden8, name="hidden8", activation=tf.nn.relu)

        h8_do = tf.nn.dropout(hidden8, keep_prob, name = "hidden8_dropout")
        logits = tf.layers.dense(h8_do, n_outputs, name="outputs")

    elif dropout == False:

        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.relu)
        hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4", activation=tf.nn.relu)
        hidden5 = tf.layers.dense(hidden4, n_hidden4, name="hidden5", activation=tf.nn.relu)
        hidden6 = tf.layers.dense(hidden5, n_hidden4, name="hidden6", activation=tf.nn.relu)
        hidden7 = tf.layers.dense(hidden6, n_hidden4, name="hidden7", activation=tf.nn.relu)
        hidden8 = tf.layers.dense(hidden7, n_hidden4, name="hidden8", activation=tf.nn.relu)

        logits = tf.layers.dense(hidden8, n_outputs, name="outputs")

    # Cross entropy cost function for binary classification
    unweighted_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=x[1], logits=logits, name = "unweighted_xentropy")
    loss = tf.reduce_mean(unweighted_xentropy, name="loss")

    return loss, logits


def iterator_creator(x, y, batch_size, create_iter = True):
	dx = tf.data.Dataset.from_tensor_slices(np.float32(x.values))
    dy = tf.data.Dataset.from_tensor_slices(np.float32(y))
    dcomb = tf.data.Dataset.zip((dx, dy)).repeat().batch(batch_size)

    iter = tf.data.Iterator.from_structure(dcomb.output_types, dcomb.output_shapes)

    if create_iter == True:
	    next_el
	else:
	    next_el = iter.get_next(name="next_el")

	return iter


def traindnn(x_train, y_train, x_validation, y_validation, N_epochs=5000, starter_learning_rate=0.01, batch_size=128,
             stat_step=50, saver_step=500, directory="C:/Users/b/AFP/", restart_flag=0,
             start_epoch=0, epoch_decay=500, display_step=20, meta_count=2, N_classes = 2, dropout_val = 0.5):

    cpu_device = "/device:CPU:0"
    gpu_device = "/device:GPU:0"

    N_features = x_train.shape[1]
    N_samples = x_train.shape[0]
    N_batches = int(len(x_train) / batch_size)
    N_batches_validation = int(len(x_validation) / batch_size)
    n_outputs = N_classes

    keep_prob = tf.placeholder(tf.float32)

    iter = iterator_creator(x_train, y_train, batch_size, create_iter = True)
    next_el = iter.get_next(name="next_el")
    training_init_op = iter.make_initializer(next_el, name="training_init_op")

    # create tf Dataset iterators from numpy arrays - validation set
    iter_valid = iterator_creator(x_validation, y_validation, batch_size, create_iter = True)
    validation_init_op = iter.make_initializer(iter_valid, name="validation_init_op")

    loss, logits = network(next_el, n_outputs, dropout=True, keep_prob = keep_prob)

    prediction = tf.nn.softmax(logits, name="prediction")
    actual_y = next_el[1]

    # Exponential Adaptive Learning Rate Schedule
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               N_batches*epoch_decay, 0.98, staircase=True, name="learning_rate")
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, name="optimizer")
    training_op = optimizer.minimize(loss, global_step=global_step, name="training_op")

    if restart_flag == 1:
        pass
    else:
        init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=None, name="saver")

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        if restart_flag == 1:
            saver.restore(sess, tf.train.latest_checkpoint(directory + 'saver/'))
            print("Model Restored")
        else:
            start_epoch = 0
            sess.run(init)

        # define containers to store average cost function/ ys per epoch
        epoch_cost_array = [0] * (N_epochs - start_epoch)
        epoch_cost_validation_array = [0] * (N_epochs - start_epoch)

        res_mat_train_array = [[0] * 4] * (int((N_epochs - start_epoch) / stat_step) + 10)
        res_mat_validation_array = [[0] * 4] * (int((N_epochs - start_epoch) / stat_step) + 10)

        act_y_train_epoch_level = np.zeros(shape=(N_batches, batch_size, N_classes))
        act_y_validation_epoch_level = np.zeros(shape=(N_batches_validation, batch_size, N_classes))
        pred_y_train_epoch_level = np.zeros(shape=(N_batches, batch_size, N_classes))
        pred_y_validation_epoch_level = np.zeros(shape=(N_batches_validation, batch_size, N_classes))

        for i in range(start_epoch, N_epochs):
            if i % display_step == 0:
                print('epoch: ' + str(i))
            batch_cost_epoch = 0.0
            batch_cost_epoch_validation = 0.0
            sess.run(training_init_op)
            if (i % stat_step) != 0 or i == 0:
                for j in range(N_batches):
                    _, batch_cost = sess.run([training_op, loss], feed_dict={keep_prob : dropout_val})
             
    return 0
