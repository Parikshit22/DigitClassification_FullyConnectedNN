# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:57:05 2019

@author: MUJ
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
x = tf.placeholder(tf.float32,shape = [None,784])
y_true = tf.placeholder(tf.float32,shape = [None,10])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,w)+b
 
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true,logits = y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = .01)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches = 500
    for i in range(batches):
        batch_x,batch_y = mnist.train.images,mnist.train.labels
        feed = {x:batch_x,y_true:batch_y}
        sess.run(train,feed_dict=feed)
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_true,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))