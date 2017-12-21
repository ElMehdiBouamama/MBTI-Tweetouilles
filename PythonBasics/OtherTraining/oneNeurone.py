#%% cell 0
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(),dtype= tf.float32)
B = tf.Variable(np.random.randn(), dtype= tf.float32)

prediction = x * W + B

Loss = tf.reduce_mean(tf.square(y - prediction))
trainer = tf.train.GradientDescentOptimizer(0.1).minimize(Loss)

x_train = [0,1,2,3,4,5]
y_train = [2,7,12,17,22,27]

x_test = [6,7,8,9,10]
y_test = [32,37,42,47,52]

session = tf.Session()
session.run(tf.global_variables_initializer())
for i in range(10000):
    session.run(trainer, {x:x_train, y:y_train})
    
print(session.run(Loss, {x:x_train, y:y_train}))

''' Testing'''
print(session.run(Loss, {x:x_test , y:y_test}))
print(session.run([prediction,W,B], {x:[6]}))

