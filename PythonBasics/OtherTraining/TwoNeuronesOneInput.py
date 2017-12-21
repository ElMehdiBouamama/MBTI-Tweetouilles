#%% cell 0
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,2])

W = tf.Variable(tf.random_normal([1,2]), dtype=tf.float32)
B = tf.Variable(tf.random_normal([1,2]), dtype=tf.float32)

prediction = tf.matmul(x, W) + B
loss = tf.reduce_mean(tf.square(y-prediction))
trainer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

TrainData = {x:[[0],[1],[2],[3],[4]] ,y:[[3,6],[5,7],[7,8],[9,9],[11,10]]}

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10000):
    sess.run(trainer, TrainData)

print(sess.run(loss, TrainData))
''' Testing '''
TestData = {x:[[6],[7],[8],[9],[10]] ,y:[[13,11],[15,12],[17,13],[19,14],[21,15]]}
print(sess.run([loss, W, B], TestData))

print(sess.run(prediction, {x:[[11]]}))

sess.close()
