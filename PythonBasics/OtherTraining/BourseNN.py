#%% cell 0
import tensorflow as tf
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(tf.random_normal([1]), tf.float32)
b = tf.Variable(tf.random_normal([1]), tf.float32)
prediction = x * w + b
loss = tf.reduce_mean(tf.square(y-prediction))
trainer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(trainer, {x:[[0],[2]], y:[[30],[50]]})
    sess.run([loss], {x:[[0],[2]], y:[[30],[50]]})
