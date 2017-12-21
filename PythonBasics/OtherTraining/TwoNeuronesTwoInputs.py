#%% cell 0
import tensorflow as tf

x = tf.placeholder(tf.float32, [None,2])
y = tf.placeholder(tf.float32, [None,2])

W = tf.Variable(tf.random_normal([2,2]), dtype=tf.float32)
B = tf.Variable(tf.random_normal([2]), dtype=tf.float32)

prediction = tf.matmul(x,W) + B
loss = tf.reduce_mean(tf.square(y - prediction))

trainer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

x_train = [[0,0],[1,4],[1,5],[2,5],[1,1],[3,2]]
y_train = [[1,2],[7,11],[8,13],[10,14],[4,5],[9,9]]

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(trainer, {x:x_train , y:y_train})
    print(sess.run([loss,W,B], {x:x_train, y:y_train}))
    print(sess.run(prediction, {x:[[10,10]]}))