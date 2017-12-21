#%% cell 0
import tensorflow as tf

x = tf.placeholder(tf.float32, [None,3])
y = tf.placeholder(tf.float32, [None,3])

W = tf.Variable(tf.random_normal([3,3]))
B = tf.Variable(tf.random_normal([3]))

prediction = tf.matmul(x,W) + B


loss = tf.reduce_mean(tf.square(y-prediction))
trainer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

x_train = [[1,4,3],[2,4,9],[2,3,1],[4,4,7],[6,2,2],[1,7,6],[2,2,3],[3,4,9],[10,12,1],[1,1,0],[0,5,7],[1,0,3]]
y_train = [[13,21,16],[26,29,29],[9,18,11],[21,31,27],[14,24,15],[22,33,28],[12,17,13],[27,31,30],[26,61,37],[4,9,4],[21,26,25],[9,9,8]]

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(trainer, {x:x_train , y:y_train})
    
    print(sess.run(loss,  {x:x_train , y:y_train}))
    print(sess.run(prediction, {x:[[1,1,1]]}))
