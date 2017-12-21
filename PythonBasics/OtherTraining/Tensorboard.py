#%% cell 0
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 3])
y = tf.placeholder(tf.float32, [None, 1])

with tf.name_scope('hidden') as scope:
    w = tf.Variable(tf.random_normal([3]), name="weights")
    b = tf.Variable(tf.random_normal([1]), name="biases")

prediction = w*x + b

loss = tf.reduce_mean(tf.square(y-prediction))
trainer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
tf.summary.histogram("loss", loss)

x_train = [[1,4,3],[2,4,9],[2,3,1],[4,4,7],[6,2,2],[1,7,6]]
y_train = [[13],[26],[9],[21],[14],[22]]

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./test")
    summaries = tf.summary.merge_all()
    for i in range(10000):
        sess.run(trainer, {x:x_train, y:y_train})
        summ = sess.run(summaries,{x:x_train, y:y_train})
        writer.add_summary(summ, global_step=i)
    print(sess.run(loss, {x:x_train, y:y_train}))
    