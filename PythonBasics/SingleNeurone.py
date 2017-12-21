#%% cell 0
import tensorflow as tf

with tf.name_scope('Inputs'):
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

with tf.name_scope('Neurone'):
    w = tf.Variable(tf.random_normal([1]), tf.float32)
    b = tf.Variable(tf.random_normal([1]), tf.float32)

with tf.name_scope('Prediction'):
    prediction = x * w + b

with tf.name_scope('Optimiser'):
    loss = tf.reduce_mean(tf.square(y-prediction))
    trainer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

x_train = [[1],[2]]
y_train = [[5],[8]]

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("c:/test/singleNeurone", graph=tf.get_default_graph())
    for i in range(10000):
        _,summary = sess.run([trainer, summary_op], {x:x_train , y:y_train})
        writer.add_summary(summary)
    
