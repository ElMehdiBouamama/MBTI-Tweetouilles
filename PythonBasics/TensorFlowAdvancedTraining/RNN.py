#%% cell 0
import tensorflow as tf

with tf.name_scope('Inputs'):
    x = tf.placeholder(tf.float32, [None,1])
    y = tf.placeholder(tf.float32, [None,1])

with tf.name_scope('Model'):
    result = tf.contrib.rnn.LSTMCell(1)
    print(result.add_variable('krkr',[1,10],tf.float32))
    loss = tf.reduce_mean(tf.square(result.call(x) - y))
    trainer = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08).minimize(loss)

X_Train = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23]]
Y_Train = [[0],[1],[1],[2],[3],[5],[8],[13],[21],[34],[55],[89],[144],[233],[377],[610],[987],[1597],[2584],[4181],[6765],[10946],[17711],[28657]]
X_Test = [[24],[25],[26]]
Y_Test = [[46368],[75025],[121393]]

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(trainer, {x:X_Train, y:Y_Train})
    print(sess.run(result.apply(X_Test)))
