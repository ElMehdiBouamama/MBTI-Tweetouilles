#%% cell 0
import tensorflow as tf
#Global Variables
log_path = "C:/Tests/Fibonnaci/InitializedExpanded2"
training_steps = 100000
with tf.name_scope('Inputs'):
    X = tf.placeholder(tf.float32, [None,1])
    Y = tf.placeholder(tf.float32, [None,1])
with tf.name_scope('Variables'):
    Wh_1 = tf.Variable(tf.random_normal([1,20],mean=0.75,stddev=0.25), tf.float32)
    Bh_1 = tf.Variable(tf.random_normal([20],mean=-7,stddev=2), tf.float32)
    Wh_2 = tf.Variable(tf.random_normal([20,20],mean=2,stddev=1), tf.float32)
    Bh_2 = tf.Variable(tf.random_normal([20],mean=-5,stddev=1), tf.float32)
    Wf_2 = tf.Variable(tf.random_normal([20,1],mean=6,stddev=1), tf.float32)
    Bf_2 = tf.Variable(tf.random_normal([1],mean=-4,stddev=1), tf.float32)
with tf.name_scope('Model'):
    h_1 = tf.nn.elu(tf.cast(tf.matmul(X,Wh_1),tf.float32) + Bh_1)
    h_2 = tf.nn.elu(tf.cast(tf.matmul(h_1,Wh_2),tf.float32) + Bh_2)
    Y_ = tf.nn.elu(tf.cast(tf.matmul(h_2,Wf_2),tf.float32)+ Bf_2)
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.square(Y-Y_))
with tf.name_scope('Optimizer'):
    trainer = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08).minimize(loss)
#Tensorboard
tf.summary.histogram('H1W', Wh_1)
tf.summary.histogram('H1B', Bh_1)
tf.summary.histogram('H2W', Wh_2)
tf.summary.histogram('H2B', Bh_2)
tf.summary.histogram('HfW', Wf_2)
tf.summary.histogram('HfB', Bf_2)
tf.summary.scalar('Loss', loss)
#Dataset
X_Train = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23]]
Y_Train = [[0],[1],[1],[2],[3],[5],[8],[13],[21],[34],[55],[89],[144],[233],[377],[610],[987],[1597],[2584],[4181],[6765],[10946],[17711],[28657]]
X_Test = [[24],[25],[26]]
Y_Test = [[46368],[75025],[121393]]
#GPU handling
gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.allow_growth = True
with tf.Session(config=gpuconfig) as sess :
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(log_path,tf.get_default_graph())
    summarizer = tf.summary.merge_all()
    for step in range(training_steps):
        _,summary = sess.run([trainer,summarizer], {X:X_Train , Y:Y_Train})
        writer.add_summary(summary,step)
    print(sess.run(loss, {X:X_Train , Y:Y_Train}))
    print(sess.run(loss, {X:X_Test , Y:Y_Test}))
    print(sess.run(Y_, {X:X_Test}))
    writer.close()