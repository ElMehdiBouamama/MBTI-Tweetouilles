import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2D(x,W):
    return tf.nn.conv2d(x,W,strides=[-1,28,28,1])

def pool2D(x):
    return tf.nn.pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
conv1 = tf.nn.relu(conv2D(x,W_conv1) + b_conv1)
pool1 = pool2D(conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
conv2 = tf.nn.relu(conv2D(pool1,W_conv2) + b_conv2)
pool2 = pool2D(conv2)

W_fc1 = weight_variable([7*7*64,1024])
B_fc1 = bias_variable([1024])
pool2_flaten = tf.reshape(pool2,[-1,7*7*64])
fc1 = tf.nn.relu(tf.matmul(pool2_flaten,W_fc1) + B_fc1)

keep_prob = tf.placeholder(tf.float32)
fc1_dropout = tf.nn.dropout(fc1, keep_prob)
W_fc2 = weight_variable([1024,10])
B_fc2 = bias_variable([10])
y_conv = tf.matmul(fc1_dropout,W_fc2) + B_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
