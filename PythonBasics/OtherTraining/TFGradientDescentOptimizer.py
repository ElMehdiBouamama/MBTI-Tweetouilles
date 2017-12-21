#%% cell 0
import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
print(node3)

with tf.Session() as sess :
    sess.run(node3)

#%% cell 1
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
with tf.Session() as sess :
    print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node, {a: [1,3], b: [4,5]}))

#%% cell 2
add_and_triple = adder_node * 3
with tf.Session() as sess :
    print(sess.run(add_and_triple, {a:[1,2], b:[5,1]}))

#%% cell 3
w = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = x * w + b

y = tf.placeholder(tf.float32)
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)
with tf.Session() as sess :
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
with tf.Session() as sess :
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

    print(sess.run([w,b]))