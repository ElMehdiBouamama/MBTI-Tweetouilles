#%% cell 0
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None])
W1 = tf.Variable(np.random.randn())
B1 = tf.Variable(np.random.randn())
W2 = tf.Variable(np.random.randn())
B2 = tf.Variable(np.random.randn())
W3 = tf.Variable(np.random.randn())
B3 = tf.Variable(np.random.randn())
W4 = tf.Variable(np.random.randn())
B4 = tf.Variable(np.random.randn())
W5 = tf.Variable(np.random.randn())
B5 = tf.Variable(np.random.randn())

result1 = tf.nn.relu(x * W1 + B1)
result2 = tf.nn.relu(result1 * W2 + B2)
result3 = tf.nn.relu(result2 * W3 + B3)
result4 = tf.nn.relu(result3 * W4 + B4)
prediction = tf.nn.relu(result4 * W5 + B5)

y = tf.placeholder(tf.float32, [None])
Loss = tf.reduce_mean(tf.square(y-prediction))

x_train = [0,1,2,3,4,5,6,7,8,9,10]
y_train = [2,6,40,254,1039,3142,7796,16830,32794,59078,100032]

Train = tf.train.GradientDescentOptimizer(0.1)
training = Train.minimize(Loss)

init = tf.global_variables_initializer()
with tf.Session() as sess1 :
    sess1.run(init)
    print(sess1.run(Loss , {x:x_train, y:y_train}))
    for i in range(1000000):
        sess1.run(training, {x:x_train , y:y_train})
        
    print(sess1.run([Loss,W5,B5] , {x:x_train, y:y_train}))