#%% cell 0
import tensorflow as tf
from random import *
#Generating Dataset
MinValue = 0
MaxValue = 1000

def InputBuilder(RangeOfX):
    y = []
    x = []
    for iteration in range(RangeOfX):
        value = randomValueGenerator(MinValue,MaxValue)
        result1 = 7*value*value+3*value+8
        result2 = 3*value*value+2
        y.append([result1, result2])
        x.append([value])
    return y,x

def randomValueGenerator(MinValue, MaxValue):
    return Random().random()*(MaxValue - MinValue) + MinValue

#Global Variable Initialization
log_path = "C:/Tests/FunctionApproximator/Dynamic"
training_steps = 500
Training_Dataset_Size = 4000
Testing_Dataset_Size = 1600
Dataset_Train = InputBuilder(Training_Dataset_Size)
X_Train = Dataset_Train[1]
Y_Train = Dataset_Train[0]
Dataset_Test = InputBuilder(Testing_Dataset_Size)
X_Test = Dataset_Test[1]
Y_Test = Dataset_Test[0]

with tf.name_scope('InputAndOutput'):
    X = tf.placeholder(tf.float32,[None,1])
    Y = tf.placeholder(tf.float32,[None,2])

class NeuralLayer():
    def __init__(self,numberOfInputs,numberOfNeurones):
        self.W = tf.Variable(tf.random_normal([numberOfInputs,numberOfNeurones]),tf.float32)
        self.B = tf.Variable(tf.random_normal([numberOfNeurones]),tf.float32)
    def forwardPass(self, x):
        return tf.nn.selu(tf.matmul(x,self.W) + self.B)

class NeuralNetwork():
    Layers = []
    def __init__(self,numberOfNeuronesBetweenIntermediateLayers, numberOfLayers, numberOfInputs, numberOfOutputs):
        FirstLayer = NeuralLayer(numberOfInputs,numberOfNeuronesBetweenIntermediateLayers)
        self.Layers.append(FirstLayer)
        for i in range(numberOfLayers-2):
            iLayer = NeuralLayer(numberOfNeuronesBetweenIntermediateLayers,numberOfNeuronesBetweenIntermediateLayers)
            self.Layers.append(iLayer)
        LastLayer = NeuralLayer(numberOfNeuronesBetweenIntermediateLayers,numberOfOutputs)
        self.Layers.append(LastLayer)
    def process(self, input):
        result = self.Layers[0].forwardPass(input)
        for i in range(len(self.Layers)-1):
            result = self.Layers[i+1].forwardPass(result)
        return result

with tf.name_scope('NeuralNetwork'):
    FFNN = NeuralNetwork(10,2,1,2)
    Prediction = FFNN.process(X)
with tf.name_scope('LossAndAccuracy'):
    loss = tf.reduce_mean(tf.square(Prediction-Y))

trainer = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08).minimize(loss)

for i in range(len(FFNN.Layers)):
    tf.summary.histogram('W'+ str(i), FFNN.Layers[i].W)
    tf.summary.histogram('B'+ str(i), FFNN.Layers[i].B)
tf.summary.scalar('LossFunction', loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(log_path,tf.get_default_graph())
    summarizer = tf.summary.merge_all()
    for i in range(training_steps):
        _,summary = sess.run([trainer,summarizer], {X:X_Train, Y:Y_Train})
        writer.add_summary(summary,i)
    print(sess.run(loss, {X:X_Test, Y:Y_Test}))
    writer.close()