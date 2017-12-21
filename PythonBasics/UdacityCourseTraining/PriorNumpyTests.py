#%% cell 0
import numpy as np

scores = np.array([[1, 2, 3, 6],[2, 4, 5, 6],[3, 8, 7, 6]])
#scores = [1.0, 2.0, 3.0]

def createOneSizedMatrixOfSize(size):
    return np.matrix(np.ones([len(size)])).T

def softmax(x):
    score = np.matrix(x)
    TEscores = np.exp(score.T)
    scoreOneSizedMatrix = createOneSizedMatrixOfSize(score)
    if(len(score)==1):
        SumSoftMax = np.reciprocal(np.dot(TEscores.T,createOneSizedMatrixOfSize(TEscores)))
    else:
        SumSoftMax = np.reciprocal(np.dot(TEscores,scoreOneSizedMatrix))
    EnlargedSumSoftMax = np.dot(scoreOneSizedMatrix,SumSoftMax.T).T
    result = np.multiply(TEscores,EnlargedSumSoftMax).T
    return result

def softmax2(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)/10

print(softmax2(scores))

x = 1000000000
y = 0.000001

z = x+y
z = z-x
print(z)