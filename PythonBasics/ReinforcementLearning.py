#%% cell 0
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

width = 5
height = 4
LearningRate = 0.1
Lambda = 0.9

S = np.zeros([width,height])
A = 
A = np.random.rand([4,width,height])
R = np.zeros([width,height])

R[1,0] = R[1,1] = R[1,2] = R[3,1] = R[3,2] = R[4,3] = -50
R[4,1] = 100

iterator = (0,0)
nextIterator = (0,0)

def T():
    S[iterator] = S[iterator] + LearningRate * (R[iterator] + Lambda*(argmax()- V[iterator]))
    iterator = nextIterator
    if(iterator == (4,1)):
        iterator = (0,0)
        print(S)


def argmax():
   np.max(np.multiply(A[iterator,:], S[nextIterator]))


for i in range(5000):
    T()
    
