import sys
sys.path.append('C:/Users/BOUÂMAMAElMehdi/documents/visual studio 2017/Projects/PythonBasics/PythonBasics/MBTI/')
import numpy as np
import threading
import tensorflow as tf
from pprint import pprint
from DatabaseManager import *
from TwitterManager import *
import matplotlib.pyplot as plt
import collections

DataFolder = "M:/Datasets/TwitterMBTI/Parallel"
datas = ReadJsonFile("M:/Datasets/TwitterMBTI/MBTINotExtracted/twisty-2016-03/TwiSty-FR.json")
userIds = GetUserIds(datas)

MBTI = [GetMbtiOfUser(datas,x) for x in userIds]
typesCount = collections.Counter(MBTI).most_common()
MBTITypes = []
for k in typesCount:
    MBTITypes.append(k[0])

MbtiDict = dict({MBTITypes[x]:[] for x in range(len(MBTITypes))})
for i,j in enumerate(MBTI):
    MbtiDict[j].append(userIds[i])
#Dictionnary of Types / UsersIds done
MbtiDict[MBTITypes[0]]


