#%% cell 0
import sys
sys.path.append('C:/Users/BOUÂMAMAElMehdi/documents/visual studio 2017/Projects/PythonBasics/PythonBasics/')
import numpy as np
from DatabaseManager import *
import matplotlib.pyplot as plt
import collections
import os

ProjectFolder = "C:/Users/BOUÂMAMAElMehdi/Documents/Visual Studio 2017/Projects/PythonBasics/PythonBasics/"
DataFolder = ProjectFolder + "ExtractedTweets/"
datas = ReadJsonFile(ProjectFolder + "TwiSty-FR.json")
userIds = np.loadtxt(ProjectFolder + 'ValidUserIds.txt' ,dtype=np.str)

FilesToTreat = os.listdir(DataFolder)
AllTweets = []
for x in userIds:
    with open(DataFolder + x + ".txt","r",encoding="UTF-8") as f:
        Tweets = f.readlines()
        for Tweet in Tweets:
            AllTweets = AllTweets + Tweet.split(" ")

len(FilesToTreat)


'''
# Building dictionnary of types
MBTI = [GetMbtiOfUser(datas,x) for x in userIds]
MBTITypes = []
#Initialize the dictionnary with 16 keys
typesCount = collections.Counter(MBTI).most_common()
for k in typesCount:
    MBTITypes.append(k[0])
del typesCount
#Initialize the keys with 16 arrays
MbtiDict = dict({MBTITypes[x]:[] for x in range(len(MBTITypes))})
#Fill the keys with respective values
for i,j in enumerate(MBTI):
    MbtiDict[j].append(userIds[i])

z = [len(MbtiDict[MBTITypes[x]])/len(userIds) for x in range(16)]
plt.bar(MBTITypes,z)
plt.show()
'''

len(AllTweets)