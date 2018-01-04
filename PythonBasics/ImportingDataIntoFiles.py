#%% cell 0
from multiprocessing import Pool
import sys
sys.path.append('C:/Users/BOUÂMAMAElMehdi/documents/visual studio 2017/Projects/PythonBasics/PythonBasics/MBTI/')
from DatabaseManager import *
from TwitterManager import *
from functools import partial
import os

TweetFolderPath = "M:/Datasets/TwitterMBTI/Parallel/"
datas = ReadJsonFile("M:/Datasets/TwitterMBTI/MBTINotExtracted/twisty-2016-03/TwiSty-FR.json")
userIds = GetUserIds(datas)
os.chdir(TweetFolderPath)
# We need to build a dictionnary of users / tweets first

for x in userIds[40:]:
    print("Processing user : " + x)
    stringToSave = ""
    with Pool(100) as p:
        UserTweets = GetConfirmedTweetIdsOfUser(datas,x)
        checker = GetPostFromTwitter(UserTweets[0],x)
        print(checker)
        if(checker != ""):
            GetPostPartial = partial(GetPostFromTwitter, userId=x)
            result =  p.map(GetPostPartial,UserTweets)
        else:
            result = ""
    with open(x + ".txt","w+", encoding="utf-8") as file:
        file.writelines(result)

