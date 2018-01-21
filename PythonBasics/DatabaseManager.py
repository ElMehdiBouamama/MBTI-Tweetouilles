#%% cell 0
import json as js
import numpy as np

def ReadJsonFile(FilePath):
    with open(FilePath, encoding='utf-8') as data_file:
        datas = js.load(data_file)
    return datas

def GetUserIds(datas):
    userIds = []
    for User in datas:
        userId = ""
        for caracter in User:
            userId = userId + caracter
        userIds.append(userId)
    return userIds
        
def GetMbtiOfUser(Datas,User):
    return Datas[User]["mbti"]

def GetConfirmedTweetIdsOfUser(Datas,User):
    return Datas[User]["confirmed_tweet_ids"]

def GetOtherTweetIdsOfUser(Datas,User):
    return Datas[User]["other_tweet_ids"]

def GetUserIdOfUser(Datas,User):
    return Datas[User]["user_id"]

def GetGenderOfUser(Datas,User): 
    return Datas[User]["gender"]

def GetNumberOfUsers(datas):
    return len(datas)

def GetCountArrayOfConfirmedTweet(datas):
    userIds = GetUserIds(datas)
    return [len(GetConfirmedTweetIdsOfUser(datas,userIds[x])) for x in range(GetNumberOfUsers(datas))]

def GetCountArrayOfOtherTweet(datas):
    userIds = GetUserIds(datas)
    return [len(GetOtherTweetIdsOfUser(datas,userIds[x])) for x in range(GetNumberOfUsers(datas))]

def GetCountOfConfirmedTweetOfUser(datas, userId):
    return len(GetConfirmedTweetIdsOfUser(datas,userId))

def GetTotalConfirmedTweetsCount(datas):
    return np.sum(GetCountArrayOfConfirmedTweet(datas))
